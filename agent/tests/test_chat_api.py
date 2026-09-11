"""POST /chat: the key, the request, the stream, and what a closed tab does to a run.

The graph is the real one (graph.py) around fakes: a scripted planner, a scripted search and a
scripted chat model whose tokens go through LangChain's real callbacks, so LangGraph's
"messages" mode sees them as it would see ChatOpenAI's.

Three kinds of test:
  - in memory over httpx.ASGITransport: the key, validation, headers, what reaches the graph;
  - golden streams: the exact bytes /chat sends for six scenarios, frozen in
    tests/golden/chat-stream/. The Vitest contract test (tests/python-stream-contract.test.ts)
    feeds the same bytes to the real AI SDK client and compares the message it builds with the
    one the TypeScript route produces for the same scenario. Rewrite them with
        UPDATE_GOLDEN=1 uv run pytest tests/test_chat_api.py
    and commit them: this test fails while the files and the code disagree;
  - a real uvicorn on a socket: ASGITransport buffers the whole response before returning it,
    so it cannot show that tokens stream, or what a disconnect does.
"""

import asyncio
import json
import os
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, Field

from copilot_agent import ui_stream
from copilot_agent.api import (
    MAX_CHARS_PER_MESSAGE,
    MAX_MESSAGES,
    MAX_TOTAL_CHARS,
    create_app,
    in_own_task,
)
from copilot_agent.generation import REFUSAL_MESSAGE
from copilot_agent.graph import build_graph
from copilot_agent.planner import GREETING_MESSAGE
from copilot_agent.retrieval import RetrievalResult, RetrievedChunk
from copilot_agent.settings import Settings

pytestmark = pytest.mark.anyio

GOLDEN = Path(__file__).parent / "golden" / "chat-stream"
POOLER_URL = "postgresql://u:not-real@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"
AGENT_KEY = "k" * 64
AUTH = {"authorization": f"Bearer {AGENT_KEY}"}
# The secret tests/setup.ts gives the Vitest process, so the TypeScript verifier accepts the
# golden streams' signatures. The contract test checks the two still agree.
TS_TEST_SIGNING_SECRET = "test-signing-secret"
GOLDEN_MESSAGE_ID = "PyGoldenMessage1"
LEAK = "sk-must-never-reach-the-client"

USAGE = {"input_tokens": 1500, "output_tokens": 60, "total_tokens": 1560}


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-not-real",
        cohere_api_key="co-test-not-real",
        database_url=POOLER_URL,
        agent_api_key=AGENT_KEY,
        assistant_signing_secret=TS_TEST_SIGNING_SECRET,
    )


# ---- fakes ------------------------------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """Streams the given deltas, then a last chunk with usage, like ChatOpenAI on the Responses
    API. Can wait on a gate before a delta, fail after N deltas, and records cancellation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deltas: list[str] = Field(default_factory=list)
    fail_after: int | None = None
    gates: dict[int, asyncio.Event] = Field(default_factory=dict)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    cancelled: asyncio.Event = Field(default_factory=asyncio.Event)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("the graph streams")

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.seen.append(messages)
        try:
            for i, delta in enumerate(self.deltas):
                if i in self.gates:
                    await self.gates[i].wait()
                if i == self.fail_after:
                    raise RuntimeError(f"upstream said: {LEAK}")
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if self.fail_after == len(self.deltas):
                raise RuntimeError(f"upstream said: {LEAK}")
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", usage_metadata=USAGE, chunk_position="last")
            )
            self.events.append("finished")
        except asyncio.CancelledError:
            self.events.append("cancelled")
            self.cancelled.set()
            raise


class ScriptedSearch:
    """Answers every query with the same chunks; can fail, or hang until cancelled."""

    def __init__(self, relevant: list[RetrievedChunk], *, error: Exception | None = None) -> None:
        self.relevant = relevant
        self.error = error
        self.hang = False
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def __call__(self, query: str, embed_text: str | None = None) -> RetrievalResult:
        self.calls.append(query)
        self.started.set()
        if self.hang:
            try:
                await asyncio.Event().wait()  # never set
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        if self.error is not None:
            raise self.error
        return RetrievalResult(
            candidates=[], relevant=self.relevant, mode="reranked", timings_ms={}
        )


def scripted_planner(intent: str, queries: list[str]) -> tuple[Any, list[list[BaseMessage]]]:
    calls: list[list[BaseMessage]] = []

    async def reply(messages: list[BaseMessage]) -> dict[str, Any]:
        calls.append(messages)
        parsed = {"intent": intent, "queries": [{"query": q, "hypothetical": ""} for q in queries]}
        raw = AIMessage(content="", usage_metadata=USAGE)
        return {"raw": raw, "parsed": parsed, "parsing_error": None}

    return RunnableLambda(reply), calls


@dataclass
class Rig:
    """One app with its fakes, so a test can look at what each of them saw."""

    app: FastAPI
    planner_calls: list[list[BaseMessage]]
    search: ScriptedSearch
    model: ScriptedChatModel


def make_rig(
    intent: str = "search",
    queries: tuple[str, ...] = ("How do I stream text?",),
    relevant: list[RetrievedChunk] | None = None,
    model: ScriptedChatModel | None = None,
    search_error: Exception | None = None,
) -> Rig:
    planner, planner_calls = scripted_planner(intent, list(queries))
    search = ScriptedSearch(relevant if relevant is not None else [], error=search_error)
    model = model or ScriptedChatModel(deltas=["Use ", "streamText."])

    @asynccontextmanager
    async def search_factory(settings: Settings) -> AsyncIterator[ScriptedSearch]:
        yield search

    def graph_factory(settings: Settings, search_docs: Any) -> Any:
        return build_graph(planner=planner, search=search_docs, model=model)

    app = create_app(make_settings(), search_factory=search_factory, graph_factory=graph_factory)
    return Rig(app, planner_calls, search, model)


@asynccontextmanager
async def in_memory(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        yield client


def chunks_of(body: str) -> list[Any]:
    """The JSON payloads of an SSE body, [DONE] kept as the string it is."""
    payloads = [line[len("data: ") :] for line in body.split("\n") if line.startswith("data: ")]
    return [p if p == ui_stream.DONE else json.loads(p) for p in payloads]


def page(n: int, score: float, url: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"Chunk {n} of the docs.",
        title=f"Page {url or n}",
        source_url=f"https://ai-sdk.dev/docs/page-{url or n}",
        score=score,
    )


# ---- the key ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-header"),
        pytest.param({"authorization": "Bearer "}, id="empty-key"),
        pytest.param({"authorization": "Bearer " + "k" * 63}, id="one-short"),
        pytest.param({"authorization": "Bearer " + "k" * 65}, id="one-long"),
        pytest.param({"authorization": AGENT_KEY}, id="no-scheme"),
        pytest.param({"authorization": f"Basic {AGENT_KEY}"}, id="other-scheme"),
    ],
)
async def test_chat_without_the_key_is_401_and_costs_nothing(headers: dict[str, str]) -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json={"question": "q"}, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert rig.planner_calls == [] and rig.search.calls == [] and rig.model.seen == []


async def test_the_scheme_is_case_insensitive() -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post(
            "/chat", json={"question": "q"}, headers={"authorization": f"bearer {AGENT_KEY}"}
        )
    assert response.status_code == 200


async def test_malformed_json_is_422_even_without_the_key() -> None:
    # FastAPI reads and decodes the body before any dependency runs. Documented in
    # api.require_key; pinned so a change in FastAPI's order shows up here.
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post(
            "/chat", content=b'{"question": ', headers={"content-type": "application/json"}
        )
    assert response.status_code == 422
    assert rig.planner_calls == []


@pytest.mark.parametrize(
    ("missing", "name"),
    [("agent_api_key", "AGENT_API_KEY"), ("assistant_signing_secret", "ASSISTANT_SIGNING_SECRET")],
)
def test_the_service_does_not_start_without_its_secrets(missing: str, name: str) -> None:
    settings = make_settings().model_copy(update={missing: None})
    with pytest.raises(RuntimeError, match=name):
        create_app(settings)


# ---- the request ------------------------------------------------------------------------------


def turns(n: int, text: str = "t") -> list[dict[str, str]]:
    return [{"role": ("user", "assistant")[i % 2], "text": text} for i in range(n)]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing-question"),
        pytest.param({"question": ""}, id="empty-question"),
        pytest.param({"question": " \n"}, id="blank-question"),
        pytest.param({"question": "q" * (MAX_CHARS_PER_MESSAGE + 1)}, id="question-over-cap"),
        pytest.param({"question": "q", "history": turns(MAX_MESSAGES)}, id="too-many-turns"),
        pytest.param({"question": "q", "history": [{"role": "system", "text": "x"}]}, id="system"),
        pytest.param(
            {"question": "q", "history": [{"role": "user", "text": " "}]}, id="blank-turn"
        ),
        pytest.param(
            {"question": "q", "history": turns(1, "t" * (MAX_CHARS_PER_MESSAGE + 1))},
            id="turn-over-cap",
        ),
        pytest.param(
            {"question": "q", "history": turns(6, "t" * MAX_CHARS_PER_MESSAGE)}, id="total-over-cap"
        ),
        pytest.param({"question": "q", "messages": []}, id="the-usechat-body"),
        pytest.param(
            {"question": "q", "history": [{"role": "user", "text": "t", "sig": "v1.x"}]},
            id="extra-key-in-a-turn",
        ),
    ],
)
async def test_a_bad_request_is_422_before_any_paid_work(payload: dict[str, Any]) -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json=payload, headers=AUTH)
    assert response.status_code == 422
    assert rig.planner_calls == [] and rig.search.calls == []


async def test_a_request_exactly_at_every_cap_is_accepted() -> None:
    # 19 turns + the question = MAX_MESSAGES; 5 full turns + a 4000-character question = 24000.
    history = turns(MAX_MESSAGES - 1, "t")
    history[:5] = turns(5, "t" * MAX_CHARS_PER_MESSAGE)
    question = "q" * (MAX_TOTAL_CHARS - 5 * MAX_CHARS_PER_MESSAGE - (MAX_MESSAGES - 1 - 5))
    assert len(question) <= MAX_CHARS_PER_MESSAGE
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post(
            "/chat", json={"question": question, "history": history}, headers=AUTH
        )
    assert response.status_code == 200


async def test_question_and_history_reach_the_planner_and_the_model() -> None:
    rig = make_rig(relevant=[page(1, 0.8)])
    history = [
        {"role": "user", "text": "how do I stream text"},
        {"role": "assistant", "text": "Use streamText (Source 1)."},
    ]
    async with in_memory(rig.app) as client:
        response = await client.post(
            "/chat", json={"question": "and configure it?", "history": history}, headers=AUTH
        )
    assert response.status_code == 200
    planner_prompt = rig.planner_calls[0][1].content[0]["text"]
    assert "user: how do I stream text\nassistant: Use streamText (Source 1)." in planner_prompt
    assert "and configure it?" in planner_prompt
    [messages] = rig.model.seen
    assert [m.type for m in messages] == ["system", "human", "ai", "human"]
    assert messages[2].content == [{"type": "text", "text": "Use streamText (Source 1)."}]


# ---- the response -----------------------------------------------------------------------------


async def test_headers_are_the_ai_sdk_ones() -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json={"question": "q"}, headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"


async def test_every_event_is_one_data_line_and_the_stream_ends_with_done() -> None:
    # A delta with newlines must not break the framing: JSON escapes them inside the string.
    rig = make_rig(model=ScriptedChatModel(deltas=["line one\n", "\nline two\r\n"]))
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json={"question": "q"}, headers=AUTH)
    events = response.text.split("\n\n")
    assert events[-1] == ""
    assert all(e.startswith("data: ") and "\n" not in e for e in events[:-1])
    assert chunks_of(response.text)[-1] == ui_stream.DONE
    deltas = [c["delta"] for c in chunks_of(response.text)[:-1] if c["type"] == "text-delta"]
    assert "".join(deltas) == "line one\n\nline two\r\n"


async def test_each_response_gets_its_own_message_id() -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        ids = []
        for _ in range(2):
            response = await client.post("/chat", json={"question": "q"}, headers=AUTH)
            ids.append(chunks_of(response.text)[0]["messageId"])
    assert ids[0] != ids[1]
    assert all(len(i) == 16 and i.isalnum() for i in ids)


# ---- golden streams: the input of the cross-language contract test ----------------------------


@dataclass(frozen=True)
class Scenario:
    """One /chat run, described so that both sides can replay it.

    Python replays it through the real app with fakes; the Vitest contract test replays it
    through the TypeScript route with plannedRetrieve and the model mocked, then compares the
    two messages the real client builds.
    """

    id: str
    question: str
    intent: str
    sub_queries: list[str] = field(default_factory=list)
    relevant: list[RetrievedChunk] = field(default_factory=list)
    deltas: list[str] = field(default_factory=list)
    # "none", "model" (the answer fails after its deltas), "retrieval" (the search fails)
    failure: str = "none"

    def rig(self) -> Rig:
        fail_after = len(self.deltas) if self.failure == "model" else None
        return make_rig(
            intent=self.intent,
            queries=tuple(self.sub_queries),
            relevant=self.relevant,
            model=ScriptedChatModel(deltas=self.deltas, fail_after=fail_after),
            search_error=RuntimeError(f"db said: {LEAK}") if self.failure == "retrieval" else None,
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "intent": self.intent,
            "subQueries": self.sub_queries,
            "relevant": [c.model_dump() for c in self.relevant],
            "deltas": self.deltas,
            "failure": self.failure,
        }


# Awkward text on purpose: JSON escapes, a backslash, braces, a backtick, non-ASCII (e acute, an
# em dash, a check mark), U+2028 (a line terminator to JavaScript source, not to SSE or JSON)
# and an astral emoji, which is two UTF-16 units in the TypeScript signature and one code point
# here. chr() and not escapes, so this file stays ASCII.
AWKWARD = 'Quotes "like this", a \\ backslash, {braces}, `code`, ' + " ".join(
    ["caf" + chr(0xE9), chr(0x2014), chr(0x2713), chr(0x2028), chr(0x1F680)]
)

SCENARIOS = [
    Scenario(
        id="answered",
        question="how do I stream text and pick a model?",
        intent="search",
        sub_queries=["How do I stream text?", "How do I choose a model?"],
        # Chunks 1 and 3 share a page: one pill carrying chunks [1, 3]. In score order, as the
        # union leaves them (the TypeScript side receives this list from plannedRetrieve).
        relevant=[page(1, 0.79, url=1), page(2, 0.76, url=2), page(3, 0.71, url=1)],
        deltas=["Use `streamText` (Source 1).", "\n\n", AWKWARD, " Pick a model (Source 2)."],
    ),
    Scenario(id="greeting", question="hi", intent="greeting"),
    Scenario(id="off-topic", question="how do I deploy to AWS?", intent="off-topic"),
    Scenario(
        id="no-context",
        question="what is the airspeed of a swallow in the SDK?",
        intent="search",
        sub_queries=["What is the airspeed of a swallow in the SDK?"],
        deltas=[REFUSAL_MESSAGE],
    ),
    Scenario(
        id="error-mid-answer",
        question="how do I stream text?",
        intent="search",
        sub_queries=["How do I stream text?"],
        relevant=[page(1, 0.76)],
        deltas=["Use ", "streamText"],
        failure="model",
    ),
    Scenario(
        id="retrieval-fails",
        question="how do I stream text?",
        intent="search",
        sub_queries=["How do I stream text?"],
        failure="retrieval",
    ),
]


async def golden_body(scenario: Scenario, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(ui_stream, "new_message_id", lambda: GOLDEN_MESSAGE_ID)
    rig = scenario.rig()
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json={"question": scenario.question}, headers=AUTH)
    assert response.status_code == 200
    return response.text


def golden_meta() -> str:
    meta = {
        "about": "Written by agent/tests/test_chat_api.py (UPDATE_GOLDEN=1). Read by "
        "tests/python-stream-contract.test.ts. Do not edit by hand.",
        "signingSecret": TS_TEST_SIGNING_SECRET,
        "messageId": GOLDEN_MESSAGE_ID,
        "scenarios": [s.as_json() for s in SCENARIOS],
    }
    return json.dumps(meta, indent=2, ensure_ascii=False) + "\n"


def check_or_write(path: Path, content: str) -> None:
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return
    assert path.exists(), f"{path.name} missing: run with UPDATE_GOLDEN=1"
    assert path.read_text(encoding="utf-8") == content, (
        f"{path.name} is stale: the stream changed. Rerun with UPDATE_GOLDEN=1, then run "
        "npm test (the contract test reads it) and commit both."
    )


def test_the_scenario_file_is_current() -> None:
    check_or_write(GOLDEN / "scenarios.json", golden_meta())


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
async def test_the_golden_stream_is_current(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    body = await golden_body(scenario, monkeypatch)
    check_or_write(GOLDEN / f"{scenario.id}.sse", body)

    chunks = chunks_of(body)
    assert chunks[-1] == ui_stream.DONE
    # The scenario file says what plannedRetrieve returns in TypeScript: the merged chunks.
    # Hold it to that, so both sides replay the same run.
    sources = [c["data"] for c in chunks[:-1] if c["type"] == "data-sources"]
    assert sources == ([ui_stream.source_pills(scenario.relevant)] if scenario.relevant else [])
    # An error's text never reaches the client; the log has it.
    assert LEAK not in body
    if scenario.failure == "none":
        assert chunks[0] == {"type": "start", "messageId": GOLDEN_MESSAGE_ID}
        assert [c for c in chunks if c != ui_stream.DONE and c["type"] == "start"] == [chunks[0]]
    else:
        assert {"type": "error", "errorText": ui_stream.STREAM_FAILED} in chunks
        assert LEAK in caplog.text
    if scenario.failure == "retrieval":
        # No `start`: an opened message would be an empty bubble above the error box.
        assert chunks == [{"type": "error", "errorText": ui_stream.STREAM_FAILED}, ui_stream.DONE]


async def test_the_answer_is_signed_as_streamed() -> None:
    from copilot_agent.signing import sign_assistant_text

    scenario = SCENARIOS[0]
    rig = scenario.rig()
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json={"question": scenario.question}, headers=AUTH)
    chunks = chunks_of(response.text)[:-1]
    streamed = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
    [sig] = [c["data"]["sig"] for c in chunks if c["type"] == "data-signature"]
    assert streamed == "".join(scenario.deltas)
    assert sig == sign_assistant_text(streamed, TS_TEST_SIGNING_SECRET)


@pytest.mark.parametrize(
    ("intent", "reply"), [("greeting", GREETING_MESSAGE), ("off-topic", REFUSAL_MESSAGE)]
)
async def test_canned_replies_are_signed(intent: str, reply: str) -> None:
    from copilot_agent.signing import sign_assistant_text

    rig = make_rig(intent=intent, queries=())
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json={"question": "hi"}, headers=AUTH)
    chunks = chunks_of(response.text)[:-1]
    [sig] = [c["data"]["sig"] for c in chunks if c["type"] == "data-signature"]
    assert sig == sign_assistant_text(reply, TS_TEST_SIGNING_SECRET)
    assert rig.search.calls == [] and rig.model.seen == []


# ---- over a real socket: streaming and disconnects ---------------------------------------------


@asynccontextmanager
async def served(app: FastAPI) -> AsyncIterator[str]:
    """Run app on a real uvicorn (h11, as in production), on a free port of 127.0.0.1."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        app, lifespan="on", log_level="warning", http="h11", timeout_graceful_shutdown=2
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        # uvicorn exposes a flag, not an event, so poll it.
        while not server.started:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
        sock.close()


@asynccontextmanager
async def real_client() -> AsyncIterator[httpx.AsyncClient]:
    # trust_env=False: an HTTP(S)_PROXY in the environment must not route 127.0.0.1.
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        yield client


async def lines_until(lines: AsyncIterator[str], found: Callable[[Any], bool]) -> list[Any]:
    """Read SSE lines until a chunk matches. Pass the SAME response.aiter_lines() iterator to
    each call: httpx lets a response body be iterated once."""
    seen = []
    async for line in lines:
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            chunk = payload if payload == ui_stream.DONE else json.loads(payload)
            seen.append(chunk)
            if found(chunk):
                return seen
    raise AssertionError(f"stream ended first: {seen}")


def is_delta(chunk: Any) -> bool:
    return isinstance(chunk, dict) and chunk["type"] == "text-delta"


async def test_tokens_reach_the_client_while_the_model_is_still_answering() -> None:
    # The model holds its second token until the client has the first. Anything that buffered
    # the response (a middleware, a missing flush) would deadlock here, and fail on the timeout.
    release = asyncio.Event()
    rig = make_rig(model=ScriptedChatModel(deltas=["first", " second"], gates={1: release}))
    async with served(rig.app) as url, real_client() as client:
        async with asyncio.timeout(5):
            async with client.stream(
                "POST", f"{url}/chat", json={"question": "q"}, headers=AUTH
            ) as response:
                lines = response.aiter_lines()
                first = await lines_until(lines, is_delta)
                assert first[-1]["delta"] == "first"
                assert rig.model.events == []  # still answering
                release.set()
                rest = await lines_until(lines, lambda c: c == ui_stream.DONE)
    assert [c["delta"] for c in rest if is_delta(c)] == [" second"]
    assert rig.model.events == ["finished"]


async def test_a_disconnect_during_retrieval_cancels_the_search() -> None:
    rig = make_rig(relevant=[page(1, 0.8)])
    rig.search.hang = True
    async with served(rig.app) as url, real_client() as client:
        async with asyncio.timeout(5):
            async with client.stream(
                "POST", f"{url}/chat", json={"question": "q"}, headers=AUTH
            ) as response:
                # Headers arrive at once; nothing else can until retrieval is done.
                assert response.status_code == 200
                await rig.search.started.wait()
            # Leaving the block closes the connection mid-response: the tab was closed.
            await rig.search.cancelled.wait()
    assert rig.model.seen == []  # no answer was paid for


async def test_a_disconnect_during_the_answer_cancels_the_model_stream() -> None:
    never = asyncio.Event()
    rig = make_rig(
        relevant=[page(1, 0.8)],
        model=ScriptedChatModel(deltas=["first", " second"], gates={1: never}),
    )
    async with served(rig.app) as url, real_client() as client:
        async with asyncio.timeout(5):
            async with client.stream(
                "POST", f"{url}/chat", json={"question": "q"}, headers=AUTH
            ) as response:
                await lines_until(response.aiter_lines(), is_delta)
            await rig.model.cancelled.wait()
    assert rig.model.events == ["cancelled"]


# ---- why the graph runs in its own task --------------------------------------------------------


async def run_cancelled_by_a_cancel_scope(wrap: bool) -> ScriptedSearch:
    """Consume a graph run inside an anyio task group, as FastAPI's SSE producer does, and
    cancel the group's scope while the search hangs, as a disconnect does."""
    planner, _ = scripted_planner("search", ["q"])
    search = ScriptedSearch([page(1, 0.8)])
    search.hang = True
    graph = build_graph(planner=planner, search=search, model=ScriptedChatModel())

    async def consume() -> None:
        parts = graph.astream(
            {"question": "q", "history": []}, stream_mode=["updates", "messages"], version="v2"
        )
        async for _ in in_own_task(parts) if wrap else parts:
            pass

    async with anyio.create_task_group() as group:
        group.start_soon(consume)
        await search.started.wait()
        group.cancel_scope.cancel()
    return search


async def test_in_its_own_task_a_cancelled_scope_cancels_the_run() -> None:
    search = await run_cancelled_by_a_cancel_scope(wrap=True)
    with anyio.fail_after(2):
        await search.cancelled.wait()


async def test_canary_langgraph_leaks_a_run_cancelled_by_a_cancel_scope() -> None:
    # The reason for in_own_task, pinned: consumed directly inside a cancelled anyio scope, the
    # run's node task is never cancelled. If this starts failing, LangGraph unwinds correctly
    # under anyio now, and in_own_task may be unnecessary: check before removing it.
    before = asyncio.all_tasks()
    search = await run_cancelled_by_a_cancel_scope(wrap=False)
    await asyncio.sleep(0.2)
    leaked = [t for t in asyncio.all_tasks() - before if not t.done()]
    assert not search.cancelled.is_set()
    assert leaked, "no leaked task: LangGraph now cancels its nodes here"
    for task in leaked:  # clean up the leak so it does not outlive the test
        task.cancel()
    await asyncio.gather(*leaked, return_exceptions=True)
