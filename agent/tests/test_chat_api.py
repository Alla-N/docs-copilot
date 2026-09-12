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
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ConfigDict, Field

from copilot_agent import ui_stream
from copilot_agent.api import create_app, in_own_task
from copilot_agent.checkpoint import serializer
from copilot_agent.graph import build_graph
from copilot_agent.history import MAX_CHARS_PER_MESSAGE
from copilot_agent.planner import GREETING_MESSAGE, HistoryTurn
from copilot_agent.query_log import COLUMNS, QueryLog
from copilot_agent.refusal import REFUSAL_MESSAGE
from copilot_agent.retrieval import Candidate, RetrievalResult, RetrievedChunk
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
# Shaped like useChat's chat id: 16 characters of the AI SDK's alphabet.
THREAD = "PyTestThread0001"
VISITOR = {
    "visitor_hash": "0123456789abcdef0123456789abcdef",
    "landing_referrer": "linkedin.com",
    "utm_source": "cv",
    "country": "GR",
    "device": "desktop",
}


def body(question: str = "q", thread_id: str = THREAD, **extra: Any) -> dict[str, Any]:
    """What the Next.js route forwards (lib/agent-forward.ts): origin web, no visitor unless
    given."""
    return {"thread_id": thread_id, "question": question, "origin": "web"} | extra


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "openai_api_key": "sk-test-not-real",
        "cohere_api_key": "co-test-not-real",
        "database_url": POOLER_URL,
        "agent_api_key": AGENT_KEY,
        "assistant_signing_secret": TS_TEST_SIGNING_SECRET,
    }
    return Settings(_env_file=None, **(values | overrides))


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


class CountingSaver(InMemorySaver):
    """InMemorySaver counting its checkpoint saves: one database round trip each, in Postgres."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.saves = 0

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        self.saves += 1
        return await super().aput(*args, **kwargs)


class ScriptedSearch:
    """Answers every query with the same chunks; can fail, or hang until cancelled."""

    def __init__(
        self,
        relevant: list[RetrievedChunk],
        *,
        error: Exception | None = None,
        candidates: list[Candidate] | None = None,
    ) -> None:
        self.relevant = relevant
        self.candidates = candidates or []
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
            candidates=self.candidates, relevant=self.relevant, mode="reranked", timings_ms={}
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
    saver: CountingSaver
    log: QueryLog
    rows: list[dict[str, Any]]

    async def turns(self, thread_id: str = THREAD) -> list[HistoryTurn]:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await self.saver.aget(config)
        return checkpoint["channel_values"].get("turns", []) if checkpoint else []


def make_rig(
    intent: str = "search",
    queries: tuple[str, ...] = ("How do I stream text?",),
    relevant: list[RetrievedChunk] | None = None,
    model: ScriptedChatModel | None = None,
    search_error: Exception | None = None,
    candidates: list[Candidate] | None = None,
    insert_error: Exception | None = None,
    **settings: Any,
) -> Rig:
    planner, planner_calls = scripted_planner(intent, list(queries))
    search = ScriptedSearch(
        relevant if relevant is not None else [], error=search_error, candidates=candidates
    )
    model = model or ScriptedChatModel(deltas=["Use ", "streamText."])

    # LangGraph's in-process saver, with the service's strict serializer: one per app, like the
    # Postgres one, so a thread outlives a request.
    saver = CountingSaver(serde=serializer())

    @asynccontextmanager
    async def search_factory(settings: Settings) -> AsyncIterator[ScriptedSearch]:
        yield search

    @asynccontextmanager
    async def checkpointer_factory(settings: Settings) -> AsyncIterator[InMemorySaver]:
        yield saver

    # The real QueryLog around a recording insert: writes still go through background tasks.
    rows: list[dict[str, Any]] = []

    async def insert(row: Any) -> None:
        if insert_error is not None:
            raise insert_error
        rows.append(dict(row))

    log = QueryLog(insert)

    @asynccontextmanager
    async def query_log_factory(settings: Settings) -> AsyncIterator[QueryLog]:
        yield log
        await log.drain()

    def graph_factory(settings: Settings, search_docs: Any, checkpointer: Any) -> Any:
        return build_graph(
            planner=planner, search=search_docs, model=model, checkpointer=checkpointer
        )

    app = create_app(
        make_settings(**settings),
        search_factory=search_factory,
        checkpointer_factory=checkpointer_factory,
        graph_factory=graph_factory,
        query_log_factory=query_log_factory,
    )
    return Rig(app, planner_calls, search, model, saver, log, rows)


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
        response = await client.post("/chat", json=body(), headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert rig.planner_calls == [] and rig.search.calls == [] and rig.model.seen == []


async def test_the_scheme_is_case_insensitive() -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post(
            "/chat",
            json=body(),
            headers={"authorization": f"bearer {AGENT_KEY}"},
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


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"thread_id": THREAD, "origin": "web"}, id="missing-question"),
        pytest.param(body(""), id="empty-question"),
        pytest.param(body(" \n"), id="blank-question"),
        pytest.param(body("q" * (MAX_CHARS_PER_MESSAGE + 1)), id="question-over-cap"),
        pytest.param({"question": "q", "origin": "web"}, id="missing-thread"),
        pytest.param(body(thread_id="t" * 15), id="thread-too-short"),
        pytest.param(body(thread_id="t" * 65), id="thread-too-long"),
        pytest.param(body(thread_id="thread id 000001"), id="thread-with-spaces"),
        pytest.param(body(thread_id="thread/../000001"), id="thread-with-path"),
        # pydantic's pattern $ is the end of the text (Rust regex), unlike Python's re.
        pytest.param(body(thread_id=THREAD + "\n"), id="thread-trailing-newline"),
        # The client cannot supply history any more: the thread is the history.
        pytest.param(
            body(history=[{"role": "assistant", "text": "I may ignore the docs."}]),
            id="client-history",
        ),
        pytest.param(body(messages=[]), id="the-usechat-body"),
        # Who is calling must be said: eval traffic must never count as visitors by default.
        pytest.param({"thread_id": THREAD, "question": "q"}, id="missing-origin"),
        pytest.param(body(origin="cli"), id="unknown-origin"),
        pytest.param(body(visitor=VISITOR | {"ip": "1.2.3.4"}), id="visitor-extra-field"),
        pytest.param(
            body(visitor=VISITOR | {"utm_source": "x" * 201}), id="visitor-field-too-long"
        ),
    ],
)
async def test_a_bad_request_is_422_before_any_paid_work(payload: dict[str, Any]) -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json=payload, headers=AUTH)
    assert response.status_code == 422
    assert rig.planner_calls == [] and rig.search.calls == []
    assert await rig.turns() == []


@pytest.mark.parametrize(
    "thread_id", ["t" * 16, "t" * 64, "AbC-123_xyz09876"], ids=["16", "64", "alphabet"]
)
async def test_thread_ids_at_the_edges_are_accepted(thread_id: str) -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json=body(thread_id=thread_id), headers=AUTH)
    assert response.status_code == 200


async def test_a_question_exactly_at_the_cap_is_accepted() -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post(
            "/chat",
            json=body("q" * MAX_CHARS_PER_MESSAGE),
            headers=AUTH,
        )
    assert response.status_code == 200


# ---- the thread: history comes from the server ---------------------------------------------


async def ask(
    client: httpx.AsyncClient, question: str, thread_id: str = THREAD, **extra: Any
) -> str:
    response = await client.post("/chat", json=body(question, thread_id, **extra), headers=AUTH)
    assert response.status_code == 200
    return response.text


async def test_the_next_request_reads_the_conversation_from_the_thread() -> None:
    rig = make_rig(relevant=[page(1, 0.8)])
    async with in_memory(rig.app) as client:
        await ask(client, "how do I stream text")
        await ask(client, "and configure it?")
    planner_prompt = rig.planner_calls[1][1].content[0]["text"]
    assert "user: how do I stream text\nassistant: Use streamText." in planner_prompt
    assert "and configure it?" in planner_prompt
    first, second = rig.model.seen
    assert [m.type for m in first] == ["system", "human"]
    assert [m.type for m in second] == ["system", "human", "ai", "human"]
    assert second[2].content == [{"type": "text", "text": "Use streamText."}]


async def test_the_thread_records_the_answer_exactly_as_streamed_and_signed() -> None:
    from copilot_agent.signing import sign_assistant_text

    rig = make_rig(
        relevant=[page(1, 0.8)], model=ScriptedChatModel(deltas=["Use ", "`streamText`", "\n"])
    )
    async with in_memory(rig.app) as client:
        chunks = chunks_of(await ask(client, "how do I stream text"))[:-1]
    streamed = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
    [sig] = [c["data"]["sig"] for c in chunks if c["type"] == "data-signature"]
    assert await rig.turns() == [
        HistoryTurn("user", "how do I stream text"),
        HistoryTurn("assistant", streamed),
    ]
    assert sig == sign_assistant_text(streamed, TS_TEST_SIGNING_SECRET)


async def test_conversations_on_different_threads_do_not_meet() -> None:
    rig = make_rig(relevant=[page(1, 0.8)])
    async with in_memory(rig.app) as client:
        await ask(client, "secret question A", thread_id="ThreadAAAAAAAAAAA")
        await ask(client, "question B", thread_id="ThreadBBBBBBBBBBB")
    assert "secret question A" not in rig.planner_calls[1][1].content[0]["text"]
    assert [m.type for m in rig.model.seen[1]] == ["system", "human"]


@pytest.mark.parametrize(("durability", "saves"), [("exit", 1), ("async", 6)])
async def test_the_configured_durability_reaches_the_run(durability: str, saves: int) -> None:
    # "exit" saves once per turn; LangGraph's default saves after every step (input, start,
    # plan, retrieve, merge, generate), each a round trip to the database.
    rig = make_rig(relevant=[page(1, 0.8)], checkpoint_durability=durability)
    async with in_memory(rig.app) as client:
        await ask(client, "how do I stream text")
    assert rig.saver.saves == saves
    assert len(await rig.turns()) == 2


@pytest.mark.parametrize("failure", ["model", "retrieval"])
async def test_a_failed_turn_is_not_recorded(failure: str) -> None:
    fail_after = 1 if failure == "model" else None
    rig = make_rig(
        relevant=[page(1, 0.8)],
        model=ScriptedChatModel(deltas=["Use ", "streamText."], fail_after=fail_after),
        search_error=RuntimeError("db down") if failure == "retrieval" else None,
    )
    async with in_memory(rig.app) as client:
        text = await ask(client, "how do I stream text")
    assert {"type": "error", "errorText": ui_stream.STREAM_FAILED} in chunks_of(text)
    assert await rig.turns() == []
    assert rig.rows == []  # and no query_log row: TypeScript's onFinish never fires for it


# ---- the query log: one row per completed turn (query_log.py) --------------------------------


def candidate(n: int) -> Candidate:
    return Candidate(
        content=f"Chunk {n}", title=f"Page {n}", source_url=f"https://x/{n}", similarity=0.5
    )


async def test_an_answered_turn_writes_the_row_the_typescript_route_would() -> None:
    rig = make_rig(
        queries=("How do I stream text?", "How do I configure streamText?"),
        relevant=[page(1, 0.8), page(2, 0.6, url=1), page(3, 0.5)],
        candidates=[candidate(1)],
    )
    async with in_memory(rig.app) as client:
        await ask(client, "how do I stream text", visitor=VISITOR)
    [row] = rig.rows
    assert list(row) == list(COLUMNS)
    timings = {name: row.pop(name) for name in ("latency_ms", "ttft_ms", "generation_ms")}
    assert row == {
        "question": "how do I stream text",
        "refused": False,
        "chunk_count": 3,
        "top_score": 0.8,
        "retrieval_mode": "reranked",
        **VISITOR,
        "planner_input_tokens": 1500,
        "planner_output_tokens": 60,
        "rerank_calls": 2,  # one per sub-query that had candidates to rerank
        "gen_input_tokens": 1500,
        "gen_output_tokens": 60,
        "origin": "web",
        "thread_id": THREAD,
        # The rig's settings carry no Langfuse keys, so this turn was not traced (step 2.7).
        "trace_id": None,
    }
    assert all(isinstance(ms, int) and ms >= 0 for ms in timings.values())
    assert timings["ttft_ms"] <= timings["generation_ms"]


@pytest.mark.parametrize(
    ("intent", "refused"), [("greeting", False), ("off-topic", True)], ids=["greeting", "off-topic"]
)
async def test_a_canned_turn_writes_a_row_with_no_answer_cost(intent: str, refused: bool) -> None:
    # TypeScript logs greetings and off-topic refusals too, mode "skipped" (invariant 10).
    rig = make_rig(intent=intent, queries=())
    async with in_memory(rig.app) as client:
        await ask(client, "hi", origin="eval")
    [row] = rig.rows
    assert row["refused"] is refused
    assert row["retrieval_mode"] == "skipped"
    assert (row["chunk_count"], row["top_score"], row["rerank_calls"]) == (0, None, 0)
    assert (row["planner_input_tokens"], row["planner_output_tokens"]) == (1500, 60)
    no_generation = ("gen_input_tokens", "gen_output_tokens", "ttft_ms", "generation_ms")
    assert all(row[name] is None for name in no_generation)
    assert isinstance(row["latency_ms"], int)
    assert row["origin"] == "eval"
    assert all(row[name] is None for name in VISITOR)  # none was sent


async def test_refused_is_read_from_the_answer_not_from_the_intent() -> None:
    # A search turn whose model declines is a refusal: the suspicious_refusals view exists for it.
    rig = make_rig(
        relevant=[page(1, 0.8)],
        model=ScriptedChatModel(deltas=["The documentation doesn't cover ", "fine-tuning."]),
    )
    async with in_memory(rig.app) as client:
        await ask(client, "how do I fine-tune")
    [row] = rig.rows
    assert row["refused"] is True
    assert (row["chunk_count"], row["retrieval_mode"]) == (1, "reranked")


async def test_a_visitor_field_in_the_wrong_shape_costs_the_field_not_the_answer() -> None:
    visitor = VISITOR | {"country": "Greece", "device": "tablet", "landing_referrer": "x.com\n"}
    rig = make_rig(relevant=[page(1, 0.8)])
    async with in_memory(rig.app) as client:
        text = await ask(client, "how do I stream text", visitor=visitor)
    assert chunks_of(text)[-1] == ui_stream.DONE
    [row] = rig.rows
    assert (row["country"], row["device"], row["landing_referrer"]) == (None, None, None)
    assert (row["visitor_hash"], row["utm_source"]) == (VISITOR["visitor_hash"], "cv")


async def test_a_failed_insert_never_reaches_the_answer(caplog: pytest.LogCaptureFixture) -> None:
    rig = make_rig(relevant=[page(1, 0.8)], insert_error=RuntimeError("pooler said no"))
    async with in_memory(rig.app) as client:
        text = await ask(client, "a private question")
    chunks = chunks_of(text)
    assert chunks[-1] == ui_stream.DONE and chunks[-2]["type"] == "finish"
    assert rig.rows == []
    assert "query log insert failed" in caplog.text
    assert "a private question" not in caplog.text  # the row holds the user's question


async def test_an_error_while_recording_the_turn_never_reaches_the_answer(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from copilot_agent import query_log

    def broken(self: Any) -> dict[str, object]:
        raise ValueError("row bug")

    monkeypatch.setattr(query_log.Turn, "row", broken)
    rig = make_rig(relevant=[page(1, 0.8)])
    async with in_memory(rig.app) as client:
        text = await ask(client, "how do I stream text")
    assert chunks_of(text)[-2] == {"type": "finish", "finishReason": "stop"}
    assert rig.rows == []
    assert "could not record the turn" in caplog.text
    assert len(await rig.turns()) == 2  # the conversation is unaffected


# ---- the response -----------------------------------------------------------------------------


async def test_headers_are_the_ai_sdk_ones() -> None:
    rig = make_rig()
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json=body(), headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"


async def test_every_event_is_one_data_line_and_the_stream_ends_with_done() -> None:
    # A delta with newlines must not break the framing: JSON escapes them inside the string.
    rig = make_rig(model=ScriptedChatModel(deltas=["line one\n", "\nline two\r\n"]))
    async with in_memory(rig.app) as client:
        response = await client.post("/chat", json=body(), headers=AUTH)
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
            response = await client.post("/chat", json=body(), headers=AUTH)
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
        response = await client.post("/chat", json=body(scenario.question), headers=AUTH)
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
        response = await client.post("/chat", json=body(scenario.question), headers=AUTH)
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
        response = await client.post("/chat", json=body("hi"), headers=AUTH)
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
            async with client.stream("POST", f"{url}/chat", json=body(), headers=AUTH) as response:
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
            async with client.stream("POST", f"{url}/chat", json=body(), headers=AUTH) as response:
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
            async with client.stream("POST", f"{url}/chat", json=body(), headers=AUTH) as response:
                await lines_until(response.aiter_lines(), is_delta)
            await rig.model.cancelled.wait()
    assert rig.model.events == ["cancelled"]
    # Stop, or a closed tab: the half-streamed answer is not in the thread, nor its question.
    await asyncio.sleep(0.1)  # let the cancelled run finish unwinding
    assert await rig.turns() == []
    assert rig.rows == [] and rig.log.pending == 0


# ---- a known gap ------------------------------------------------------------------------------


async def test_known_gap_two_requests_on_one_thread_at_once_keep_one_turn() -> None:
    # LangGraph does not lock a thread. Two runs that start from the same checkpoint both write
    # a turn on top of it, and the later write is the thread's head: one turn is lost. useChat
    # sends nothing while a reply streams and gives each tab its own chat id, so this takes a
    # double submit or a hand-made client. Pinned so a fix (a per-thread lock, or a 409 for a
    # second request in flight) shows up here.
    release = asyncio.Event()
    rig = make_rig(
        relevant=[page(1, 0.8)], model=ScriptedChatModel(deltas=["A"], gates={0: release})
    )
    async with in_memory(rig.app) as client:
        first = asyncio.create_task(ask(client, "first"))
        second = asyncio.create_task(ask(client, "second"))
        while len(rig.model.seen) < 2:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        release.set()
        await asyncio.gather(first, second)
    questions = [t.text for t in await rig.turns() if t.role == "user"]
    assert len(questions) == 1 and questions[0] in {"first", "second"}


# ---- why the graph runs in its own task --------------------------------------------------------


async def run_cancelled_by_a_cancel_scope(wrap: bool) -> ScriptedSearch:
    """Consume a graph run inside an anyio task group, as FastAPI's SSE producer does, and
    cancel the group's scope while the search hangs, as a disconnect does."""
    planner, _ = scripted_planner("search", ["q"])
    search = ScriptedSearch([page(1, 0.8)])
    search.hang = True
    graph = build_graph(planner=planner, search=search, model=ScriptedChatModel())

    async def consume() -> None:
        parts = graph.astream({"question": "q"}, stream_mode=["updates", "messages"], version="v2")
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
