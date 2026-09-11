"""The chat graph with the REAL LangChain models, over a fake network.

Both ChatOpenAI models (planner and answer) share one httpx MockTransport that answers like
OpenAI: the planner's structured-output request gets the planner golden's canned reply, the
streaming answer request gets the generation golden's canned event stream. Retrieval is a fake.
This is where the "nostream" tag is tested for real: the planner is a chat model inside a node,
so without the tag its JSON would appear in the "messages" stream next to the answer.
"""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from copilot_agent.generation import openai_generation_model
from copilot_agent.graph import build_graph
from copilot_agent.planner import TokenUsage, build_planner, openai_planner_model
from copilot_agent.retrieval import RetrievalResult, RetrievedChunk
from copilot_agent.settings import Settings

pytestmark = pytest.mark.anyio

GOLDEN = Path(__file__).parent / "golden"
PLANNER_REPLY = json.loads((GOLDEN / "planner-requests.json").read_text())["meta"]["cannedReply"]
GENERATION = json.loads((GOLDEN / "generation-requests.json").read_text())
STREAM: str = GENERATION["meta"]["cannedStream"]
ANSWER: str = GENERATION["cases"][0]["result"]["text"]


class FakeOpenAI:
    """One fake endpoint for both models, told apart by the request body."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if "text" in body:  # structured output: the planner
            return httpx2.Response(200, json=PLANNER_REPLY)
        return httpx2.Response(
            200, content=STREAM.encode(), headers={"content-type": "text/event-stream"}
        )


async def search(query: str, embed_text: str | None = None) -> RetrievalResult:
    chunk = RetrievedChunk(
        content="streamText streams.",
        title="Generating Text",
        source_url="https://ai-sdk.dev/docs/generating-text",
        score=0.759,
    )
    return RetrievalResult(candidates=[], relevant=[chunk], mode="reranked", timings_ms={})


@pytest.fixture(autouse=True)
def default_models(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PLANNER_MODEL", "GENERATION_MODEL", "MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(name, raising=False)


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-not-real",
        cohere_api_key="co-test-not-real",
        database_url="postgresql://u:p@aws-0-eu-west-1.pooler.supabase.com:5432/postgres",
    )


async def test_only_the_answer_streams_and_the_state_is_complete() -> None:
    settings = make_settings()
    openai = FakeOpenAI()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(openai)) as client:
        graph = build_graph(
            planner=build_planner(openai_planner_model(settings, http_async_client=client)),
            search=search,
            model=openai_generation_model(settings, http_async_client=client),
        )
        streamed: list[tuple[str, str]] = []
        final: dict[str, Any] = {}
        async for part in graph.astream(
            {"question": "how do I stream text", "history": []},
            stream_mode=["messages", "values"],
            version="v2",
        ):
            if part["type"] == "messages":
                message, metadata = part["data"]
                streamed.append((metadata["langgraph_node"], message.text))
            else:
                final = part["data"]

    assert [b.get("stream") for b in openai.bodies] == [False, True]  # planner, then answer
    # Nothing from the planner in the answer stream: the "nostream" tag holds.
    assert {node for node, _ in streamed} == {"generate"}
    assert "".join(text for _, text in streamed) == ANSWER
    assert final["plan"].queries[0].query == "golden query"
    assert final["answer"] == ANSWER
    assert final["generation"].usage == TokenUsage(input_tokens=1234, output_tokens=56)
    assert final["generation"].ttft_ms is not None
    assert final["generation"].finish_reason == "stop"


# The golden stream's last event, rewritten the way OpenAI ends a response that hit
# max_output_tokens: event response.incomplete, status incomplete, and the reason.
INCOMPLETE_STREAM = (
    STREAM.replace("event: response.completed", "event: response.incomplete")
    .replace('"type":"response.completed"', '"type":"response.incomplete"')
    .replace(
        '"object":"response","created_at":1789137600,"status":"completed"',
        '"object":"response","created_at":1789137600,"status":"incomplete",'
        '"incomplete_details":{"reason":"max_output_tokens"}',
    )
)


async def test_an_answer_cut_off_by_the_token_cap_finishes_with_length() -> None:
    # Where LangChain puts incomplete_details is read in its source (langchain_openai 1.6.2);
    # this runs the real parser on the event OpenAI sends, so a move there shows up here.
    assert INCOMPLETE_STREAM.count("max_output_tokens") == 1
    settings = make_settings()

    def reply(request: httpx2.Request) -> httpx2.Response:
        if "text" in json.loads(request.content):
            return httpx2.Response(200, json=PLANNER_REPLY)
        return httpx2.Response(
            200, content=INCOMPLETE_STREAM.encode(), headers={"content-type": "text/event-stream"}
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(reply)) as client:
        graph = build_graph(
            planner=build_planner(openai_planner_model(settings, http_async_client=client)),
            search=search,
            model=openai_generation_model(settings, http_async_client=client),
        )
        final = await graph.ainvoke({"question": "how do I stream text", "history": []})
    assert final["answer"] == ANSWER
    assert final["generation"].finish_reason == "length"
