"""Request parity: the Python answer step sends the request the TypeScript route sends, and
reads the streamed reply the same way.

scripts/experiments/generation-requests.ts (npm run exp:generation-requests, from the repo root)
ran the route's streamText call (generationSettings + generationMessages) on six synthetic
cases with fetch replaced by a recorder, answered each with the same canned Responses API event
stream, and froze the requests, the text and the usage the AI SDK read back. This test builds
the same messages with generation_messages(), streams them through the real ChatOpenAI over an
httpx MockTransport that replays the canned stream byte for byte, and compares all three.

Free and deterministic: it runs in the pre-commit hook and in CI, like the planner's.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx2  # openai>=3 is built on httpx2, so the client handed to it must be an httpx2 one
import openai
import pytest
from langchain_core.messages import AIMessageChunk

from copilot_agent.generation import (
    GENERATION_MAX_RETRIES,
    generation_messages,
    openai_generation_model,
)
from copilot_agent.planner import HistoryTurn
from copilot_agent.retrieval import RetrievedChunk
from copilot_agent.settings import Settings

pytestmark = pytest.mark.anyio

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN: dict[str, Any] = json.loads(
    (Path(__file__).parent / "golden" / "generation-requests.json").read_text()
)
CASES = GOLDEN["cases"]
CANNED_STREAM: str = GOLDEN["meta"]["cannedStream"]
# The deltas the canned stream carries, in order: streaming must deliver them one by one.
DELTAS = [
    event["delta"]
    for line in CANNED_STREAM.splitlines()
    if line.startswith("data: ")
    and (event := json.loads(line.removeprefix("data: ")))["type"] == "response.output_text.delta"
]


def encoded_by_langchain(ts_body: dict[str, Any]) -> dict[str, Any]:
    """The TypeScript body as LangChain encodes the same request.

    Three differences are expected, all encodings of the same request, and nothing else:
      - "type": "message" on every input item (as in the planner's parity test);
      - "annotations": [] on every output_text part of an assistant turn: LangChain writes the
        empty list out, the AI SDK leaves it off, and an empty list means none;
      - (none for "stream": both send true, because the generation model streams.)
    """
    expected = json.loads(json.dumps(ts_body))  # a deep copy
    for item in expected["input"]:
        item["type"] = "message"
        if item["role"] == "assistant":
            for part in item["content"]:
                part["annotations"] = []
    return expected


@pytest.fixture(autouse=True)
def default_generation_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # The golden was recorded with the defaults; the shell must not change the model or the cap.
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    monkeypatch.delenv("MAX_OUTPUT_TOKENS", raising=False)


def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-not-real",
        cohere_api_key="co-test-not-real",
        database_url="postgresql://u:p@aws-0-eu-west-1.pooler.supabase.com:5432/postgres",
    )


class StreamingServer:
    """The network, replaced: records each request and replays the golden's event stream."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200:
            # retry-after-ms keeps the client's retry wait at 1 ms instead of its backoff.
            return httpx2.Response(
                self.status,
                headers={"retry-after-ms": "1"},
                json={"error": {"message": "boom", "type": "server_error", "code": "x"}},
            )
        return httpx2.Response(
            200,
            content=CANNED_STREAM.encode(),
            headers={"content-type": "text/event-stream"},
        )


def messages_for(case: dict[str, Any]) -> list[Any]:
    return generation_messages(
        [RetrievedChunk(**chunk) for chunk in case["relevant"]],
        [HistoryTurn(turn["role"], turn["content"]) for turn in case["history"]],
        case["question"],
        case["subQueries"],
    )


def sha256(path: str) -> str:
    return hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()


@pytest.mark.parametrize("path", sorted(GOLDEN["meta"]["sourcesSha256"]))
def test_the_golden_matches_the_current_typescript(path: str) -> None:
    assert sha256(path) == GOLDEN["meta"]["sourcesSha256"][path], (
        f"{path} changed after the golden file was written; regenerate it from the repo root: "
        "npm run exp:generation-requests"
    )


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
async def test_same_request_and_same_stream_reading_as_typescript(case: dict[str, Any]) -> None:
    server = StreamingServer()
    deltas: list[str] = []
    answer: AIMessageChunk | None = None
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(server)) as client:
        model = openai_generation_model(settings(), http_async_client=client)
        async for chunk in model.astream(messages_for(case)):
            if chunk.text:
                deltas.append(chunk.text)
            answer = chunk if answer is None else answer + chunk

    assert len(server.requests) == 1
    request = server.requests[0]
    assert (request.method, str(request.url)) == (case["request"]["method"], case["request"]["url"])
    sent = json.loads(request.content)
    expected = encoded_by_langchain(case["request"]["body"])
    differing = sorted(k for k in sent.keys() | expected.keys() if sent.get(k) != expected.get(k))
    assert sent == expected, f"request differs from TypeScript in: {differing}"

    # What came out of the stream: token by token, the same text, the same usage.
    assert deltas == DELTAS
    assert answer is not None
    assert answer.text == case["result"]["text"]
    assert answer.usage_metadata is not None
    assert (answer.usage_metadata["input_tokens"], answer.usage_metadata["output_tokens"]) == (
        case["result"]["inputTokens"],
        case["result"]["outputTokens"],
    )


async def test_ainvoke_streams_too() -> None:
    """A LangGraph node will call ainvoke, not astream. With streaming=True the request is still
    a streaming one (the same body as TypeScript), and the result is the joined text."""
    case = CASES[0]
    server = StreamingServer()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(server)) as client:
        model = openai_generation_model(settings(), http_async_client=client)
        answer = await model.ainvoke(messages_for(case))
    assert json.loads(server.requests[0].content) == encoded_by_langchain(case["request"]["body"])
    assert answer.text == case["result"]["text"]


async def test_a_server_error_is_retried_twice_then_raised() -> None:
    """Three attempts, like streamText (maxRetries 2). Unlike the planner there is no fallback:
    an answer that cannot be generated is an error the route reports, as in TypeScript."""
    server = StreamingServer(status=500)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(server)) as client:
        model = openai_generation_model(settings(), http_async_client=client)
        with pytest.raises(openai.InternalServerError):
            await model.ainvoke(messages_for(CASES[0]))
    assert len(server.requests) == 1 + GENERATION_MAX_RETRIES == 3
