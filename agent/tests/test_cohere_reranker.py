"""cohere_reranker against a fake Cohere: the request it sends, and when it retries.

httpx.MockTransport swaps out only the network layer of a REAL httpx.AsyncClient. The client
still builds real requests and parses real responses, but a Python function answers instead
of a socket. In JS terms this is msw: intercept below fetch, leave the calling code as it is.
"""

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from copilot_agent import retrieval
from copilot_agent.retrieval import COHERE_RERANK_URL, Ranking, Rerank, cohere_reranker

pytestmark = pytest.mark.anyio

API_KEY = "co-test-not-real"
QUERY = "how do I stream text"
DOCUMENTS = ["chunk about generateText", "chunk about streamText"]
OK_BODY = {"results": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.2}]}
OK_RANKING = [Ranking(index=1, score=0.91), Ranking(index=0, score=0.2)]


class FakeCohere:
    """Answers each request with the next step of a script and records everything.

    A step is an httpx.Response, or an exception to raise instead (a dropped connection).
    """

    def __init__(self) -> None:
        self.script: list[httpx.Response | Exception] = []
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)  # recorded, not waited: a retry test takes no 2 s


@pytest.fixture
def cohere() -> FakeCohere:
    return FakeCohere()


@pytest.fixture
async def rerank(cohere: FakeCohere, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Rerank]:
    """The real cohere_reranker, wired to the fake. Fixtures compose: this one asks for
    `cohere`, and a test asking for both gets the SAME FakeCohere (one instance per test).
    """
    # Patch the name where it is LOOKED UP: retrieval.py calls asyncio.sleep through its own
    # module-level `asyncio` name, so replacing that name touches nothing outside retrieval.py
    # (patching asyncio.sleep itself would also change it for the event loop running the test).
    # retrieval.py uses asyncio for nothing but this sleep; add to the namespace if that changes.
    monkeypatch.setattr(retrieval, "asyncio", SimpleNamespace(sleep=cohere.sleep))
    async with httpx.AsyncClient(transport=httpx.MockTransport(cohere.handle)) as http:
        yield cohere_reranker(http, API_KEY)
    # Teardown: the client is closed here, whether the test passed or failed.


def ok() -> httpx.Response:
    return httpx.Response(200, json=OK_BODY)


# ---- the request ----------------------------------------------------------------------


async def test_sends_the_ai_sdk_request_body(cohere: FakeCohere, rerank: Rerank) -> None:
    cohere.script = [ok()]

    ranking = await rerank(QUERY, DOCUMENTS, 5)

    [request] = cohere.requests
    assert request.method == "POST"
    assert str(request.url) == COHERE_RERANK_URL
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    # Field for field what @ai-sdk/cohere sends. It also sets max_tokens_per_doc and priority,
    # but to undefined, and JSON.stringify drops undefined fields, so they never go out.
    assert json.loads(request.content) == {
        "model": "rerank-v3.5",
        "query": QUERY,
        "documents": DOCUMENTS,
        "top_n": 5,
    }
    assert ranking == OK_RANKING
    assert cohere.sleeps == []


async def test_unexpected_response_shape_fails_by_name(cohere: FakeCohere, rerank: Rerank) -> None:
    cohere.script = [httpx.Response(200, json={"results": [{"index": 0}]})]

    with pytest.raises(ValidationError, match="relevance_score"):
        await rerank(QUERY, DOCUMENTS, 5)


# ---- retries: the AI SDK rule, maxRetries 1 -------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(408, id="408-timeout"),
        pytest.param(409, id="409-conflict"),
        pytest.param(429, id="429-rate-limited"),
        pytest.param(500, id="500"),
        pytest.param(503, id="503"),
        pytest.param(501, id="501-every-5xx-like-the-ai-sdk"),
    ],
)
async def test_retryable_status_is_retried_once(
    cohere: FakeCohere, rerank: Rerank, status: int
) -> None:
    cohere.script = [httpx.Response(status), ok()]

    ranking = await rerank(QUERY, DOCUMENTS, 5)

    assert ranking == OK_RANKING
    assert len(cohere.requests) == 2
    assert cohere.requests[0].content == cohere.requests[1].content  # the same request again
    assert cohere.sleeps == [2.0]  # the AI SDK's initial retry delay


async def test_network_error_is_retried_once(cohere: FakeCohere, rerank: Rerank) -> None:
    cohere.script = [httpx.ConnectError("connection reset"), ok()]

    assert await rerank(QUERY, DOCUMENTS, 5) == OK_RANKING
    assert len(cohere.requests) == 2
    assert cohere.sleeps == [2.0]


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(400, id="400-bad-request"),
        pytest.param(401, id="401-bad-key"),
        pytest.param(404, id="404"),
        pytest.param(422, id="422"),
    ],
)
async def test_client_errors_are_not_retried(
    cohere: FakeCohere, rerank: Rerank, status: int
) -> None:
    # Asking again cannot fix a bad key or a bad body; it only doubles the wait before the
    # cosine fallback takes over.
    cohere.script = [httpx.Response(status)]

    with pytest.raises(httpx.HTTPStatusError) as error:
        await rerank(QUERY, DOCUMENTS, 5)

    assert error.value.response.status_code == status
    assert len(cohere.requests) == 1
    assert cohere.sleeps == []


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        pytest.param(
            [httpx.Response(503), httpx.Response(503)], httpx.HTTPStatusError, id="503-twice"
        ),
        pytest.param(
            [httpx.ConnectError("reset"), httpx.ConnectError("reset")],
            httpx.ConnectError,
            id="network-twice",
        ),
        pytest.param(
            [httpx.ConnectError("reset"), httpx.Response(429)],
            httpx.HTTPStatusError,
            id="network-then-429",
        ),
    ],
)
async def test_gives_up_after_one_retry(
    cohere: FakeCohere,
    rerank: Rerank,
    script: list[httpx.Response | Exception],
    expected: type[Exception],
) -> None:
    # search_docs catches whatever this raises and falls back to cosine order.
    cohere.script = list(script)

    with pytest.raises(expected):
        await rerank(QUERY, DOCUMENTS, 5)

    assert len(cohere.requests) == 2  # never a third
    assert cohere.sleeps == [2.0]
