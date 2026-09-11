"""The HTTP layer: routes, request validation, the lifespan, and invariant 2 for /search.

No network, no secrets. Apps are built with a fake search_factory, and requests go through
httpx.ASGITransport: straight into the ASGI app in memory, no server and no port.
ASGITransport does not send lifespan events, so serve() enters the app's lifespan itself:
the same context manager uvicorn runs at startup and shutdown.

Not Starlette's TestClient: it runs the app in a background thread with its own event loop,
and Starlette 1.6 warns when it runs on httpx (it now wants httpx2). These tests stay on the
anyio plugin like the rest of the suite. POST /chat has its own file, test_chat_api.py, which
also runs a real uvicorn: ASGITransport buffers a whole response, so it cannot show streaming.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from copilot_agent import api
from copilot_agent.api import create_app
from copilot_agent.history import MAX_CHARS_PER_MESSAGE as MAX_TEXT_CHARS
from copilot_agent.retrieval import RetrievalResult, RetrievedChunk
from copilot_agent.settings import Settings

pytestmark = pytest.mark.anyio

POOLER_URL = "postgresql://u:not-real@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"


@pytest.fixture(autouse=True)
def _no_flag_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    # Init arguments beat env vars in pydantic-settings, but the default-off test passes none:
    # an ENABLE_SEARCH_ENDPOINT=1 left exported in the shell must not flip its result.
    monkeypatch.delenv("ENABLE_SEARCH_ENDPOINT", raising=False)


AGENT_KEY = "k" * 64
AUTH = {"authorization": f"Bearer {AGENT_KEY}"}


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "openai_api_key": "sk-test-not-real",
        "cohere_api_key": "co-test-not-real",
        "database_url": POOLER_URL,
        "agent_api_key": AGENT_KEY,
        "assistant_signing_secret": "test-signing-secret",
    }
    return Settings(_env_file=None, **(values | overrides))


def no_graph(settings: Settings, search: Any, checkpointer: Any) -> Any:
    """The graph factory for these tests: /chat is tested in test_chat_api.py."""
    return None


@asynccontextmanager
async def no_checkpointer(settings: Settings) -> AsyncIterator[None]:
    """The checkpointer factory for these tests: nothing here reads a thread."""
    yield None


class FakeSearch:
    """Stands in for the function open_search() yields. Records every call."""

    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.error: Exception | None = None
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(self, query: str, embed_text: str | None = None) -> RetrievalResult:
        self.calls.append((query, embed_text))
        if self.error is not None:
            raise self.error
        return self.result


class FakeFactory:
    """Stands in for open_search(): records when it opens and closes, and with what."""

    def __init__(self, search: FakeSearch) -> None:
        self.search = search
        self.fail_on_open: Exception | None = None
        self.events: list[str] = []
        self.opened_with: Settings | None = None

    @asynccontextmanager
    async def __call__(self, settings: Settings) -> AsyncIterator[FakeSearch]:
        self.opened_with = settings
        if self.fail_on_open is not None:
            raise self.fail_on_open
        self.events.append("open")
        try:
            yield self.search
        finally:
            self.events.append("close")


@asynccontextmanager
async def serve(
    app: FastAPI, *, raise_app_exceptions: bool = True
) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        yield client


@pytest.fixture
def result(make_candidate: Any) -> RetrievalResult:
    first = make_candidate(0.61, title="Generating Text")
    second = make_candidate(0.58, title="streamText")
    return RetrievalResult(
        candidates=[first, second],
        relevant=[
            RetrievedChunk(
                content=second.content, title=second.title, source_url=second.source_url, score=0.74
            )
        ],
        mode="reranked",
        timings_ms={"embed": 1.5, "vector": 2.5, "rerank": 3.5},
    )


@pytest.fixture
def fake(result: RetrievalResult) -> FakeSearch:
    return FakeSearch(result)


@pytest.fixture
def factory(fake: FakeSearch) -> FakeFactory:
    return FakeFactory(fake)


@pytest.fixture
async def client(factory: FakeFactory) -> AsyncIterator[httpx.AsyncClient]:
    """A client for an app WITH /search enabled, lifespan running."""
    app = create_app(
        make_settings(enable_search_endpoint=True),
        search_factory=factory,
        graph_factory=no_graph,
        checkpointer_factory=no_checkpointer,
    )
    async with serve(app) as client:
        client.headers.update(AUTH)
        yield client


# ---- /health and the lifespan -------------------------------------------------------------


async def test_health_is_ok_and_calls_nothing(client: httpx.AsyncClient, fake: FakeSearch) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert fake.calls == []


async def test_lifespan_opens_search_once_and_closes_it(factory: FakeFactory) -> None:
    settings = make_settings()
    app = create_app(
        settings,
        search_factory=factory,
        graph_factory=no_graph,
        checkpointer_factory=no_checkpointer,
    )
    assert factory.events == []  # building the app opens nothing; startup does

    async with serve(app) as client:
        assert factory.events == ["open"]
        assert factory.opened_with is settings
        await client.get("/health")
        await client.get("/health")
        assert factory.events == ["open"]  # once per server life, not once per request

    assert factory.events == ["open", "close"]


async def test_startup_fails_when_search_cannot_open(factory: FakeFactory) -> None:
    # The real open_search() raises here when the pool cannot connect (pool.open(wait=True)).
    # The error must stop startup, so the service never answers /health without a database.
    factory.fail_on_open = OSError("connection refused")
    app = create_app(
        make_settings(),
        search_factory=factory,
        graph_factory=no_graph,
        checkpointer_factory=no_checkpointer,
    )
    with pytest.raises(OSError, match="connection refused"):
        async with serve(app):
            pass


async def test_startup_fails_when_the_checkpoint_database_is_not_ready(
    factory: FakeFactory,
) -> None:
    # The real open_checkpointer() raises when the tables are missing, behind the saver's
    # migrations, or have Row Level Security off (checkpoint.readiness_problems).
    savers = FakeFactory(None)  # type: ignore[arg-type]
    savers.fail_on_open = RuntimeError("checkpoint database not ready: tables missing")
    app = create_app(
        make_settings(), search_factory=factory, checkpointer_factory=savers, graph_factory=no_graph
    )
    with pytest.raises(RuntimeError, match="not ready"):
        async with serve(app):
            pass
    assert factory.events == ["open", "close"]  # what opened before it is closed again


def test_create_app_without_arguments_reads_settings_then(
    monkeypatch: pytest.MonkeyPatch, factory: FakeFactory
) -> None:
    # `uvicorn --factory copilot_agent.api:create_app` calls create_app() with no arguments.
    monkeypatch.setattr(api, "get_settings", lambda: make_settings(enable_search_endpoint=True))
    app = create_app(
        search_factory=factory, graph_factory=no_graph, checkpointer_factory=no_checkpointer
    )
    assert "/search" in app.openapi()["paths"]


# ---- /search: invariant 2 -----------------------------------------------------------------


async def test_search_route_does_not_exist_unless_enabled(
    factory: FakeFactory, fake: FakeSearch
) -> None:
    app = create_app(
        make_settings(),
        search_factory=factory,
        graph_factory=no_graph,
        checkpointer_factory=no_checkpointer,
    )
    async with serve(app) as client:
        response = await client.post("/search", json={"query": "how do I stream text"})
    assert response.status_code == 404
    assert fake.calls == []
    assert set(app.openapi()["paths"]) == {"/health", "/chat"}


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-header"),
        pytest.param({"authorization": "Bearer wrong"}, id="wrong-key"),
        pytest.param({"authorization": AGENT_KEY}, id="no-scheme"),
        pytest.param({"authorization": f"Basic {AGENT_KEY}"}, id="other-scheme"),
    ],
)
async def test_search_needs_the_agent_key_too(
    factory: FakeFactory, fake: FakeSearch, headers: dict[str, str]
) -> None:
    app = create_app(
        make_settings(enable_search_endpoint=True),
        search_factory=factory,
        graph_factory=no_graph,
        checkpointer_factory=no_checkpointer,
    )
    async with serve(app) as client:
        response = await client.post("/search", json={"query": "q"}, headers=headers)
    assert response.status_code == 401
    assert fake.calls == []


async def test_search_is_post_only(client: httpx.AsyncClient, fake: FakeSearch) -> None:
    response = await client.get("/search", params={"query": "how do I stream text"})
    assert response.status_code == 405
    assert fake.calls == []


# ---- /search: behaviour -------------------------------------------------------------------


async def test_search_returns_the_result_with_slim_candidates(
    client: httpx.AsyncClient, result: RetrievalResult
) -> None:
    response = await client.post("/search", json={"query": "how do I stream text"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "reranked"
    assert body["timings_ms"] == result.timings_ms
    # The surviving chunks keep their text; the candidates do not, and keep cosine order.
    assert body["relevant"] == [chunk.model_dump() for chunk in result.relevant]
    assert body["candidates"] == [
        {"title": c.title, "source_url": c.source_url, "similarity": c.similarity}
        for c in result.candidates
    ]


@pytest.mark.parametrize(
    ("payload", "expected_call"),
    [
        pytest.param({"query": "q"}, ("q", None), id="query-only"),
        pytest.param({"query": "q", "embed_text": "an answer"}, ("q", "an answer"), id="hyde"),
        pytest.param({"query": "q", "embed_text": None}, ("q", None), id="explicit-null"),
    ],
)
async def test_search_passes_query_and_embed_text_through(
    client: httpx.AsyncClient,
    fake: FakeSearch,
    payload: dict[str, Any],
    expected_call: tuple[str, str | None],
) -> None:
    response = await client.post("/search", json=payload)
    assert response.status_code == 200
    assert fake.calls == [expected_call]


async def test_search_accepts_text_exactly_at_the_cap(
    client: httpx.AsyncClient, fake: FakeSearch
) -> None:
    query, embed_text = "q" * MAX_TEXT_CHARS, "e" * MAX_TEXT_CHARS
    response = await client.post("/search", json={"query": query, "embed_text": embed_text})
    assert response.status_code == 200
    assert fake.calls == [(query, embed_text)]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing-query"),
        pytest.param({"query": ""}, id="empty-query"),
        pytest.param({"query": " \n\t"}, id="blank-query"),
        pytest.param({"query": "q" * (MAX_TEXT_CHARS + 1)}, id="query-over-cap"),
        pytest.param({"query": 42}, id="query-not-a-string"),
        pytest.param({"query": "q", "embed_text": ""}, id="empty-embed-text"),
        pytest.param({"query": "q", "embed_text": "  "}, id="blank-embed-text"),
        pytest.param({"query": "q", "embed_text": "e" * (MAX_TEXT_CHARS + 1)}, id="embed-over-cap"),
        pytest.param({"query": "q", "embedText": "an answer"}, id="typescript-spelling"),
        pytest.param(["q"], id="not-an-object"),
    ],
)
async def test_search_rejects_a_bad_body_before_any_paid_work(
    client: httpx.AsyncClient, fake: FakeSearch, payload: Any
) -> None:
    response = await client.post("/search", json=payload)
    assert response.status_code == 422
    assert fake.calls == []


async def test_search_rejects_malformed_json(client: httpx.AsyncClient, fake: FakeSearch) -> None:
    response = await client.post(
        "/search", content=b'{"query": ', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert fake.calls == []


async def test_unexpected_error_is_a_bare_500(factory: FakeFactory, fake: FakeSearch) -> None:
    # Invariant 9's rule on this side: the error text never reaches the client. It can carry
    # anything, an upstream message quoting a key included. uvicorn logs the traceback.
    fake.error = RuntimeError("upstream said: sk-leaked-key")
    app = create_app(
        make_settings(enable_search_endpoint=True),
        search_factory=factory,
        graph_factory=no_graph,
        checkpointer_factory=no_checkpointer,
    )
    async with serve(app, raise_app_exceptions=False) as client:
        response = await client.post("/search", json={"query": "q"}, headers=AUTH)
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
