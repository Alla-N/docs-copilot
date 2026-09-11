"""The agent service's HTTP surface. Phase 1: GET /health and a local-only POST /search.

    cd agent
    ENABLE_SEARCH_ENDPOINT=1 uv run uvicorn --factory copilot_agent.api:create_app --reload

--factory: uvicorn calls create_app() at startup instead of importing a module-level `app`.
Settings are read then, not at import time (the same rule as get_settings()), so a test can
import this module without any secrets set.

Phase 2 adds POST /chat here. The Next.js route stays the public front door either way.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from copilot_agent.retrieval import RetrievalMode, RetrievedChunk, SearchDocs, open_search
from copilot_agent.settings import Settings, get_settings

# The shape of open_search(): given settings, an async context manager yielding a search.
# create_app() takes one so tests can hand it a fake that opens nothing and costs nothing.
OpenSearch = Callable[[Settings], AbstractAsyncContextManager[SearchDocs]]

# The chat route's per-message cap (MAX_CHARS_PER_MESSAGE in lib/chat-request.ts, invariant 9).
# The route truncates; this debug endpoint rejects instead, so what you sent is what ran.
MAX_TEXT_CHARS = 4000


# ---- request and response models ----------------------------------------------------------


class SearchRequest(BaseModel):
    # forbid: a misspelt field is a 422, not silently ignored. `embedText` (the TypeScript
    # spelling) would otherwise run a plain search while you believe HyDE text was used.
    model_config = ConfigDict(extra="forbid")

    query: str = Field(max_length=MAX_TEXT_CHARS)
    embed_text: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)

    @field_validator("query", "embed_text")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        # Empty or whitespace-only. Rejected, not stripped: the reranker must see exactly the
        # text that was sent. (No min_length=1 as well: this check covers it, and a mutation
        # run showed a second check for the same thing is one no test can pin.)
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value


class CandidateOut(BaseModel):
    """A cosine candidate without its text: 100 chunk texts (~250 tokens each) would bury
    the part a person reads. The chunks that survived keep their text, in `relevant`."""

    title: str
    source_url: str
    similarity: float


class SearchResponse(BaseModel):
    mode: RetrievalMode
    timings_ms: dict[str, float]
    relevant: list[RetrievedChunk]
    candidates: list[CandidateOut]  # every candidate, in cosine order


class HealthResponse(BaseModel):
    status: Literal["ok"]


# ---- dependencies -------------------------------------------------------------------------


def get_search(request: Request) -> SearchDocs:
    """The search function the lifespan opened. A dependency, so a route never reaches for
    global state itself, and a test can swap it with app.dependency_overrides."""
    return request.app.state.search


Search = Annotated[SearchDocs, Depends(get_search)]


# ---- routes -------------------------------------------------------------------------------

router = APIRouter()


@router.get("/health")
async def health() -> HealthResponse:
    """Liveness: the process is up and serving. It calls nothing, on purpose.

    A health check that queried Supabase would let a database blip mark every container
    unhealthy at once, and the orchestrator (ECS in phase 2b) would replace all of them
    together. Startup already proves the dependencies: the lifespan opens the pool with
    wait=True, so a service that cannot reach the database never starts serving at all.
    """
    return HealthResponse(status="ok")


search_router = APIRouter()


@search_router.post("/search")
async def search(body: SearchRequest, search_docs: Search) -> SearchResponse:
    """Run search_docs on one query, for debugging retrieval over HTTP. Spends credits.

    POST, not GET: a GET that costs money can be fired by anything that follows a link
    (a link preview, a crawler, a prefetch). The old GET /api/ingest was that bug.
    """
    result = await search_docs(body.query, body.embed_text)
    return SearchResponse(
        mode=result.mode,
        timings_ms=result.timings_ms,
        relevant=result.relevant,
        candidates=[
            CandidateOut(title=c.title, source_url=c.source_url, similarity=c.similarity)
            for c in result.candidates
        ],
    )


# ---- the app ------------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None, *, search_factory: OpenSearch = open_search
) -> FastAPI:
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once around the whole life of the server: open the pool and clients before
        # the first request, close them after the last. If opening fails, the server exits.
        async with search_factory(settings) as search_docs:
            app.state.search = search_docs
            yield

    app = FastAPI(title="docs-copilot agent", lifespan=lifespan)
    app.include_router(router)
    # Invariant 2: a paid route is registered only when asked for. No flag, no route (404).
    if settings.enable_search_endpoint:
        app.include_router(search_router)
    return app
