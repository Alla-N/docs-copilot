"""The agent service's HTTP surface: GET /health, POST /chat, and a local-only POST /search.

    cd agent
    uv run uvicorn --factory copilot_agent.api:create_app --reload

--factory: uvicorn calls create_app() at startup instead of importing a module-level `app`.
Settings are read then, not at import time (the same rule as get_settings()), so a test can
import this module without any secrets set.

The Next.js route stays the public front door: it rate-limits and parses the useChat request
(invariant 9), then (step 2.6) forwards {thread_id, question} here and passes the stream back.
So /chat trusts its caller's parsing but not its identity: every paid route needs the shared
AGENT_API_KEY.

History does not come in the request at all (step 2.5): the thread id names a conversation the
checkpointer keeps, and the turns before the question are read from there (graph.py,
history.py). A request carrying a `history` field is a 422.
"""

import asyncio
import hmac
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Durability
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SecretStr, StringConstraints

from copilot_agent import ui_stream
from copilot_agent.checkpoint import open_checkpointer
from copilot_agent.graph import ChatGraph, openai_chat_graph
from copilot_agent.history import MAX_CHARS_PER_MESSAGE
from copilot_agent.retrieval import RetrievalMode, RetrievedChunk, SearchDocs, open_search
from copilot_agent.settings import Settings, get_settings
from copilot_agent.signing import sign_assistant_text

# The shape of open_search(): given settings, an async context manager yielding a search.
# create_app() takes one so tests can hand it a fake that opens nothing and costs nothing.
OpenSearch = Callable[[Settings], AbstractAsyncContextManager[SearchDocs]]
# The shape of open_checkpointer(): given settings, an async context manager yielding a saver.
OpenCheckpointer = Callable[[Settings], AbstractAsyncContextManager[BaseCheckpointSaver]]
# The shape of openai_chat_graph(): the graph around a search and a saver. Tests pass fakes.
BuildGraph = Callable[[Settings, SearchDocs, BaseCheckpointSaver], ChatGraph]

# The caps of lib/chat-request.ts (invariant 9) live in history.py since the history moved
# server-side. The route cuts the question to MAX_CHARS_PER_MESSAGE, so what it forwards always
# fits. (TypeScript counts UTF-16 units, Python code points, and a code point is never more
# units than one, so the Python count is never the larger.) /search rejects instead of
# truncating, so what you sent is what ran.


def _not_blank(value: str) -> str:
    # Empty or whitespace-only. Rejected, not stripped: the reranker must see exactly the text
    # that was sent. (No min_length=1 as well: this check covers it, and a mutation run showed a
    # second check for the same thing is one no test can pin.)
    if not value.strip():
        raise ValueError("must not be blank")
    return value


# A text field of any request here: within the per-message cap, and not blank.
Text = Annotated[str, Field(max_length=MAX_CHARS_PER_MESSAGE), AfterValidator(_not_blank)]


# ---- request and response models ----------------------------------------------------------


class SearchRequest(BaseModel):
    # forbid: a misspelt field is a 422, not silently ignored. `embedText` (the TypeScript
    # spelling) would otherwise run a plain search while you believe HyDE text was used.
    model_config = ConfigDict(extra="forbid")

    query: Text
    embed_text: Text | None = None


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


# The conversation's key. The Next.js route forwards useChat's chat id (2.6): 16 characters of
# the AI SDK's alphabet by default, random per chat. It is the only thing that ties a request to
# a conversation, so it is a bearer capability: whoever has it continues that conversation. The
# format check keeps it a plain key everywhere it goes (logs, SQL parameters, Langfuse) and
# rules out short ones anybody could hit by accident; it cannot tell a random id from a chosen
# one, and a client that picks a guessable id only shares its own conversation.
THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{16,64}$"
ThreadId = Annotated[str, StringConstraints(pattern=THREAD_ID_PATTERN)]


class ChatRequest(BaseModel):
    """What the Next.js route forwards: the conversation's thread id and the new question.

    Not the useChat body. Parsing that stays in TypeScript, in front of the rate limiter's
    decision. No history: the turns before the question are the thread's (history.py), and with
    extra="forbid" a request that tries to supply them is a 422, so there is no assistant text
    a client could forge or replay from another conversation.
    """

    model_config = ConfigDict(extra="forbid")

    thread_id: ThreadId
    question: Text


# ---- dependencies -------------------------------------------------------------------------


def get_search(request: Request) -> SearchDocs:
    """The search function the lifespan opened. A dependency, so a route never reaches for
    global state itself, and a test can swap it with app.dependency_overrides."""
    return request.app.state.search


Search = Annotated[SearchDocs, Depends(get_search)]


def get_graph(request: Request) -> ChatGraph:
    return request.app.state.graph


def get_signer(request: Request) -> Callable[[str], str]:
    return request.app.state.sign


def get_durability(request: Request) -> Durability:
    return request.app.state.durability


Graph = Annotated[ChatGraph, Depends(get_graph)]
Signer = Annotated[Callable[[str], str], Depends(get_signer)]
DurabilityMode = Annotated[Durability, Depends(get_durability)]


def require_key(key: SecretStr) -> Callable[..., None]:
    """A dependency that lets a request through only with "Authorization: Bearer <key>".

    Compared in constant time, like verifyAssistantText. It runs before the body is VALIDATED
    and before any paid work, but after FastAPI has read and JSON-decoded the body: malformed
    JSON is a 422 even without a key. That leaks only that the route exists.
    """
    expected = key.get_secret_value().encode()

    def check(authorization: Annotated[str | None, Header()] = None) -> None:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), expected):
            raise HTTPException(
                status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"}
            )

    return check


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


chat_router = APIRouter()

# Strong references to graph runs still unwinding after their stream ended. asyncio keeps only
# weak references to tasks, and a pending task nobody references can be garbage collected.
_unwinding: set[asyncio.Task[None]] = set()


async def in_own_task(parts: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Iterate `parts` (a graph run) in a separate asyncio task, and cancel that task ONCE when
    the consumer stops early.

    Why: FastAPI runs an SSE endpoint's generator in an anyio task group, and a disconnect
    cancels it through an anyio cancel scope. anyio cancellation is level-triggered: every await
    inside a cancelled scope raises CancelledError again. LangGraph unwinds a cancelled run by
    creating a task that cancels the running nodes and awaiting it (AsyncPregelLoop.__aexit__);
    inside the cancelled scope that await is cancelled at once, which cancels the unwinding task
    before it starts, and the nodes are never told. Found by the disconnect tests in
    tests/test_chat_api.py: after a disconnect the search kept running. A canary test there pins
    the mechanism, so it shows when LangGraph handles this itself.

    In its own task the run sees one plain task.cancel(), which is what LangGraph handles.
    """
    queue: asyncio.Queue[tuple[bool, Any]] = asyncio.Queue(maxsize=1)

    async def pump() -> None:
        try:
            async for part in parts:
                await queue.put((True, part))
            await queue.put((False, None))
        except Exception as exc:  # handed to the consumer, which raises it
            await queue.put((False, exc))

    task = asyncio.create_task(pump())
    _unwinding.add(task)
    task.add_done_callback(_unwinding.discard)
    try:
        while True:
            is_part, value = await queue.get()
            if is_part:
                yield value
            elif isinstance(value, BaseException):
                raise value
            else:
                return
    finally:
        # Not awaited: inside a cancelled scope the await would be cancelled too. The run
        # unwinds in its own task, and _unwinding keeps it alive until it has.
        task.cancel()


def ui_message_stream_headers(response: Response) -> None:
    """Add the AI SDK's stream header. A dependency, because it has to run BEFORE the endpoint:
    the body of a generator endpoint only starts once the response (headers included) has been
    built, so a header set there is silently dropped. The header test caught exactly that."""
    response.headers.update(ui_stream.UI_MESSAGE_STREAM_HEADERS)


@chat_router.post(
    "/chat",
    response_class=EventSourceResponse,
    dependencies=[Depends(ui_message_stream_headers)],
)
async def chat(
    body: ChatRequest, graph: Graph, sign: Signer, durability: DurabilityMode
) -> AsyncIterator[ServerSentEvent]:
    """Run the chat graph on one question and stream the answer as the AI SDK's UI message
    stream, so useChat can read it (ui_stream.py has the protocol and the chunk order).

    EventSourceResponse (FastAPI's own SSE support) runs this generator in a producer task,
    inserts a ": ping" comment after 15 s of silence (the AI SDK's parser skips comments), and
    sets the no-cache and no-buffering headers. The status is 200 from the first byte on, so a
    failure after that travels as an error chunk (ui_stream.ui_message_chunks).

    A closed tab cancels the run: uvicorn reports ASGI spec 2.3, so Starlette watches for
    http.disconnect next to the body and cancels it, FastAPI cancels this generator, and
    in_own_task turns that into one cancel of the graph run, wherever it is (the planner, the
    searches, the model's stream). A cancelled turn is not recorded in the thread.
    """
    parts = graph.astream(
        {"question": body.question},
        {"configurable": {"thread_id": body.thread_id}},
        stream_mode=["updates", "messages"],
        version="v2",
        durability=durability,
    )
    parts = in_own_task(parts)
    chunks = ui_stream.ui_message_chunks(parts, message_id=ui_stream.new_message_id(), sign=sign)
    async for chunk in chunks:
        yield ServerSentEvent(raw_data=ui_stream.encode(chunk))
    yield ServerSentEvent(raw_data=ui_stream.DONE)


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


def _required(value: SecretStr | None, name: str) -> SecretStr:
    if value is None:
        raise RuntimeError(
            f"{name} is not set. The service will not start without it: see agent/README.md."
        )
    return value


def create_app(
    settings: Settings | None = None,
    *,
    search_factory: OpenSearch = open_search,
    checkpointer_factory: OpenCheckpointer = open_checkpointer,
    graph_factory: BuildGraph = openai_chat_graph,
) -> FastAPI:
    if settings is None:
        settings = get_settings()
    # Fail at startup, like the TypeScript signing module in a deployment: without the key a
    # paid route would be open, and without the secret every answer would be unsigned, so the
    # next request would silently drop it from the history.
    api_key = _required(settings.agent_api_key, "AGENT_API_KEY")
    signing_secret = _required(settings.assistant_signing_secret, "ASSISTANT_SIGNING_SECRET")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once around the whole life of the server: open the pools and clients before
        # the first request, close them after the last. If opening fails, the server exits,
        # and that includes a checkpoint database that is not set up (checkpoint.py).
        async with (
            search_factory(settings) as search_docs,
            checkpointer_factory(settings) as checkpointer,
        ):
            app.state.search = search_docs
            app.state.graph = graph_factory(settings, search_docs, checkpointer)
            yield

    app = FastAPI(title="docs-copilot agent", lifespan=lifespan)
    app.state.sign = partial(sign_assistant_text, secret=signing_secret.get_secret_value())
    app.state.durability = settings.checkpoint_durability
    paid = [Depends(require_key(api_key))]
    app.include_router(router)
    app.include_router(chat_router, dependencies=paid)
    # Invariant 2: a paid route is registered only when asked for. No flag, no route (404).
    # And, like /chat, it needs the key.
    if settings.enable_search_endpoint:
        app.include_router(search_router, dependencies=paid)
    return app
