"""search_docs: the Python port of retrieve() in lib/retrieve.ts.

    embed -> pgvector top 100 -> Cohere rerank top 5 -> threshold 0.30
    (cosine order + the stricter 0.45 gate when the reranker is unavailable)

Same models, same SQL function, same request body to Cohere, same numbers
(retrieval_config.py, guarded by tests/test_ts_parity.py). Until the TypeScript retrieve()
is retired in phase 2, the two must return the same chunks for the same input.

Shape: search_docs() does the logic and takes its three I/O steps (embed, vector search,
rerank) as plain async functions. The real ones are built by the *_embedder / *_search /
*_reranker factories below; tests pass fakes. open_search() wires the real ones together.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal, NamedTuple, Protocol

import httpx
from openai import AsyncOpenAI
from psycopg.rows import class_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from copilot_agent import retrieval_config
from copilot_agent.settings import Settings

logger = logging.getLogger(__name__)

# Same embedding model as ingestion, non-negotiable: the stored vectors were made with it.
EMBEDDING_MODEL = "text-embedding-3-small"
# 0.30 is calibrated on THIS model's scores. A newer reranker needs a new sweep, not a swap.
RERANK_MODEL = "rerank-v3.5"
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"

# lib/retrieve.ts passes maxRetries: 1 to the AI SDK. Mirrored here explicitly: one retry,
# after the AI SDK's 2 s initial delay, on what its APICallError marks retryable: 408, 409,
# 429 and EVERY 5xx (statusCode >= 500 in @ai-sdk/provider). Not mirrored: when the response
# carries retry-after-ms or retry-after under 60 s, the AI SDK waits that long instead.
RERANK_RETRYABLE_4XX = frozenset({408, 409, 429})
RERANK_RETRY_DELAY_S = 2.0
RERANK_TIMEOUT_S = 15.0

# Connection pool: small on purpose. In session mode every pooled connection holds a real
# Postgres connection on Supabase's side, and the project has a fixed number of them.
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 4
POOL_OPEN_TIMEOUT_S = 10.0

RetrievalMode = Literal["reranked", "cosine-fallback", "skipped"]


class Candidate(BaseModel):
    """One row from match_documents(), in cosine order."""

    content: str
    title: str
    source_url: str
    similarity: float


class RetrievedChunk(BaseModel):
    """What survived reranking and the threshold. This is what the model sees."""

    content: str
    title: str
    source_url: str
    score: float


class RetrievalResult(BaseModel):
    candidates: list[Candidate]
    relevant: list[RetrievedChunk]
    mode: RetrievalMode
    # Per-stage latency, so the port can be compared with the TypeScript baseline.
    timings_ms: dict[str, float]


class Ranking(NamedTuple):
    index: int  # position in the documents list sent to the reranker
    score: float


# The three I/O steps. Plain async function types: anything with this call shape fits.
Embed = Callable[[str], Awaitable[list[float]]]
VectorSearch = Callable[[list[float], int], Awaitable[list[Candidate]]]
Rerank = Callable[[str, list[str], int], Awaitable[list[Ranking]]]


# ---- pure logic -----------------------------------------------------------------------


def gate_reranked(
    candidates: list[Candidate], ranking: list[Ranking], threshold: float
) -> list[RetrievedChunk]:
    """Keep ranked chunks scoring at least the threshold, in the reranker's order."""
    relevant = []
    for ranked in ranking:
        if ranked.score < threshold:
            continue
        chunk = candidates[ranked.index]
        relevant.append(
            RetrievedChunk(
                content=chunk.content,
                title=chunk.title,
                source_url=chunk.source_url,
                score=ranked.score,
            )
        )
    return relevant


def cosine_fallback(candidates: list[Candidate], top_n: int) -> list[RetrievedChunk]:
    """Without the cross-encoder, scores are noisier, so the stricter Day-6 gate applies."""
    kept = [c for c in candidates if c.similarity >= retrieval_config.COSINE_THRESHOLD]
    return [
        RetrievedChunk(
            content=c.content, title=c.title, source_url=c.source_url, score=c.similarity
        )
        for c in kept[:top_n]
    ]


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def search_docs(
    query: str,
    embed_text: str | None = None,
    *,
    embed: Embed,
    vector_search: VectorSearch,
    rerank: Rerank,
    candidates: int = retrieval_config.VECTOR_CANDIDATES,
    top_n: int = retrieval_config.RERANK_TOP_N,
    threshold: float = retrieval_config.RERANK_THRESHOLD,
) -> RetrievalResult:
    """Retrieve documentation chunks for one (sub-)query.

    query       the real question. Used for RERANKING: a cross-encoder judges
                question-to-passage relevance, so the true question belongs there.
    embed_text  what to EMBED for vector search; defaults to query. HyDE passes a
                hypothetical answer here, because an answer embeds closer to the answer
                chunk than a question does.
    """
    timings: dict[str, float] = {}

    start = time.perf_counter()
    embedding = await embed(query if embed_text is None else embed_text)
    timings["embed"] = _ms_since(start)

    start = time.perf_counter()
    found = await vector_search(embedding, candidates)
    timings["vector"] = _ms_since(start)

    if not found:
        return RetrievalResult(candidates=[], relevant=[], mode="reranked", timings_ms=timings)

    start = time.perf_counter()
    try:
        ranking = await rerank(query, [c.content for c in found], top_n)
    except Exception:
        # Reranking is an enhancement, not a dependency: an outage, a rate limit or an
        # exhausted key must degrade the answer, not break it. Unlike lib/retrieve.ts, the
        # try covers ONLY the network call, so a bug in the gating code below surfaces as a
        # bug instead of being disguised as a Cohere outage.
        timings["rerank"] = _ms_since(start)
        logger.warning("Rerank failed; falling back to cosine ordering", exc_info=True)
        return RetrievalResult(
            candidates=found,
            relevant=cosine_fallback(found, top_n),
            mode="cosine-fallback",
            timings_ms=timings,
        )
    timings["rerank"] = _ms_since(start)

    return RetrievalResult(
        candidates=found,
        relevant=gate_reranked(found, ranking, threshold),
        mode="reranked",
        timings_ms=timings,
    )


# ---- the real I/O steps ---------------------------------------------------------------


def openai_embedder(client: AsyncOpenAI) -> Embed:
    async def embed(text: str) -> list[float]:
        # "float" is the encoding the AI SDK requests. Asking for the same one keeps the two
        # requests identical, so one more difference stays out of the parity comparison.
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL, input=text, encoding_format="float"
        )
        return response.data[0].embedding

    return embed


# The same SQL function the TypeScript side calls through supabase-js. Casts are explicit:
# the embedding travels as text ('[0.1,0.2,...]'), and the count must match the integer
# parameter for Postgres to find the function.
MATCH_DOCUMENTS_SQL = (
    "select content, title, source_url, similarity from match_documents(%s::vector, %s::int)"
)


def to_vector_literal(embedding: list[float]) -> str:
    """pgvector's text format. repr() is the shortest string that round-trips the float."""
    return "[" + ",".join(map(repr, embedding)) + "]"


def pgvector_search(pool: AsyncConnectionPool) -> VectorSearch:
    async def search(embedding: list[float], count: int) -> list[Candidate]:
        async with (
            pool.connection() as conn,
            conn.cursor(row_factory=class_row(Candidate)) as cur,
        ):
            await cur.execute(MATCH_DOCUMENTS_SQL, (to_vector_literal(embedding), count))
            return await cur.fetchall()

    return search


class _CohereResult(BaseModel):
    index: int
    relevance_score: float


class _CohereRerankResponse(BaseModel):
    results: list[_CohereResult]


def _is_retryable(status: int) -> bool:
    return status in RERANK_RETRYABLE_4XX or status >= 500


async def _post_with_one_retry(
    http: httpx.AsyncClient, url: str, *, json: dict[str, object], headers: dict[str, str]
) -> httpx.Response:
    """POST; on a network error or a retryable status, wait and try exactly once more."""
    try:
        response = await http.post(url, json=json, headers=headers)
        if not _is_retryable(response.status_code):
            return response
        logger.info("Rerank returned %s; retrying once", response.status_code)
    except httpx.TransportError:
        logger.info("Rerank request failed at the transport level; retrying once")
    await asyncio.sleep(RERANK_RETRY_DELAY_S)
    return await http.post(url, json=json, headers=headers)


def cohere_reranker(http: httpx.AsyncClient, api_key: str) -> Rerank:
    async def rerank(query: str, documents: list[str], top_n: int) -> list[Ranking]:
        # The body the AI SDK's Cohere provider sends, field for field.
        response = await _post_with_one_retry(
            http,
            COHERE_RERANK_URL,
            json={"model": RERANK_MODEL, "query": query, "documents": documents, "top_n": top_n},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        # Parse, then construct: an unexpected response shape fails HERE, by name.
        parsed = _CohereRerankResponse.model_validate(response.json())
        return [Ranking(index=r.index, score=r.relevance_score) for r in parsed.results]

    return rerank


# ---- wiring ---------------------------------------------------------------------------


class SearchDocs(Protocol):
    async def __call__(self, query: str, embed_text: str | None = None) -> RetrievalResult: ...


@asynccontextmanager
async def open_search(settings: Settings) -> AsyncIterator[SearchDocs]:
    """Open the DB pool and HTTP clients, yield a ready search function, close them all.

    async with open_search(get_settings()) as search:
        result = await search("how do I stream text")
    """
    # The transaction pooler (port 6543) cannot keep prepared statements between
    # transactions; psycopg would otherwise start preparing after 5 runs of the same query.
    connect_kwargs = {"prepare_threshold": None} if settings.uses_transaction_pooler else {}
    pool = AsyncConnectionPool(
        settings.database_url.get_secret_value(),
        min_size=POOL_MIN_SIZE,
        max_size=POOL_MAX_SIZE,
        kwargs=connect_kwargs,
        open=False,
    )
    # wait=True: fail now, with the real connection error, instead of on the first query
    # after a 30 s pool timeout.
    await pool.open(wait=True, timeout=POOL_OPEN_TIMEOUT_S)
    try:
        async with (
            httpx.AsyncClient(timeout=RERANK_TIMEOUT_S) as http,
            AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()) as openai,
        ):
            embed = openai_embedder(openai)
            vector_search = pgvector_search(pool)
            rerank = cohere_reranker(http, settings.cohere_api_key.get_secret_value())

            async def search(query: str, embed_text: str | None = None) -> RetrievalResult:
                return await search_docs(
                    query,
                    embed_text,
                    embed=embed,
                    vector_search=vector_search,
                    rerank=rerank,
                    candidates=settings.vector_candidates,
                    top_n=settings.rerank_top_n,
                )

            yield search
    finally:
        await pool.close()
