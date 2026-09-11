"""The query log, written by the service for every turn it completes (step 2.6).

lib/query-log.ts writes one query_log row per request the TypeScript route answers. Once the route
forwards to this service (AGENT_URL), the route is a byte pipe: it no longer sees the planner's
tokens, the rerank calls, the answer's tokens or its timings, so it cannot write that row. This
service can, and does, with the same columns and the same meaning, plus two (db/006_origin.sql):

  - origin: "web" for the Next.js route, "eval" for the eval harness. Every view over the table
    counts web rows only, so an eval run neither looks like traffic nor mines itself as eval
    cases, and the harness reads its own rows back for a MEASURED cost per request.
  - thread_id: the conversation. One thread, many rows; the same key Langfuse sessions will use.

One row per COMPLETED turn, the same rule the thread follows (invariant 13): the row is handed
to the writer when the graph's last node (canned or generate) reports, and a turn that fails or
is cancelled writes nothing, as TypeScript's onFinish never fires for one.

Writing never touches the response. QueryLog.write() starts the insert in a background task and
returns at once (the Python counterpart of Next's after(), invariant 10); a failed insert is
logged and dropped, because a telemetry bug should not become an outage. At shutdown the lifespan
waits a bounded time for inserts still running (QueryLog.drain), so a deploy that stops the
container does not silently lose the last rows.
"""

import asyncio
import logging
import math
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from psycopg_pool import AsyncConnectionPool

from copilot_agent.checkpoint import open_pool
from copilot_agent.graph import GenerationMetrics
from copilot_agent.planner import NO_USAGE, TokenUsage
from copilot_agent.refusal import is_refusal
from copilot_agent.retrieval import RetrievalMode, RetrievedChunk
from copilot_agent.settings import Settings

logger = logging.getLogger(__name__)

Origin = Literal["web", "eval"]

# Every column this service writes, in insert order. db/002, 003 and 005 created all but the last
# two, which db/006_origin.sql adds; the service refuses to start without any of them.
COLUMNS = (
    "question",
    "refused",
    "chunk_count",
    "top_score",
    "retrieval_mode",
    "latency_ms",
    "visitor_hash",
    "landing_referrer",
    "utm_source",
    "country",
    "device",
    "planner_input_tokens",
    "planner_output_tokens",
    "rerank_calls",
    "gen_input_tokens",
    "gen_output_tokens",
    "ttft_ms",
    "generation_ms",
    "origin",
    "thread_id",
)

INSERT_SQL = (
    f"insert into query_log ({', '.join(COLUMNS)}) "
    f"values ({', '.join(f'%({c})s' for c in COLUMNS)})"
)

SETUP_HINT = "run db/006_origin.sql in the Supabase SQL editor"

# How long shutdown waits for inserts still running. One insert is one round trip (~70 ms on the
# pooler, measured in step 1e); ECS gives a stopping task 30 s in all.
DRAIN_TIMEOUT_S = 5.0

Row = Mapping[str, object]


def js_round(value: float) -> int:
    """JavaScript's Math.round: halves go UP. Python's round() sends them to the even neighbour
    (round(2.5) is 2), which would make every x.5 ms timing differ from a TypeScript row."""
    floor = math.floor(value)
    return floor + 1 if value - floor >= 0.5 else floor


# The shapes lib/visitor.ts lets through, without their ^ and $: matched with re.fullmatch, because
# Python's $ also matches before a trailing newline ("linkedin.com\n" would pass ^...$ here).
# tests/test_ts_parity.py compares ^<shape>$ with the TypeScript regexes.
REFERRER_HOST = "[a-z0-9.-]{1,100}"
UTM_SOURCE = "[a-z0-9_-]{1,40}"
COUNTRY_ISO2 = "[A-Z]{2}"
# clientKey() in lib/rate-limit.ts: the first 32 hex characters of a sha256.
VISITOR_HASH = "[0-9a-f]{32}"
DEVICES = ("mobile", "desktop")


def _shaped(value: str | None, shape: str) -> str | None:
    return value if value is not None and re.fullmatch(shape, value) else None


@dataclass(frozen=True)
class Visitor:
    """Visitor attribution for the row (db/003): which channel sent the person, never who they
    are. The Next.js route computes it from request headers (lib/visitor.ts) and forwards it."""

    visitor_hash: str | None
    landing_referrer: str | None
    utm_source: str | None
    country: str | None
    device: str | None

    @classmethod
    def sanitised(
        cls,
        visitor_hash: str | None,
        landing_referrer: str | None,
        utm_source: str | None,
        country: str | None,
        device: str | None,
    ) -> "Visitor":
        """Every field in the shape lib/visitor.ts allows, or None. The route already sanitised
        them, so a mismatch is a bug on one side; it costs the field, not the request: telemetry
        must not turn into a failed answer."""
        return cls(
            visitor_hash=_shaped(visitor_hash, VISITOR_HASH),
            landing_referrer=_shaped(landing_referrer, REFERRER_HOST),
            utm_source=_shaped(utm_source, UTM_SOURCE),
            country=_shaped(country, COUNTRY_ISO2),
            device=device if device in DEVICES else None,
        )


NO_VISITOR = Visitor(None, None, None, None, None)


@dataclass
class Turn:
    """What one /chat request did, gathered from the graph's stream as it goes past."""

    question: str
    thread_id: str
    origin: Origin
    visitor: Visitor = NO_VISITOR
    started: float = field(default_factory=time.perf_counter)
    planner_usage: TokenUsage = NO_USAGE
    relevant: list[RetrievedChunk] = field(default_factory=list)
    mode: RetrievalMode | None = None
    rerank_calls: int = 0
    # Planner + retrieval, like TypeScript's latency_ms (plannedRetrieve's duration): from the
    # start of the run to the merged sources, or to the canned reply.
    retrieval_ms: float | None = None
    answer: str | None = None
    generation: GenerationMetrics | None = None

    def see(self, part: Mapping[str, Any]) -> bool:
        """Take what the row needs from one stream part. True when the turn just completed."""
        if part["type"] != "updates":
            return False
        completed = False
        for node, update in part["data"].items():
            if node == "plan":
                self.planner_usage = update["plan"].usage
            elif node in ("merge", "canned"):
                self.relevant = update["relevant"]
                self.mode = update["mode"]
                self.rerank_calls = update["rerank_calls"]
                self.retrieval_ms = (time.perf_counter() - self.started) * 1000
                if node == "canned":
                    self.answer = update["answer"]
                    completed = True
            elif node == "generate":
                self.answer = update["answer"]
                self.generation = update["generation"]
                completed = True
        return completed

    def row(self) -> dict[str, object]:
        """The query_log row, column for column what lib/query-log.ts writes for the same turn."""
        if self.answer is None or self.mode is None or self.retrieval_ms is None:
            raise ValueError("the turn has not completed")
        generation = self.generation
        return {
            "question": self.question,
            "refused": is_refusal(self.answer),
            "chunk_count": len(self.relevant),
            "top_score": self.relevant[0].score if self.relevant else None,
            "retrieval_mode": self.mode,
            "latency_ms": js_round(self.retrieval_ms),
            "visitor_hash": self.visitor.visitor_hash,
            "landing_referrer": self.visitor.landing_referrer,
            "utm_source": self.visitor.utm_source,
            "country": self.visitor.country,
            "device": self.visitor.device,
            "planner_input_tokens": self.planner_usage.input_tokens,
            "planner_output_tokens": self.planner_usage.output_tokens,
            "rerank_calls": self.rerank_calls,
            "gen_input_tokens": generation.usage.input_tokens if generation else None,
            "gen_output_tokens": generation.usage.output_tokens if generation else None,
            "ttft_ms": (
                js_round(generation.ttft_ms)
                if generation and generation.ttft_ms is not None
                else None
            ),
            "generation_ms": js_round(generation.generation_ms) if generation else None,
            "origin": self.origin,
            "thread_id": self.thread_id,
        }


async def observe(
    parts: AsyncIterator[dict[str, Any]],
    turn: Turn,
    on_complete: Callable[[Turn], None],
) -> AsyncIterator[dict[str, Any]]:
    """Pass a graph run's stream through unchanged, filling `turn` from it, and call on_complete
    once the turn has completed (before passing that last update on, so a client that leaves
    right then does not lose the row of a turn the thread has recorded).

    Nothing here may fail the answer: an error while reading a part or recording the turn is
    logged, and the part goes on to the client all the same.
    """
    async with aclosing(parts) as events:
        async for part in events:
            try:
                if turn.see(part):
                    on_complete(turn)
            except Exception:
                logger.exception("query log: could not record the turn")
            yield part


Insert = Callable[[Row], Awaitable[None]]


class QueryLog:
    """Inserts rows in the background and never lets a failure reach the caller."""

    def __init__(self, insert: Insert) -> None:
        self._insert = insert
        # Strong references: asyncio keeps only weak ones to tasks (see api._unwinding).
        self._pending: set[asyncio.Task[None]] = set()

    def write(self, row: Row) -> None:
        task = asyncio.create_task(self._safe_insert(row))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _safe_insert(self, row: Row) -> None:
        try:
            await self._insert(row)
        except Exception:
            # Not the row: it holds the user's question.
            logger.exception("query log insert failed")

    @property
    def pending(self) -> int:
        return len(self._pending)

    async def drain(self, wait_s: float = DRAIN_TIMEOUT_S) -> int:
        """Wait up to `wait_s` seconds for inserts still running; return how many were left."""
        if not self._pending:
            return 0
        _, still_running = await asyncio.wait(set(self._pending), timeout=wait_s)
        if still_running:
            logger.error("query log: %d rows not written at shutdown", len(still_running))
        return len(still_running)


def postgres_insert(pool: AsyncConnectionPool) -> Insert:
    async def insert(row: Row) -> None:
        async with pool.connection() as conn:
            await conn.execute(INSERT_SQL, row)

    return insert


def readiness_problems(columns: set[str], row_security: bool | None) -> list[str]:
    """Why the service must not start on this database; empty when it may."""
    if row_security is None:
        return [f"the query_log table does not exist; run db/002 to db/006 ({SETUP_HINT})"]
    problems = []
    missing = [c for c in COLUMNS if c not in columns]
    if missing:
        problems.append(f"query_log has no column {missing}; {SETUP_HINT}")
    if not row_security:
        problems.append(
            "Row Level Security is off on query_log: the Supabase Data API would serve every "
            "question ever asked to anyone with the anon key (db/002_query_log.sql)"
        )
    return problems


async def read_readiness(pool: AsyncConnectionPool) -> tuple[set[str], bool | None]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "select column_name from information_schema.columns "
            "where table_schema = current_schema() and table_name = 'query_log'"
        )
        columns = {name for (name,) in await cur.fetchall()}
        cur = await conn.execute(
            "select c.relrowsecurity from pg_class c "
            "join pg_namespace n on n.oid = c.relnamespace "
            "where n.nspname = current_schema() and c.relkind = 'r' and c.relname = 'query_log'"
        )
        row = await cur.fetchone()
    return columns, (bool(row[0]) if row else None)


@asynccontextmanager
async def open_query_log(settings: Settings) -> AsyncIterator[QueryLog]:
    """A query log for the service's lifespan, or a RuntimeError saying what to fix.

    Its own pool of one connection, like the checkpointer's: inserts are one statement each and
    run after the answer, and on their own pool they never wait behind a search. That makes at
    most six connections per process (search 4, checkpoints 1, log 1) on Supabase's pooler.
    """
    async with open_pool(settings) as pool:
        problems = readiness_problems(*await read_readiness(pool))
        if problems:
            raise RuntimeError("query log database not ready: " + " | ".join(problems))
        log = QueryLog(postgres_insert(pool))
        try:
            yield log
        finally:
            await log.drain()
