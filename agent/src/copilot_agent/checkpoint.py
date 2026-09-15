"""Conversation state in Postgres: LangGraph's AsyncPostgresSaver, on the same Supabase database.

    uv run python -m copilot_agent.checkpoint setup    # once per database, from a terminal
    uv run python -m copilot_agent.checkpoint check    # what the service checks at startup

The saver keeps every thread's checkpoints in four tables (checkpoint_migrations, checkpoints,
checkpoint_blobs, checkpoint_writes). Three things here are decisions, not defaults:

1. The tables are created by `setup`, from a terminal, never by the service at startup. Creating
   them is DDL (and CREATE INDEX CONCURRENTLY), which several containers starting at once would
   race on, and which a service should not need the rights for. Like ingestion (invariant 2),
   changing the database is a deliberate act. The service only CHECKS at startup, and refuses to
   start if the tables are missing, behind the saver's latest migration, or readable by the
   Supabase Data API (point 2).

2. Row Level Security on, with no policies, on all four tables. The saver creates them in the
   `public` schema, which Supabase's Data API (PostgREST) exposes to the anon key, and that key
   is designed to be publishable. The tables hold every question and answer (like query_log,
   db/002_query_log.sql), and checkpoint blobs are deserialised by the service (point 3), so
   write access to them would be worse than a leak. The service connects as the owner, which
   RLS does not restrict.

3. A strict serializer. LangGraph's default one rebuilds any class named in a checkpoint blob by
   importing it ("any Python callable stored in checkpoint data will be imported and executed on
   load", langgraph/checkpoint/serde/_msgpack.py). serializer() allows only LangGraph's safe types
   plus the classes ChatState is made of. A class it does not allow comes back as its raw data,
   and one that fails to rebuild (a field renamed since the blob was written) comes back as None,
   silently: the persisted state is a schema now, and changing HistoryTurn changes how existing
   threads load.

4. Retention is a scheduled job in the database, not code in here. The four tables grow
   without bound and LangGraph ships no TTL: adelete_thread(thread_id) takes a thread id, never
   an age, and it is the only delete the library has. db/008_checkpoint_retention.sql deletes
   whole THREADS whose newest checkpoint is over 30 days old, daily, from pg_cron, in a schema
   Supabase's Data API does not expose. Deliberately NOT part of readiness_problems: pg_cron is
   a Supabase extension, and CI, Docker and a laptop all run against databases without one, so
   a service that refused to start without a retention job would be a service that cannot be
   tested. `check` reports it instead, and deploying with it in place is a checklist item
   (agent/README.md).

One pool of ONE connection: the saver serialises every database call behind a single asyncio.Lock
(AsyncPostgresSaver._cursor in langgraph/checkpoint/postgres/aio.py), even when handed a pool, so
it never uses more than one connection at a time. A separate pool keeps checkpoint traffic from
queueing behind searches for the search pool's connections.
"""

import argparse
import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import psycopg
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import errors
from psycopg_pool import AsyncConnectionPool

from copilot_agent.github_agent import GitHubEvidence
from copilot_agent.graph import GenerationMetrics, SubQueryRetrieval
from copilot_agent.planner import HistoryTurn, Plan, SubQuery, TokenUsage
from copilot_agent.retrieval import (
    POOL_CLOSE_TIMEOUT_S,
    POOL_OPEN_TIMEOUT_S,
    RetrievedChunk,
    connection_kwargs,
)
from copilot_agent.settings import Settings, get_settings

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

CHECKPOINT_TABLES = (
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
)

# Every class a ChatState value is built from. tests/test_checkpoint.py checks this against the
# types LangGraph itself finds in the schema, so a new field type cannot be forgotten here.
PERSISTED_TYPES: tuple[type, ...] = (
    HistoryTurn,
    Plan,
    SubQuery,
    TokenUsage,
    SubQueryRetrieval,
    RetrievedChunk,
    GenerationMetrics,
    # Step 3.5. Only the evidence: the subagent's messages, attempts and query data never reach
    # this list, because the subgraph compiles with checkpointer=False and its channels are not
    # this graph's. Measured, both ways (experiments/subgraph_stream.py, 2026-09-15): with
    # checkpointer=None the same run wrote 9 checkpoints instead of 5 and put `messages`,
    # `result` and `steps` into the PARENT's channel values. The wrapper node counts as nesting,
    # which was the open half of decision 15.
    GitHubEvidence,
)

SETUP_HINT = "run: cd agent && uv run python -m copilot_agent.checkpoint setup"

# db/008_checkpoint_retention.sql: the pg_cron job's name, and the table it writes a row to on
# every run. Both live in schemas the Data API does not serve, which is why a function that
# deletes conversations is not also an HTTP endpoint (the file says more).
RETENTION_JOB = "prune-checkpoint-threads"
RETENTION_LOG = "maintenance.checkpoint_retention_log"
RETENTION_HINT = "run db/008_checkpoint_retention.sql in the Supabase SQL editor"


@dataclass(frozen=True)
class Retention:
    """What db/008 has done to this database. Reported by `check`, never enforced (point 4)."""

    scheduled: str | None = None  # the job's cron schedule; None when there is no job
    last_ran: datetime | None = None
    last_retain_days: int | None = None
    last_threads_deleted: int | None = None
    unreadable: str | None = None  # the psycopg error class, on a database without pg_cron or 008


def describe_retention(retention: Retention) -> list[str]:
    """The two lines `check` prints about retention. Pure, so a test can pin the wording."""
    if retention.scheduled is None:
        head = f"retention: NO {RETENTION_JOB} job - the tables grow without bound"
    else:
        head = f"retention: {RETENTION_JOB} on {retention.scheduled}"
    if retention.unreadable is not None:
        return [head, f"  not readable here ({retention.unreadable}); {RETENTION_HINT}"]
    if retention.scheduled is None:
        return [head, f"  {RETENTION_HINT}"]
    if retention.last_ran is None:
        return [head, "  never run"]
    return [
        head,
        f"  last run {retention.last_ran.astimezone(UTC):%Y-%m-%d %H:%M} UTC: "
        f"{retention.last_threads_deleted} threads over {retention.last_retain_days} days",
    ]


def estimated_rows(reltuples: int) -> str:
    """Postgres writes -1 in reltuples for a table it has never analysed, and it means UNKNOWN.

    The first run of `check` printed "0 rows" for checkpoint_migrations, a table that cannot be
    empty because the migration number on the line above is read out of it. A diagnostic that
    turns "I do not know" into a confident zero is worse than one that says nothing, so an
    unanalysed table says so. `analyse checkpoints;` in the SQL editor refreshes the estimate.
    """
    return "?" if reltuples < 0 else f"{reltuples:,}"


def human_bytes(size: float) -> str:
    if size < 1024:
        return f"{size:.0f} B"
    for unit in ("KiB", "MiB"):
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size / 1024:.1f} GiB"


def serializer() -> JsonPlusSerializer:
    """msgpack, rebuilding only LangGraph's safe types and PERSISTED_TYPES."""
    return JsonPlusSerializer(allowed_msgpack_modules=None).with_msgpack_allowlist(PERSISTED_TYPES)


def latest_migration() -> int:
    """The version the installed saver's migrations end at (the list index of the last one)."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    return len(AsyncPostgresSaver.MIGRATIONS) - 1


def readiness_problems(
    version: int | None, row_security: Mapping[str, bool], latest: int
) -> list[str]:
    """Why the service must not start on this database; empty when it may."""
    if version is None:
        return [f"the checkpoint tables do not exist; {SETUP_HINT}"]
    problems = []
    if version < latest:
        problems.append(
            f"the checkpoint tables are at migration {version}, the saver needs {latest}; "
            f"{SETUP_HINT}"
        )
    missing = [t for t in CHECKPOINT_TABLES if t not in row_security]
    if missing:
        problems.append(f"missing tables {missing}; {SETUP_HINT}")
    exposed = [t for t in CHECKPOINT_TABLES if row_security.get(t) is False]
    if exposed:
        problems.append(
            f"Row Level Security is off on {exposed}: the Supabase Data API would serve every "
            f"conversation to anyone with the anon key; {SETUP_HINT}"
        )
    return problems


async def read_readiness(pool: AsyncConnectionPool) -> tuple[int | None, dict[str, bool]]:
    async with pool.connection() as conn:
        try:
            cur = await conn.execute("select max(v) from checkpoint_migrations")
            row = await cur.fetchone()
            version = row[0] if row else None
        except errors.UndefinedTable:
            version = None
        cur = await conn.execute(
            "select c.relname, c.relrowsecurity from pg_class c "
            "join pg_namespace n on n.oid = c.relnamespace "
            "where n.nspname = current_schema() and c.relkind = 'r' and c.relname = any(%s)",
            (list(CHECKPOINT_TABLES),),
        )
        row_security = {name: bool(on) for name, on in await cur.fetchall()}
    return version, row_security


async def read_retention(pool: AsyncConnectionPool) -> Retention:
    """The retention job and its last run, or the reason neither could be read.

    A database without pg_cron has no cron schema, and one where db/008 has not been run has no
    maintenance schema. Both are ordinary states here (CI, Docker, a laptop), not errors, so the
    psycopg error becomes a line of output instead of an exception. The pool is autocommit
    (retrieval.connection_kwargs), so a failed statement does not leave the next one inside an
    aborted transaction.
    """
    async with pool.connection() as conn:
        try:
            cur = await conn.execute(
                "select schedule from cron.job where jobname = %s", (RETENTION_JOB,)
            )
            row = await cur.fetchone()
        except psycopg.Error as exc:
            return Retention(unreadable=type(exc).__name__)
        scheduled = row[0] if row else None
        try:
            cur = await conn.execute(
                f"select ran_at, retain_days, threads_deleted from {RETENTION_LOG} "
                "order by ran_at desc limit 1"
            )
            row = await cur.fetchone()
        except psycopg.Error as exc:
            return Retention(scheduled=scheduled, unreadable=type(exc).__name__)
    if row is None:
        return Retention(scheduled=scheduled)
    return Retention(
        scheduled=scheduled,
        last_ran=row[0],
        last_retain_days=row[1],
        last_threads_deleted=row[2],
    )


async def read_sizes(pool: AsyncConnectionPool) -> dict[str, tuple[int, int]]:
    """Estimated live rows and total bytes per table, so `check` shows what retention is for.

    reltuples is the planner's estimate, refreshed by analyse and autovacuum, and it is -1 on a
    table that has never been analysed; estimated_rows keeps that distinction. Exact counts would
    be four sequential scans to print a diagnostic line; an estimate is all this line is.
    """
    async with pool.connection() as conn:
        cur = await conn.execute(
            "select c.relname, c.reltuples::bigint, pg_total_relation_size(c.oid) "
            "from pg_class c join pg_namespace n on n.oid = c.relnamespace "
            "where n.nspname = current_schema() and c.relkind = 'r' and c.relname = any(%s)",
            (list(CHECKPOINT_TABLES),),
        )
        return {name: (rows, size) for name, rows, size in await cur.fetchall()}


@asynccontextmanager
async def open_pool(settings: Settings) -> AsyncIterator[AsyncConnectionPool]:
    pool = AsyncConnectionPool(
        settings.database_url.get_secret_value(),
        min_size=1,
        max_size=1,
        kwargs=connection_kwargs(settings),
        open=False,
    )
    await pool.open(wait=True, timeout=POOL_OPEN_TIMEOUT_S)
    try:
        yield pool
    finally:
        await pool.close(timeout=POOL_CLOSE_TIMEOUT_S)


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator["AsyncPostgresSaver"]:
    """A ready saver for the service's lifespan, or a RuntimeError saying what to fix."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with open_pool(settings) as pool:
        problems = readiness_problems(*await read_readiness(pool), latest=latest_migration())
        if problems:
            raise RuntimeError("checkpoint database not ready: " + " | ".join(problems))
        yield AsyncPostgresSaver(pool, serde=serializer())


async def setup(settings: Settings) -> None:
    """Create or migrate the saver's tables, then turn Row Level Security on. Idempotent."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with open_pool(settings) as pool:
        await AsyncPostgresSaver(pool).setup()
        async with pool.connection() as conn:
            for table in CHECKPOINT_TABLES:
                await conn.execute(f"alter table {table} enable row level security")
        await check(pool)


async def check(pool: AsyncConnectionPool) -> None:
    version, row_security = await read_readiness(pool)
    sizes = await read_sizes(pool)
    print(f"migration {version} (the saver needs {latest_migration()})")
    for table in CHECKPOINT_TABLES:
        rows, size = sizes.get(table, (-1, 0))
        security = str(row_security.get(table, "missing"))
        print(
            f"  {table:24} row level security {security:7} "
            f"{estimated_rows(rows):>9} rows {human_bytes(size):>10}"
        )
    for line in describe_retention(await read_retention(pool)):
        print(line)
    problems = readiness_problems(version, row_security, latest=latest_migration())
    print("ready" if not problems else "NOT READY:\n  " + "\n  ".join(problems))


async def _check_only(settings: Settings) -> None:
    async with open_pool(settings) as pool:
        await check(pool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["setup", "check"])
    args = parser.parse_args()
    settings = get_settings()
    asyncio.run(setup(settings) if args.command == "setup" else _check_only(settings))


if __name__ == "__main__":
    main()
