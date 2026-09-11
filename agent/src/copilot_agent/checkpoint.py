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

One pool of ONE connection: the saver serialises every database call behind a single asyncio.Lock
(AsyncPostgresSaver._cursor in langgraph/checkpoint/postgres/aio.py), even when handed a pool, so
it never uses more than one connection at a time. A separate pool keeps checkpoint traffic from
queueing behind searches for the search pool's connections.
"""

import argparse
import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import errors
from psycopg_pool import AsyncConnectionPool

from copilot_agent.graph import GenerationMetrics, SubQueryRetrieval
from copilot_agent.planner import HistoryTurn, Plan, SubQuery, TokenUsage
from copilot_agent.retrieval import POOL_OPEN_TIMEOUT_S, RetrievedChunk, connection_kwargs
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
)

SETUP_HINT = "run: cd agent && uv run python -m copilot_agent.checkpoint setup"


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
        await pool.close()


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
    print(f"migration {version} (the saver needs {latest_migration()})")
    for table in CHECKPOINT_TABLES:
        print(f"  {table:24} row level security {row_security.get(table, 'missing')}")
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
