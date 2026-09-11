"""What does the Postgres checkpointer cost per turn? Time, round trips and bytes, per durability.

    cd agent && uv run python experiments/checkpoint_overhead.py

The real AsyncPostgresSaver on the real Supabase database (checkpoint.open_checkpointer: its own
one-connection pool, the strict serializer); every other service is an instant fake, so what is
left is the checkpointer. A turn is shaped like a real answered one: two sub-queries, each
retrieval carrying 100 candidates and 5 kept chunks of REAL chunk text (read from the documents
table, so sizes and compressibility are the corpus's), and a 1200-character answer.

Variants: no checkpointer (the baseline), then the saver with durability "sync", "async" and
"exit". Each runs THREADS conversations of TURNS turns (history grows, like a real chat), and the
whole sequence is repeated REPEATS times; one repeat is not a measurement. Per variant it prints
the median time per turn (the whole run: the stream ends only after LangGraph's last save), the
median time to the first answer token (what a user feels), and what the turns left in the
database, per turn: rows, serialized bytes (octet_length) and stored bytes (pg_column_size, after
Postgres's compression). The threads are deleted at the end.

Free: database only, no model or API calls.
"""

import asyncio
import itertools
import secrets
import statistics
import time
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from psycopg_pool import AsyncConnectionPool

from copilot_agent.checkpoint import open_checkpointer, open_pool
from copilot_agent.graph import build_graph
from copilot_agent.retrieval import Candidate, RetrievalResult, RetrievedChunk
from copilot_agent.settings import get_settings

THREADS = 3
TURNS = 4
REPEATS = 2
VARIANTS = ["none", "sync", "async", "exit"]
ANSWER = " ".join(["Use streamText to stream the answer (Source 1)."] * 25)  # ~1200 chars
PLAN = {
    "intent": "search",
    "queries": [
        {"query": "How do I stream text?", "hypothetical": "h"},
        {"query": "What is generateText?", "hypothetical": "h"},
    ],
}


async def planner_reply(_: Any) -> dict[str, Any]:
    usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    return {"raw": AIMessage(content="", usage_metadata=usage), "parsed": PLAN}


async def real_chunks(pool: AsyncConnectionPool) -> list[tuple[str, str, str]]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "select content, title, source_url from documents order by id limit 100"
        )
        return list(await cur.fetchall())


def fake_search(rows: list[tuple[str, str, str]]) -> Any:
    candidates = [Candidate(content=c, title=t, source_url=u, similarity=0.5) for c, t, u in rows]
    kept = [RetrievedChunk(content=c, title=t, source_url=u, score=0.7) for c, t, u in rows[:5]]

    async def search(query: str, embed_text: str | None = None) -> RetrievalResult:
        return RetrievalResult(
            candidates=candidates, relevant=kept, mode="reranked", timings_ms={"embed": 1.0}
        )

    return search


async def one_turn(graph: Any, thread_id: str | None, durability: str) -> tuple[float, float]:
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    kwargs = {"durability": durability} if thread_id else {}
    started = time.perf_counter()
    first: float | None = None
    async for part in graph.astream(
        {"question": "how do I stream text"},
        config,
        stream_mode=["updates", "messages"],
        version="v2",
        **kwargs,
    ):
        if part["type"] == "messages" and first is None:
            first = time.perf_counter()
    ended = time.perf_counter()
    return (ended - started) * 1000, ((first or ended) - started) * 1000


async def stored(pool: AsyncConnectionPool, thread_ids: list[str]) -> dict[str, float]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            select
              (select count(*) from checkpoints where thread_id = any(%(t)s)),
              (select count(*) from checkpoint_blobs where thread_id = any(%(t)s)),
              (select count(*) from checkpoint_writes where thread_id = any(%(t)s)),
              (select coalesce(sum(octet_length(checkpoint::text)
                                   + octet_length(metadata::text)), 0)
                 from checkpoints where thread_id = any(%(t)s)),
              (select coalesce(sum(pg_column_size(checkpoint) + pg_column_size(metadata)), 0)
                 from checkpoints where thread_id = any(%(t)s)),
              (select coalesce(sum(octet_length(blob)), 0)
                 from checkpoint_blobs where thread_id = any(%(t)s)),
              (select coalesce(sum(pg_column_size(blob)), 0)
                 from checkpoint_blobs where thread_id = any(%(t)s)),
              (select coalesce(sum(octet_length(blob)), 0)
                 from checkpoint_writes where thread_id = any(%(t)s)),
              (select coalesce(sum(pg_column_size(blob)), 0)
                 from checkpoint_writes where thread_id = any(%(t)s))
            """,
            {"t": thread_ids},
        )
        row = await cur.fetchone()
    assert row is not None
    names = ["checkpoints", "blobs", "writes", "ck_bytes", "ck_stored", "blob_bytes",
             "blob_stored", "write_bytes", "write_stored"]  # fmt: skip
    return dict(zip(names, (float(v) for v in row), strict=True))


async def main() -> None:
    settings = get_settings()
    async with open_pool(settings) as pool, open_checkpointer(settings) as saver:
        search = fake_search(await real_chunks(pool))
        created: list[str] = []
        results: dict[str, list[tuple[float, float]]] = {v: [] for v in VARIANTS}
        threads_of: dict[str, list[str]] = {v: [] for v in VARIANTS}
        try:
            for repeat in range(REPEATS):
                for variant in VARIANTS:
                    graph = build_graph(
                        planner=RunnableLambda(planner_reply),
                        search=search,
                        # Endless: the warm-up turn answers too.
                        model=GenericFakeChatModel(messages=itertools.repeat(ANSWER)),
                        checkpointer=None if variant == "none" else saver,
                    )
                    await one_turn(graph, None if variant == "none" else "warmup-x", "async")
                    for _ in range(THREADS):
                        thread_id = f"overhead-{variant}-{secrets.token_hex(6)}"
                        created.append(thread_id)
                        threads_of[variant].append(thread_id)
                        for _ in range(TURNS):
                            tid = None if variant == "none" else thread_id
                            results[variant].append(await one_turn(graph, tid, variant))
                    totals = [t for t, _ in results[variant][-THREADS * TURNS :]]
                    firsts = [f for _, f in results[variant][-THREADS * TURNS :]]
                    print(
                        f"repeat {repeat + 1}  {variant:5}"
                        f"  turn {statistics.median(totals):7.1f} ms"
                        f"  first token {statistics.median(firsts):7.1f} ms"
                    )
            print(f"\nper turn, {REPEATS * THREADS * TURNS} turns per variant:")
            n = REPEATS * THREADS * TURNS
            for variant in VARIANTS[1:]:
                s = await stored(pool, threads_of[variant])
                kib = {k: v / n / 1024 for k, v in s.items()}
                serialized = kib["ck_bytes"] + kib["blob_bytes"] + kib["write_bytes"]
                on_disk = kib["ck_stored"] + kib["blob_stored"] + kib["write_stored"]
                print(
                    f"  {variant:5}  rows {s['checkpoints'] / n:.1f} checkpoints"
                    f" + {s['blobs'] / n:.1f} blobs + {s['writes'] / n:.1f} writes"
                    f"   serialized {serialized:6.1f} KiB (blobs {kib['blob_bytes']:.1f},"
                    f" writes {kib['write_bytes']:.1f})   stored {on_disk:6.1f} KiB"
                )
        finally:
            for thread_id in [*created, "warmup-x"]:
                await saver.adelete_thread(thread_id)


if __name__ == "__main__":
    asyncio.run(main())
