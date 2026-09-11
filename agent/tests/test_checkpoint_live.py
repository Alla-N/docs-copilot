"""The checkpointer against the REAL Supabase database: set up, locked down, and a conversation
saved and read back through the strict serializer.

Marked integration (network, secrets), so the pre-commit hook skips it. Costs nothing: the
planner, search and model are fakes, only the database is real. Run it on purpose, after
`uv run python -m copilot_agent.checkpoint setup`:

    uv run pytest -m integration tests/test_checkpoint_live.py
"""

import secrets
from collections.abc import AsyncIterator
from typing import Any

import psycopg
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from copilot_agent.checkpoint import open_checkpointer
from copilot_agent.graph import build_graph
from copilot_agent.planner import HistoryTurn
from copilot_agent.retrieval import RetrievalResult, RetrievedChunk
from copilot_agent.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def planner(messages: Any) -> dict[str, Any]:
    parsed = {"intent": "search", "queries": [{"query": "stream text", "hypothetical": "h"}]}
    return {"raw": AIMessage(content=""), "parsed": parsed}


async def search(query: str, embed_text: str | None = None) -> RetrievalResult:
    chunk = RetrievedChunk(content="Use streamText.", title="T", source_url="https://x", score=0.7)
    return RetrievalResult(candidates=[], relevant=[chunk], mode="reranked", timings_ms={})


@pytest.fixture
async def thread_id() -> AsyncIterator[str]:
    """A fresh thread, deleted afterwards, so the test leaves no rows behind."""
    thread_id = "live-test-" + secrets.token_hex(8)
    yield thread_id
    async with open_checkpointer(get_settings()) as saver:
        await saver.adelete_thread(thread_id)


async def test_a_conversation_survives_the_round_trip(thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    async with open_checkpointer(get_settings()) as saver:
        graph = build_graph(
            planner=RunnableLambda(planner),
            search=search,
            model=GenericFakeChatModel(messages=iter(["first answer", "second answer"])),
            checkpointer=saver,
        )
        await graph.ainvoke({"question": "how do I stream text"}, config)
    # A new saver and a new graph, as after a restart: only the database remembers.
    async with open_checkpointer(get_settings()) as saver:
        graph = build_graph(
            planner=RunnableLambda(planner),
            search=search,
            model=GenericFakeChatModel(messages=iter(["second answer"])),
            checkpointer=saver,
        )
        await graph.ainvoke({"question": "and configure it?"}, config)
        turns = (await graph.aget_state(config)).values["turns"]
    assert turns == [
        HistoryTurn("user", "how do I stream text"),
        HistoryTurn("assistant", "first answer"),
        HistoryTurn("user", "and configure it?"),
        HistoryTurn("assistant", "second answer"),
    ]


async def test_the_data_api_roles_see_no_rows(thread_id: str) -> None:
    # RLS with no policies: the anon role (what the Supabase Data API runs a request with the
    # anon key as) sees nothing, while the owner, which the service connects as, sees the rows.
    config = {"configurable": {"thread_id": thread_id}}
    async with open_checkpointer(get_settings()) as saver:
        graph = build_graph(
            planner=RunnableLambda(planner),
            search=search,
            model=GenericFakeChatModel(messages=iter(["an answer"])),
            checkpointer=saver,
        )
        await graph.ainvoke({"question": "how do I stream text"}, config)

    url = get_settings().database_url.get_secret_value()
    counts: dict[str, int] = {}
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        for role in ("owner", "anon", "authenticated"):
            if role != "owner":
                await conn.execute(f"set role {role}")
            cur = await conn.execute(
                "select (select count(*) from checkpoints where thread_id = %s)"
                " + (select count(*) from checkpoint_blobs where thread_id = %s)",
                (thread_id, thread_id),
            )
            row = await cur.fetchone()
            counts[role] = row[0] if row else 0
            await conn.execute("reset role")
    assert counts["owner"] > 0
    assert counts["anon"] == counts["authenticated"] == 0
