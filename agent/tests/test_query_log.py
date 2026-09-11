"""query_log.py: the row, the shapes it accepts, and a writer that never touches the answer.

What /chat writes, end to end, is tested in tests/test_chat_api.py ("the query log" section);
that the columns are TypeScript's is tested in tests/test_ts_parity.py. Here: the parts.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from copilot_agent.graph import GenerationMetrics
from copilot_agent.planner import NO_USAGE, Plan, SubQuery, TokenUsage
from copilot_agent.query_log import (
    COLUMNS,
    INSERT_SQL,
    QueryLog,
    Turn,
    Visitor,
    js_round,
    observe,
    readiness_problems,
)
from copilot_agent.retrieval import RetrievedChunk

pytestmark = pytest.mark.anyio

HASH = "0123456789abcdef0123456789abcdef"


# ---- rounding like Math.round ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Values and results checked in node: Math.round(x).
        (0.5, 1),
        (1.5, 2),
        (2.5, 3),  # Python's round(2.5) is 2
        (1234.5, 1235),
        (1234.4999, 1234),
        (0.49999999999999994, 0),  # floor(x + 0.5) would say 1
        (-0.5, 0),
        (-1.5, -1),
        (7.0, 7),
    ],
)
def test_js_round_is_math_round(value: float, expected: int) -> None:
    assert js_round(value) == expected


# ---- visitor shapes ---------------------------------------------------------------------------


def test_a_visitor_in_the_right_shape_is_kept() -> None:
    fields = dict(
        visitor_hash=HASH,
        landing_referrer="linkedin.com",
        utm_source="cv_2026",
        country="GR",
        device="mobile",
    )
    assert Visitor.sanitised(**fields) == Visitor(**fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("visitor_hash", HASH[:-1]),
        ("visitor_hash", HASH.upper()),
        ("visitor_hash", HASH + "\n"),
        ("landing_referrer", "LinkedIn.com"),
        ("landing_referrer", "linkedin.com/feed"),
        # Python's re $ matches before a trailing newline; fullmatch does not.
        ("landing_referrer", "linkedin.com\n"),
        ("landing_referrer", "x" * 101),
        ("utm_source", "cv 2026"),
        ("utm_source", "x" * 41),
        ("country", "gr"),
        ("country", "GRC"),
        ("device", "tablet"),
    ],
)
def test_a_field_in_the_wrong_shape_becomes_none(field: str, value: str) -> None:
    good = dict(
        visitor_hash=HASH,
        landing_referrer="linkedin.com",
        utm_source="cv",
        country="GR",
        device="desktop",
    )
    visitor = Visitor.sanitised(**(good | {field: value}))
    assert getattr(visitor, field) is None
    assert all(getattr(visitor, name) == good[name] for name in good if name != field)


def test_missing_fields_stay_none() -> None:
    assert Visitor.sanitised(None, None, None, None, None) == Visitor(None, None, None, None, None)


# ---- the turn and its row ---------------------------------------------------------------------


def updates(node: str, update: dict[str, Any]) -> dict[str, Any]:
    return {"type": "updates", "ns": (), "data": {node: update}}


def chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(content="c", title="t", source_url="https://x", score=score)


PLAN = Plan(
    intent="search", queries=(SubQuery(query="q", hypothetical=""),), usage=TokenUsage(10, 2)
)


def test_a_turn_completes_only_on_its_last_node() -> None:
    turn = Turn(question="q", thread_id="t" * 16, origin="web")
    assert turn.see(updates("plan", {"plan": PLAN})) is False
    assert turn.see({"type": "messages", "ns": (), "data": (object(), {})}) is False
    merged = {"relevant": [chunk(0.7)], "mode": "reranked", "rerank_calls": 1}
    assert turn.see(updates("merge", merged)) is False
    with pytest.raises(ValueError, match="not completed"):
        turn.row()
    generation = GenerationMetrics(
        usage=TokenUsage(100, 20), ttft_ms=812.5, generation_ms=1500.5, finish_reason="stop"
    )
    assert turn.see(updates("generate", {"answer": "Use it.", "generation": generation})) is True
    turn.retrieval_ms = 2.5  # measured in the test run; set to a half to pin the rounding
    row = turn.row()
    assert (row["latency_ms"], row["ttft_ms"], row["generation_ms"]) == (3, 813, 1501)
    assert (row["planner_input_tokens"], row["gen_output_tokens"]) == (10, 20)
    assert (row["top_score"], row["chunk_count"], row["refused"]) == (0.7, 1, False)


def test_a_planner_that_fell_back_logs_no_planner_tokens() -> None:
    # plan_query's fallback carries NO_USAGE, as TypeScript's does: unknown, not zero.
    turn = Turn(question="q", thread_id="t" * 16, origin="web")
    turn.see(updates("plan", {"plan": Plan(intent="search", queries=(), usage=NO_USAGE)}))
    canned = {"answer": "hi", "relevant": [], "mode": "skipped", "rerank_calls": 0}
    assert turn.see(updates("canned", canned)) is True
    row = turn.row()
    assert (row["planner_input_tokens"], row["planner_output_tokens"]) == (None, None)


def test_the_insert_names_every_column_once() -> None:
    assert len(set(COLUMNS)) == len(COLUMNS)
    assert INSERT_SQL.count("%(") == len(COLUMNS)
    for column in COLUMNS:
        assert f"%({column})s" in INSERT_SQL


# ---- observe(): the stream goes past unchanged ------------------------------------------------


async def parts_of(*parts: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    for part in parts:
        yield part


RUN = (
    updates("plan", {"plan": PLAN}),
    updates("canned", {"answer": "hi", "relevant": [], "mode": "skipped", "rerank_calls": 0}),
)


async def test_observe_records_the_turn_before_passing_its_last_part_on() -> None:
    completed: list[Turn] = []
    turn = Turn(question="q", thread_id="t" * 16, origin="web")
    stream = observe(parts_of(*RUN), turn, completed.append)
    assert await anext(stream) is RUN[0]
    assert completed == []
    assert await anext(stream) is RUN[1]
    assert completed == [turn]  # already, when the client gets the last part


async def test_observe_never_fails_the_stream(caplog: pytest.LogCaptureFixture) -> None:
    def broken(turn: Turn) -> None:
        raise RuntimeError("log bug")

    turn = Turn(question="q", thread_id="t" * 16, origin="web")
    seen = [p async for p in observe(parts_of(*RUN), turn, broken)]
    assert seen == list(RUN)
    assert "could not record the turn" in caplog.text


async def test_closing_observe_closes_the_graph_stream() -> None:
    closed = asyncio.Event()

    async def graph_stream() -> AsyncIterator[dict[str, Any]]:
        try:
            yield RUN[0]
            yield RUN[1]
        finally:
            closed.set()

    stream = observe(graph_stream(), Turn("q", "t" * 16, "web"), lambda _: None)
    await anext(stream)
    await stream.aclose()
    assert closed.is_set()


# ---- the writer ----------------------------------------------------------------------------


async def test_write_returns_before_the_insert_and_drain_waits_for_it() -> None:
    release = asyncio.Event()
    written: list[Any] = []

    async def insert(row: Any) -> None:
        await release.wait()
        written.append(row)

    log = QueryLog(insert)
    log.write({"question": "q"})
    assert log.pending == 1 and written == []
    release.set()
    assert await log.drain() == 0
    assert written == [{"question": "q"}] and log.pending == 0


async def test_drain_gives_up_after_its_wait_and_says_how_many(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def never(row: Any) -> None:
        await asyncio.Event().wait()

    log = QueryLog(never)
    log.write({})
    log.write({})
    assert await log.drain(wait_s=0.05) == 2
    assert "2 rows not written" in caplog.text


async def test_a_failed_insert_is_logged_without_the_row(caplog: pytest.LogCaptureFixture) -> None:
    async def failing(row: Any) -> None:
        raise ConnectionError("pooler down")

    log = QueryLog(failing)
    with caplog.at_level(logging.ERROR):
        log.write({"question": "my private question"})
        assert await log.drain() == 0
    assert "query log insert failed" in caplog.text
    assert "my private question" not in caplog.text


# ---- readiness ----------------------------------------------------------------------------------


def test_ready_when_every_column_exists_and_rls_is_on() -> None:
    assert readiness_problems(set(COLUMNS) | {"id", "created_at"}, row_security=True) == []


def test_not_ready_without_the_006_columns() -> None:
    [problem] = readiness_problems(set(COLUMNS) - {"origin", "thread_id"}, row_security=True)
    assert "'origin'" in problem and "'thread_id'" in problem and "db/006" in problem


def test_not_ready_without_the_table() -> None:
    [problem] = readiness_problems(set(), row_security=None)
    assert "does not exist" in problem


def test_not_ready_with_row_level_security_off() -> None:
    [problem] = readiness_problems(set(COLUMNS), row_security=False)
    assert "Row Level Security is off" in problem


# ---- open_query_log(): the database glue, on a fake pool ----------------------------------------


class FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None


class FakeDatabase:
    """Answers the two readiness queries and records every other statement."""

    def __init__(self, columns: set[str], row_security: bool | None) -> None:
        self.columns = columns
        self.row_security = row_security
        self.executed: list[tuple[str, Any]] = []
        self.gate: asyncio.Event | None = None

    async def execute(self, sql: str, params: Any = None) -> FakeCursor:
        if "information_schema.columns" in sql:
            return FakeCursor([(c,) for c in self.columns])
        if "relrowsecurity" in sql:
            return FakeCursor([] if self.row_security is None else [(self.row_security,)])
        if self.gate is not None:
            await self.gate.wait()
        self.executed.append((sql, params))
        return FakeCursor([])

    @asynccontextmanager
    async def connection(self) -> AsyncIterator["FakeDatabase"]:
        yield self


def fake_pool(monkeypatch: pytest.MonkeyPatch, database: FakeDatabase) -> list[str]:
    from copilot_agent import query_log

    events: list[str] = []

    @asynccontextmanager
    async def open_pool(settings: Any) -> AsyncIterator[FakeDatabase]:
        events.append("open")
        try:
            yield database
        finally:
            events.append("close")

    monkeypatch.setattr(query_log, "open_pool", open_pool)
    return events


async def test_open_query_log_refuses_a_database_without_the_006_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copilot_agent.query_log import open_query_log

    database = FakeDatabase(set(COLUMNS) - {"origin"}, row_security=True)
    events = fake_pool(monkeypatch, database)
    with pytest.raises(RuntimeError, match="db/006"):
        async with open_query_log(None):  # type: ignore[arg-type]
            pass
    assert events == ["open", "close"]


async def test_open_query_log_inserts_every_column_and_drains_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copilot_agent.query_log import open_query_log

    database = FakeDatabase(set(COLUMNS), row_security=True)
    database.gate = asyncio.Event()
    events = fake_pool(monkeypatch, database)
    row = dict.fromkeys(COLUMNS)
    async with open_query_log(None) as log:  # type: ignore[arg-type]
        log.write(row)
        await asyncio.sleep(0)
        assert database.executed == []  # still in flight when the lifespan ends
        asyncio.get_running_loop().call_later(0.05, database.gate.set)
    # Leaving waited for it, and only then closed the pool.
    assert database.executed == [(INSERT_SQL, row)]
    assert events == ["open", "close"]
