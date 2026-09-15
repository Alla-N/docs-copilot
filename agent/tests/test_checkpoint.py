"""checkpoint.py: what the service checks before it serves, and what its serializer rebuilds.

The Postgres side itself (setup, Row Level Security, a turn saved and loaded) runs against the
real database in test_checkpoint_live.py (integration).
"""

import dataclasses
from datetime import UTC, datetime

import ormsgpack
import pytest
from langgraph._internal._serde import collect_allowlist_from_schemas

from copilot_agent.checkpoint import (
    CHECKPOINT_TABLES,
    PERSISTED_TYPES,
    RETENTION_JOB,
    Retention,
    describe_retention,
    estimated_rows,
    human_bytes,
    readiness_problems,
    serializer,
)
from copilot_agent.github_agent import GitHubEvidence
from copilot_agent.graph import ChatState, GenerationMetrics, SubQueryRetrieval
from copilot_agent.planner import HistoryTurn, Plan, SubQuery, TokenUsage
from copilot_agent.retrieval import RetrievedChunk

LATEST = 9
ALL_ON = dict.fromkeys(CHECKPOINT_TABLES, True)

# The cloud stand-in runs without the compiled ormsgpack and replaces it with a pickle stub,
# which rebuilds anything: the serializer tests only mean something with the real one.
real_msgpack = pytest.mark.skipif(
    getattr(ormsgpack, "STAND_IN", False), reason="ormsgpack is a stand-in stub here"
)


# ---- readiness ----------------------------------------------------------------------------------


def test_a_migrated_locked_down_database_is_ready() -> None:
    assert readiness_problems(LATEST, ALL_ON, latest=LATEST) == []


@pytest.mark.parametrize(
    ("version", "row_security", "expected"),
    [
        pytest.param(None, {}, "do not exist", id="never-set-up"),
        pytest.param(LATEST - 1, ALL_ON, "migration 8, the saver needs 9", id="behind"),
        pytest.param(
            LATEST, ALL_ON | {"checkpoint_blobs": False}, "Row Level Security is off", id="rls-off"
        ),
        pytest.param(
            LATEST,
            {t: True for t in CHECKPOINT_TABLES if t != "checkpoint_writes"},
            "missing tables ['checkpoint_writes']",
            id="table-missing",
        ),
    ],
)
def test_the_service_refuses_an_unready_database(
    version: int | None, row_security: dict[str, bool], expected: str
) -> None:
    problems = readiness_problems(version, row_security, latest=LATEST)
    assert any(expected in p for p in problems), problems
    assert all("copilot_agent.checkpoint setup" in p for p in problems)


def test_every_table_with_rls_off_is_named() -> None:
    [problem] = readiness_problems(LATEST, dict.fromkeys(CHECKPOINT_TABLES, False), latest=LATEST)
    assert all(t in problem for t in CHECKPOINT_TABLES)


# ---- the serializer ---------------------------------------------------------------------------


def test_the_allowlist_is_every_class_in_the_state() -> None:
    # LangGraph's own walk over the schema (what compile() allowlists in strict mode), so a
    # new field type in ChatState that PERSISTED_TYPES lacks fails here, not on a live thread.
    found = collect_allowlist_from_schemas(schemas=[ChatState])
    ours = {(t.__module__, t.__name__) for t in PERSISTED_TYPES}
    assert ours == found


def full_state() -> dict[str, object]:
    chunk = RetrievedChunk(content="c", title="T", source_url="https://x", score=0.5)
    return {
        "question": "q",
        "turns": [HistoryTurn("user", "q0"), HistoryTurn("assistant", "a0")],
        "plan": Plan("search", (SubQuery(query="q", hypothetical="h"),), TokenUsage(1, 2)),
        "retrievals": [
            SubQueryRetrieval(
                "q", relevant=[chunk], mode="reranked", reranked=True, timings_ms={"embed": 1.0}
            )
        ],
        "relevant": [chunk],
        "mode": "reranked",
        "rerank_calls": 1,
        "answer": "a",
        "generation": GenerationMetrics(TokenUsage(3, 4), 1.5, 2.5, "stop"),
        # Step 3.5. The subagent's evidence rides in the thread; its messages, attempts and
        # query data do not, because the subgraph compiles with checkpointer=False.
        "route": "both",
        "router_usage": TokenUsage(300, 4),
        "github": GitHubEvidence(
            question="when was v7 released",
            ok=True,
            evidence="GitHub GraphQL result for: when was v7 released",
            query="query { repository { name } }",
            attempts=2,
            repairs=1,
            first_try_valid=False,
            stages=["field-error", "ok"],
            lookups=2,
            # A LIST on purpose. It went in as a tuple for one commit, and this test failed:
            # msgpack has no tuple (2.5), so the value came back a list and the frozen dataclass
            # no longer equalled itself across a checkpoint.
            types_read=["Repository", "Release"],
            points_spent=1,
            node_count=5,
            usage=TokenUsage(900, 60),
        ),
    }


@real_msgpack
@pytest.mark.parametrize("key", list(full_state()))
def test_every_state_value_comes_back_as_it_went_in(key: str) -> None:
    value = full_state()[key]
    loaded = serializer().loads_typed(serializer().dumps_typed(value))
    assert loaded == value
    assert type(loaded) is type(value)


def test_a_plan_rebuilt_from_a_list_still_holds_a_tuple() -> None:
    # How the serializer rebuilds a dataclass: the class called with the stored fields, and a
    # msgpack array comes back as a list. The round trip above shows it with the real ormsgpack.
    plan = Plan("search", [SubQuery(query="q", hypothetical="h")], TokenUsage(1, 2))  # type: ignore[arg-type]
    assert plan.queries == (SubQuery(query="q", hypothetical="h"),)
    assert hash(plan.queries)


@dataclasses.dataclass(frozen=True)
class NotInTheState:
    text: str


@real_msgpack
def test_a_class_not_in_the_allowlist_is_not_rebuilt() -> None:
    # What strict mode buys: a blob naming any other class does not get it imported and called.
    loaded = serializer().loads_typed(serializer().dumps_typed(NotInTheState("x")))
    assert not isinstance(loaded, NotInTheState)


# ---- retention (db/008), reported and never enforced ---------------------------------------------


def test_no_retention_job_says_the_tables_grow_without_bound() -> None:
    # The whole point of the line. A deploy check that printed nothing here would let a database
    # collect conversations for a year before anybody noticed.
    head, detail = describe_retention(Retention())
    assert "NO " + RETENTION_JOB in head
    assert "grow without bound" in head
    assert "db/008" in detail


def test_a_database_without_pg_cron_is_reported_not_raised() -> None:
    # CI, Docker and a laptop run against databases with no cron schema and no db/008. That is an
    # ordinary state for this command, so read_retention turns the psycopg error into this line.
    head, detail = describe_retention(Retention(unreadable="InvalidSchemaName"))
    assert "InvalidSchemaName" in detail
    assert "NO " + RETENTION_JOB in head


def test_a_scheduled_job_that_has_never_run_is_not_the_same_as_no_job() -> None:
    head, detail = describe_retention(Retention(scheduled="17 3 * * *"))
    assert "17 3 * * *" in head
    assert detail.strip() == "never run"


def test_the_last_run_is_reported_in_utc_whatever_the_session_timezone() -> None:
    # timestamptz comes back in the connection's timezone. The line says UTC, so it converts.
    ran = datetime.fromisoformat("2026-09-13T06:17:00+03:00")
    _, detail = describe_retention(
        Retention(scheduled="17 3 * * *", last_ran=ran, last_retain_days=30, last_threads_deleted=4)
    )
    assert "2026-09-13 03:17 UTC" in detail
    assert "4 threads over 30 days" in detail
    assert ran.astimezone(UTC).hour == 3


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0 B"), (1023, "1023 B"), (1024, "1.0 KiB"), (9532, "9.3 KiB"), (5 << 20, "5.0 MiB")],
)
def test_human_bytes(size: int, expected: str) -> None:
    # 9532 is the measured 9.3 KiB a turn stores at durability=exit: the unit the growth is in.
    assert human_bytes(size) == expected


@pytest.mark.parametrize(
    ("reltuples", "expected"),
    [(-1, "?"), (0, "0"), (916, "916"), (4462, "4,462")],
)
def test_an_unanalysed_table_says_so_instead_of_zero(reltuples: int, expected: str) -> None:
    # The first run of `check` reported 0 rows for checkpoint_migrations, which cannot be empty:
    # the migration number printed on the line above is read out of it. -1 is Postgres saying it
    # has never analysed the table, and a diagnostic must not round that to a confident zero.
    assert estimated_rows(reltuples) == expected
