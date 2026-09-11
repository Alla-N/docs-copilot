"""checkpoint.py: what the service checks before it serves, and what its serializer rebuilds.

The Postgres side itself (setup, Row Level Security, a turn saved and loaded) runs against the
real database in test_checkpoint_live.py (integration).
"""

import dataclasses

import ormsgpack
import pytest
from langgraph._internal._serde import collect_allowlist_from_schemas

from copilot_agent.checkpoint import (
    CHECKPOINT_TABLES,
    PERSISTED_TYPES,
    readiness_problems,
    serializer,
)
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
