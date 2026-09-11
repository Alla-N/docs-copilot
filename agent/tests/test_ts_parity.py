"""The Python port must use the same calibrated numbers as lib/retrieve.ts and lib/plan.ts.

While two retrieval implementations exist, a threshold changed on one side only would be
a silent drift (CLAUDE.md invariant 3). This test reads the TypeScript source and compares.
"""

import re
from pathlib import Path

import pytest

from copilot_agent import retrieval_config

# agent/tests/this_file.py -> parents[2] is the repo root.
LIB = Path(__file__).resolve().parents[2] / "lib"


def ts_default(file: str, name: str) -> float:
    """Read a constant's default from a TypeScript file in lib/.

    Matches the three shapes used there:
        export const RERANK_THRESHOLD = 0.3;
        export const VECTOR_CANDIDATES = Number(process.env.VECTOR_CANDIDATES ?? 100);
        const UNION_CAP = 8;
    """
    source = (LIB / file).read_text()
    pattern = rf"(?:export )?const {name} = (?:Number\(process\.env\.{name} \?\? )?([\d.]+)"
    match = re.search(pattern, source)
    assert match, f"{name} not found in lib/{file}"
    return float(match.group(1))


@pytest.mark.parametrize(
    ("file", "name", "python_value"),
    [
        ("retrieve.ts", "RERANK_THRESHOLD", retrieval_config.RERANK_THRESHOLD),
        ("retrieve.ts", "COSINE_THRESHOLD", retrieval_config.COSINE_THRESHOLD),
        ("retrieve.ts", "VECTOR_CANDIDATES", retrieval_config.VECTOR_CANDIDATES),
        ("retrieve.ts", "RERANK_TOP_N", retrieval_config.RERANK_TOP_N),
        ("plan.ts", "UNION_CAP", retrieval_config.UNION_CAP),
    ],
)
def test_matches_typescript(file: str, name: str, python_value: float) -> None:
    assert python_value == ts_default(file, name)


def ts_string(file: str, name: str) -> str:
    """Read a string constant from lib/: one literal, or several joined with +."""
    source = (LIB / file).read_text()
    match = re.search(rf"const {name} =\s*((?:\"(?:[^\"\\]|\\.)*\"\s*\+?\s*)+);", source)
    assert match, f"{name} not found in lib/{file}"
    return "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", match.group(1)))


def test_greeting_matches_typescript() -> None:
    from copilot_agent.planner import GREETING_MESSAGE

    assert ts_string("plan.ts", "GREETING_MESSAGE") == GREETING_MESSAGE


def test_refusal_matches_typescript() -> None:
    from copilot_agent.refusal import REFUSAL_MESSAGE

    assert ts_string("refusal.ts", "REFUSAL_MESSAGE") == REFUSAL_MESSAGE


@pytest.mark.parametrize("name", ["MAX_MESSAGES", "MAX_CHARS_PER_MESSAGE", "MAX_TOTAL_CHARS"])
def test_chat_caps_match_typescript(name: str) -> None:
    # The thread's history is capped as lib/chat-request.ts caps a request (invariant 9), and
    # POST /chat refuses a question the route would never forward. A cap changed on one side
    # only would change what the model reads, or turn long questions into 422s.
    from copilot_agent import history

    assert getattr(history, name) == ts_default("chat-request.ts", name)


def test_signature_version_matches_typescript() -> None:
    from copilot_agent.signing import VERSION

    assert ts_string("assistant-signature.ts", "VERSION") == VERSION


def ts_regex(file: str, name: str) -> str:
    """Read a regex literal's source from lib/: const NAME = /source/;"""
    source = (LIB / file).read_text()
    match = re.search(rf"const {name} = /(.+?)/;", source)
    assert match, f"{name} not found in lib/{file}"
    return match.group(1)


@pytest.mark.parametrize("name", ["REFERRER_HOST", "UTM_SOURCE", "COUNTRY_ISO2"])
def test_visitor_shapes_match_typescript(name: str) -> None:
    # The route sanitises the visitor with these (lib/visitor.ts) and the service checks them
    # again before the row is written (query_log.Visitor.sanitised). Python keeps the shape
    # without ^ and $ and uses re.fullmatch: Python's $ also matches before a trailing newline.
    from copilot_agent import query_log

    assert ts_regex("visitor.ts", name) == f"^{getattr(query_log, name)}$"


def test_the_row_has_the_columns_the_typescript_route_writes() -> None:
    # One table, two writers: lib/query-log.ts for the TypeScript route, query_log.py for the
    # service. Same columns in the same order, plus the two db/006 added.
    from copilot_agent.query_log import COLUMNS

    source = (LIB / "query-log.ts").read_text()
    insert = source[source.index('.from("query_log").insert({') : source.index("} as never)")]
    ts_columns = re.findall(r"^\s+([a-z_]+):", insert, re.MULTILINE)
    assert (*ts_columns, "origin", "thread_id") == COLUMNS


def test_every_column_the_service_writes_is_created_by_a_migration() -> None:
    from copilot_agent.query_log import COLUMNS

    db = LIB.parent / "db"
    sql = "\n".join(p.read_text() for p in sorted(db.glob("*.sql")))
    table = sql[sql.index("create table if not exists query_log") :]
    table = table[: table.index(");")]
    for column in COLUMNS:
        created = re.search(rf"^\s+{column}\s", table, re.MULTILINE)
        added = re.search(rf"add column if not exists {column}\s", sql)
        assert created or added, f"no migration in db/ creates query_log.{column}"


def test_thread_id_pattern_matches_typescript() -> None:
    # The route checks useChat's chat id with this before forwarding it (lib/agent-forward.ts),
    # so an id the service would 422 is a 400 there, before anything is paid for.
    from copilot_agent.api import THREAD_ID_PATTERN

    assert ts_regex("agent-forward.ts", "CHAT_ID_PATTERN") == THREAD_ID_PATTERN
