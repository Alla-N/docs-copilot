"""refusal.py: is_refusal() gives the verdict lib/refusal.ts gives, answer by answer.

The oracle is isRefusal itself: scripts/experiments/refusal-verdicts.ts ran it on every answer in
tests/golden/refusal-verdicts.json and recorded what it said. The query log's `refused` column
comes from this function on the Python path, and the eval set is mined from that column.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from copilot_agent.refusal import REFUSAL_MESSAGE, is_refusal

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((Path(__file__).parent / "golden" / "refusal-verdicts.json").read_text())
CASES = GOLDEN["cases"]


@pytest.mark.parametrize("path", sorted(GOLDEN["meta"]["sourcesSha256"]))
def test_the_golden_matches_the_current_typescript(path: str) -> None:
    digest = hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
    assert digest == GOLDEN["meta"]["sourcesSha256"][path], (
        f"{path} changed after the golden file was written; regenerate it from the repo root: "
        "npm run exp:refusal-verdicts"
    )


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_same_verdict_as_is_refusal(case: dict[str, Any]) -> None:
    assert is_refusal(case["answer"]) is case["refused"]


def test_the_golden_has_both_verdicts_and_the_canonical_message() -> None:
    # A golden of all refusals (or none) would pass a detector that always says the same thing.
    verdicts = {c["refused"] for c in CASES}
    assert verdicts == {True, False}
    by_id = {c["id"]: c for c in CASES}
    assert by_id["canonical"]["answer"] == REFUSAL_MESSAGE
