"""history.py: the thread's stored turns, capped exactly as lib/chat-request.ts caps a request.

The oracle is parseChatRequest itself: scripts/experiments/history-caps.ts ran it on the same
conversations and froze what it kept in tests/golden/history-caps.json.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from copilot_agent.history import (
    MAX_CHARS_PER_MESSAGE,
    MAX_MESSAGES,
    MAX_STORED_TURNS,
    MAX_TOTAL_CHARS,
    capped_history,
    keep_recent,
)
from copilot_agent.planner import HistoryTurn

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((Path(__file__).parent / "golden" / "history-caps.json").read_text())
CASES = GOLDEN["cases"]


def turns_of(items: list[dict[str, Any]]) -> list[HistoryTurn]:
    return [HistoryTurn(t["role"], t["text"]) for t in items]


@pytest.mark.parametrize("path", sorted(GOLDEN["meta"]["sourcesSha256"]))
def test_the_golden_matches_the_current_typescript(path: str) -> None:
    digest = hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
    assert digest == GOLDEN["meta"]["sourcesSha256"][path], (
        f"{path} changed after the golden file was written; regenerate it from the repo root: "
        "npm run exp:history-caps"
    )


def test_the_golden_was_written_with_these_caps() -> None:
    assert GOLDEN["meta"]["caps"] == {
        "MAX_MESSAGES": MAX_MESSAGES,
        "MAX_CHARS_PER_MESSAGE": MAX_CHARS_PER_MESSAGE,
        "MAX_TOTAL_CHARS": MAX_TOTAL_CHARS,
    }


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_same_history_as_parse_chat_request(case: dict[str, Any]) -> None:
    # The question reaches Python as TypeScript parsed it, so that is the one to count.
    assert case["parsedQuestion"] == case["question"]
    kept = capped_history(turns_of(case["turns"]), case["question"])
    assert kept == turns_of(case["history"])


def test_every_cap_is_exercised_by_the_golden() -> None:
    # A golden that never hits a cap would let a broken cap pass.
    by_id = {c["id"]: c for c in CASES}
    assert len(by_id["more-than-max-messages"]["history"]) == MAX_MESSAGES - 1
    cut = by_id["long-answer-cut-then-trimmed"]
    assert len(cut["turns"][1]["text"]) > MAX_CHARS_PER_MESSAGE > len(cut["history"][1]["text"])
    exact = by_id["total-cap-exact"]
    kept_total = len(exact["question"]) + sum(len(t["text"]) for t in exact["history"])
    assert kept_total == MAX_TOTAL_CHARS
    assert len(exact["history"]) < len(exact["turns"])
    misfit = by_id["total-cap-stops-at-first-misfit"]
    assert misfit["turns"][0]["text"] == "tiny"
    assert "tiny" not in [t["text"] for t in misfit["history"]]


def test_the_cut_is_in_code_points_not_utf16_units() -> None:
    # The named difference (history.py): an emoji straddling the 4000 cut. TypeScript would keep
    # 3999 characters and half the emoji; Python keeps the whole emoji.
    rocket = chr(0x1F680)
    text = "a" * (MAX_CHARS_PER_MESSAGE - 1) + rocket + "tail"
    [kept] = capped_history([HistoryTurn("assistant", text)], "q")
    assert kept.text == "a" * (MAX_CHARS_PER_MESSAGE - 1) + rocket


def test_the_stored_turns_are_bounded() -> None:
    stored = [HistoryTurn("user", f"{i}") for i in range(MAX_STORED_TURNS)]
    new = [HistoryTurn("user", "new q"), HistoryTurn("assistant", "new a")]
    kept = keep_recent(stored, new)
    assert len(kept) == MAX_STORED_TURNS
    assert kept[-2:] == new
    assert kept[0] == stored[2]


def test_bounding_the_store_changes_nothing_the_model_reads() -> None:
    # 30 stored turns vs the last MAX_STORED_TURNS of them: the same history, for any question.
    many = [HistoryTurn(("user", "assistant")[i % 2], f"turn {i}") for i in range(30)]
    bounded = keep_recent([], many)
    for question in ["q", "q" * MAX_CHARS_PER_MESSAGE]:
        assert capped_history(many, question) == capped_history(bounded, question)


def test_the_store_holds_nothing_a_request_cannot_read() -> None:
    # The other half of the bound: with short turns, every stored turn is read.
    many = [HistoryTurn(("user", "assistant")[i % 2], f"turn {i}") for i in range(30)]
    stored = keep_recent([], many)
    assert capped_history(stored, "q") == stored
