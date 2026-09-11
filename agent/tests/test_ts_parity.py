"""The Python port must use the same calibrated numbers as lib/retrieve.ts.

While two retrieval implementations exist, a threshold changed on one side only would be
a silent drift (CLAUDE.md invariant 3). This test reads the TypeScript source and compares.
"""

import re
from pathlib import Path

import pytest

from copilot_agent import retrieval_config

# agent/tests/this_file.py -> parents[2] is the repo root.
TS_RETRIEVE = Path(__file__).resolve().parents[2] / "lib" / "retrieve.ts"


def ts_default(name: str) -> float:
    """Read a constant's default from lib/retrieve.ts.

    Matches both shapes used there:
        export const RERANK_THRESHOLD = 0.3;
        export const VECTOR_CANDIDATES = Number(process.env.VECTOR_CANDIDATES ?? 100);
    """
    source = TS_RETRIEVE.read_text()
    pattern = rf"export const {name} = (?:Number\(process\.env\.{name} \?\? )?([\d.]+)"
    match = re.search(pattern, source)
    assert match, f"{name} not found in {TS_RETRIEVE}"
    return float(match.group(1))


@pytest.mark.parametrize(
    ("name", "python_value"),
    [
        ("RERANK_THRESHOLD", retrieval_config.RERANK_THRESHOLD),
        ("COSINE_THRESHOLD", retrieval_config.COSINE_THRESHOLD),
        ("VECTOR_CANDIDATES", retrieval_config.VECTOR_CANDIDATES),
        ("RERANK_TOP_N", retrieval_config.RERANK_TOP_N),
    ],
)
def test_matches_typescript(name: str, python_value: float) -> None:
    assert python_value == ts_default(name)
