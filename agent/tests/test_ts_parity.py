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
