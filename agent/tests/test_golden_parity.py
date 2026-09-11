"""Golden-file parity: the Python search_docs against what the TypeScript retrieve() returned.

scripts/experiments/retrieval-golden.ts (npm run exp:golden, from the repo root) ran the planner
ONCE per recall case and froze, in tests/golden/retrieval-golden.json, every sub-query, the
exact text it embedded (HyDE) and what retrieve() returned. This test retrieves for the SAME
strings with the Python port, so a difference is the port, not HyDE. Once phase 2 deletes the
TypeScript retrieve(), the file is the only oracle left.

    uv run pytest -m integration -s tests/test_golden_parity.py

Costs one embedding and one rerank per sub-query (about 25 of each). On a Cohere TRIAL key set
RERANK_INTERVAL_MS=6500, as for the TypeScript script.

Both services behind the pipeline are slightly noisy (measured below), so the test does not
demand bit-equal floats. It asserts that every difference is one noise can explain:
  - reranked mode on every call (a fallback run is a different pipeline, never a parity result);
  - kept chunks: the same set, except a chunk AT the cut-off; the same order, except between
    near-tied scores;
  - candidates: the same set, except at the cut-off (order is only printed);
  - rerank and cosine scores within NOISE_BOUND of the golden;
  - recall on the union.
PRINTED with -s: exact-match counts, the largest deltas, and warm latency medians.
"""

import asyncio
import json
import os
import statistics
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from copilot_agent.content_hash import hash_chunk
from copilot_agent.retrieval import (
    RetrievalResult,
    RetrievedChunk,
    SearchDocs,
    open_search,
    union_relevant,
)
from copilot_agent.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

GOLDEN_FILE = Path(__file__).parent / "golden" / "retrieval-golden.json"
GOLDEN: dict[str, Any] | None = (
    json.loads(GOLDEN_FILE.read_text()) if GOLDEN_FILE.exists() else None
)
INTERVAL_S = float(os.environ.get("RERANK_INTERVAL_MS", "250")) / 1000

# Three runs against the same golden file, 2026-09-11, 13 sub-queries each:
#   cosine   max |TS - PY|   0.0, 1.0e-04, 0.0   (OpenAI: same text, not always the same vector)
#   rerank   max |TS - PY|   0.0, 0.0, 3.1e-04   (Cohere: same request, not always the same score)
# The database and SQL are fixed and the Python requests are byte-identical every run, so both
# are service-side noise, not the port. Two earlier versions of this file set tight tolerances
# from one and then two runs, and the next run broke each: a few runs are not a measurement of
# a noise ceiling.
#
# So NOISE_BOUND is not an estimate of the noise. It is 30x the largest seen, and it is sized
# against what the pipeline decides on: in this golden file the kept score nearest the 0.30 gate
# is 0.33 above it. Near-ties are what noise CAN flip, and they are real here: the closest pair
# of kept scores is 2.6e-04 apart (typo, multi-intent-noise), candidates #99 and #100 of
# changed-7 are 1.1e-05 apart. A port bug (another model, the wrong text embedded or reranked,
# another SQL function) changes scores by far more than this or changes the chunk set outright.
NOISE_BOUND = 0.01


def golden_cases() -> list[Any]:
    """One test per golden case, named by its id. Without the file, one skipped placeholder
    that says how to make it (a module-level skip would also show up in the no-network run)."""
    if GOLDEN is None:
        reason = "no golden file: run npm run exp:golden from the repo root first"
        return [pytest.param(None, id="no-golden-file", marks=pytest.mark.skip(reason=reason))]
    return [pytest.param(case, id=case["id"]) for case in GOLDEN["cases"]]


def key_of(chunk: Any) -> str:
    """The golden file's chunk key: hash_chunk (invariant 1) cut to the file's key length."""
    assert GOLDEN is not None
    return hash_chunk(chunk.source_url, chunk.content)[: GOLDEN["knobs"]["keyLength"]]


@dataclass
class SubQueryComparison:
    """One sub-query, retrieved by both sides."""

    case_id: str
    query: str
    ts_relevant: list[str]
    py_relevant: list[str]
    kept_off_edge: list[str]  # kept by one side only, and not at the other side's cut-off
    kept_inversions: list[tuple[str, str]]  # kept in opposite order without being near-tied
    score_delta: float  # max |TS - Python| rerank score, over chunks both sides kept
    candidate_overlap: int  # candidates present on both sides, out of the TypeScript count
    candidate_count: int
    candidates_off_edge: list[str]  # found by one side only, and not at the other side's cut-off
    candidate_order_same: bool
    similarity_delta: float  # max |TS - Python| cosine similarity, over shared candidates
    ts_ms: float
    py_timings_ms: dict[str, float]


def off_edge(ts: dict[str, float], py: dict[str, float], tolerance: float) -> list[str]:
    """Keys on one side only that noise cannot explain.

    Works for any ranked, cut list: the 100 candidates (cut by count) and the kept chunks (cut at
    top 5 or the 0.30 gate). Noise of size `tolerance` can push an item across the cut, but only
    an item within that distance of the other side's cut-off (the lowest score it kept). A
    missing item far above the cut means the two sides ranked differently.
    """
    if not ts or not py:
        return sorted(ts.keys() ^ py.keys())
    ts_cut, py_cut = min(ts.values()), min(py.values())
    only_ts = [k for k in ts.keys() - py.keys() if ts[k] - py_cut > tolerance]
    only_py = [k for k in py.keys() - ts.keys() if py[k] - ts_cut > tolerance]
    return sorted(only_ts + only_py)


def inversions(
    ts_order: list[str], py_order: list[str], ts_scores: dict[str, float], tolerance: float
) -> list[tuple[str, str]]:
    """Pairs both sides kept, in opposite order, whose scores are NOT within noise of each other.

    Noise can swap two chunks scored 0.8750 and 0.8748; it cannot swap 0.87 and 0.83.
    """
    py_rank = {k: i for i, k in enumerate(py_order)}
    shared = [k for k in ts_order if k in py_rank]
    return [
        (a, b)
        for i, a in enumerate(shared)
        for b in shared[i + 1 :]
        if py_rank[a] > py_rank[b] and abs(ts_scores[a] - ts_scores[b]) > tolerance
    ]


def compare(case_id: str, sub_query: dict[str, Any], result: RetrievalResult) -> SubQueryComparison:
    ts_scores = {r["key"]: r["score"] for r in sub_query["relevant"]}
    py_scores = {key_of(c): c.score for c in result.relevant}
    ts_sims = dict(sub_query["candidates"])
    py_candidates = [(key_of(c), c.similarity) for c in result.candidates]
    py_sims = dict(py_candidates)
    shared_relevant = ts_scores.keys() & py_scores.keys()
    shared_candidates = ts_sims.keys() & py_sims.keys()
    ts_relevant = [r["key"] for r in sub_query["relevant"]]
    py_relevant = list(py_scores)
    return SubQueryComparison(
        case_id=case_id,
        query=sub_query["query"],
        ts_relevant=ts_relevant,
        py_relevant=py_relevant,
        kept_off_edge=off_edge(ts_scores, py_scores, NOISE_BOUND),
        kept_inversions=inversions(ts_relevant, py_relevant, ts_scores, NOISE_BOUND),
        score_delta=max((abs(ts_scores[k] - py_scores[k]) for k in shared_relevant), default=0.0),
        candidate_overlap=len(shared_candidates),
        candidate_count=len(ts_sims),
        candidates_off_edge=off_edge(ts_sims, py_sims, NOISE_BOUND),
        candidate_order_same=[k for k, _ in sub_query["candidates"]]
        == [k for k, _ in py_candidates],
        similarity_delta=max(
            (abs(ts_sims[k] - py_sims[k]) for k in shared_candidates), default=0.0
        ),
        ts_ms=sub_query["ms"],
        py_timings_ms=result.timings_ms,
    )


def ts_relevant_chunks(sub_query: dict[str, Any]) -> list[RetrievedChunk]:
    """The TypeScript side's kept chunks, rebuilt from the golden file for the union.

    The file stores keys, not text; the key stands in for content, and it is unique per chunk,
    so the union dedupes exactly as it would on the real text.
    """
    assert GOLDEN is not None
    chunks = GOLDEN["chunks"]
    return [
        RetrievedChunk(
            content=r["key"],
            title=chunks[r["key"]]["title"],
            source_url=chunks[r["key"]]["source_url"],
            score=r["score"],
        )
        for r in sub_query["relevant"]
    ]


def recalled(expected_source: list[str], chunks: list[RetrievedChunk]) -> bool:
    """Same rule as expectedFound() in evals/run.ts: any expected slug in any kept source_url."""
    return any(slug in chunk.source_url for chunk in chunks for slug in expected_source)


# ---- fixtures -------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def search() -> AsyncIterator[SearchDocs]:
    """ONE pool and one set of HTTP clients for every case: warm connections after the first
    call, which is what makes the latency medians mean something. Module scope needs the
    module-scoped anyio_backend in conftest.py."""
    async with open_search(get_settings()) as search_docs:
        yield search_docs


@dataclass
class Report:
    comparisons: list[SubQueryComparison]
    recall: dict[str, tuple[bool, bool]]  # case id -> (TypeScript, Python)


@pytest.fixture(scope="module")
def report() -> Iterator[Report]:
    """Collects every comparison; prints the summary once, after the last case."""
    collected = Report(comparisons=[], recall={})
    yield collected
    print_report(collected)


def print_report(report: Report) -> None:
    rows = report.comparisons
    if not rows or GOLDEN is None:
        return
    n = len(rows)
    same = sum(r.ts_relevant == r.py_relevant for r in rows)
    ts_recall = sum(ts for ts, _ in report.recall.values())
    py_recall = sum(py for _, py in report.recall.values())
    # Warm: drop the first sub-query. It pays for the pool, TLS and DNS on the Python side, and
    # for the same on the TypeScript side, so it measures the network, not the code.
    warm = rows[1:]

    def median(values: list[float]) -> str:
        return f"{statistics.median(values):.0f}" if values else "n/a"

    stages = {
        stage: median([r.py_timings_ms.get(stage, 0.0) for r in warm])
        for stage in ("embed", "vector", "rerank")
    }
    py_total = median([sum(r.py_timings_ms.values()) for r in warm])
    ts_total = median([r.ts_ms for r in warm])
    worst_overlap = min(rows, key=lambda r: r.candidate_overlap)

    cases = len(report.recall)
    dirty = " (dirty tree)" if GOLDEN["dirty"] else ""
    lines = [
        "",
        f"GOLDEN PARITY  golden from commit {GOLDEN['commit']}{dirty}, {GOLDEN['date'][:10]}",
        f"  {cases} cases, {n} sub-queries",
        f"  kept chunks, exactly equal     {same}/{n}",
        f"  kept, unexplained by noise     "
        f"{sum(len(r.kept_off_edge) + len(r.kept_inversions) for r in rows)}",
        f"  rerank score max |TS - PY|     {max(r.score_delta for r in rows):.2e}",
        f"  cosine max |TS - PY|           {max(r.similarity_delta for r in rows):.2e}",
        f"  candidates overlap, worst      {worst_overlap.candidate_overlap}"
        f"/{worst_overlap.candidate_count} ({worst_overlap.case_id})",
        f"  candidates, off the cut-off    {sum(len(r.candidates_off_edge) for r in rows)}",
        f"  candidates in same order       {sum(r.candidate_order_same for r in rows)}/{n}",
        f"  recall                         TS {ts_recall}/{cases}   PY {py_recall}/{cases}",
        f"  warm medians, ms (n={len(warm)})      PY embed {stages['embed']}"
        f" + vector {stages['vector']} + rerank {stages['rerank']} = {py_total}",
        f"                                 TS retrieve() {ts_total}",
    ]
    titles = {key: chunk["title"] for key, chunk in GOLDEN["chunks"].items()}
    for r in rows:
        if r.ts_relevant != r.py_relevant:
            lines.append(f"  DIFF {r.case_id}: {r.query!r}")
            lines += [f"    TS {k} {titles.get(k, '?')}" for k in r.ts_relevant]
            lines += [f"    PY {k} {titles.get(k, '?')}" for k in r.py_relevant]
    print("\n".join(lines))


# ---- the test -------------------------------------------------------------------------


@pytest.mark.parametrize("case", golden_cases())
async def test_python_retrieves_what_typescript_retrieved(
    case: dict[str, Any], search: SearchDocs, report: Report
) -> None:
    comparisons: list[SubQueryComparison] = []
    py_kept: list[list[RetrievedChunk]] = []
    for sub_query in case["subQueries"]:
        result = await search(sub_query["query"], sub_query["embedText"])
        assert result.mode == "reranked", (
            f"{case['id']}: the Python rerank call failed, so this ran the cosine fallback. "
            "That is a different pipeline, not a parity result: fix Cohere and rerun."
        )
        comparisons.append(compare(case["id"], sub_query, result))
        py_kept.append(result.relevant)
        await asyncio.sleep(INTERVAL_S)

    # Record before asserting, so the printed report covers every case, failing ones included.
    report.comparisons.extend(comparisons)
    ts_union = union_relevant([ts_relevant_chunks(sq) for sq in case["subQueries"]])
    py_union = union_relevant(py_kept)
    ts_hit = recalled(case["expectedSource"], ts_union)
    py_hit = recalled(case["expectedSource"], py_union)
    report.recall[case["id"]] = (ts_hit, py_hit)

    for c in comparisons:
        where = f"{case['id']}, sub-query {c.query!r}"
        kept = f"Python kept {c.py_relevant}, TypeScript kept {c.ts_relevant}"
        assert not c.kept_off_edge, f"{where}: kept by one side only {c.kept_off_edge}; {kept}"
        assert not c.kept_inversions, f"{where}: order flipped {c.kept_inversions}; {kept}"
        assert c.score_delta <= NOISE_BOUND, f"{where}: rerank score moved {c.score_delta:.2e}"
        assert c.similarity_delta <= NOISE_BOUND, (
            f"{where}: cosine similarity moved {c.similarity_delta:.2e}"
        )
        assert not c.candidates_off_edge, (
            f"{where}: candidates on one side only, off the cut-off: {c.candidates_off_edge}"
        )
    assert py_hit, f"{case['id']}: no page from {case['expectedSource']} in the Python union"
