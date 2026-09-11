"""Planner-only eval for the Python planner: the port of evals/planner.ts.

    cd agent && uv run python evals/planner_eval.py
    PLANNER_RUNS=5 EVAL_ONLY=off-aws,expand-sdk uv run python evals/planner_eval.py

Same cases, same checks, same verdicts as `npm run eval:planner`: N runs per case (planner
output is model output), PASS only if every run passes, FLAKY if some do, FAIL if none. The
cases come from tests/golden/planner-requests.json, which scripts/experiments/planner-requests.ts
writes from evals/planner-cases.ts, so the two suites cannot drift apart.

This is the statistical half of the planner port's parity; test_planner_request_parity.py is
the exact half. Compare two runs of this with two runs of `npm run eval:planner`: the requests
are identical, so a difference is sampling, and one run of either is not a measurement.

Costs one gpt-4o-mini call per run: 23 cases x 5 runs = 115 calls, a few cents. Needs the three
variables the service reads (.env.local is picked up from the repo root).

Lives outside src/ on purpose: eval code is not part of the service, and the image's
.dockerignore allowlist leaves it out.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from copilot_agent.planner import (
    HistoryTurn,
    Plan,
    Planner,
    build_planner,
    openai_planner_model,
    plan_query,
)
from copilot_agent.settings import get_settings

GOLDEN_FILE = Path(__file__).resolve().parents[1] / "tests" / "golden" / "planner-requests.json"
# 5, not 3, as in evals/planner.ts: the failure it guards showed up on a 4th sample once.
RUNS = int(os.environ.get("PLANNER_RUNS", "5"))
ONLY = [x.strip() for x in os.environ.get("EVAL_ONLY", "").split(",") if x.strip()]


def check(case: dict[str, Any], plan: Plan) -> list[str]:
    """check() in evals/planner.ts, line for line."""
    expect = case["expect"]
    problems: list[str] = []
    wanted = expect["intent"] if isinstance(expect["intent"], list) else [expect["intent"]]
    if plan.intent not in wanted:
        problems.append(f"intent {plan.intent}, expected {'|'.join(wanted)}")
    if "count" in expect and len(plan.queries) != expect["count"]:
        problems.append(f"{len(plan.queries)} queries, expected {expect['count']}")
    joined = " | ".join(q.query.lower() for q in plan.queries)
    problems += [f'missing "{s}"' for s in expect.get("mustContain", []) if s.lower() not in joined]
    problems += [f'contains "{s}"' for s in expect.get("mustNotContain", []) if s.lower() in joined]
    return problems


async def run_case(case: dict[str, Any], planner: Planner) -> tuple[list[Plan], list[list[str]]]:
    history = [HistoryTurn(t["role"], t["text"]) for t in case["history"]]
    plans = await asyncio.gather(
        *(plan_query(case["question"], history, planner=planner) for _ in range(RUNS))
    )
    return list(plans), [check(case, p) for p in plans]


async def main() -> int:
    cases = json.loads(GOLDEN_FILE.read_text())["cases"]
    active = [c for c in cases if c["id"] in ONLY] if ONLY else cases
    planner = build_planner(openai_planner_model(get_settings()))
    print(f"planner eval (python): {len(active)} cases x {RUNS} runs, temp 0\n")

    failures = 0
    fallbacks = 0
    for case in active:
        plans, problems = await run_case(case, planner)
        # A fallback plan has no usage. It can still pass a case, so count it separately:
        # a run full of silent fallbacks is not a planner result (the same rule as for rerank).
        fallbacks += sum(p.usage.input_tokens is None for p in plans)
        passes = sum(not p for p in problems)
        verdict = "PASS" if passes == RUNS else "FAIL" if passes == 0 else "FLAKY"
        if verdict != "PASS":
            failures += 1
        print(f"  {verdict:<5} {case['id']:<24} {'' if verdict == 'PASS' else f'{passes}/{RUNS}'}")
        if verdict != "PASS":
            i = next(i for i, p in enumerate(problems) if p)
            print(f"         -> {'; '.join(problems[i])}")
            queries = json.dumps([q.query for q in plans[i].queries])
            print(f"         -> planner: {plans[i].intent} {queries}")

    print(f"\n{f'{failures} failing' if failures else 'all green'}.", end="")
    print(f" {fallbacks} fallback runs." if fallbacks else "")
    return 1 if failures or fallbacks else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
