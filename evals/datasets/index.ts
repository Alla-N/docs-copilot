/**
 * The dataset registry: three labelled sets, and what each one claims.
 *
 * They existed as datasets before the phase named them — different cases, different evaluators,
 * different denominators, their own cost lines and their own exit codes. The work was to make
 * the structure say what the code already did.
 *
 * This registry is read by `tests/evaluators.test.ts` and by anyone who wants the overview in
 * one screen. The runners still import their own cases directly: routing `run.ts` through this
 * file would change how the suite is assembled, and decision 7 keeps assembly changes out of
 * the renaming half.
 */
import { CASES, type EvalCase } from "./golden";
import { GITHUB_CASES, FROZEN_AT, type GitHubCase } from "./github";
import { PLANNER_CASES, type PlannerCase } from "./planner";
import type { Dataset } from "./types";

export const GOLDEN: Dataset<EvalCase> = {
    id: "golden",
    title: "the golden set — retrieval, answering, guardrails and injection end to end",
    cases: CASES,
    evaluators: ["recall", "recallEveryRun", "coverage", "guardrails", "injection", "falseRefusals", "faithfulness"],
    denominators:
        "answerable, guardrail and injection cases are three disjoint populations; parked (expectFail) " +
        "and ERROR cases are in none of them",
    exitCode: 1,
    runs:
        "3 generations per case, 8 for an adversarial one. An attack that works one time in five is a " +
        "working attack, so a clean 3-run sample there means 'did not reproduce', not 'resistant'.",
    note: "The series every stored run belongs to. Its cost figure is a series across runs and nothing else is folded into it.",
};

export const GITHUB: Dataset<GitHubCase> = {
    id: "github",
    title: "the labelled GitHub set — frozen answers about vercel/ai, asked through the subagent",
    cases: GITHUB_CASES,
    evaluators: ["githubAccuracy", "githubFirstTryValid", "githubValidAfterRepairs", "githubRouting", "githubPoints"],
    denominators:
        "accuracy is over observations; validity is over the turns whose subagent RAN; routing is over " +
        "the turns that were ROUTED, pooled with the golden set's docs label",
    exitCode: 5,
    runs:
        "2 per case, which is not a per-case rate — n=10 cannot separate 0.85 from 1.0, so n=2 certainly " +
        "cannot. It is two observations so that a case differing between them reports as VARIED rather " +
        "than silently as whichever run came first.",
    note: `Answers frozen ${FROZEN_AT}. Exit code 5, not 1: the documentation pipeline being fine and this set not being fine are two different claims.`,
};

export const PLANNER: Dataset<PlannerCase> = {
    id: "planner",
    title: "the planner set — intent and sub-queries, asserted directly",
    cases: PLANNER_CASES,
    evaluators: ["plannerPlan"],
    denominators: "every case, every run; no retrieval and no answer, so nothing is excluded",
    exitCode: 1,
    runs:
        "5, not 3: a planner call costs a fraction of a cent, and the failure this set guards showed up " +
        "on a 4th sample after 3 had passed.",
    note:
        "Keeps its own runner (npm run eval:planner) and its own command. It takes no target and does no " +
        "retrieval, so folding it into run.ts would mean a target-shaped hole in it (decision 9). This " +
        "file is also SHA-pinned from the Python side — see datasets/types.ts.",
};

export const DATASETS = { golden: GOLDEN, github: GITHUB, planner: PLANNER };
