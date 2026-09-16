/**
 * The dataset vocabulary: the case types, and what a dataset IS.
 *
 * ── why the case types are re-exported rather than moved here ──────────────────────────────
 *
 * The spec's target tree (specs/eval-subsystem.md section 3) drew this file as the home of
 * `EvalCase`, `GitHubCase` and `PlannerCase`. It is a barrel instead, for one hard reason and
 * one soft one.
 *
 * The hard one: `evals/datasets/planner.ts` is pinned by SHA256 from the Python side.
 * `agent/tests/golden/planner-requests.json` records a hash of that file and of `lib/plan.ts`,
 * and `agent/tests/test_planner_request_parity.py` fails until the golden is regenerated if
 * either moves by one byte. Lifting a type out of it would cost a regeneration — a paid run of
 * `exp:planner-requests` — to buy nothing but symmetry. A file pinned by hash cannot be
 * refactored for free, and that is a property of the file, not an oversight.
 *
 * The soft one: `EvalCase`'s field comments are the criteria rationale for the golden set —
 * why `shouldAnswer` has three values, why `expectedSource` is any-of, what `mustContain`
 * exists to catch. That is documentation about the cases, and it belongs beside them. Moving
 * it here would have produced an eighty-line diff with no behaviour in it and left the cases
 * file explaining nothing.
 *
 * So: one import surface, three homes. Read a type here; read why it is shaped that way in the
 * dataset that defines it.
 */
export type { EvalCase } from "./golden";
export type { GitHubCase } from "./github";
export type { PlannerCase } from "./planner";

/**
 * A dataset is a descriptor, not a bare array (decision 5).
 *
 * The three sets already had different denominators, different cost lines and different exit
 * codes before this phase named any of it — the golden set exits 1 and the labelled GitHub set
 * exits 5, and that difference is load-bearing. They are two different claims about the system,
 * deliberately not collapsed: "the documentation pipeline is fine and the GitHub set is not" is
 * a sentence the exit code can say only because the codes differ.
 *
 * This is where that stops being a comment inside `main()` and becomes data.
 */
export type Dataset<Case> = {
    id: string;
    title: string;
    cases: Case[];
    /** Evaluator ids from `evals/evaluators/index.ts`. */
    evaluators: string[];
    /** Who is in the denominators this set reports over, in one sentence. */
    denominators: string;
    /** What a failure of THIS set exits with. Different sets, different claims, different codes. */
    exitCode: number;
    /** How many times each case is observed by default, and why it is that number. */
    runs: string;
    note?: string;
};
