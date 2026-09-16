/**
 * The evaluator registry: what this suite measures, in one readable list.
 *
 * An evaluator is a pure function over cases and results, with the verdict computed in code
 * (decision 6). `faithfulness.ts` already worked that way before the phase — a model produces
 * claims and quotes, and the pass/fail is decided by `isQuoteFound` in TypeScript — and that
 * is the shape the others were given.
 *
 * ── why a table and not a dispatcher ───────────────────────────────────────────────────────
 *
 * The obvious design is `Record<metricId, (ctx) => Ratio>` with the record built by iterating
 * it. That was deliberately not built here, because decision 7 says the renaming half must not
 * move a single number, and rewiring how the summary object is ASSEMBLED is exactly the kind
 * of change that moves one quietly — a key that stops being written, a conditional that loses
 * its guard. run.ts still writes its summary literally; each value now comes from one
 * evaluator instead of being re-spelled.
 *
 * What the table does instead is make a missing number findable. Every entry names the metric
 * id it produces, and `tests/evaluators.test.ts` asserts that each of those ids exists in
 * `METRICS` in evals/record.ts. That closes a real gap: before this, a number could be computed
 * and printed and stored and still be invisible to `eval:diff`, because nothing connected the
 * thing that computes it to the table that reads it back.
 */
import type { MetricDef } from "../record";

export { partition, ran } from "./partition";
export { expectedFound, scoreRuns, verdictOf, erroredResult } from "./case-verdict";
export { evaluateRecall, type RecallScore } from "./recall";
export { evaluateCoverage } from "./coverage";
export { evaluateGuardrails, type GuardrailScore, type GuardrailHold, type GuardrailLayer } from "./guardrails";
export { evaluateInjection, type InjectionScore } from "./injection";
export { evaluateFalseRefusals } from "./refusal";
export { judgeFaithfulness, aggregateFaithfulness, reportUnfaithful, type Verdict } from "./faithfulness";
export { declined, answeredCorrectly, evaluateGitHubAccuracy, githubVerdict, type GitHubRun } from "./github-answer";
export { evaluateGitHubProcess, type GitHubProcessScore, type GitHubCaseFacts } from "./github-process";
export type { Result, Scored, Scoreboard, Populations, GoldenContext, FailedClaim } from "./types";

export type EvaluatorDescriptor = {
    id: string;
    /** The line it prints. */
    label: string;
    dataset: "golden" | "github" | "planner";
    /**
     * The id in `METRICS` (evals/record.ts) this evaluator's number is stored and diffed under.
     * `null` means it is reported but not stored — and every one of those is a thing a future
     * diff cannot see, which is why they are listed rather than omitted.
     */
    metricId: MetricDef["id"] | null;
    /** Who is in the denominator. The half of a fraction that is easiest to change by accident. */
    denominator: string;
    /** Opt-in evaluators cost a model call and do not run by default. */
    optIn?: boolean;
};

export const EVALUATORS: EvaluatorDescriptor[] = [
    {
        id: "recall",
        label: "retrieval recall",
        dataset: "golden",
        metricId: "recall",
        denominator: "answerable cases: shouldAnswer true, not adversarial, not parked, ran",
    },
    {
        id: "recallEveryRun",
        label: "recall every run",
        dataset: "golden",
        metricId: "recallEveryRun",
        denominator: "the same answerable cases; only meaningful on a per-run-retrieval target",
    },
    {
        id: "coverage",
        label: "answer coverage",
        dataset: "golden",
        metricId: "coverage",
        denominator: "the same answerable cases",
    },
    {
        id: "guardrails",
        label: "guardrails held",
        dataset: "golden",
        metricId: "guardrails",
        denominator: "shouldAnswer false, not adversarial, not parked, ran",
    },
    {
        id: "injection",
        label: "injection resisted",
        dataset: "golden",
        metricId: "injection",
        denominator: "adversarial cases, not parked, ran — its own axis, not folded into guardrails",
    },
    {
        id: "falseRefusals",
        label: "false refusals",
        dataset: "golden",
        metricId: "falseRefusals",
        denominator: "a count, not a rate: answerable cases that retrieved the page and refused anyway",
    },
    {
        id: "faithfulness",
        label: "faithfulness",
        dataset: "golden",
        metricId: "faithful",
        denominator: "cases that were JUDGED — a refusal has no claims, and absent is not zero",
        optIn: true,
    },
    {
        id: "plannerPlan",
        label: "planner intent and sub-queries",
        dataset: "planner",
        // Not a mistake and not an omission: `npm run eval:planner` writes NO record at all, so
        // there is nothing in evals/results/ for a metric id to point at and nothing for
        // `eval:diff` to compare. The 23x5 planner suite is the one set whose numbers have only
        // ever lived in a terminal. Listed here rather than left out, because a number that
        // cannot be diffed is a fact about the suite worth being able to find.
        metricId: null,
        denominator: "every planner case, every run; PASS only if all 5 runs pass",
    },
    {
        id: "githubAccuracy",
        label: "answer accuracy",
        dataset: "github",
        metricId: "ghAccuracy",
        denominator: "observations in THIS run (13 cases x 2), never a pooled pair",
    },
    {
        id: "githubFirstTryValid",
        label: "first-try query valid",
        dataset: "github",
        metricId: "ghFirstTryValid",
        denominator: "turns whose subagent RAN — a canned turn wrote no query to be invalid",
    },
    {
        id: "githubValidAfterRepairs",
        label: "valid after repairs",
        dataset: "github",
        metricId: "ghValidAfterRepairs",
        denominator: "the same turns whose subagent ran",
    },
    {
        id: "githubRouting",
        label: "routing accuracy",
        dataset: "github",
        metricId: "ghRouting",
        denominator: "turns that were ROUTED, pooled across both datasets; canned turns on their own line",
    },
    {
        id: "githubPoints",
        label: "points per question",
        dataset: "github",
        metricId: "ghPointsMedian",
        denominator: "a median over the turns that spent points, not a rate",
    },
];
