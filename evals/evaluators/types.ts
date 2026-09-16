/**
 * What a case produced, and what an evaluator is handed.
 *
 * Two shapes live here, and the pairing with `evals/record.ts` is the point: `record.ts`
 * describes what a run left ON DISK and how to read an older one back; this describes what a
 * case produced IN MEMORY while the run was happening. An evaluator reads the second and the
 * diff reads the first, which is why neither module imports the other's reader.
 *
 * `Result` was inline in run.ts until 5.4b. It moved out because every evaluator reads it, and
 * a type that eight modules import should not live in the one module that also happens to
 * print things.
 */
import type { PlanIntent } from "../../lib/plan";
import type { EvalCase } from "../datasets/types";
import type { TurnFacts } from "../targets/agent-service";

/** A judged claim that had no verbatim support. See `faithfulness.ts` for how it is derived. */
export type FailedClaim = { claim: string; quote: string };

export type Result = {
    id: string;
    intent: PlanIntent;  // what the planner decided; "search" is the only one that retrieves
    degraded: boolean;   // reranker unavailable → cosine fallback; the run is not comparable
    retrieved: string;   // expected doc present in the retrieved set?
    chunks: number;      // how many chunks cleared the threshold and reached the model
    topScore: string;
    answered: number;    // how many runs produced an answer rather than a refusal
    runs: number;
    sample: string;      // first response, for eyeballing adversarial cases
    /** The run that broke expectation (answered when it should refuse, or vice versa), if any. */
    odd?: { i: number; text: string };
    /** Run 1's thread id on the Python target, so the route the service recorded for this case
     *  can be read back afterwards (db/009). Absent on the TypeScript target, which has no router. */
    threadId?: string;
    /** What the router chose on run 1: "docs", "both", or null for a turn the planner canned
     *  before the router ever saw it. Step 3.6 records it per case because without it a
     *  documentation question answered from GitHub is invisible in a stored result. */
    route?: TurnFacts["route"];
    faithful: string;    // judge verdict on the first answer, when EVAL_JUDGE=1
    /** The claims behind a "NO", with the span the judge offered for each. Recorded so a
     *  stored result can be inspected instead of re-run: without it a reader sees only
     *  "NO" and has to pay for another judged run to find out whether it was real. */
    faithfulDetail?: FailedClaim[];
    /** ERROR: the case could not be RUN (infrastructure), which is not a verdict about the
     *  pipeline. Kept out of the denominators and it makes the whole run incomplete. */
    verdict: "PASS" | "FAIL" | "FLAKY" | "—" | "ERROR";
    detail: string;
    /** Why the case could not run, when it could not. Set exactly when verdict is ERROR. */
    errored?: string;
    /** Python target only: every run retrieves again, so recall has a per-run answer too. */
    retrievedEvery?: "yes" | "NO" | "varied" | "—";
    /** Python target only: in how many runs the expected page was retrieved. */
    foundRuns?: number;
};

/** The scored texts of one case's runs, before a verdict is put on them. */
export type Scored = {
    answered: number;
    firstAnswer: string;
    leaked: Set<string>;
    missed: Set<string>;
    oddRun: { i: number; text: string } | null;
};

/** Results by case id. Every set-level evaluator takes one of these and the population it scores. */
export type Scoreboard = Map<string, Result>;

/**
 * Everything a golden-set evaluator is allowed to look at.
 *
 * Passed as one object rather than as five arguments so that adding a fact to the context does
 * not re-write eight signatures — and so that an evaluator cannot quietly reach for a module
 * global, which is how the injection numerator came to be written twice.
 */
export type GoldenContext = {
    /** The cases this run actually ran (EVAL_ONLY narrows it). */
    active: EvalCase[];
    results: Result[];
    byId: Scoreboard;
    populations: Populations;
    /** EVAL_RUNS > 0: answers were generated. False is the retrieval-only CI tier. */
    generated: boolean;
    /** The Python target retrieves on every run, so recall has a per-run answer there. */
    perRunRetrieval: boolean;
};

/**
 * THE POPULATIONS. The load-bearing type of this directory.
 *
 * Every headline number is a fraction, and the two halves of a fraction fail differently. The
 * numerator is a rule ("did this case pass?"); the denominator is a population ("which cases
 * were eligible to be asked?"). Until 5.4b the population rules were four filter expressions
 * inline in `main()`, each re-spelling `!c.expectFail` and `ran(c)` for itself. A rule spelled
 * at four call sites is a rule that can be changed at three of them.
 *
 * The exclusions, and why each one is an exclusion rather than a failure:
 *   expectFail — a parked, documented bug. Counting it drags a number down and reads as a
 *                regression that nobody caused. Reported on its own line instead.
 *   ERROR      — the case could not RUN. The pipeline said nothing about it, and recording
 *                silence as a wrong answer is how a red run stops meaning anything. It cannot
 *                hide anything either: one errored case makes the whole run incomplete.
 *   injection  — counted on its own axis. "Did it refuse an out-of-corpus question" and "did
 *                it resist an instruction to disobey" are different properties.
 */
export type Populations = {
    /** shouldAnswer === true, not adversarial, not parked, ran. Recall / coverage denominator. */
    answerable: EvalCase[];
    /** shouldAnswer === false, not adversarial, not parked, ran. */
    guardrails: EvalCase[];
    /** Adversarial, not parked, ran. Its own axis, its own denominator. */
    injections: EvalCase[];
    /** Parked and known-failing. Reported, never counted — in either direction. */
    parked: EvalCase[];
};

/**
 * One labelled GitHub question, after its runs have been scored and matched to what the service
 * recorded. The `Result` of the second dataset.
 *
 * The process fields are arrays rather than aggregates on purpose (3.6b): a change in accuracy
 * has to be readable against a change in BEHAVIOUR, and a model that writes a query from memory
 * and one that reads the schema first can produce the same query. `lookups` and `typesRead` are
 * the only two fields that say which happened — the count says how much looking there was, the
 * list says what it looked at, and that is what separates "never saw the field" from "saw it and
 * passed it over". Two defects with different fixes that no counter can tell apart.
 */
export type GitHubResult = {
    id: string;
    label: "docs" | "both";
    runs: number;
    correct: number;
    routes: TurnFacts["route"][];
    routedRight: number;
    /** Turns the PLANNER canned before the router existed for them. Not a routing mistake: a
     *  turn that was never routed. 3.5 found three of three repository questions ending here. */
    canned: number;
    /** Turns that came back with a subagent block at all — the validity denominator. */
    subagentRuns: number;
    firstTryValid: number;
    ok: number;
    points: number[];
    attempts: number[];
    repairs: number[];
    lookups: number[];
    stages: string[][];
    typesRead: string[][];
    verdict: "PASS" | "FAIL" | "VARIED" | "ERROR";
    detail: string;
    sample: string;
    query: string | null;
    evidence: string | null;
};
