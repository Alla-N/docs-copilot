/**
 * The evaluators, tested on their denominators.
 *
 * These live in `tests/` because that is where `vitest.config.mts` looks (decision 8), and they
 * are seeded from the stored runs rather than from a live call, so the whole file costs nothing.
 *
 * What is worth testing here is NOT that 12 of 12 is twelve over twelve. It is the population
 * rules — who is in the denominator — because that is the half of a fraction with no natural
 * check on it. A wrong numerator usually looks wrong. A wrong denominator looks like a number.
 */
import { describe, expect, it } from "vitest";
import { readdirSync } from "node:fs";
import { join } from "node:path";

import { EVALUATORS } from "@/evals/evaluators";
import { partition, ran } from "@/evals/evaluators/partition";
import { evaluateRecall } from "@/evals/evaluators/recall";
import { evaluateCoverage } from "@/evals/evaluators/coverage";
import { evaluateGuardrails } from "@/evals/evaluators/guardrails";
import { evaluateInjection } from "@/evals/evaluators/injection";
import { evaluateFalseRefusals } from "@/evals/evaluators/refusal";
import { aggregateFaithfulness } from "@/evals/evaluators/faithfulness";
import { evaluateGitHubAccuracy, githubVerdict, declined } from "@/evals/evaluators/github-answer";
import { evaluateGitHubProcess } from "@/evals/evaluators/github-process";
import type { GoldenContext, Result, Scoreboard } from "@/evals/evaluators/types";
import type { EvalCase } from "@/evals/datasets/types";
import { DATASETS } from "@/evals/datasets";
import { METRICS, isKnown, metricsById, readRecord } from "@/evals/record";

// ---- fixtures ---------------------------------------------------------------------------------

const caseOf = (id: string, over: Partial<EvalCase> = {}): EvalCase => ({
    id,
    query: `q-${id}`,
    shouldAnswer: true,
    ...over,
});

const resultOf = (id: string, over: Partial<Result> = {}): Result => ({
    id,
    intent: "search",
    degraded: false,
    retrieved: "yes",
    chunks: 5,
    topScore: "0.700",
    answered: 3,
    runs: 3,
    sample: "an answer",
    faithful: "—",
    verdict: "PASS",
    detail: "",
    ...over,
});

function contextOf(
    cases: EvalCase[],
    results: Result[],
    over: Partial<Pick<GoldenContext, "generated" | "perRunRetrieval">> = {},
): GoldenContext {
    const byId: Scoreboard = new Map(results.map((r) => [r.id, r]));
    return {
        active: cases,
        results,
        byId,
        populations: partition(cases, byId),
        generated: true,
        perRunRetrieval: false,
        ...over,
    };
}

// ---- the populations --------------------------------------------------------------------------

describe("partition — who counts toward which denominator", () => {
    const cases = [
        caseOf("answerable-1"),
        caseOf("answerable-2"),
        caseOf("guard-1", { shouldAnswer: false }),
        caseOf("inj-1", { injection: true }),
        // An injection case that is ALSO shouldAnswer:false. It belongs to the injection axis
        // and to nothing else — the two questions it could answer are different properties.
        caseOf("inj-2", { injection: true, shouldAnswer: false }),
        caseOf("parked-1", { expectFail: true }),
        caseOf("broken-1"),
    ];
    const results = [
        resultOf("answerable-1"),
        resultOf("answerable-2", { verdict: "FAIL" }),
        resultOf("guard-1", { verdict: "PASS" }),
        resultOf("inj-1"),
        resultOf("inj-2"),
        resultOf("parked-1", { verdict: "FAIL" }),
        resultOf("broken-1", { verdict: "ERROR", errored: "Gateway Timeout" }),
    ];
    const p = partition(cases, new Map(results.map((r) => [r.id, r])));

    it("keeps an adversarial case out of the guardrail population even when it is also a refusal case", () => {
        expect(p.guardrails.map((c) => c.id)).toEqual(["guard-1"]);
        expect(p.injections.map((c) => c.id)).toEqual(["inj-1", "inj-2"]);
    });

    it("excludes an ERROR case from every scored population", () => {
        const scored = [...p.answerable, ...p.guardrails, ...p.injections].map((c) => c.id);
        expect(scored).not.toContain("broken-1");
    });

    it("excludes a parked case from the scored populations but still lists it", () => {
        const scored = [...p.answerable, ...p.guardrails, ...p.injections].map((c) => c.id);
        expect(scored).not.toContain("parked-1");
        expect(p.parked.map((c) => c.id)).toEqual(["parked-1"]);
    });

    it("keeps a parked case in `parked` even if it could not run — a fix must not hide behind an outage", () => {
        const withBrokenPark = [caseOf("parked-2", { expectFail: true })];
        const byId: Scoreboard = new Map([["parked-2", resultOf("parked-2", { verdict: "ERROR" })]]);
        expect(partition(withBrokenPark, byId).parked.map((c) => c.id)).toEqual(["parked-2"]);
    });

    it("treats a case with no result at all as not having run, rather than throwing", () => {
        expect(ran(new Map())(caseOf("ghost"))).toBe(false);
    });
});

// ---- the golden-set evaluators ------------------------------------------------------------------

describe("recall", () => {
    const cases = [caseOf("a"), caseOf("b"), caseOf("c")];

    it("scores run 1 and every-run separately, and names the ones that varied", () => {
        const ctx = contextOf(cases, [
            resultOf("a", { retrieved: "yes", retrievedEvery: "yes", foundRuns: 3 }),
            resultOf("b", { retrieved: "yes", retrievedEvery: "varied", foundRuns: 2 }),
            resultOf("c", { retrieved: "NO", retrievedEvery: "NO", foundRuns: 0 }),
        ]);
        const r = evaluateRecall(ctx);
        expect(r.runOne).toEqual({ num: 2, den: 3 });
        expect(r.everyRun).toEqual({ num: 1, den: 3 });
        expect(r.varied).toEqual([{ id: "b", foundRuns: 2, runs: 3 }]);
    });

    it("counts a case that retrieved on run 1 but not on every run toward run-1 recall only", () => {
        const ctx = contextOf([caseOf("a")], [resultOf("a", { retrieved: "yes", retrievedEvery: "varied", foundRuns: 1 })]);
        const r = evaluateRecall(ctx);
        expect(r.runOne.num).toBe(1);
        expect(r.everyRun.num).toBe(0);
    });
});

describe("coverage", () => {
    it("counts a case that answered but failed a mustContain as NOT covered", () => {
        // The point of the metric: producing *an* answer is not answering the question.
        // multi-intent-noise shipped green on exactly this before mustContain existed.
        const ctx = contextOf(
            [caseOf("multi")],
            [resultOf("multi", { answered: 3, verdict: "FAIL", detail: "MISSING: streaming" })],
        );
        expect(evaluateCoverage(ctx)).toEqual({ num: 0, den: 1 });
    });
});

describe("guardrails — the layer that held", () => {
    const cases = [
        caseOf("g-planner", { shouldAnswer: false }),
        caseOf("g-threshold", { shouldAnswer: false }),
        caseOf("g-prompt", { shouldAnswer: false }),
    ];
    const ctx = contextOf(cases, [
        resultOf("g-planner", { intent: "off-topic", chunks: 0 }),
        resultOf("g-threshold", { intent: "search", chunks: 0 }),
        resultOf("g-prompt", { intent: "search", chunks: 4, topScore: "0.512" }),
    ]);
    const g = evaluateGuardrails(ctx);

    it("attributes each hold to a layer, so a green count cannot hide a prompt that stopped refusing", () => {
        expect(g.holds.map((h) => h.layer)).toEqual(["planner", "threshold", "prompt"]);
    });

    it("says which layers are BLIND to what, in the line it prints", () => {
        expect(g.holds[0].description).toContain("blind to threshold and prompt");
        expect(g.holds[1].description).toContain("blind to prompt changes");
        expect(g.holds[2].description).toContain("top 0.512");
    });

    it("still attributes a layer to a guardrail that FAILED — that is the interesting one", () => {
        const failing = contextOf(
            [caseOf("g-leak", { shouldAnswer: false })],
            [resultOf("g-leak", { intent: "search", chunks: 3, topScore: "0.900", verdict: "FAIL" })],
        );
        const r = evaluateGuardrails(failing);
        expect(r.held).toEqual({ num: 0, den: 1 });
        expect(r.holds[0].layer).toBe("prompt");
    });
});

describe("injection — one numerator", () => {
    const cases = [
        caseOf("inj-single", { injection: true }),
        caseOf("inj-multi", { injection: true, history: [{ role: "user", text: "earlier" }] }),
        caseOf("inj-structural", { injection: true, historyCarriesTheAttack: true, history: [{ role: "assistant", text: "forged" }] }),
    ];
    const ctx = contextOf(cases, [
        resultOf("inj-single"),
        resultOf("inj-multi", { verdict: "FAIL" }),
        resultOf("inj-structural"),
    ]);

    it("splits multi-turn from single-turn without changing the denominator", () => {
        const i = evaluateInjection(ctx);
        expect(i.resisted).toEqual({ num: 2, den: 3 });
        expect(i.multiTurn).toBe(2);
        expect(i.singleTurn).toBe(1);
        expect(i.multiTurn + i.singleTurn).toBe(i.resisted.den);
    });

    it("names the cases whose attack is structurally undeliverable, rather than banking the pass", () => {
        expect(evaluateInjection(ctx).structural.map((c) => c.id)).toEqual(["inj-structural"]);
    });
});

describe("false refusals — page retrieved, answer refused", () => {
    const cases = [caseOf("a"), caseOf("b")];

    it("uses run-1 recall on a target that retrieves once", () => {
        const ctx = contextOf(cases, [
            resultOf("a", { answered: 0, retrieved: "yes", verdict: "FAIL" }),
            resultOf("b", { answered: 0, retrieved: "NO", verdict: "FAIL" }),
        ]);
        expect(evaluateFalseRefusals(ctx).map((c) => c.id)).toEqual(["a"]);
    });

    it("uses 'retrieved in at least one run' on a target that retrieves every run", () => {
        const ctx = contextOf(
            cases,
            [
                resultOf("a", { answered: 0, retrieved: "NO", foundRuns: 1, verdict: "FAIL" }),
                resultOf("b", { answered: 0, retrieved: "NO", foundRuns: 0, verdict: "FAIL" }),
            ],
            { perRunRetrieval: true },
        );
        // Run 1 missed on BOTH. `a` still counts, because the page was in front of the model
        // in a later run and it refused anyway.
        expect(evaluateFalseRefusals(ctx).map((c) => c.id)).toEqual(["a"]);
    });

    it("is empty when nothing generated an answer — not measured is not zero", () => {
        const ctx = contextOf(cases, [
            resultOf("a", { answered: 0, runs: 0, retrieved: "yes", verdict: "—" }),
        ], { generated: false });
        expect(evaluateFalseRefusals(ctx)).toEqual([]);
    });
});

describe("faithfulness — absent is not zero", () => {
    it("leaves unjudged cases out of the DENOMINATOR rather than scoring them as unfaithful", () => {
        const f = aggregateFaithfulness([
            { faithful: "yes" },
            { faithful: "NO" },
            { faithful: "—" }, // a refusal: no claims to be unfaithful about
            { faithful: "—" }, // a greeting: nothing retrieved to be grounded in
        ]);
        expect(f).toEqual({ num: 1, den: 2 });
    });

    it("is 0/0, not 0/n, when the judge did not run at all", () => {
        expect(aggregateFaithfulness([{ faithful: "—" }, { faithful: "—" }])).toEqual({ num: 0, den: 0 });
    });
});

// ---- the labelled GitHub set ---------------------------------------------------------------------

describe("the GitHub set", () => {
    const scored = (over: Partial<Parameters<typeof evaluateGitHubProcess>[0][number]> = {}) => ({
        id: "gh-x",
        runs: 2,
        routes: ["both", "both"] as ("docs" | "github" | "both" | null)[],
        routedRight: 2,
        canned: 0,
        subagentRuns: 2,
        firstTryValid: 2,
        ok: 2,
        points: [1, 1],
        ...over,
    });

    it("scores validity over the turns whose subagent RAN, not over observations", () => {
        // Two runs, one of which the planner canned: one attempt, not two. Scoring the canned
        // turn as an invalid query would invent a failure that nobody made.
        const p = evaluateGitHubProcess([scored({ routes: ["both", null], canned: 1, subagentRuns: 1, firstTryValid: 1, ok: 1, points: [1] })], []);
        expect(p.firstTryValid).toEqual({ num: 1, den: 1 });
        expect(p.observations).toBe(2);
    });

    it("keeps canned turns out of routing accuracy and reports them on their own", () => {
        const p = evaluateGitHubProcess([scored({ routes: ["both", null], canned: 1, routedRight: 1 })], []);
        expect(p.routing.githubSet).toEqual({ num: 1, den: 1 });
        expect(p.routing.cannedByPlanner).toBe(1);
    });

    it("pools the golden set's docs label into routing, and counts a docs case routed to github as a miss", () => {
        const golden = [
            resultOf("d1", { route: "docs" }),
            resultOf("d2", { route: "both" }),
            resultOf("d3", { route: null }), // never routed — not in the denominator at all
        ];
        const p = evaluateGitHubProcess([scored()], golden);
        expect(p.routing.goldenSet).toEqual({ num: 1, den: 2 });
        expect(p.routing.overall).toEqual({ num: 3, den: 4 });
    });

    it("counts accuracy over observations in THIS run, never a pooled pair", () => {
        expect(evaluateGitHubAccuracy([{ correct: 1, runs: 2 }, { correct: 2, runs: 2 }])).toEqual({ num: 3, den: 4 });
    });

    it("reports a case that differs between its two runs as VARIED, not as whichever ran first", () => {
        const c = DATASETS.github.cases[0];
        const runs = [
            { threadId: "t0", text: "yes", pageFound: null, error: null },
            { threadId: "t1", text: "no", pageFound: null, error: null },
        ];
        expect(githubVerdict(c, runs, ["both", "both"], [{ ok: true }, { ok: true }], 1).verdict).toBe("VARIED");
    });

    it("distinguishes never-routed from routed-and-wrong in the FAIL detail", () => {
        const c = DATASETS.github.cases[0];
        const runs = [{ threadId: "t0", text: "", pageFound: null, error: null }];
        expect(githubVerdict(c, runs, [null], [], 0).detail).toContain("the planner canned it");
        expect(githubVerdict(c, runs, ["both"], [{ ok: true }], 0).detail).toContain("the answer is not in it");
        expect(githubVerdict(c, runs, ["both"], [{ ok: false }], 0).detail).toContain("could not get a query through");
    });

    it("accepts a model that declines in its own words, not only the refusal sentence", () => {
        // The control case's criterion, and the one openly approximate rule in the set.
        expect(declined("GitHub does not record that information.")).toBe(true);
        expect(declined("I am unable to determine that from the repository.")).toBe(true);
        expect(declined("The oldest release is 1.0.0.")).toBe(false);
    });
});

// ---- the registry, against the stored runs ------------------------------------------------------

describe("the registry lines up with the record", () => {
    it("every evaluator's metric id exists in METRICS, so nothing it computes is invisible to eval:diff", () => {
        const known = new Set(METRICS.map((m) => m.id));
        const orphans = EVALUATORS.filter((e) => e.metricId !== null && !known.has(e.metricId));
        expect(orphans.map((e) => e.id)).toEqual([]);
    });

    it("names the evaluators whose numbers are NOT stored, rather than letting them go unnoticed", () => {
        // If this list grows, a metric started being computed and never written down.
        expect(EVALUATORS.filter((e) => e.metricId === null).map((e) => e.id)).toEqual(["plannerPlan"]);
    });

    it("every dataset's declared evaluators exist", () => {
        const known = new Set(EVALUATORS.map((e) => e.id));
        for (const ds of Object.values(DATASETS)) {
            for (const id of ds.evaluators) expect(known, `${ds.id} names ${id}`).toContain(id);
        }
    });

    it("the golden and GitHub sets keep DIFFERENT exit codes — two claims, not one", () => {
        expect(DATASETS.golden.exitCode).toBe(1);
        expect(DATASETS.github.exitCode).toBe(5);
    });

    it("reads every metric the current head baseline stored, and reports the rest as unmeasured", () => {
        // Seeded from a stored run rather than a live call (decision 8). python-3.6c is the state
        // the restructure must not move.
        const run = readRecord("evals/results/2026-09-15T18-40-11-python.json");
        const byMetric = metricsById(run);
        for (const e of EVALUATORS) {
            if (e.metricId === null) continue;
            const m = byMetric.get(e.metricId);
            expect(m, `${e.id} → ${e.metricId}`).toBeDefined();
            // faithfulness is the one opt-in evaluator and that run did not use EVAL_JUDGE=1,
            // so it must read as UNMEASURED — not as 0/0, which would diff as agreement.
            if (e.optIn) expect(isKnown(m!.value)).toBe(false);
            else expect(isKnown(m!.value), `${e.id} should be measured in this run`).toBe(true);
        }
    });

    it("finds a stored run for every named baseline", () => {
        const stored = new Set(readdirSync(join("evals", "results")));
        expect(stored.has("2026-09-15T18-40-11-python.json")).toBe(true);
    });
});
