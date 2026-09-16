/**
 * The run record: what a full eval run leaves behind, and how an OLDER one is read back.
 *
 * `evals/results/` holds 23 stored runs written across six sub-steps, and they do not all have
 * the same shape. The five TypeScript runs have no `target`, `dirty`, `incomplete` or `errored`;
 * `errored` arrives 2026-09-14 with the ERROR verdict; `github` and `githubCost` arrive
 * 2026-09-15 with the labelled set. Comparing two of them therefore has a three-valued problem
 * before it has a one-line answer:
 *
 *   present on both and equal        → unchanged
 *   present on both and different    → a change
 *   absent from one                  → NOT KNOWN, which is neither of the above
 *
 * Collapsing that third case into either of the first two is how a diff reports a regression
 * that never happened, or reports agreement it never checked. So a measurement here is
 * `Maybe<T>`: either a value, or a reason there isn't one. There is deliberately no way to
 * spell "absent" as `0` — that is the entire point of the type, and the reason the diff can be
 * trusted against a file written before the field existed.
 *
 * Nothing in this module computes a metric. It reads what a run recorded and says what it
 * found; the evaluators decide what the numbers mean. Reading costs nothing, so this is the
 * one part of phase 5 that can be exercised against all 23 files for free, which is why it is
 * built first (spec section 5, sub-step 5.2).
 */

import { readFileSync } from "node:fs";

/**
 * Bumped when a field changes MEANING, not when one is added. A reader that finds a version it
 * does not know about should say so rather than guess; a reader that finds no version at all is
 * looking at a file written before this existed, and treats it as version 0.
 */
export const SCHEMA_VERSION = 1;

/** A run that predates `schemaVersion`. All 23 files on disk today are this. */
export const UNVERSIONED = 0;

// ---- a measurement that may not have been made ------------------------------------------------

export type Measured<T> = { known: true; value: T };
export type Unmeasured = { known: false; why: string };
export type Maybe<T> = Measured<T> | Unmeasured;

export const measured = <T>(value: T): Measured<T> => ({ known: true, value });
export const unmeasured = (why: string): Unmeasured => ({ known: false, why });

/**
 * Narrowing helper, so callers read as prose: `if (isKnown(m)) ... m.value`.
 *
 * Constrained over the whole shape rather than over one `T`, because a Metric's value is the
 * UNION `Maybe<Ratio> | Maybe<number>` and a `<T>` signature cannot infer a single T from it.
 * `Extract` distributes over that union and drops the `known: false` member, so the guard works
 * on a concrete measurement and on a metric read out of the table alike. (tsc found this, not
 * a test — the generic version type-checked everywhere it was first used.)
 */
export function isKnown<M extends { known: boolean }>(m: M): m is Extract<M, { known: true }> {
    return m.known;
}

/** "12/12" — stored as a string so a result file reads like the terminal did. */
export type Ratio = { num: number; den: number };

const RATIO = /^(\d+)\s*\/\s*(\d+)$/;

/**
 * `where` is the path as a human would cite it ("summary.recall"), because the reason a metric
 * is missing is the only thing the diff can print in its place, and "not recorded" without a
 * name is not a reason.
 */
export function asRatio(v: unknown, where: string): Maybe<Ratio> {
    if (v === undefined) return unmeasured(`${where} was not recorded by this run`);
    if (v === null) return unmeasured(`${where} was recorded as null`);
    if (typeof v !== "string") return unmeasured(`${where} was recorded as a ${typeof v}, not a ratio`);
    const m = RATIO.exec(v.trim());
    if (!m) return unmeasured(`${where} was recorded as ${JSON.stringify(v)}, which is not n/d`);
    const den = Number(m[2]);
    // A zero denominator is a real state (no injection cases in an EVAL_ONLY subset), not an
    // error — but it is not a rate either, and dividing by it downstream would produce NaN and
    // print as a change. Keep it, and let the diff decide.
    return measured({ num: Number(m[1]), den });
}

export function asNumber(v: unknown, where: string): Maybe<number> {
    if (v === undefined) return unmeasured(`${where} was not recorded by this run`);
    // Not pedantry: `retrievalMsWorst` is written as Math.max of a possibly-empty population,
    // which is -Infinity, which JSON.stringify writes as null. The console line guards that case
    // and the record does not (survey finding 2). A null here is a real thing to say out loud.
    if (v === null) return unmeasured(`${where} was recorded as null`);
    if (typeof v !== "number" || !Number.isFinite(v))
        return unmeasured(`${where} was recorded as ${JSON.stringify(v)}, which is not a finite number`);
    return measured(v);
}

export function asString(v: unknown, where: string): Maybe<string> {
    if (v === undefined) return unmeasured(`${where} was not recorded by this run`);
    if (v === null) return unmeasured(`${where} was recorded as null`);
    if (typeof v !== "string") return unmeasured(`${where} was recorded as a ${typeof v}, not a string`);
    return measured(v);
}

export function asBoolean(v: unknown, where: string): Maybe<boolean> {
    if (v === undefined) return unmeasured(`${where} was not recorded by this run`);
    if (v === null) return unmeasured(`${where} was recorded as null`);
    if (typeof v !== "boolean") return unmeasured(`${where} was recorded as a ${typeof v}, not a boolean`);
    return measured(v);
}

export function asStringArray(v: unknown, where: string): Maybe<string[]> {
    if (v === undefined) return unmeasured(`${where} was not recorded by this run`);
    if (!Array.isArray(v)) return unmeasured(`${where} was recorded as a ${typeof v}, not a list`);
    return measured(v.map((x) => String(x)));
}

/** Walks a dotted path without throwing on a missing parent. Returns undefined, never null. */
function at(root: unknown, path: string): unknown {
    let cur: unknown = root;
    for (const key of path.split(".")) {
        if (cur === null || cur === undefined || typeof cur !== "object") return undefined;
        cur = (cur as Record<string, unknown>)[key];
    }
    return cur === null ? null : cur;
}

// ---- the metrics a run reports ----------------------------------------------------------------

export type MetricKind = "ratio" | "count" | "ms" | "usd";

export type MetricDef = {
    id: string;
    label: string;
    /** Which dataset the number belongs to. The two are never folded together: the golden
     *  suite's figures are a series across stored runs, and the labelled set has its own
     *  denominators, its own cost line and its own exit code. */
    dataset: "golden" | "github";
    kind: MetricKind;
    /** The path as a human would cite it. Doubles as the reason text when the field is absent. */
    where: string;
    /** For the diff's arrow. Absent means higher is better. */
    lowerIsBetter?: boolean;
    /** Not a score — a population size, printed so a reader can see what n the score had. */
    isSampleSize?: boolean;
};

/**
 * Declared once, read by the diff and by the report. The duplication this replaces is real:
 * the injection numerator is currently computed twice in run.ts, once for the console line and
 * once as a separately written expression inside the record (survey finding 1). Two spellings
 * of one metric is what a table like this exists to prevent.
 */
export const METRICS: MetricDef[] = [
    { id: "recall", label: "recall run 1", dataset: "golden", kind: "ratio", where: "summary.recall" },
    { id: "recallEveryRun", label: "recall every run", dataset: "golden", kind: "ratio", where: "summary.recallEveryRun" },
    { id: "coverage", label: "answer coverage", dataset: "golden", kind: "ratio", where: "summary.coverage" },
    { id: "guardrails", label: "guardrails held", dataset: "golden", kind: "ratio", where: "summary.guardrails" },
    { id: "injection", label: "injection resisted", dataset: "golden", kind: "ratio", where: "summary.injection" },
    { id: "faithful", label: "faithfulness", dataset: "golden", kind: "ratio", where: "summary.faithful" },
    { id: "falseRefusals", label: "false refusals", dataset: "golden", kind: "count", where: "summary.falseRefusals", lowerIsBetter: true },
    { id: "retrievalMsMedian", label: "retrieval median", dataset: "golden", kind: "ms", where: "summary.retrievalMsMedian", lowerIsBetter: true },
    { id: "retrievalMsWorst", label: "retrieval worst", dataset: "golden", kind: "ms", where: "summary.retrievalMsWorst", lowerIsBetter: true },
    { id: "toSources", label: "to sources", dataset: "golden", kind: "ms", where: "summary.answeredMs.toSources", lowerIsBetter: true },
    { id: "firstToken", label: "first token", dataset: "golden", kind: "ms", where: "summary.answeredMs.firstToken", lowerIsBetter: true },
    { id: "done", label: "answer done", dataset: "golden", kind: "ms", where: "summary.answeredMs.done", lowerIsBetter: true },
    { id: "cannedDone", label: "canned done", dataset: "golden", kind: "ms", where: "summary.cannedMs.done", lowerIsBetter: true },
    { id: "answeredN", label: "answered runs", dataset: "golden", kind: "count", where: "summary.answeredMs.n", isSampleSize: true },
    { id: "costUsd", label: "suite cost", dataset: "golden", kind: "usd", where: "summary.cost.usd", lowerIsBetter: true },
    { id: "costPerRequest", label: "cost per request", dataset: "golden", kind: "usd", where: "summary.cost.usdPerRequest", lowerIsBetter: true },

    { id: "ghAccuracy", label: "answer accuracy", dataset: "github", kind: "ratio", where: "summary.github.accuracy" },
    { id: "ghFirstTryValid", label: "first-try query valid", dataset: "github", kind: "ratio", where: "summary.github.firstTryValid" },
    { id: "ghValidAfterRepairs", label: "valid after repairs", dataset: "github", kind: "ratio", where: "summary.github.validAfterRepairs" },
    { id: "ghRouting", label: "routing accuracy", dataset: "github", kind: "ratio", where: "summary.github.routing.overall" },
    { id: "ghCannedByPlanner", label: "canned by the planner", dataset: "github", kind: "count", where: "summary.github.routing.cannedByPlanner", lowerIsBetter: true },
    { id: "ghPointsMedian", label: "points per question", dataset: "github", kind: "count", where: "summary.github.points.median", lowerIsBetter: true },
    { id: "ghPointsMax", label: "points worst question", dataset: "github", kind: "count", where: "summary.github.points.max", lowerIsBetter: true },
    { id: "ghObservations", label: "observations", dataset: "github", kind: "count", where: "summary.github.observations", isSampleSize: true },
    { id: "ghCostUsd", label: "set cost", dataset: "github", kind: "usd", where: "summary.githubCost.usd", lowerIsBetter: true },
    { id: "ghCostPerRequest", label: "cost per request", dataset: "github", kind: "usd", where: "summary.githubCost.usdPerRequest", lowerIsBetter: true },
];

export type Metric =
    | { def: MetricDef; kind: "ratio"; value: Maybe<Ratio> }
    | { def: MetricDef; kind: "count" | "ms" | "usd"; value: Maybe<number> };

/**
 * `falseRefusals` is stored as the LIST of case ids, not a count. Reading its length is the
 * only derivation in this module, and it is written down rather than inferred at the call site.
 */
function readMetric(record: unknown, def: MetricDef): Metric {
    if (def.id === "falseRefusals") {
        const ids = asStringArray(at(record, def.where), def.where);
        return { def, kind: "count", value: isKnown(ids) ? measured(ids.value.length) : ids };
    }
    const raw = at(record, def.where);
    if (def.kind === "ratio") return { def, kind: "ratio", value: asRatio(raw, def.where) };
    return { def, kind: def.kind, value: asNumber(raw, def.where) };
}

// ---- one run, normalised ----------------------------------------------------------------------

export type NormalisedCase = {
    id: string;
    verdict: string;
    detail: string;
    /** Per case this is "answered/runs", e.g. "3/3". */
    answered: Maybe<Ratio>;
    retrieved: Maybe<string>;
    /** What the router decided, recorded since 3.6. `null` in the file means the row was missing. */
    route: Maybe<string>;
};

export type NormalisedGitHubCase = {
    id: string;
    verdict: string;
    /** "4/4" across the run's observations of that case. */
    correct: Maybe<Ratio>;
    detail: string;
};

export type NormalisedRun = {
    /** The file this came from, as cited. */
    file: string;
    schemaVersion: number;
    date: string;
    commit: string;
    /** Did the tree have uncommitted changes? Unknown before 2.6 added the field. */
    dirty: Maybe<boolean>;
    /** Did every case run? A run that did not measure everything is not a baseline. */
    incomplete: Maybe<boolean>;
    /** "ts" or "python". Unknown on the four runs written before there was a second target. */
    target: Maybe<string>;
    knobs: Record<string, unknown>;
    caseCount: Maybe<number>;
    metrics: Metric[];
    cases: NormalisedCase[];
    githubCases: Maybe<NormalisedGitHubCase[]>;
    parked: Maybe<string[]>;
    errored: Maybe<string[]>;
    failing: Maybe<string[]>;
};

function normaliseCase(raw: unknown): NormalisedCase {
    const o = (raw ?? {}) as Record<string, unknown>;
    const id = typeof o.id === "string" ? o.id : "(unnamed)";
    return {
        id,
        verdict: typeof o.verdict === "string" ? o.verdict : "(none)",
        detail: typeof o.detail === "string" ? o.detail : "",
        answered: asRatio(o.answered, `case ${id} answered`),
        retrieved: asString(o.retrieved, `case ${id} retrieved`),
        route: asString(o.route, `case ${id} route`),
    };
}

function normaliseGitHubCase(raw: unknown): NormalisedGitHubCase {
    const o = (raw ?? {}) as Record<string, unknown>;
    const id = typeof o.id === "string" ? o.id : "(unnamed)";
    return {
        id,
        verdict: typeof o.verdict === "string" ? o.verdict : "(none)",
        correct: asRatio(o.correct, `github case ${id} correct`),
        detail: typeof o.detail === "string" ? o.detail : "",
    };
}

/**
 * Parse a record that is already in memory. Separate from `readRecord` so the tests can hand it
 * a hand-built object — including deliberately broken ones — without writing files.
 *
 * This never throws on a missing field. It throws only when the thing it was handed is not a
 * record at all, because THAT is a mistake about which file you are pointing at, and a diff
 * against the wrong file is worse than no diff.
 */
export function parseRecord(raw: unknown, file: string): NormalisedRun {
    if (raw === null || typeof raw !== "object" || Array.isArray(raw))
        throw new Error(`${file}: not an eval record (expected an object, got ${Array.isArray(raw) ? "an array" : typeof raw})`);
    const o = raw as Record<string, unknown>;
    if (!("summary" in o) || !("cases" in o))
        throw new Error(`${file}: not an eval record (no summary and cases). A subset or retrieval-only run is not recorded at all — see run.ts.`);

    const version = typeof o.schemaVersion === "number" ? o.schemaVersion : UNVERSIONED;
    const cases = Array.isArray(o.cases) ? o.cases.map(normaliseCase) : [];
    const ghRaw = at(o, "summary.github.cases");

    return {
        file,
        schemaVersion: version,
        date: typeof o.date === "string" ? o.date : "(undated)",
        commit: typeof o.commit === "string" ? o.commit : "unknown",
        dirty: asBoolean(o.dirty, "dirty"),
        incomplete: asBoolean(o.incomplete, "incomplete"),
        target: asString(o.target, "target"),
        knobs: (o.knobs ?? {}) as Record<string, unknown>,
        caseCount: asNumber(at(o, "summary.cases"), "summary.cases"),
        metrics: METRICS.map((def) => readMetric(o, def)),
        cases,
        githubCases: Array.isArray(ghRaw)
            ? measured(ghRaw.map(normaliseGitHubCase))
            : unmeasured("this run has no labelled GitHub set (it predates 3.6, or ran with EVAL_GITHUB=0)"),
        parked: asStringArray(at(o, "summary.parked"), "summary.parked"),
        errored: asStringArray(at(o, "summary.errored"), "summary.errored"),
        failing: asStringArray(at(o, "summary.failing"), "summary.failing"),
    };
}

export function readRecord(file: string): NormalisedRun {
    let text: string;
    try {
        text = readFileSync(file, "utf8");
    } catch {
        throw new Error(`${file}: cannot be read. Stored runs live in evals/results/; baselines.json names them.`);
    }
    let json: unknown;
    try {
        json = JSON.parse(text);
    } catch (err) {
        throw new Error(`${file}: is not valid JSON (${err instanceof Error ? err.message : String(err)})`);
    }
    return parseRecord(json, file);
}

/** Metrics by id, for a caller that wants one by name rather than the whole list. */
export function metricsById(run: NormalisedRun): Map<string, Metric> {
    return new Map(run.metrics.map((m) => [m.def.id, m]));
}

// ---- formatting --------------------------------------------------------------------------------

export const formatRatio = (r: Ratio): string => `${r.num}/${r.den}`;

/** One metric's value as a reader sees it. An unmeasured one is a dash, never a zero. */
export function formatMetric(m: Metric): string {
    if (!isKnown(m.value)) return "—";
    if (m.kind === "ratio") return formatRatio(m.value.value);
    if (m.kind === "ms") return `${Math.round(m.value.value)}ms`;
    if (m.kind === "usd") return `$${m.value.value.toFixed(4)}`;
    return String(m.value.value);
}
