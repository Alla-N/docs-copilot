/**
 * Run-to-run diff.
 *
 *   npm run eval:diff -- --list
 *   npm run eval:diff -- --baseline python-3.6c
 *   npm run eval:diff -- --baseline python-3.6c --variant 2026-09-16T09-12-00-python.json
 *   EVAL_TARGET=python AGENT_URL=http://127.0.0.1:8000 npm run eval:diff -- --baseline python-3.6c --run
 *
 * Compare-only by DEFAULT (decision 3). Two stored runs cost nothing to read, so the expensive
 * thing is opt-in and the formatting below could be developed against the 23 files already on
 * disk without paying once. `--run` executes the suite first and then diffs what it wrote.
 *
 * Three rules this exists to enforce, all of them learned from stored runs rather than invented:
 *
 *   1. A metric absent from either side is NOT COMPARABLE. It prints under its own heading with
 *      the reason, never as `=` and never as a change. Six of the 23 files carry the GitHub set
 *      and five have no `target` at all, so this is the common case, not an edge one.
 *   2. A changed DENOMINATOR is a changed population, not a changed score. 12/12 against 11/12
 *      is a regression; 12/12 against 12/13 is a different suite. They get different words.
 *   3. Latency and cost print their delta and get no verdict. Retrieval is nondeterministic and
 *      token counts move; calling a 12 ms move a regression would be the harness lying. The
 *      ratio metrics and counts are the ones with a direction.
 *
 * Exit code is always 0. This REPORTS; it is not a gate. A gate that fires on HyDE variance
 * teaches you to ignore it, and the suite already has five exit codes that mean something.
 */

import { execSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import {
    METRICS,
    formatMetric,
    formatRatio,
    isKnown,
    metricsById,
    readRecord,
    type Metric,
    type MetricDef,
    type NormalisedRun,
} from "./record";

const RESULTS = "evals/results";
const BASELINES = "evals/baselines.json";

type BaselineEntry = { file: string; commit?: string; note: string; pairedWith?: string };

function loadBaselines(): Map<string, BaselineEntry> {
    const raw = JSON.parse(readFileSync(BASELINES, "utf8")) as Record<string, unknown>;
    const out = new Map<string, BaselineEntry>();
    for (const [name, value] of Object.entries(raw)) {
        if (name.startsWith("_")) continue; // the comment block
        const e = value as BaselineEntry;
        out.set(name, e);
    }
    return out;
}

function storedRuns(): string[] {
    return readdirSync(RESULTS).filter((f) => f.endsWith(".json")).sort();
}

/** A name from baselines.json, a filename inside evals/results/, or a path. In that order. */
function resolveRef(ref: string, baselines: Map<string, BaselineEntry>): string {
    const named = baselines.get(ref);
    if (named) return join(RESULTS, named.file);
    if (existsSync(join(RESULTS, ref))) return join(RESULTS, ref);
    if (existsSync(ref)) return ref;
    throw new Error(
        `Unknown run "${ref}". Named baselines: ${[...baselines.keys()].join(", ")}.\n` +
        `Or give a filename from ${RESULTS}/, or a path.`
    );
}

// ---- comparing ---------------------------------------------------------------------------------

type State = "same" | "better" | "worse" | "moved" | "population" | "incomparable";

type Comparison = {
    def: MetricDef;
    baseline: Metric;
    variant: Metric;
    state: State;
    /** What to print in the third column. */
    delta: string;
    /** Only set when state is "incomparable". */
    why?: string;
};

/** The reason a measurement is missing, or "" when it is not missing. */
function reason(v: Metric["value"]): string {
    return "why" in v ? v.why : "";
}

function directionOf(def: MetricDef, deltaUp: boolean): State {
    if (def.isSampleSize) return "moved";
    return deltaUp === !def.lowerIsBetter ? "better" : "worse";
}

function compare(def: MetricDef, baseline: Metric, variant: Metric): Comparison {
    // Kept as one inline condition rather than two named booleans: assigning a type guard to a
    // const loses the narrowing for the branches below, and tsc said so.
    if (!isKnown(baseline.value) || !isKnown(variant.value)) {
        // Which SIDE is missing it matters. "Neither run recorded this" is a metric that did not
        // exist yet; "the variant did not record it" is a metric that STOPPED being written, and
        // that is something to go and look at rather than a fact about history.
        const bWhy = reason(baseline.value);
        const vWhy = reason(variant.value);
        const why = bWhy && vWhy
            ? `neither run recorded it — ${bWhy}`
            : bWhy
                ? `baseline: ${bWhy}`
                : `variant: ${vWhy}`;
        return { def, baseline, variant, state: "incomparable", delta: "", why };
    }

    if (baseline.kind === "ratio" && variant.kind === "ratio") {
        const b = baseline.value.value;
        const v = variant.value.value;
        if (b.den !== v.den)
            // Rule 2. A different denominator is a different population — a subset run, a case
            // added, a case excluded because it could not run. Comparing the numerators would
            // report a regression that is really a change of what was measured.
            return { def, baseline, variant, state: "population", delta: `denominator ${b.den} → ${v.den}` };
        if (b.num === v.num) return { def, baseline, variant, state: "same", delta: "=" };
        const d = v.num - b.num;
        return { def, baseline, variant, state: directionOf(def, d > 0), delta: `${d > 0 ? "+" : ""}${d}` };
    }

    if (baseline.kind !== "ratio" && variant.kind !== "ratio") {
        const b = baseline.value.value;
        const v = variant.value.value;
        if (b === v) return { def, baseline, variant, state: "same", delta: "=" };
        const d = v - b;
        const sign = d > 0 ? "+" : "−";
        const size = Math.abs(d);
        const text =
            def.kind === "ms" ? `${sign}${Math.round(size)}ms`
                : def.kind === "usd" ? `${sign}$${size.toFixed(4)}`
                    : `${sign}${size}`;
        // Rule 3: ms and usd get the number and no verdict.
        const state: State = def.kind === "ms" || def.kind === "usd" ? "moved" : directionOf(def, d > 0);
        return { def, baseline, variant, state, delta: text };
    }

    // The same metric id read as two different kinds means the table changed under a stored
    // file. Say so rather than compare a ratio with a number.
    return { def, baseline, variant, state: "incomparable", delta: "", why: `kind changed: ${baseline.kind} → ${variant.kind}` };
}

// ---- printing ----------------------------------------------------------------------------------

const MARK: Record<State, string> = {
    same: " ",
    better: "▲",
    worse: "▼",
    moved: " ",
    population: "!",
    incomparable: " ",
};

const pad = (s: string, n: number) => s.padEnd(n);
const padStart = (s: string, n: number) => s.padStart(n);

function describe(run: NormalisedRun, label: string, ref: string, entry?: BaselineEntry): string {
    const target = isKnown(run.target) ? run.target.value : "unrecorded target";
    const dirty = isKnown(run.dirty) && run.dirty.value ? ", WITH UNCOMMITTED CHANGES" : "";
    const note = entry?.note ? `\n            ${entry.note}` : "";
    return `${pad(label, 10)} ${pad(ref, 22)} ${run.commit}  ${run.date.slice(0, 19)}  ${target}${dirty}${note}`;
}

/** Everything the reader needs before they are allowed to believe a single line below. */
function caveats(baseline: NormalisedRun, variant: NormalisedRun): string[] {
    const out: string[] = [];
    const incomplete = [baseline, variant].filter((r) => isKnown(r.incomplete) && r.incomplete.value);
    for (const r of incomplete)
        out.push(`${r.file} is INCOMPLETE — at least one case could not run. A run that did not measure everything is not a baseline.`);

    const bt = isKnown(baseline.target) ? baseline.target.value : null;
    const vt = isKnown(variant.target) ? variant.target.value : null;
    if (bt && vt && bt !== vt)
        out.push(`different targets (${bt} → ${vt}): this is a port comparison, not a regression check.`);
    if (bt === null || vt === null)
        out.push(`one side does not record its target — it predates there being a second one, so it is the TypeScript pipeline.`);

    for (const key of new Set([...Object.keys(baseline.knobs), ...Object.keys(variant.knobs)])) {
        const b = JSON.stringify(baseline.knobs[key]);
        const v = JSON.stringify(variant.knobs[key]);
        if (b !== v) out.push(`knob ${key}: ${b} → ${v} — the two runs did not measure the same thing.`);
    }
    return out;
}

function printMetricBlock(title: string, rows: Comparison[]): void {
    const comparable = rows.filter((r) => r.state !== "incomparable");
    if (!comparable.length) return;
    console.log(`\n${title}`);
    for (const r of comparable) {
        console.log(
            `  ${MARK[r.state]} ${pad(r.def.label, 22)} ${padStart(formatMetric(r.baseline), 9)}  →  ` +
            `${padStart(formatMetric(r.variant), 9)}   ${r.delta}`
        );
    }
}

function printCaseChanges(baseline: NormalisedRun, variant: NormalisedRun): void {
    const b = new Map(baseline.cases.map((c) => [c.id, c]));
    const v = new Map(variant.cases.map((c) => [c.id, c]));
    const lines: string[] = [];
    for (const [id, bc] of b) {
        const vc = v.get(id);
        if (!vc) { lines.push(`  ${pad(id, 24)} ${bc.verdict} → (not in the variant)`); continue; }
        if (bc.verdict !== vc.verdict) lines.push(`  ${pad(id, 24)} ${bc.verdict} → ${vc.verdict}   ${vc.detail}`);
    }
    for (const id of v.keys()) if (!b.has(id)) lines.push(`  ${pad(id, 24)} (new case) → ${v.get(id)!.verdict}`);
    // Rule: only what CHANGED. A 27-line table where 26 lines say "=" is how a reader stops
    // reading diffs at all.
    console.log(`\nper case  (${lines.length ? `${lines.length} changed` : "no verdict changed, 27 cases"})`);
    for (const l of lines) console.log(l);
}

function printGitHubCaseChanges(baseline: NormalisedRun, variant: NormalisedRun): void {
    if (!isKnown(baseline.githubCases) || !isKnown(variant.githubCases)) return;
    const b = new Map(baseline.githubCases.value.map((c) => [c.id, c]));
    const v = new Map(variant.githubCases.value.map((c) => [c.id, c]));
    const lines: string[] = [];
    for (const [id, bc] of b) {
        const vc = v.get(id);
        if (!vc) { lines.push(`  ${pad(id, 24)} ${bc.verdict} → (not in the variant)`); continue; }
        const bs = isKnown(bc.correct) ? formatRatio(bc.correct.value) : "—";
        const vs = isKnown(vc.correct) ? formatRatio(vc.correct.value) : "—";
        if (bc.verdict !== vc.verdict || bs !== vs)
            lines.push(`  ${pad(id, 24)} ${pad(`${bc.verdict} ${bs}`, 14)} → ${pad(`${vc.verdict} ${vs}`, 14)} ${vc.detail}`);
    }
    for (const id of v.keys()) if (!b.has(id)) lines.push(`  ${pad(id, 24)} (new case) → ${v.get(id)!.verdict}`);
    console.log(`\ngithub per case  (${lines.length ? `${lines.length} changed` : "nothing changed"})`);
    for (const l of lines) console.log(l);
}

// ---- main ---------------------------------------------------------------------------------------

type Args = { baseline?: string; variant?: string; run: boolean; list: boolean; cases: boolean };

function parseArgs(argv: string[]): Args {
    const a: Args = { run: false, list: false, cases: true };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === "--baseline" || arg === "-b") a.baseline = argv[++i];
        else if (arg === "--variant" || arg === "-v") a.variant = argv[++i];
        else if (arg === "--run") a.run = true;
        else if (arg === "--list") a.list = true;
        else if (arg === "--no-cases") a.cases = false;
        else throw new Error(`Unknown argument "${arg}". Use --list, --baseline, --variant, --run, --no-cases.`);
    }
    return a;
}

function main(): void {
    const args = parseArgs(process.argv.slice(2));
    const baselines = loadBaselines();

    if (args.list) {
        console.log("named baselines (evals/baselines.json):\n");
        for (const [name, e] of baselines)
            console.log(`  ${pad(name, 14)} ${pad(e.commit ?? "", 9)} ${e.file}\n                 ${e.note}\n`);
        return;
    }

    if (!args.baseline) throw new Error(`--baseline is required. Run with --list to see the names.`);
    const baselineFile = resolveRef(args.baseline, baselines);

    let variantRef = args.variant;
    if (args.run) {
        // The one expensive path, and it is opt-in. The suite reads its own environment
        // (EVAL_TARGET, AGENT_URL); this just runs it and then takes what it wrote.
        const before = new Set(storedRuns());
        console.log(`running the suite first (--run)...\n`);
        execSync("npm run eval", { stdio: "inherit" });
        const written = storedRuns().filter((f) => !before.has(f));
        if (!written.length)
            throw new Error(`the run recorded nothing. Subsets (EVAL_ONLY) and retrieval-only runs (EVAL_RUNS=0) are diagnostics and are not recorded — see run.ts.`);
        variantRef = written[written.length - 1];
        console.log(`\nvariant is the run just recorded: ${variantRef}`);
    }
    if (!variantRef) {
        const all = storedRuns();
        variantRef = all[all.length - 1];
    }
    const variantFile = resolveRef(variantRef, baselines);

    if (baselineFile === variantFile) {
        console.log(`baseline and variant are the same file (${baselineFile}). Nothing to diff.`);
        return;
    }

    const baseline = readRecord(baselineFile);
    const variant = readRecord(variantFile);

    console.log(`eval diff`);
    console.log(describe(baseline, "  baseline", args.baseline, baselines.get(args.baseline)));
    console.log(describe(variant, "  variant", variantRef, baselines.get(variantRef)));

    const notes = caveats(baseline, variant);
    if (notes.length) {
        console.log(`\nread these first`);
        for (const n of notes) console.log(`  ! ${n}`);
    }

    const bm = metricsById(baseline);
    const vm = metricsById(variant);
    const rows = METRICS.map((def) => compare(def, bm.get(def.id)!, vm.get(def.id)!));

    printMetricBlock("golden set", rows.filter((r) => r.def.dataset === "golden"));
    printMetricBlock("github set", rows.filter((r) => r.def.dataset === "github"));

    if (args.cases) {
        printCaseChanges(baseline, variant);
        printGitHubCaseChanges(baseline, variant);
    }

    // Rule 1, printed last so it is the thing left on screen: what this diff did NOT compare.
    const incomparable = rows.filter((r) => r.state === "incomparable");
    if (incomparable.length) {
        console.log(`\nnot comparable  (${incomparable.length})`);
        for (const r of incomparable) console.log(`  ${pad(r.def.label, 22)} ${r.why}`);
    } else {
        console.log(`\nnot comparable  (none — both runs recorded every metric in the table)`);
    }

    const changed = rows.filter((r) => r.state === "better" || r.state === "worse" || r.state === "population").length;
    console.log(`\n${changed} metric(s) changed, ${rows.filter((r) => r.state === "same").length} unchanged, ${incomparable.length} not comparable.`);
}

try {
    main();
} catch (err) {
    console.error(err instanceof Error ? err.message : err);
    // Bad arguments are the caller's mistake and worth a non-zero code; a printed diff is not.
    process.exit(2);
}
