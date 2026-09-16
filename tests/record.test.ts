import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
    METRICS,
    SCHEMA_VERSION,
    UNVERSIONED,
    asNumber,
    asRatio,
    formatMetric,
    isKnown,
    metricsById,
    parseRecord,
    readRecord,
    type NormalisedRun,
} from "@/evals/record";

/**
 * The point of these tests is not that the reader compiles. It is that a field an older run
 * never wrote comes back as UNMEASURED and not as zero — because zero is a number a diff would
 * print as a change, and "not recorded" is not a change.
 *
 * They run against every file actually in evals/results/, so the day a new field is added the
 * assertions below start describing the real history rather than a fixture of it. No network,
 * no model, no cost.
 */

const RESULTS = "evals/results";

const files = readdirSync(RESULTS)
    .filter((f) => f.endsWith(".json"))
    .sort();

const runs: NormalisedRun[] = files.map((f) => readRecord(join(RESULTS, f)));

const dateOf = (r: NormalisedRun) => r.file.split("/").pop()!.slice(0, 10);

describe("the stored runs", () => {
    it("has the corpus these assertions were written against", () => {
        // If this fails, a run was added or deleted. That is fine — update the number and read
        // the rest of this file, because the history the assertions below describe just moved.
        expect(files.length).toBeGreaterThanOrEqual(23);
    });

    it("parses every one of them", () => {
        for (const r of runs) expect(r.commit).not.toBe("");
    });

    it("finds no schemaVersion in any file written before this module existed", () => {
        // All 23 are version 0. The first run recorded after 5.2 lands will be SCHEMA_VERSION.
        const versioned = runs.filter((r) => r.schemaVersion !== UNVERSIONED);
        for (const r of versioned) expect(r.schemaVersion).toBe(SCHEMA_VERSION);
    });
});

describe("a field an older run never wrote", () => {
    const ts = runs.filter((r) => !r.file.includes("-python"));

    it("reads as unmeasured, with a reason, on the TypeScript runs", () => {
        // Five of them: four from 2026-09-07 and the 2026-09-08 baseline the README cites.
        // `dirty` arrives with the Python target in 2.6, so none of these has it either.
        expect(ts.length).toBe(5);
        for (const r of ts) {
            expect(isKnown(r.target)).toBe(false);
            expect(isKnown(r.dirty)).toBe(false);
            expect(isKnown(r.incomplete)).toBe(false);
            if (!isKnown(r.target)) expect(r.target.why).toContain("target");
        }
    });

    it("never comes back as a zero", () => {
        // The assertion this whole module exists for. Every unmeasured metric in every stored
        // run must format as a dash; a single "0" here would be a silent regression in a diff.
        for (const r of runs)
            for (const m of r.metrics)
                if (!isKnown(m.value)) expect(formatMetric(m)).toBe("—");
    });

    it("carries a reason that names the field", () => {
        for (const r of runs)
            for (const m of r.metrics)
                if (!isKnown(m.value)) expect(m.value.why).toContain(m.def.where);
    });

    it("marks errored absent before the ERROR verdict existed (2026-09-14)", () => {
        const before = runs.filter((r) => dateOf(r) < "2026-09-14");
        expect(before.length).toBeGreaterThan(0);
        for (const r of before) expect(isKnown(r.errored)).toBe(false);
    });

    it("marks the labelled GitHub set absent before 3.6 (2026-09-15)", () => {
        const before = runs.filter((r) => dateOf(r) < "2026-09-15");
        for (const r of before) expect(isKnown(r.githubCases)).toBe(false);
        // NOT a census, and it was one until 5.5.
        //
        // This read `expect(withSet.length).toBe(6)` and it failed on the done-when run: the
        // one commit of the phase that changed nothing whatever about this reader. Two more
        // stored runs carried the set, the count went to 8, and the assertion broke because
        // the project did exactly what it is supposed to do.
        //
        // The trap is what the repair looks like. Bump 6 to 8, green again, breaks at 10. An
        // assertion whose fix is always "edit the number" teaches you to edit the test rather
        // than read it, and a test nobody reads cannot report anything.
        //
        // The contract was never "six runs have it". It is "the reader agrees with the file",
        // and that holds for every run there will ever be.
        for (const r of runs) {
            const raw = JSON.parse(readFileSync(r.file, "utf8"));
            const inFile = Array.isArray(raw?.summary?.github?.cases);
            expect(isKnown(r.githubCases), r.file).toBe(inFile);
        }
        // A CLOSED set can be counted exactly -- see the five TypeScript runs above, which can
        // never grow again because that target stopped being written to. This set is open, so
        // the only honest bound is a floor.
        expect(runs.filter((r) => isKnown(r.githubCases)).length).toBeGreaterThanOrEqual(6);
    });

    it("has recall on every single run, all the way back", () => {
        // The counterpart assertion: the series that must stay comparable actually does.
        for (const r of runs) {
            const recall = metricsById(r).get("recall");
            expect(recall && isKnown(recall.value)).toBe(true);
        }
    });
});

describe("prediction P2 — which metrics need normalising at all", () => {
    it("names them, so the spec's list can be checked against reality", () => {
        const partial = METRICS.filter((def) => {
            const present = runs.filter((r) => isKnown(metricsById(r).get(def.id)!.value)).length;
            return present > 0 && present < runs.length;
        }).map((d) => d.id);
        // P2 named four fields in advance: target, errored, github, githubCost. Every metric
        // below is a field that some stored run has and another does not.
        expect(partial).toContain("ghAccuracy");
        expect(partial).toContain("ghCostUsd");
        expect(partial).toContain("recallEveryRun");
        expect(partial).toContain("costUsd");
    });
});

describe("the parsers", () => {
    it("tells undefined from null, because they mean different things", () => {
        const missing = asNumber(undefined, "summary.x");
        const nulled = asNumber(null, "summary.x");
        expect(isKnown(missing)).toBe(false);
        expect(isKnown(nulled)).toBe(false);
        if (!isKnown(missing)) expect(missing.why).toContain("not recorded");
        // -Infinity through JSON.stringify: the record writer's empty-population case.
        if (!isKnown(nulled)) expect(nulled.why).toContain("null");
    });

    it("refuses a non-finite number rather than comparing one", () => {
        expect(isKnown(asNumber(Number.NaN, "summary.x"))).toBe(false);
        expect(isKnown(asNumber(Number.POSITIVE_INFINITY, "summary.x"))).toBe(false);
    });

    it("reads a ratio and keeps a zero denominator as itself", () => {
        const r = asRatio("12/12", "summary.recall");
        expect(isKnown(r) && r.value).toEqual({ num: 12, den: 12 });
        const empty = asRatio("0/0", "summary.injection");
        // A subset run really can have no injection cases. That is a state, not a failure to
        // record, and turning it into a rate here would produce NaN downstream.
        expect(isKnown(empty) && empty.value).toEqual({ num: 0, den: 0 });
    });

    it("rejects a ratio that is not n/d", () => {
        expect(isKnown(asRatio("twelve", "summary.recall"))).toBe(false);
        expect(isKnown(asRatio(12, "summary.recall"))).toBe(false);
    });
});

describe("parseRecord", () => {
    it("throws on something that is not a record at all", () => {
        // Pointing the diff at the wrong file is worse than not running it.
        expect(() => parseRecord([], "x.json")).toThrow(/not an eval record/);
        expect(() => parseRecord({ hello: 1 }, "x.json")).toThrow(/not an eval record/);
        expect(() => parseRecord(null, "x.json")).toThrow(/not an eval record/);
    });

    it("does not throw on a record whose summary is empty", () => {
        // Every metric is then unmeasured, which is a readable state; refusing the file would
        // make the reader stricter than the writer has ever been.
        const r = parseRecord({ summary: {}, cases: [] }, "x.json");
        expect(r.metrics.every((m) => !isKnown(m.value))).toBe(true);
        expect(r.commit).toBe("unknown");
    });

    it("reads false refusals as a count of the ids it stores", () => {
        const r = parseRecord({ summary: { falseRefusals: ["changed-7", "new-7"] }, cases: [] }, "x.json");
        const m = metricsById(r).get("falseRefusals")!;
        expect(isKnown(m.value) && m.value.value).toBe(2);
    });
});
