/**
 * RETRIEVAL RECALL — did the expected page survive rerank + threshold?
 *
 * Two numbers, not one, and the difference is the whole reason the second exists. `runOne`
 * counts one retrieval per case, which is the population the TypeScript baseline measured and
 * therefore the only one a figure from 2026-09-08 can be compared against. `everyRun` counts a
 * case only if the expected page came back in ALL of its runs, which is a question the
 * TypeScript target cannot be asked at all: it retrieves once and generates N times, while the
 * Python service plans and retrieves again on every run.
 *
 * Both are computed on both targets. Only the report decides which to print, because a metric
 * that is computed only when it is printed is a metric nobody can diff later.
 */
import type { Ratio } from "../record";
import type { GoldenContext, Result } from "./types";

export type RecallScore = {
    /** Run 1 only — comparable to every stored run since the TypeScript baseline. */
    runOne: Ratio;
    /** The expected page in EVERY run of the case. Meaningful on a per-run-retrieval target. */
    everyRun: Ratio;
    /** Cases where it came back in some runs and not others, with the count. HyDE moved the page. */
    varied: { id: string; foundRuns: number; runs: number }[];
};

export function evaluateRecall(ctx: GoldenContext): RecallScore {
    const { answerable } = ctx.populations;
    const den = answerable.length;
    const resultOf = (id: string): Result => ctx.byId.get(id)!;

    const varied = answerable
        .filter((c) => resultOf(c.id).retrievedEvery === "varied")
        .map((c) => ({ id: c.id, foundRuns: resultOf(c.id).foundRuns ?? 0, runs: resultOf(c.id).runs }));

    return {
        runOne: { num: answerable.filter((c) => resultOf(c.id).retrieved === "yes").length, den },
        everyRun: { num: answerable.filter((c) => resultOf(c.id).retrievedEvery === "yes").length, den },
        varied,
    };
}
