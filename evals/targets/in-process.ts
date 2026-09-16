/**
 * The in-process target (EVAL_TARGET=ts, the default): the pipeline run in this process.
 *
 * `plannedRetrieve` + `generateText` — the same functions the Next.js route calls, not a copy
 * of them (invariants #3 and #4). Retrieval runs ONCE per case and generation runs N times,
 * because that is where the non-determinism lives: temp 0 lowers variance, it does not remove
 * it. The planner's HyDE hypothetical is model output, so retrieval is only *nearly*
 * deterministic and a different hypothetical can reorder near-tied pages.
 *
 * Extracted from run.ts in 5.4b. The one thing that had to change shape is the latency
 * population: `retrievalMs` was a module-level array in run.ts that this function pushed into,
 * so the caller now gets the measurement back and owns the array. A target that reaches into a
 * global to record its own timings is a target you cannot run twice.
 */
import { generateText } from "ai";

import { generationMessages, generationSettings } from "../../lib/generation";
import { GREETING_MESSAGE, plannedRetrieve } from "../../lib/plan";
import { isRefusal, REFUSAL_MESSAGE } from "../../lib/retrieve";
import type { EvalCase } from "../datasets/types";
import { expectedFound, judgeFaithfulness, reportUnfaithful, scoreRuns, verdictOf } from "../evaluators";
import type { FailedClaim, Result } from "../evaluators/types";

export type InProcessKnobs = {
    /** EVAL_RUNS. Zero is the retrieval-only diagnostic: planner and retrieval ran, no answer. */
    runs: number;
    /** EVAL_ADVERSARIAL_RUNS, used instead of `runs` for an injection case. */
    adversarialRuns: number;
    /** EVAL_JUDGE. Costs one extra model call per answered case. */
    judge: boolean;
};

export type InProcessOutcome = {
    result: Result;
    /** Planner + embed + search + rerank, for this case's one retrieval. */
    retrievalMs: number;
};

export async function runCaseInProcess(c: EvalCase, knobs: InProcessKnobs): Promise<InProcessOutcome> {
    const t0 = performance.now();
    const { relevant, mode, intent, subQueries } = await plannedRetrieve(c.query, c.history ?? []);
    const retrievalMs = performance.now() - t0;

    // greeting / off-topic are answered by the pipeline without a model call — the route
    // writes the fixed text straight to the stream. The harness mirrors that exactly, so a
    // canned reply is scored the same way it is served.
    const canned =
        intent === "greeting" ? GREETING_MESSAGE : intent === "off-topic" ? REFUSAL_MESSAGE : null;

    // Plan-and-execute, execute half: generation answers the planner's RESOLVED sub-queries.
    // The settings and the swap live in lib/generation.ts, shared with the route and the
    // judge calibration — one code path (invariants #3 and #4), not three kept in sync.
    const history = (c.history ?? []).map((h) => ({ role: h.role, content: h.text }));
    const degraded = mode === "cosine-fallback";
    if (degraded) console.log(`  !! ${c.id}: reranker unavailable, cosine fallback`);

    const found = expectedFound(c, relevant);

    const chunks = relevant.length;
    const topScore = chunks ? relevant[0].score.toFixed(3) : "—";

    // EVAL_RUNS=0 → retrieval-only diagnostic. No answer generation (planner + retrieval ran).
    if (knobs.runs === 0) {
        // Inspect the data, don't trust the count: a guardrail is only useful if the
        // chunks reaching the model look plausible enough to tempt it.
        console.log(`  ${c.id}`);
        if (intent !== "search") console.log(`      planner → [${intent}]`);
        else if (subQueries.length > 1 || (subQueries[0] && subQueries[0] !== c.query))
            console.log(`      planner → ${JSON.stringify(subQueries)}`);
        for (const r of relevant) console.log(`      ${r.score.toFixed(3)}  ${r.title}`);
        if (!relevant.length) console.log(intent === "search" ? "      (nothing cleared the threshold)" : "      (retrieval skipped)");
        return {
            result: { id: c.id, intent, degraded, retrieved: found === null ? "—" : found ? "yes" : "NO", chunks, topScore, answered: 0, runs: 0, sample: "", faithful: "—", verdict: "—", detail: "" },
            retrievalMs,
        };
    }

    const runs = c.injection ? knobs.adversarialRuns : knobs.runs;

    // Multi-turn cases replay their history before the query. Retrieval above still
    // used c.query alone, matching production, which embeds only the latest message.
    // The final user turn is the planner's RESOLVED question, not the raw message.
    const texts: string[] = [];
    for (let i = 0; i < runs; i++) {
        texts.push(
            canned ??
                (
                    await generateText({
                        ...generationSettings(relevant),
                        messages: generationMessages(history, c.query, subQueries),
                    })
                ).text
        );
    }
    const scored = scoreRuns(c, texts);
    const { answered, firstAnswer } = scored;

    // Judge only answered cases: a refusal contains no claims to be unfaithful about.
    // One judgement per case, not per run — generation varies, but not usually in
    // whether it stayed grounded, and this keeps the cost linear in cases.
    let faithful = "—";
    let faithfulDetail: FailedClaim[] | undefined;
    if (knobs.judge && answered > 0 && !isRefusal(firstAnswer)) {
        const v = await judgeFaithfulness(c.query, relevant, firstAnswer);
        faithful = v.supported ? "yes" : "NO";
        if (!v.supported) faithfulDetail = reportUnfaithful(c.id, v);
    }

    const { verdict, detail } = verdictOf(c, runs, scored, found);

    if (verdict !== "PASS" && !canned && relevant.length) {
        console.log(`      [context] ${relevant.length} chunks:`);
        for (const r of relevant)
            console.log(`        ${r.score.toFixed(3)} ${r.title}: ${r.content.replace(/\s+/g, " ").slice(0, 130)}`);
    }

    return {
        result: {
            id: c.id,
            intent,
            degraded,
            retrieved: found === null ? "—" : found ? "yes" : "NO",
            chunks,
            topScore,
            answered,
            runs,
            sample: firstAnswer,
            ...(scored.oddRun ? { odd: scored.oddRun } : {}),
            faithful,
            ...(faithfulDetail ? { faithfulDetail } : {}),
            verdict,
            detail,
        },
        retrievalMs,
    };
}
