/**
 * Eval harness — deterministic layer.
 *
 *   npm run eval
 *
 * Measures two things, neither of which needs an LLM judge:
 *
 *   RETRIEVAL RECALL  did the expected doc survive rerank + threshold?
 *   REFUSAL CORRECTNESS  did it answer when it should, refuse when it shouldn't?
 *
 * Refusal is detected by comparing against REFUSAL_MESSAGE — the same constant the
 * system prompt instructs. If that sentence is ever reworded, this follows it instead
 * of silently scoring every refusal as an answer.
 *
 * Retrieval runs ONCE per case (it is deterministic — same embedding, same rerank).
 * Generation runs N times, because that is where non-determinism actually lives:
 * temperature 0 lowers variance, it does not eliminate it. A case that passes 2/3 is
 * not passing, it is flaky, and a harness that hides that is worse than no harness.
 */
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

import { retrieve, buildSystemPrompt, isRefusal, RERANK_THRESHOLD, VECTOR_CANDIDATES, RERANK_TOP_N } from "../lib/retrieve";
import { CASES, type EvalCase } from "./dataset";
import { judgeFaithfulness } from "./judge";

const RUNS = Number(process.env.EVAL_RUNS ?? 3);

/**
 * Faithfulness is opt-in: EVAL_JUDGE=1 npm run eval
 * It costs one extra model call per answered case, and — unlike every other metric
 * here — the number comes from a model rather than a comparison. Calibrated at
 * 14–15/15 with 0 false alarms (see eval:calibrate), which is good enough to act on,
 * not good enough to publish without saying n=15.
 */
const JUDGE = process.env.EVAL_JUDGE === "1";

/** Widening the candidate pool costs latency on every production query. Price it. */
const retrievalMs: number[] = [];

/**
 * The Cohere trial key allows 10 rerank calls/minute. The harness makes one per case,
 * so an unthrottled run trips the limit and reports a fake failure. Space them out —
 * a harness that fails for its own reasons teaches you nothing about the system.
 */
const RERANK_INTERVAL_MS = Number(process.env.RERANK_INTERVAL_MS ?? 6500);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Result = {
    id: string;
    retrieved: string;   // expected doc present in the retrieved set?
    chunks: number;      // how many chunks cleared the threshold and reached the model
    topScore: string;
    answered: number;    // how many of RUNS produced an answer rather than a refusal
    faithful: string;    // judge verdict on the first answer, when EVAL_JUDGE=1
    verdict: "PASS" | "FAIL" | "FLAKY" | "—";
    detail: string;
};

async function runCase(c: EvalCase): Promise<Result> {
    const t0 = performance.now();
    const { relevant, mode } = await retrieve(c.query);
    retrievalMs.push(performance.now() - t0);
    if (mode === "cosine-fallback") console.log(`  !! ${c.id}: reranker unavailable, cosine fallback`);

    const found = c.expectedSource
        ? relevant.some((r) => r.source_url.includes(c.expectedSource!))
        : null;

    const chunks = relevant.length;
    const topScore = chunks ? relevant[0].score.toFixed(3) : "—";

    // EVAL_RUNS=0 → retrieval-only diagnostic. Costs no generation calls.
    if (RUNS === 0) {
        // Inspect the data, don't trust the count: a guardrail is only useful if the
        // chunks reaching the model look plausible enough to tempt it.
        console.log(`  ${c.id}`);
        for (const r of relevant) console.log(`      ${r.score.toFixed(3)}  ${r.title}`);
        if (!relevant.length) console.log("      (nothing cleared the threshold)");
        return { id: c.id, retrieved: found === null ? "—" : found ? "yes" : "NO", chunks, topScore, answered: 0, faithful: "—", verdict: "—", detail: "" };
    }

    let answered = 0;
    let firstAnswer = "";
    for (let i = 0; i < RUNS; i++) {
        const { text } = await generateText({
            model: openai("gpt-4o-mini"),
            temperature: 0,
            system: buildSystemPrompt(relevant),
            prompt: c.query,
        });
        if (!isRefusal(text)) answered++;
        if (i === 0) firstAnswer = text;
    }

    // Judge only answered cases: a refusal contains no claims to be unfaithful about.
    // One judgement per case, not per run — generation varies, but not usually in
    // whether it stayed grounded, and this keeps the cost linear in cases.
    let faithful = "—";
    if (JUDGE && answered > 0 && !isRefusal(firstAnswer)) {
        const v = await judgeFaithfulness(c.query, relevant, firstAnswer);
        faithful = v.supported ? "yes" : "NO";
        if (!v.supported) {
            console.log(`  UNFAITHFUL ${c.id}: ${v.reasoning}`);
            for (const claim of v.unsupportedClaims) console.log(`      unsupported: ${claim}`);
        }
    }

    const expected = c.shouldAnswer ? RUNS : 0;
    const verdict = answered === expected ? "PASS" : answered === RUNS - expected ? "FAIL" : "FLAKY";

    // Retrieval succeeding while generation fails is the interesting failure —
    // it is exactly the parked bug, and it is invisible without both metrics.
    const detail =
        verdict === "PASS"
            ? ""
            : found === true && c.shouldAnswer
                ? "retrieval OK, generation refused"
                : found === false
                    ? "expected doc not retrieved"
                    : `answered ${answered}/${RUNS}, expected ${expected}`;

    return {
        id: c.id,
        retrieved: found === null ? "—" : found ? "yes" : "NO",
        chunks,
        topScore,
        answered,
        faithful,
        verdict,
        detail: detail || firstAnswer.slice(0, 0),
    };
}

async function main() {
    // Print the knobs. A pasted result that doesn't say what produced it is not a result.
    console.log(
        `eval: ${CASES.length} cases × ${RUNS} generations, temp 0  |  ` +
        `candidates ${VECTOR_CANDIDATES} → rerank top ${RERANK_TOP_N} → threshold ${RERANK_THRESHOLD}\n`
    );

    const results: Result[] = [];
    for (const [i, c] of CASES.entries()) {
        if (i > 0) await sleep(RERANK_INTERVAL_MS);
        const r = await runCase(c);
        results.push(r);
        if (RUNS > 0) console.log(`  ${r.verdict.padEnd(5)} ${r.id.padEnd(14)} ${r.detail}`);
    }

    console.log();
    console.table(
        results.map((r) => ({
            case: r.id,
            "expected doc": r.retrieved,
            "chunks past threshold": r.chunks,
            "top rerank": r.topScore,
            [`answered / ${RUNS}`]: r.answered,
            ...(JUDGE ? { faithful: r.faithful } : {}),
            verdict: r.verdict,
        }))
    );

    const answerable = CASES.filter((c) => c.shouldAnswer);
    const guardrails = CASES.filter((c) => !c.shouldAnswer);
    const byId = new Map(results.map((r) => [r.id, r]));

    const coverage = answerable.filter((c) => byId.get(c.id)!.verdict === "PASS").length;
    const held = guardrails.filter((c) => byId.get(c.id)!.verdict === "PASS").length;
    const recall = answerable.filter((c) => byId.get(c.id)!.retrieved === "yes").length;

    const sorted = [...retrievalMs].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    console.log(`retrieval latency  ${median.toFixed(0)}ms median, ${Math.max(...sorted).toFixed(0)}ms worst  (embed + search + rerank)`);
    console.log(`retrieval recall   ${recall}/${answerable.length}   expected doc survived rerank + threshold`);
    if (RUNS > 0) {
        console.log(`answer coverage    ${coverage}/${answerable.length}   answerable questions actually answered`);
        console.log(`guardrails held    ${held}/${guardrails.length}   out-of-corpus questions refused`);
    }


    // The tradeoff this harness exists to protect: loosening the prompt to raise
    // coverage must not lower guardrails. Either number moving alone is a regression.
    // A guardrail that never reaches the model is enforced by the threshold, not the
    // prompt — so it cannot detect the prompt being loosened. Say so out loud.
    for (const g of guardrails) {
        const r = byId.get(g.id)!;
        console.log(
            `  guardrail ${g.id}: ${r.chunks} chunk(s) reached the model` +
            (r.chunks === 0 ? "  → held by THRESHOLD; blind to prompt changes" : `  → held by PROMPT (top ${r.topScore})`)
        );
    }

    if (RUNS === 0) return;

    if (JUDGE) {
        const judged = results.filter((r) => r.faithful !== "—");
        const grounded = judged.filter((r) => r.faithful === "yes").length;
        console.log(`faithfulness       ${grounded}/${judged.length}   answers fully supported by their own retrieved chunks (judge, n=15 calibration)`);
    }

    const unfaithful = results.filter((r) => r.faithful === "NO");
    const failed = results.filter((r) => r.verdict !== "PASS");
    if (unfaithful.length) {
        console.log(`\n${unfaithful.length} unfaithful: ${unfaithful.map((f) => f.id).join(", ")}`);
        process.exit(1);
    }
    if (failed.length) {
        console.log(`\n${failed.length} failing: ${failed.map((f) => f.id).join(", ")}`);
        process.exit(1);
    }
    console.log("\nall green.");
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
