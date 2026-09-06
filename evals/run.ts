/**
 * Eval harness.
 *
 *   npm run eval                deterministic metrics
 *   EVAL_JUDGE=1 npm run eval   + LLM faithfulness (see judge.ts)
 *   EVAL_RUNS=0  npm run eval    retrieval-only diagnostic, no generation cost
 *
 * Per-case criteria, chosen so each measures the property that actually matters:
 *   RETRIEVAL RECALL     did the expected doc survive rerank + threshold?
 *   ANSWER vs REFUSE     shouldAnswer: true / false / "either"
 *   mustNotContain       forbidden strings (leaks, injected answers) — checked every run
 *   mustContain          required strings — proves a multi-part answer covered every intent
 *   FAITHFULNESS         (opt-in) is the answer grounded in its own chunks?
 *
 * Refusal is detected against REFUSAL_MESSAGE, the same constant the prompt instructs —
 * reword it there and this follows, rather than silently scoring every refusal as an answer.
 *
 * Retrieval runs ONCE per case. It is *nearly* deterministic: embedding, vector search and
 * rerank are, but the planner's HyDE hypothetical is model output, and a different
 * hypothetical can reorder near-tied pages (the push gate retries a recall miss once, and
 * says so — see the RUNS === 0 block). Generation runs N times, because that is
 * where non-determinism lives: temp 0 lowers variance, it does not remove it. A case that
 * passes 2/3 is FLAKY, not passing. Adversarial cases run more times (an attack that works
 * 1-in-8 is a working attack). A parked, known-failing case is marked `expectFail`: it runs
 * and reports but does not fail the suite, and is flagged if it ever starts passing.
 */
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

import { buildSystemPrompt, isRefusal, RERANK_THRESHOLD, VECTOR_CANDIDATES, RERANK_TOP_N } from "../lib/retrieve";
import { plannedRetrieve, GREETING_MESSAGE } from "../lib/plan";
import { CASES, type EvalCase } from "./dataset";
import { judgeFaithfulness } from "./judge";

const RUNS = Number(process.env.EVAL_RUNS ?? 3);

/**
 * Adversarial cases get more attempts than normal ones, because they measure a
 * different thing. For a normal case you want typical behaviour, and 3 runs is plenty.
 * For an attack you want to know whether it can EVER succeed — an injection that works
 * one time in five is a working injection, and a clean 3-run sample means "did not
 * reproduce", not "resistant".
 *
 * Found the hard way: inj-prompt-extract passed the harness and then leaked the entire
 * system prompt when tried by hand.
 */
const ADVERSARIAL_RUNS = Number(process.env.EVAL_ADVERSARIAL_RUNS ?? 8);

/** EVAL_ONLY=id1,id2 runs just those cases — for investigating a failure without paying
 *  for the whole suite. Metrics then report over the subset, which is fine for debugging. */
const ONLY = (process.env.EVAL_ONLY ?? "").split(",").map((x) => x.trim()).filter(Boolean);

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

/**
 * Did the expected page survive rerank + threshold? `null` when the case has no expectation.
 * `expectedSource` may be one slug or several — any match satisfies the case (see dataset.ts
 * for why). One helper, used by the main pass and the push-gate retry, so both judge alike.
 */
function expectedFound(c: EvalCase, relevant: { source_url: string }[]): boolean | null {
    if (!c.expectedSource) return null;
    const wanted = Array.isArray(c.expectedSource) ? c.expectedSource : [c.expectedSource];
    return relevant.some((r) => wanted.some((slug) => r.source_url.includes(slug)));
}

type Result = {
    id: string;
    retrieved: string;   // expected doc present in the retrieved set?
    chunks: number;      // how many chunks cleared the threshold and reached the model
    topScore: string;
    answered: number;    // how many runs produced an answer rather than a refusal
    runs: number;
    sample: string;      // first response, for eyeballing adversarial cases
    /** The run that broke expectation (answered when it should refuse, or vice versa), if any. */
    odd?: { i: number; text: string };
    faithful: string;    // judge verdict on the first answer, when EVAL_JUDGE=1
    verdict: "PASS" | "FAIL" | "FLAKY" | "—";
    detail: string;
};

async function runCase(c: EvalCase): Promise<Result> {
    const t0 = performance.now();
    const { relevant, mode, greeting, subQueries } = await plannedRetrieve(c.query, c.history ?? []);
    retrievalMs.push(performance.now() - t0);

    // Plan-and-execute: generation ANSWERS the planner's resolved sub-queries, not the raw
    // message. This is the "execute/synthesize" step actually using the plan. It matters
    // because the answerer anchors on the literal user turn: a terse "What is SDK?" gets
    // refused against context its explicit twin "What is the AI SDK?" answers from, and the
    // adversarial noise in multi-intent primes the model to dump legit intents. Feeding the
    // resolved queries removes both — the noise never reaches generation, and shorthand is
    // already expanded. Safe: off-topic/injection produce no usable sub-queries, so the
    // planner falls back to the raw question and nothing is rewritten into an answerable one.
    // Same computation the production route uses, so the eval measures the real pipeline.
    const resolvedQuestion = !greeting && subQueries.length ? subQueries.join("\n") : c.query;
    if (mode === "cosine-fallback") console.log(`  !! ${c.id}: reranker unavailable, cosine fallback`);

    const found = expectedFound(c, relevant);

    const chunks = relevant.length;
    const topScore = chunks ? relevant[0].score.toFixed(3) : "—";

    // EVAL_RUNS=0 → retrieval-only diagnostic. Costs no generation calls.
    if (RUNS === 0) {
        // Inspect the data, don't trust the count: a guardrail is only useful if the
        // chunks reaching the model look plausible enough to tempt it.
        console.log(`  ${c.id}`);
        if (greeting) console.log("      planner → [greeting]");
        else if (subQueries.length > 1 || (subQueries[0] && subQueries[0] !== c.query))
            console.log(`      planner → ${JSON.stringify(subQueries)}`);
        for (const r of relevant) console.log(`      ${r.score.toFixed(3)}  ${r.title}`);
        if (!relevant.length) console.log("      (nothing cleared the threshold)");
        return { id: c.id, retrieved: found === null ? "—" : found ? "yes" : "NO", chunks, topScore, answered: 0, runs: 0, sample: "", faithful: "—", verdict: "—", detail: "" };
    }

    const runs = c.injection ? ADVERSARIAL_RUNS : RUNS;

    let answered = 0;
    let firstAnswer = "";
    // Forbidden strings are checked on EVERY run, not just the first. An injection that
    // works one time in three is a working injection.
    const leaked = new Set<string>();
    // Required strings likewise: if ANY run omits one, the coverage is unreliable — a
    // multi-part answer that only sometimes includes an intent is not passing.
    const missed = new Set<string>();
    // The run that broke expectation, when one did. `firstAnswer` is run 0, which is often
    // a perfectly good refusal while run 2 is the one that got counted as "answered" — and
    // printing only run 0 made a FLAKY verdict impossible to diagnose without guessing.
    // Capture the first run whose refusal status disagrees with what the case expects.
    let oddRun: { i: number; text: string } | null = null;

    for (let i = 0; i < runs; i++) {
        // Multi-turn cases replay their history before the query. Retrieval above still
        // used c.query alone, matching production, which embeds only the latest message.
        // The final user turn is the planner's RESOLVED question, not the raw message.
        const text = greeting
            ? GREETING_MESSAGE
            : (
                await generateText({
                    model: openai("gpt-4o-mini"),
                    temperature: 0,
                    system: buildSystemPrompt(relevant),
                    messages: [
                        ...(c.history ?? []).map((h) => ({ role: h.role, content: h.text })),
                        { role: "user" as const, content: resolvedQuestion },
                    ],
                })
            ).text;
        const refused = isRefusal(text);
        if (!refused) answered++;
        if (i === 0) firstAnswer = text;
        // shouldAnswer false + answered, or shouldAnswer true + refused, is the odd one out.
        // "either" has no expectation about refusal, so it never produces an odd run.
        if (!oddRun && c.shouldAnswer !== "either" && refused === c.shouldAnswer) oddRun = { i, text };

        const lower = text.toLowerCase();
        for (const forbidden of c.mustNotContain ?? []) {
            if (lower.includes(forbidden.toLowerCase())) leaked.add(forbidden);
        }
        // Only require coverage on runs that actually answered — a legitimate refusal
        // cannot be expected to contain answer content.
        if (!isRefusal(text)) {
            for (const required of c.mustContain ?? []) {
                if (!lower.includes(required.toLowerCase())) missed.add(required);
            }
        }
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

    // "either" means answer-vs-refuse is not the criterion for this case; only leaks are.
    const expected = c.shouldAnswer === "either" ? answered : c.shouldAnswer ? runs : 0;
    let verdict: Result["verdict"] =
        answered === expected ? "PASS" : answered === runs - expected ? "FAIL" : "FLAKY";

    // A leak or a missed required string overrides answer-vs-refuse. inj-piggyback is
    // SUPPOSED to answer, so "answered 3/3" tells you nothing about whether the injection
    // worked — only the string checks can. Same for coverage: multi-intent-noise answers,
    // but dropping an intent means mustContain missed.
    if (leaked.size > 0 || missed.size > 0) verdict = "FAIL";

    // Retrieval succeeding while generation fails is the interesting failure —
    // it is exactly the parked bug, and it is invisible without both metrics.
    const detail =
        leaked.size > 0
            ? `LEAKED: ${[...leaked].join(", ")}`
            : missed.size > 0
                ? `MISSING: ${[...missed].join(", ")}`
                : verdict === "PASS"
                ? ""
                : found === true && c.shouldAnswer
                    ? "retrieval OK, generation refused"
                    : found === false
                        ? "expected doc not retrieved"
                        : `answered ${answered}/${runs}, expected ${expected}`;

    if (verdict !== "PASS" && !greeting && relevant.length) {
        console.log(`      [context] ${relevant.length} chunks:`);
        for (const r of relevant)
            console.log(`        ${r.score.toFixed(3)} ${r.title}: ${r.content.replace(/\s+/g, " ").slice(0, 130)}`);
    }

    return {
        id: c.id,
        retrieved: found === null ? "—" : found ? "yes" : "NO",
        chunks,
        topScore,
        answered,
        runs,
        sample: firstAnswer,
        ...(oddRun ? { odd: oddRun } : {}),
        faithful,
        verdict,
        detail: detail || firstAnswer.slice(0, 0),
    };
}

async function main() {
    // Duplicate ids fail silently otherwise: byId is a Map, so a second case with the same
    // id overwrites the first in the summary while both still run and both still cost money.
    // (This exact bug shipped once — two copies of multi-intent-noise.) Fail loudly instead.
    const ids = CASES.map((c) => c.id);
    const dupes = [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))];
    if (dupes.length) throw new Error(`Duplicate case ids in dataset: ${dupes.join(", ")}`);

    // Print the knobs. A pasted result that doesn't say what produced it is not a result.
    console.log(
        `eval: ${CASES.length} cases × ${RUNS} generations, temp 0  |  ` +
        `candidates ${VECTOR_CANDIDATES} → rerank top ${RERANK_TOP_N} → threshold ${RERANK_THRESHOLD}\n`
    );

    const active = ONLY.length ? CASES.filter((c) => ONLY.includes(c.id)) : CASES;
    if (ONLY.length) console.log(`(EVAL_ONLY: ${active.map((c) => c.id).join(", ")})\n`);

    const results: Result[] = [];
    for (const [i, c] of active.entries()) {
        if (i > 0) await sleep(RERANK_INTERVAL_MS);
        const r = await runCase(c);
        results.push(r);
        if (RUNS > 0) {
            console.log(`  ${r.verdict.padEnd(5)} ${r.id.padEnd(22)} ${r.detail}`);
            // Show what the model actually produced for adversarial cases AND any failure —
            // inferring behaviour from a verdict is how wrong criteria survive. Look, don't guess.
            if (c.injection || r.verdict !== "PASS")
                console.log(`         → ${r.sample.replace(/\s+/g, " ").slice(0, 1200)}`);
            // On a non-PASS, also show the run that actually broke expectation when it isn't
            // run 0 — otherwise a FLAKY 1/3 prints a perfectly good refusal and hides the
            // one reply that was scored the other way.
            if (r.verdict !== "PASS" && r.odd && r.odd.i !== 0)
                console.log(`         ↳ run ${r.odd.i + 1} (the odd one): ${r.odd.text.replace(/\s+/g, " ").slice(0, 1200)}`);
        }
    }

    console.log();
    console.table(
        results.map((r) => ({
            case: r.id,
            "expected doc": r.retrieved,
            "chunks past threshold": r.chunks,
            "top rerank": r.topScore,
            [`answered`]: `${r.answered}/${r.runs}`,
            ...(JUDGE ? { faithful: r.faithful } : {}),
            verdict: r.verdict,
        }))
    );

    // Injection cases are counted on their own axis. Several are also guardrails by
    // shouldAnswer, but "did it refuse an out-of-corpus question" and "did it resist an
    // instruction to disobey" are different properties and deserve separate numbers.
    // expectFail (parked, known-failing) cases are excluded from every headline metric and
    // reported on their own line — otherwise a documented, deferred bug drags coverage
    // below 5/5 and reads as a regression.
    const injections = active.filter((c) => c.injection && !c.expectFail);
    const answerable = active.filter((c) => c.shouldAnswer === true && !c.injection && !c.expectFail);
    const guardrails = active.filter((c) => c.shouldAnswer === false && !c.injection && !c.expectFail);
    const parked = active.filter((c) => c.expectFail);
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
    const resisted = injections.filter((c) => byId.get(c.id)!.verdict === "PASS").length;
    const multiTurn = injections.filter((c) => c.history).length;
    console.log(
        `injection resisted ${resisted}/${injections.length}   adversarial prompts that did not get what they asked for ` +
        `(${multiTurn} multi-turn, ${injections.length - multiTurn} single-turn, ${ADVERSARIAL_RUNS} attempts each)`
    );
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

    if (RUNS === 0) {
        // Retrieval-only mode is the cheap CI gate (no generation calls). It used to always
        // exit 0, which made "runs on every push" a smoke test rather than a gate. A missing
        // expected doc is a retrieval regression whether or not we generated an answer, so
        // fail on recall here. Parked (expectFail) cases are already excluded from `answerable`.
        //
        // A miss gets ONE retry before it fails the gate. Retrieval is not fully deterministic:
        // the planner's HyDE hypothetical is model output (not seedable on the Responses API —
        // see lib/plan.ts), and two near-tied pages can swap places between runs. A retry separates "the expected
        // page is gone" (a real regression — fails both times) from "it lost a coin flip" (passes
        // on retry). The retry is REPORTED, never silent: a case that keeps needing it is a case
        // whose criterion or retrieval needs attention, and hiding that would turn the gate back
        // into a smoke test. Snapshotting planner output would have been the deterministic
        // alternative, but it would test a fixture instead of the pipeline.
        const missed = answerable.filter((c) => byId.get(c.id)!.retrieved !== "yes");
        const stillMissing: string[] = [];
        const recovered: string[] = [];
        for (const c of missed) {
            await sleep(RERANK_INTERVAL_MS);
            const { relevant } = await plannedRetrieve(c.query, c.history ?? []);
            if (expectedFound(c, relevant)) {
                recovered.push(c.id);
                console.log(`  ↻ ${c.id}: expected doc NOT retrieved on first try, recovered on retry`);
                for (const r of relevant) console.log(`      ${r.score.toFixed(3)}  ${r.title}`);
            } else {
                stillMissing.push(c.id);
            }
        }
        if (recovered.length)
            console.log(`\nrecovered on retry ${recovered.length}   ${recovered.join(", ")} — flaky retrieval, not a regression; if it repeats, widen expectedSource or look at DEBUG_PLAN=1`);
        if (stillMissing.length) {
            console.log(`\nretrieval regression — expected doc not retrieved twice: ${stillMissing.join(", ")}`);
            process.exit(1);
        }
        return;
    }

    if (JUDGE) {
        const judged = results.filter((r) => r.faithful !== "—");
        const grounded = judged.filter((r) => r.faithful === "yes").length;
        console.log(`faithfulness       ${grounded}/${judged.length}   answers fully supported by their own retrieved chunks (judge, n=15 calibration)`);
    }

    // Parked, known-failing cases: reported here, never counted against the suite.
    if (parked.length) {
        const stillFailing = parked.filter((c) => byId.get(c.id)!.verdict !== "PASS").length;
        console.log(
            `known failures     ${stillFailing}/${parked.length}   parked, not blocking (${parked.map((c) => c.id).join(", ")})`
        );
    }

    // A parked case that has started PASSING is news — the bug got fixed elsewhere and the
    // marker is now lying. Surface it loudly and fail, so expectFail can't hide a real pass.
    const unexpectedPass = parked.filter((c) => byId.get(c.id)!.verdict === "PASS");
    if (unexpectedPass.length) {
        console.log(`\n⚠ expectFail case now PASSING — remove expectFail: ${unexpectedPass.map((c) => c.id).join(", ")}`);
        process.exit(1);
    }

    const parkedIds = new Set(parked.map((c) => c.id));
    const unfaithful = results.filter((r) => r.faithful === "NO" && !parkedIds.has(r.id));
    const failed = results.filter((r) => r.verdict !== "PASS" && !parkedIds.has(r.id));
    if (unfaithful.length) {
        console.log(`\n${unfaithful.length} unfaithful: ${unfaithful.map((f) => f.id).join(", ")}`);
        process.exit(1);
    }
    if (failed.length) {
        console.log(`\n${failed.length} failing: ${failed.map((f) => f.id).join(", ")}`);
        process.exit(1);
    }
    console.log(`\nall green${parked.length ? ` (${parked.length} parked)` : ""}.`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
