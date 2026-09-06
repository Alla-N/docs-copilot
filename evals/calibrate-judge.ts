/**
 * Calibrate the faithfulness judge — BEFORE using any number it produces.
 *
 *   npm run eval:calibrate
 *
 * A judge is a model. Asking it "is this answer faithful?" and writing down the reply
 * is measuring with an uncalibrated instrument. So: build answers whose faithfulness we
 * know by construction, and check whether the judge agrees.
 *
 *   CLEAN     the real answer, judged against its own chunks        -> expect supported
 *   FABRICATED  real answer + one invented specific fact             -> expect NOT supported
 *   SWAPPED     real answer judged against ANOTHER case's chunks     -> expect NOT supported
 *
 * FABRICATED and SWAPPED fail for different reasons, on purpose. Fabrication tests whether
 * the judge notices an addition; swapping tests whether it notices that nothing lines up.
 * A judge that catches one and not the other is only half working.
 */
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

import { retrieve, buildSystemPrompt, type RetrievedChunk } from "../lib/retrieve";
import { CASES } from "./dataset";
import { judgeFaithfulness } from "./judge";

const RERANK_INTERVAL_MS = Number(process.env.RERANK_INTERVAL_MS ?? 6500);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Pages that describe the whole SDK rather than one topic. Excluded as swap partners:
 * they support fragments of almost any answer, so the "unfaithful" label becomes arguable.
 */
const CROSS_CUTTING = ["migration-guide-7-0"];

/** A specific, checkable, invented fact. Specificity matters — vague filler is genuinely ambiguous. */
const FABRICATION =
    " AI SDK 7 was released on 12 March 2026 and requires Node.js 24 or newer.";

type Sample = {
    caseId: string;
    kind: "clean" | "fabricated" | "swapped";
    question: string;
    chunks: RetrievedChunk[];
    answer: string;
    /** What a correct judge must say. */
    expectSupported: boolean;
};

async function main() {
    const answerable = CASES.filter((c) => c.shouldAnswer === true && !c.injection);

    // ── 1. Produce one real answer per answerable case ───────────
    const base: { id: string; question: string; source?: string; chunks: RetrievedChunk[]; answer: string }[] = [];
    for (const [i, c] of answerable.entries()) {
        if (i > 0) await sleep(RERANK_INTERVAL_MS);
        const { relevant } = await retrieve(c.query);
        const { text } = await generateText({
            model: openai("gpt-4o-mini"),
            temperature: 0,
            system: buildSystemPrompt(relevant),
            prompt: c.query,
        });
        // `source` only picks a topically DISTANT partner below. For an any-of case the first
        // slug is the primary page, which is all the distance check needs.
        const source = Array.isArray(c.expectedSource) ? c.expectedSource[0] : c.expectedSource;
        base.push({ id: c.id, question: c.query, source, chunks: relevant, answer: text });
        console.log(`  generated ${c.id}`);
    }

    // ── 2. Build the labelled samples ────────────────────────────
    const samples: Sample[] = [];
    for (const [i, b] of base.entries()) {
        samples.push({ caseId: b.id, kind: "clean", question: b.question, chunks: b.chunks, answer: b.answer, expectSupported: true });
        samples.push({ caseId: b.id, kind: "fabricated", question: b.question, chunks: b.chunks, answer: b.answer + FABRICATION, expectSupported: false });
        // Judge this answer against another case's chunks. The partner must be topically
        // DISTANT, which took two corrections to get right:
        //   1. new-7 and changed-7 retrieve the same page — swapping them is not unfaithful
        //   2. the Migration guide is a survey doc covering the whole SDK surface, so it
        //      partially supports almost any answer. Judged against it, "unfaithful" is
        //      genuinely arguable — and an arguable label teaches you nothing about the
        //      judge, it just adds noise you'll misread as a judge failure.
        // A calibration set may only contain samples whose correct label is beyond dispute.
        const other = base.find((o) => o.source !== b.source && !CROSS_CUTTING.includes(o.source ?? ""));
        if (other) {
            samples.push({ caseId: `${b.id}←${other.id}`, kind: "swapped", question: b.question, chunks: other.chunks, answer: b.answer, expectSupported: false });
        }
    }

    // ── 3. Judge every sample ────────────────────────────────────
    const rows: { case: string; kind: string; expected: string; judged: string; ok: string }[] = [];
    for (const s of samples) {
        const v = await judgeFaithfulness(s.question, s.chunks, s.answer);
        const correct = v.supported === s.expectSupported;
        rows.push({
            case: s.caseId,
            kind: s.kind,
            expected: s.expectSupported ? "supported" : "NOT supported",
            judged: v.supported ? "supported" : "NOT supported",
            ok: correct ? "✓" : "✗",
        });
        if (!correct) {
            // A miss is a CLAIM, not a verdict. Two things can be wrong: the judge, or my
            // label. Swapped samples assume the other case's chunks don't support this
            // answer — for docs about the same SDK that assumption often fails. So print
            // the evidence needed to adjudicate it instead of trusting the count.
            console.log(`\n  MISS  ${s.kind}  ${s.caseId}`);
            console.log(`  judge said: ${v.reasoning}`);
            console.log(`  --- claims and the quotes it offered as evidence ---`);
            for (const c of v.checked) {
                console.log(`  ${c.found ? "OK  " : "FAIL"}  claim: ${c.claim.slice(0, 90)}`);
                console.log(`        quote: ${JSON.stringify(c.quote.slice(0, 140))}`);
            }
            console.log(`  --- answer being judged ---`);
            console.log(`  ${s.answer.replace(/\n/g, "\n  ").slice(0, 500)}`);
            console.log(`  --- sources it was judged against ---`);
            for (const c of s.chunks) {
                console.log(`  [${c.title}] ${c.content.replace(/\n/g, " ").slice(0, 220)}...`);
            }
            console.log();
        }
    }

    console.table(rows);

    // ── 4. The two error types are NOT interchangeable ───────────
    const clean = rows.filter((r) => r.kind === "clean");
    const dirty = rows.filter((r) => r.kind !== "clean");
    const falseAlarms = clean.filter((r) => r.ok === "✗").length;
    const missed = dirty.filter((r) => r.ok === "✗").length;

    console.log(`
false alarms   ${falseAlarms}/${clean.length}   good answers wrongly flagged as unfaithful
missed lies    ${missed}/${dirty.length}   planted unfaithfulness the judge approved
agreement      ${rows.filter((r) => r.ok === "✓").length}/${rows.length}

A judge that misses planted lies is worse than no judge: it reports a number that
looks like safety. A judge with false alarms is merely annoying — you investigate
and find nothing. Weigh 'missed lies' far more heavily.

Before believing a miss, read the evidence printed above it. A 'swapped' miss is
often a bad LABEL rather than a bad judge: two pages about the same SDK overlap,
so the other case's chunks may genuinely support this answer.`);
}

main().catch((err) => { console.error(err); process.exit(1); });
