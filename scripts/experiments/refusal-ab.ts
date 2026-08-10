/**
 * Is the false refusal caused by QUERY WORDING or by CONTEXT QUALITY?
 *
 *   npm run exp:refusal
 *
 * Day 7 concluded "retrieval succeeded, generation failed" because both queries
 * retrieved the Migration page. True — but not equal: "changed" retrieves 5 Migration
 * chunks (0.818–0.890), "new" retrieves 3 plus two unrelated pages (top 0.768).
 * So the refusal may be about thin context, not vocabulary.
 *
 * 2x2: cross each query with each query's retrieved context. The diagonal reproduces
 * the observed behaviour; the off-diagonal isolates the cause.
 */
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

import { retrieve, buildSystemPrompt, isRefusal } from "../../lib/retrieve";

const RUNS = 3;
const Q_NEW = "what is new in AI SDK 7";
const Q_CHANGED = "what was changed in AI SDK 7";

async function main() {
    const ctxNew = (await retrieve(Q_NEW)).relevant;
    const ctxChanged = (await retrieve(Q_CHANGED)).relevant;

    console.log(`context("new")     ${ctxNew.length} chunks, top ${ctxNew[0]?.score.toFixed(3)}`);
    console.log(`context("changed") ${ctxChanged.length} chunks, top ${ctxChanged[0]?.score.toFixed(3)}\n`);

    const cells = [
        { label: 'query "new"     + context "new"     (observed: refuses)', query: Q_NEW, ctx: ctxNew },
        { label: 'query "new"     + context "changed" (isolates WORDING)  ', query: Q_NEW, ctx: ctxChanged },
        { label: 'query "changed" + context "new"     (isolates CONTEXT)  ', query: Q_CHANGED, ctx: ctxNew },
        { label: 'query "changed" + context "changed" (observed: answers) ', query: Q_CHANGED, ctx: ctxChanged },
    ];

    for (const cell of cells) {
        let answered = 0;
        for (let i = 0; i < RUNS; i++) {
            const { text } = await generateText({
                model: openai("gpt-4o-mini"),
                temperature: 0,
                system: buildSystemPrompt(cell.ctx),
                prompt: cell.query,
            });
            if (!isRefusal(text)) answered++;
        }
        console.log(`${cell.label}  answered ${answered}/${RUNS}`);
    }

    console.log(`
Read it this way:
  row 2 answers  -> wording is NOT the cause; the "new" context is too thin/noisy
  row 2 refuses  -> wording IS the cause; the prompt is matching words, not meaning
  row 3 refuses  -> confirms context quality drives it, independent of wording`);
}

main().catch((err) => { console.error(err); process.exit(1); });
