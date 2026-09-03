/**
 * Isolate retrieval's sensitivity to phrasing for the "what is the AI SDK" query.
 *   npm run exp:phrasing
 *
 * what-is-sdk fails because the retrieved chunks don't define the SDK. This asks: is that
 * the PLANNER's expansion ("Vercel AI SDK") hurting, or does NO phrasing surface the
 * definitional Core: Overview chunk — i.e. a corpus limitation? Calls retrieve() directly,
 * bypassing the planner.
 */
import { retrieve } from "../../lib/retrieve";

const PHRASINGS = [
    "What is SDK?",
    "What is the AI SDK?",
    "What is the Vercel AI SDK?",
    "AI SDK overview introduction",
    "What does the AI SDK do",
    "AI SDK Core functions text generation",
];

async function main() {
    for (const q of PHRASINGS) {
        const { relevant } = await retrieve(q);
        const hasDef = relevant.some((r) => r.title.includes("Core: Overview"));
        console.log(`\n"${q}"   ${hasDef ? "✓ Core: Overview retrieved" : "✗ no Core: Overview"}`);
        for (const r of relevant) console.log(`   ${r.score.toFixed(3)}  ${r.title}`);
    }
}
main().catch((e) => { console.error(e); process.exit(1); });
