/**
 * Refusal verdicts golden file: what isRefusal says about a set of answers, frozen for the
 * Python port.
 *
 *   npm run exp:refusal-verdicts
 *
 * Step 2.6 of the Windward plan. The Python service writes the query log for the requests it
 * serves, and the log's `refused` column is what the suspicious_refusals view (db/002) mines for
 * eval cases. So agent/src/copilot_agent/refusal.py must call an answer a refusal exactly when
 * lib/refusal.ts does. This runs the REAL isRefusal on every case below and writes
 * agent/tests/golden/refusal-verdicts.json:
 *   - per case: the answer and the verdict;
 *   - meta: commit, dirty flag, and sha256 of lib/refusal.ts (the Python test refuses a golden
 *     older than it).
 * agent/tests/test_refusal.py runs is_refusal() on the same answers and compares.
 *
 * The cases are the ones tests/refusal.test.ts pins (each an observed shape or a past bug), plus
 * the places where the same regex text means something else in Python: whitespace outside the
 * shared set (U+FEFF is whitespace only in JavaScript, \x1c only in Python), and letters that
 * Unicode case folding maps onto ASCII (the long s) or that a Unicode \b treats as word
 * characters. Each of those cases fails when the matching guard in refusal.py is removed.
 *
 * No network. Every git call uses --no-optional-locks. Non-ASCII characters are built with
 * String.fromCodePoint so this file stays ASCII.
 */
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { REFUSAL_MESSAGE, isRefusal } from "../../lib/refusal";

const OUT = "agent/tests/golden/refusal-verdicts.json";

const cp = (...codes: number[]) => String.fromCodePoint(...codes);
const CURLY = cp(0x2019); // right single quotation mark
const BOM = cp(0xfeff);
const NBSP = cp(0xa0);
const LINE_SEPARATOR = cp(0x2028);
const FILE_SEPARATOR = cp(0x1c); // whitespace to Python's str.strip() and \s, not to JavaScript
const KELVIN = cp(0x212a); // folds to "k" in Unicode case-insensitive matching
const LONG_S = cp(0x17f); // folds to "s" likewise
const E_ACUTE = cp(0xe9); // a word character to a Unicode \b, not to an ASCII one
const DOTTED_I = cp(0x130); // lowercases to "i" plus a combining dot

const CORE = REFUSAL_MESSAGE.split(". ")[0];
const OPENER = "The documentation doesn't cover fine-tuning";

const CASES: { id: string; answer: string }[] = [
    // tests/refusal.test.ts
    { id: "canonical", answer: REFUSAL_MESSAGE },
    { id: "core-alone", answer: CORE },
    { id: "core-plus-offer", answer: CORE + " I can help with AI SDK docs." },
    { id: "opener-then-canonical", answer: OPENER + ". " + REFUSAL_MESSAGE },
    { id: "opener-with-topic", answer: "The documentation doesn't cover fine-tuning a model with the Vercel AI SDK." },
    { id: "opener-no-period", answer: "The documentation doesn't cover fine-tuning a model with the Vercel AI SDK" },
    {
        id: "opener-then-tail",
        answer: "The documentation doesn't cover fine-tuning a model with the Vercel AI SDK. I can help with AI SDK docs. Ask me about those and I'll help.",
    },
    { id: "does-not-mention", answer: "The documentation does not mention fine-tuning. Ask me about those and I'll help." },
    { id: "curly-apostrophe", answer: `The documentation doesn${CURLY}t cover fine-tuning.` },
    {
        id: "partial-answer-declines-at-end",
        answer: "To stream text, use streamText. I don't have information about the second part in the documentation.",
    },
    { id: "leak-after-opener", answer: OPENER + ". The capital of France is Paris." },
    { id: "clause-punctuation-in-topic", answer: "The documentation doesn't cover X, but here's how: use streamText." },
    {
        id: "leak-after-offer",
        answer: OPENER + ". I can help with AI SDK docs. Also, Paris is the capital of France.",
    },
    { id: "does-not-require", answer: "The documentation does not require you to call convertToModelMessages." },
    { id: "describes", answer: "The documentation describes streamText as a function that streams text." },
    { id: "answer-then-offer", answer: "streamText streams text from a model. I can help with AI SDK docs." },
    { id: "empty", answer: "" },
    // More shapes of the rule
    { id: "has-no-information", answer: "The documentation has no information about pricing." },
    { id: "has-no-details-then-core", answer: "The documentation has no details on that. " + CORE + "." },
    { id: "uppercase", answer: REFUSAL_MESSAGE.toUpperCase() },
    { id: "opener-exclamation", answer: OPENER + "! " + REFUSAL_MESSAGE },
    { id: "opener-question-mark-boundary", answer: OPENER + ". Ask me about those and I'll help? Sure." },
    { id: "core-inside-opener-reply", answer: "The documentation is limited here: " + CORE + "." },
    { id: "core-not-first", answer: "Sure! " + REFUSAL_MESSAGE },
    { id: "whitespace-only", answer: " \n\t " },
    // Where Python differs unless the port says otherwise
    { id: "trailing-newline-after-opener", answer: OPENER + "\n" },
    { id: "trailing-newline-then-text", answer: OPENER + "\nParis." },
    { id: "bom-around-canonical", answer: BOM + REFUSAL_MESSAGE + BOM },
    { id: "bom-before-opener", answer: BOM + OPENER + "." },
    { id: "nbsp-sentence-boundary", answer: OPENER + "." + NBSP + REFUSAL_MESSAGE },
    // A sentence boundary only JavaScript sees: U+FEFF is \s there, not in Python.
    { id: "bom-sentence-boundary", answer: OPENER + ". Ask me about those and I'll help." + BOM + "I can help with AI SDK docs." },
    { id: "line-separator-boundary", answer: OPENER + "." + LINE_SEPARATOR + "Ask me about those and I'll help." },
    { id: "file-separator-before-opener", answer: FILE_SEPARATOR + OPENER + "." },
    { id: "file-separator-boundary", answer: OPENER + "." + FILE_SEPARATOR + "Ask me about those and I'll help." },
    { id: "kelvin-sign-in-opener", answer: `The documentation doesn't cover ${KELVIN}ubernetes.` },
    { id: "long-s-in-verb", answer: `The documentation doe${LONG_S}n't cover fine-tuning.` },
    { id: "accented-letter-after-verb", answer: `The documentation doesn't cover${E_ACUTE} anything.` },
    { id: "dotted-capital-i-in-core", answer: `${DOTTED_I} don't have information about that in the documentation.` },
    { id: "tab-inside-core", answer: "I don't have\tinformation about that in the documentation." },
];

function sh(cmd: string): string {
    return execSync(cmd, { encoding: "utf8" }).trim();
}

function sha256(path: string): string {
    return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function main() {
    const ids = CASES.map((c) => c.id);
    const dupes = ids.filter((id, i) => ids.indexOf(id) !== i);
    if (dupes.length) throw new Error(`duplicate case ids: ${dupes.join(", ")}`);

    const cases = CASES.map((c) => ({ ...c, refused: isRefusal(c.answer) }));
    const golden = {
        meta: {
            generatedAt: new Date().toISOString(),
            commit: sh("git --no-optional-locks rev-parse HEAD"),
            dirty: sh("git --no-optional-locks status --porcelain") !== "",
            sourcesSha256: { "lib/refusal.ts": sha256("lib/refusal.ts") },
        },
        cases,
    };
    mkdirSync(dirname(OUT), { recursive: true });
    writeFileSync(OUT, JSON.stringify(golden, null, 2) + "\n");
    const refused = cases.filter((c) => c.refused).length;
    console.log(`wrote ${OUT}: ${cases.length} answers, ${refused} refusals`);
}

main();
