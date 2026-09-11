/**
 * History caps golden file: what parseChatRequest keeps of a conversation, frozen for the
 * Python port.
 *
 *   npm run exp:history-caps
 *
 * Step 2.5 of the Windward plan. The Python service now reads history from the thread its
 * checkpointer stores, not from the client, and applies the same caps as lib/chat-request.ts
 * (agent/src/copilot_agent/history.py). For each case below this builds the useChat body the
 * client would send for that conversation (assistant turns signed, as the route signs them),
 * runs the REAL parseChatRequest on it, and writes agent/tests/golden/history-caps.json:
 *   - per case: the stored turns and the question (the inputs), and the history and question
 *     parseChatRequest returned;
 *   - meta: commit, dirty flag, and sha256 of lib/chat-request.ts and lib/assistant-signature.ts
 *     (the Python test refuses a golden older than either).
 * agent/tests/test_history.py runs capped_history() on the same inputs and compares.
 *
 * No network. Every git call uses --no-optional-locks. Non-ASCII characters are built with
 * String.fromCodePoint so this file stays ASCII.
 */
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { signAssistantText } from "../../lib/assistant-signature";
import { MAX_CHARS_PER_MESSAGE, MAX_MESSAGES, MAX_TOTAL_CHARS, parseChatRequest } from "../../lib/chat-request";

const OUT = "agent/tests/golden/history-caps.json";

type Turn = { role: "user" | "assistant"; text: string };
type HistoryCase = { id: string; turns: Turn[]; question: string };

const u = (text: string): Turn => ({ role: "user", text });
const a = (text: string): Turn => ({ role: "assistant", text });
const cp = (...codes: number[]) => String.fromCodePoint(...codes);
const NBSP = cp(0xa0);
const BOM = cp(0xfeff);
const LINE_SEPARATOR = cp(0x2028);
const ROCKET = cp(0x1f680);

/** n alternating turns, user first, each `size` characters long. */
const alternating = (n: number, size = 5): Turn[] =>
    Array.from({ length: n }, (_, i) => {
        const text = `${i}`.padEnd(size, "x");
        return i % 2 === 0 ? u(text) : a(text);
    });

const CASES: HistoryCase[] = [
    { id: "empty-thread", turns: [], question: "how do I stream text?" },
    {
        id: "one-turn",
        turns: [u("how do I stream text?"), a("Use streamText (Source 1).")],
        question: "and how do I configure it?",
    },
    {
        id: "canned-greeting-turn",
        turns: [u("hi"), a("Hi! I answer questions about the Vercel AI SDK documentation.")],
        question: "how do I stream text?",
    },
    // 19 stored turns plus the question is exactly MAX_MESSAGES: all kept.
    { id: "exactly-max-messages", turns: alternating(MAX_MESSAGES - 1), question: "q" },
    // More than that: only the last 19 stored turns are looked at.
    { id: "more-than-max-messages", turns: alternating(MAX_MESSAGES + 5), question: "q" },
    // An answer over the per-message cap: cut at 4000, THEN trimmed, so the spaces at the cut go.
    {
        id: "long-answer-cut-then-trimmed",
        turns: [u("explain everything"), a("a".repeat(3990) + " ".repeat(20) + "b".repeat(490))],
        question: "shorter please",
    },
    // Blank and whitespace-only turns are skipped; JavaScript trim removes NBSP, BOM and U+2028.
    {
        id: "whitespace",
        turns: [
            u("  padded question \n"),
            a(`${BOM}${NBSP}answer${LINE_SEPARATOR}`),
            u(" \n\t"),
            a(""),
        ],
        question: "next",
    },
    // Total cap: the question and five full turns are exactly 24000, the sixth does not fit.
    {
        id: "total-cap-exact",
        turns: alternating(6, MAX_CHARS_PER_MESSAGE),
        question: "q".repeat(MAX_TOTAL_CHARS - 5 * MAX_CHARS_PER_MESSAGE),
    },
    // The walk STOPS at the first turn that does not fit: the tiny oldest turn would fit, but is
    // dropped with everything older than the turn that broke the cap.
    {
        id: "total-cap-stops-at-first-misfit",
        turns: [u("tiny"), ...alternating(6, MAX_CHARS_PER_MESSAGE)],
        question: "q".repeat(100),
    },
    // Well inside every cap, so UTF-16 units and code points agree on what is kept.
    {
        id: "astral-inside-the-caps",
        turns: [u(`launch ${ROCKET}`), a(`Use streamText ${ROCKET} (Source 1).`)],
        question: `and ${ROCKET}?`,
    },
];

function bodyFor(c: HistoryCase) {
    const messages = [...c.turns, u(c.question)].map((t) => ({
        role: t.role,
        parts:
            t.role === "assistant"
                ? [{ type: "text", text: t.text }, { type: "data-signature", data: { sig: signAssistantText(t.text) } }]
                : [{ type: "text", text: t.text }],
    }));
    return { messages };
}

function sh(cmd: string): string {
    return execSync(cmd, { encoding: "utf8" }).trim();
}

function sha256(path: string): string {
    return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function main() {
    const cases = CASES.map((c) => {
        const parsed = parseChatRequest(bodyFor(c));
        const history = parsed.messages.slice(0, -1).map((m) => ({ role: m.role, text: m.content as string }));
        return { ...c, history, parsedQuestion: parsed.question };
    });
    const golden = {
        meta: {
            generatedAt: new Date().toISOString(),
            commit: sh("git --no-optional-locks rev-parse HEAD"),
            dirty: sh("git --no-optional-locks status --porcelain") !== "",
            sourcesSha256: Object.fromEntries(
                ["lib/chat-request.ts", "lib/assistant-signature.ts"].map((p) => [p, sha256(p)])
            ),
            caps: { MAX_MESSAGES, MAX_CHARS_PER_MESSAGE, MAX_TOTAL_CHARS },
        },
        cases,
    };
    mkdirSync(dirname(OUT), { recursive: true });
    writeFileSync(OUT, JSON.stringify(golden, null, 2) + "\n");
    console.log(`wrote ${OUT}: ${cases.length} cases`);
}

main();
