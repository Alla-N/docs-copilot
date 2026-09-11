/**
 * Generation request golden file: the exact request the TypeScript answer step sends, frozen
 * for the Python port.
 *
 *   npm run exp:generation-requests
 *
 * Step 2.2 of the Windward plan. For each case below, this runs the SAME streamText call as the
 * chat route (generationSettings + generationMessages from lib/generation.ts) with
 * globalThis.fetch replaced by a recorder that answers with a canned Responses API event
 * stream, and writes agent/tests/golden/generation-requests.json:
 *   - per case: the retrieved chunks, history, raw question and sub-queries (the inputs), the
 *     request body the AI SDK sent, and the text and usage it read back from the canned stream;
 *   - meta: commit, dirty flag, sha256 of lib/generation.ts, lib/retrieve.ts and lib/refusal.ts
 *     (the Python test refuses a golden older than any of them), versions, and the canned
 *     stream as raw SSE text, which the Python test replays byte for byte.
 * agent/tests/test_generation_request_parity.py sends the same stream to the Python model and
 * compares both the request and what came out of the stream.
 *
 * The cases are synthetic on purpose: the system prompt is built from chunk text and rerank
 * scores, so they are chosen to hit the formatting edges (no chunks, scores at the rounding
 * ties 0.125 and 0.625 where JavaScript's toFixed and Python's format disagree, code fences,
 * template-literal characters, unicode) rather than to be realistic. Retrieval realism is the
 * retrieval golden's job.
 *
 * Nothing leaves the machine: the recorder never calls the real fetch and throws on any URL
 * but the Responses endpoint. .env.local is loaded only because lib/retrieve.ts reads the
 * Supabase variables at import time. Every git call uses --no-optional-locks.
 */
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { streamText, type ModelMessage } from "ai";

import { generationMessages, generationSettings } from "../../lib/generation";
import type { RetrievedChunk } from "../../lib/retrieve";

const OUT = "agent/tests/golden/generation-requests.json";
const RESPONSES_URL = "https://api.openai.com/v1/responses";

type GenerationCase = {
    id: string;
    relevant: RetrievedChunk[];
    history: { role: "user" | "assistant"; content: string }[];
    question: string;
    subQueries: string[];
};

const chunk = (n: number, score: number, content: string): RetrievedChunk => ({
    content,
    title: `Page ${n}`,
    source_url: `https://ai-sdk.dev/docs/page-${n}`,
    score,
});

const CASES: GenerationCase[] = [
    {
        id: "answered",
        relevant: [
            chunk(1, 0.759, "Use `streamText` to stream text from a model.\n\n```ts\nconst result = streamText({ model, prompt });\n```"),
            chunk(2, 0.737, "streamText returns a result whose textStream is an async iterable."),
            chunk(3, 0.3, "Exactly at the rerank threshold."),
        ],
        history: [],
        question: "how do I stream text",
        subQueries: ["How do I stream text with the Vercel AI SDK?"],
    },
    {
        id: "no-context",
        relevant: [],
        history: [],
        question: "how do I fine-tune a model with the AI SDK",
        subQueries: ["How do I fine-tune a model with the Vercel AI SDK?"],
    },
    {
        id: "multi-turn-two-subqueries",
        relevant: [chunk(4, 0.9, "generateText returns the whole answer at once."), chunk(5, 0.88, "Tools are defined with the tool() helper.")],
        history: [
            { role: "user", content: "how do I stream text" },
            { role: "assistant", content: "Use the `streamText` function (Source 1)." },
            { role: "user", content: "thanks" },
            { role: "assistant", content: "You're welcome!" },
        ],
        question: "And generateText? Also tools",
        subQueries: ["What does generateText return?", "How do I define tools in the Vercel AI SDK?"],
    },
    {
        id: "no-subqueries",
        relevant: [chunk(6, 0.5, "A chunk.")],
        history: [],
        question: "raw question, used when the planner produced no sub-queries",
        subQueries: [],
    },
    {
        id: "rounding-ties",
        relevant: [
            chunk(7, 0.125, "0.125 is exactly representable: toFixed(2) gives 0.13, Python's format gives 0.12."),
            chunk(8, 0.625, "0.625 likewise: 0.63 in JavaScript, 0.62 with Python's format."),
            chunk(9, 0.375, "0.375 rounds to 0.38 in both."),
            chunk(10, 0.005, "0.005 is stored slightly above 0.005: 0.01 in both."),
            chunk(11, 1, "A perfect score prints as 1.00."),
        ],
        history: [],
        question: "rounding",
        subQueries: ["rounding"],
    },
    {
        id: "awkward-content",
        relevant: [
            chunk(12, 0.8, "Template ${literal} and a backslash \\ and a backtick ` and \"quotes\" and 'single'."),
            chunk(13, 0.7, "Unicode: — ✓ café 日本語 🙂, a line separator \u2028 and a tab\tinside."),
            chunk(14, 0.6, "Trailing spaces and blank lines   \n\n\n"),
        ],
        history: [],
        question: "awkward",
        subQueries: ["awkward"],
    },
];

/** Deltas chosen so that joining them is the only way to get the text right. */
const DELTAS = ["Use `stream", "Text` (Source 1).", "\n\n```ts\nconst r = streamText({ model });\n```", " — done ✓"];
const TEXT = DELTAS.join("");
const USAGE = {
    input_tokens: 1234,
    input_tokens_details: { cached_tokens: 0 },
    output_tokens: 56,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: 1290,
};

/** A Responses API event stream shaped like the one OpenAI sends for a one-message answer. */
function cannedStream(): string {
    const part = { type: "output_text", annotations: [], logprobs: [], text: TEXT };
    const message = { id: "msg_golden", type: "message", status: "completed", role: "assistant", content: [part] };
    const response = (status: string, extra: object) => ({
        id: "resp_golden",
        object: "response",
        created_at: 1789137600,
        status,
        model: "gpt-4o-mini-2024-07-18",
        ...extra,
    });
    const events: ({ type: string } & Record<string, unknown>)[] = [
        { type: "response.created", response: response("in_progress", { output: [], usage: null }) },
        { type: "response.in_progress", response: response("in_progress", { output: [], usage: null }) },
        {
            type: "response.output_item.added",
            output_index: 0,
            item: { ...message, status: "in_progress", content: [] },
        },
        {
            type: "response.content_part.added",
            item_id: "msg_golden",
            output_index: 0,
            content_index: 0,
            part: { ...part, text: "" },
        },
        ...DELTAS.map((delta) => ({
            type: "response.output_text.delta",
            item_id: "msg_golden",
            output_index: 0,
            content_index: 0,
            delta,
            logprobs: [],
        })),
        { type: "response.output_text.done", item_id: "msg_golden", output_index: 0, content_index: 0, text: TEXT, logprobs: [] },
        { type: "response.content_part.done", item_id: "msg_golden", output_index: 0, content_index: 0, part },
        { type: "response.output_item.done", output_index: 0, item: message },
        { type: "response.completed", response: response("completed", { output: [message], usage: USAGE }) },
    ];
    return events
        .map((e, i) => {
            const event = { ...e, sequence_number: i };
            return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
        })
        .join("");
}

const CANNED_STREAM = cannedStream();

type Recorded = { url: string; method: string; body: unknown };
let recorded: Recorded[] = [];

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (url !== RESPONSES_URL) throw new Error(`recorder: unexpected request to ${url}`);
    if (typeof init?.body !== "string") throw new Error("recorder: expected a JSON string body");
    recorded.push({ url, method: init.method ?? "GET", body: JSON.parse(init.body) });
    return new Response(CANNED_STREAM, { status: 200, headers: { "content-type": "text/event-stream" } });
}) as typeof fetch;

function sh(cmd: string): string {
    return execSync(cmd, { encoding: "utf8" }).trim();
}

function sha256(path: string): string {
    return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function version(pkg: string): string {
    return JSON.parse(readFileSync(`node_modules/${pkg}/package.json`, "utf8")).version;
}

async function main() {
    const cases = [];
    for (const c of CASES) {
        recorded = [];
        // Exactly the route's call (app/api/chat/route.ts), minus abortSignal and onFinish.
        const result = streamText({
            ...generationSettings(c.relevant),
            messages: generationMessages(c.history as ModelMessage[], c.question, c.subQueries),
        });
        const text = await result.text;
        const usage = await result.usage;
        if (text !== TEXT) throw new Error(`${c.id}: the canned stream came back as ${JSON.stringify(text)}`);
        if (recorded.length !== 1) throw new Error(`${c.id}: expected 1 request, saw ${recorded.length}`);
        cases.push({
            ...c,
            request: recorded[0],
            result: { text, inputTokens: usage.inputTokens ?? null, outputTokens: usage.outputTokens ?? null },
        });
    }

    const golden = {
        meta: {
            generatedAt: new Date().toISOString(),
            commit: sh("git --no-optional-locks rev-parse HEAD"),
            dirty: sh("git --no-optional-locks status --porcelain") !== "",
            sourcesSha256: Object.fromEntries(
                ["lib/generation.ts", "lib/retrieve.ts", "lib/refusal.ts"].map((p) => [p, sha256(p)])
            ),
            versions: { ai: version("ai"), "@ai-sdk/openai": version("@ai-sdk/openai") },
            cannedStream: CANNED_STREAM,
        },
        cases,
    };
    mkdirSync(dirname(OUT), { recursive: true });
    writeFileSync(OUT, JSON.stringify(golden, null, 2) + "\n");
    console.log(`wrote ${OUT}: ${cases.length} cases`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
