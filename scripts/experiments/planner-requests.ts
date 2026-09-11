/**
 * Planner request golden file: the exact HTTP request the TypeScript planner sends, frozen for
 * the Python port.
 *
 *   npm run exp:planner-requests
 *
 * Step 2.1 of the Windward plan. For every planner-eval case (evals/planner-cases.ts), this runs
 * planQuery() with globalThis.fetch replaced by a recorder that answers with a canned Responses
 * API reply, and writes agent/tests/golden/planner-requests.json:
 *   - per case: the question, the history, the eval's expectations, and the request the planner
 *     made (URL + JSON body: model, input messages, temperature, max_output_tokens, text.format
 *     with the JSON schema zod produced);
 *   - meta: commit, dirty flag, sha256 of lib/plan.ts and evals/planner-cases.ts (the Python test
 *     refuses a golden older than either), the AI SDK versions, and the canned reply.
 * agent/tests/test_planner_request_parity.py sends the SAME canned reply back to the Python
 * planner and asserts it made the same request, byte for byte after JSON parsing. No network on
 * either side, so it runs in CI for free. The eval (planner_eval.py) is the paid, statistical
 * half; this is the exact half.
 *
 * Nothing leaves the machine: the recorder never calls the real fetch and throws on any URL but
 * the Responses endpoint. .env.local is loaded only because lib/retrieve.ts reads the Supabase
 * variables at import time; no key is used.
 *
 * Why the request and not the plan: planner output is model output, so comparing plans is a
 * statistical question (evals/planner.ts runs every case 5 times). The request is deterministic.
 * If the two sides send the same request, any later difference in plans is the model, not the
 * port. Day 14 is the reason to check it: moving the TS planner from the Responses API to Chat
 * Completions changed its outputs for the same prompt.
 */
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { PLANNER_CASES } from "../../evals/planner-cases";
import { planQuery } from "../../lib/plan";

const OUT = "agent/tests/golden/planner-requests.json";
const RESPONSES_URL = "https://api.openai.com/v1/responses";

/** Enough to parse on both sides; the content is irrelevant, only the request is compared. */
const CANNED_PLAN = { intent: "search", queries: [{ query: "golden query", hypothetical: "golden hypothetical" }] };
const CANNED_REPLY = {
    id: "resp_golden",
    object: "response",
    created_at: 1789137600,
    status: "completed",
    model: "gpt-4o-mini-2024-07-18",
    output: [
        {
            type: "message",
            id: "msg_golden",
            status: "completed",
            role: "assistant",
            content: [{ type: "output_text", text: JSON.stringify(CANNED_PLAN), annotations: [], logprobs: [] }],
        },
    ],
    incomplete_details: null,
    error: null,
    usage: {
        input_tokens: 1000,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens: 50,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 1050,
    },
};

type Recorded = { url: string; method: string; body: unknown };
let recorded: Recorded[] = [];

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (url !== RESPONSES_URL) throw new Error(`recorder: unexpected request to ${url}`);
    if (typeof init?.body !== "string") throw new Error("recorder: expected a JSON string body");
    recorded.push({ url, method: init.method ?? "GET", body: JSON.parse(init.body) });
    return new Response(JSON.stringify(CANNED_REPLY), {
        status: 200,
        headers: { "content-type": "application/json" },
    });
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
    for (const c of PLANNER_CASES) {
        recorded = [];
        const plan = await planQuery(c.query, c.history ?? []);
        // planQuery swallows every error into the raw-query fallback, with null usage. A canned
        // reply that parsed comes back with the canned usage; anything else means the recording
        // is of a request whose reply the planner rejected, so refuse to write it.
        if (plan.usage.inputTokens !== CANNED_REPLY.usage.input_tokens) {
            throw new Error(`${c.id}: the planner fell back, so the canned reply did not parse`);
        }
        if (recorded.length !== 1) throw new Error(`${c.id}: expected 1 request, saw ${recorded.length}`);
        cases.push({
            id: c.id,
            question: c.query,
            history: c.history ?? [],
            expect: c.expect,
            ...(c.note ? { note: c.note } : {}),
            request: recorded[0],
        });
    }

    const golden = {
        meta: {
            generatedAt: new Date().toISOString(),
            commit: sh("git rev-parse HEAD"),
            dirty: sh("git --no-optional-locks status --porcelain") !== "",
            planTsSha256: sha256("lib/plan.ts"),
            casesTsSha256: sha256("evals/planner-cases.ts"),
            versions: { ai: version("ai"), "@ai-sdk/openai": version("@ai-sdk/openai"), zod: version("zod") },
            cannedReply: CANNED_REPLY,
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
