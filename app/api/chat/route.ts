import {
    streamText,
    createUIMessageStream,
    toUIMessageStream,
    createUIMessageStreamResponse,
} from "ai";
import { after } from "next/server";

import { ChatMessage } from "@/lib/chat-types";
import { REFUSAL_MESSAGE } from "@/lib/retrieve";
import { plannedRetrieve, GREETING_MESSAGE } from "@/lib/plan";
import { generationSettings, generationMessages } from "@/lib/generation";
import { logQuery, type RequestMetrics } from "@/lib/query-log";
import { parseChatRequest, BadRequestError } from "@/lib/chat-request";
import { checkRateLimit, clientKey } from "@/lib/rate-limit";
import { visitorFrom } from "@/lib/visitor";
import { toSourcePills } from "@/lib/sources";

/**
 * Explicit, not the platform default. Measured retrieval worst case is ~6 s BEFORE
 * generation starts (planner + embed + rerank), and a long answer streams for several more;
 * a 10 s default would cut answers off mid-sentence on a slow day. 60 s is generous, and
 * bounded — a hung upstream call can't hold a function open indefinitely.
 */
export const maxDuration = 60;

export async function POST(req: Request) {
    try {
        // ── 1. Rate limit BEFORE any paid work ───────────────────────
        const limit = await checkRateLimit(clientKey(req));
        if (!limit.allowed) {
            return Response.json(
                { code: "rate_limited", scope: limit.scope, retryAfter: limit.retryAfter },
                { status: 429, headers: { "Retry-After": String(limit.retryAfter) } }
            );
        }

        // ── 2. Parse and normalise — never trust the client's shape ──
        // Malformed JSON is the caller's error, not ours: `req.json()` throws a SyntaxError,
        // which used to fall through to the generic 500 branch. (Review item 15.)
        let body: unknown;
        try {
            body = await req.json();
        } catch {
            throw new BadRequestError("Request body must be JSON.");
        }
        const { question, messages } = parseChatRequest(body);

        // Prior turns (everything before the current question) feed the planner so it can
        // resolve "it"/"that" in a follow-up. Text only, user/assistant only.
        const history = messages
            .slice(0, -1)
            .filter(
                (m): m is { role: "user" | "assistant"; content: string } =>
                    (m.role === "user" || m.role === "assistant") && typeof m.content === "string"
            )
            .map((m) => ({ role: m.role, text: m.content }));

        // ── 3. Plan + retrieve — expand/split/resolve, then search each sub-query ──
        // plannedRetrieve orchestrates retrieve(); the eval harness calls the SAME function.
        // The request's abort signal rides along: a closed tab cancels the planner call and
        // the generation below instead of billing for an answer nobody will read.
        const startedAt = Date.now();
        const { intent, relevant, mode, subQueries, plannerUsage, rerankCalls } = await plannedRetrieve(
            question,
            history,
            { signal: req.signal }
        );
        const retrievalMs = Date.now() - startedAt;

        // Logging runs in `after()`: on serverless the function can be frozen the moment the
        // response finishes, and a bare `void logQuery(...)` inside onFinish raced that
        // freeze — some rows never landed. `after()` keeps the function alive until the
        // insert resolves, without delaying the response. (Review item 16.)
        // Cost and timing ride along (db/005): the planner's tokens and rerank calls from the
        // retrieval step, the generation's tokens and timings from onFinish. Measured, so the
        // spend ceiling can be re-derived from traffic instead of an estimate. (Review item 25.)
        const visitor = visitorFrom(req);
        const log = (answer: string, generation: RequestMetrics["generation"] = { inputTokens: null, outputTokens: null }, timing: Pick<RequestMetrics, "ttftMs" | "generationMs"> = { ttftMs: null, generationMs: null }) =>
            after(() =>
                logQuery({
                    question, answer, relevant, mode, latencyMs: retrievalMs, visitor,
                    metrics: { planner: plannerUsage, rerankCalls, generation, ...timing },
                })
            );

        // ── 4a. Canned replies — no model call at all ────────────────
        // greeting  → the friendly scope message instead of cold-refusing "hi".
        // off-topic → the refusal sentence. The planner decided nothing here is about the
        //             SDK, so there is no context to ground on and nothing to generate; an
        //             injection classified off-topic never reaches the answering model.
        // Both used to be a streamText call told to "reply verbatim" — a paid call whose
        // output we already knew, and one the user's text could still argue with. Writing the
        // text parts straight to the UI stream is cheaper and cannot be talked out of.
        // Both are LOGGED: the "asked hi and left" visitor is exactly what the attribution
        // spec wanted to see, and an off-topic refusal is a refusal. (Review item 17.)
        const canned =
            intent === "greeting" ? GREETING_MESSAGE : intent === "off-topic" ? REFUSAL_MESSAGE : null;

        if (canned) {
            log(canned);
            const stream = createUIMessageStream<ChatMessage>({
                execute: ({ writer }) => {
                    const id = "canned";
                    writer.write({ type: "start" });
                    writer.write({ type: "data-retrieval", data: { mode, intent } });
                    writer.write({ type: "text-start", id });
                    writer.write({ type: "text-delta", id, delta: canned });
                    writer.write({ type: "text-end", id });
                    writer.write({ type: "finish" });
                },
            });
            return createUIMessageStreamResponse({ stream });
        }

        // ── 4b. Generate, grounded in the unioned retrieved chunks ───
        // Settings and the resolved-query swap come from lib/generation.ts — the same module
        // the eval harness and the judge calibration use, so the three cannot drift apart.
        const result = streamText({
            ...generationSettings(relevant),
            messages: generationMessages(messages.slice(0, -1), question, subQueries),
            abortSignal: req.signal,
            onFinish: ({ text, usage, steps }) => {
                // The SDK measures the step: time to first output chunk and the whole call.
                const perf = steps[steps.length - 1]?.performance;
                log(
                    text,
                    { inputTokens: usage.inputTokens ?? null, outputTokens: usage.outputTokens ?? null },
                    { ttftMs: perf?.timeToFirstOutputMs ?? null, generationMs: perf?.responseTimeMs ?? null }
                );
            },
        });

        const stream = createUIMessageStream<ChatMessage>({
            onError: (err) => {
                console.error("STREAM ERROR:", err);
                return "Stream failed";
            },
            execute: async ({ writer }) => {
                // OPEN THE MESSAGE FIRST — this line is load-bearing, and its absence shipped
                // a visible bug. The client keys the streaming assistant message by id: each
                // update either REPLACES the last message (same id) or PUSHES a new one, and
                // the `start` chunk is what sets that id (createUIMessageStream stamps its
                // generated messageId onto the first start chunk it sees). Writing a data part
                // before any start therefore pushed the message under the client's own
                // provisional id; the start chunk that arrived later — from the merged
                // generation stream, where `sendStart` defaults to true — renamed the
                // in-flight message, so the next update no longer matched the last message and
                // pushed it AGAIN. The reader saw an empty bubble carrying the source pills,
                // then the real answer with the same pills below it. The canned path above
                // never showed this because it writes `start` first. (Review item 35.)
                writer.write({ type: "start" });

                // Which retrieval path produced this answer. The UI tells the reader when the
                // reranker was unavailable — that path uses cosine order with a stricter cut,
                // so it refuses more and ranks worse, and silence would blame the docs.
                writer.write({ type: "data-retrieval", data: { mode, intent } });
                if (relevant.length > 0) {
                    // One pill per page, not per chunk; the prompt's "[Source N]" numbering is
                    // untouched and each pill lists the N's it stands for (lib/sources.ts).
                    writer.write({ type: "data-sources", data: toSourcePills(relevant) });
                }
                // sendStart: false — the message is already open. One `start` per message is
                // the invariant; a second one is what caused the duplicate above.
                writer.merge(toUIMessageStream({ stream: result.stream, sendStart: false }));
            },
        });

        return createUIMessageStreamResponse({ stream });
    } catch (err) {
        if (err instanceof BadRequestError) {
            return Response.json({ error: err.message }, { status: 400 });
        }
        console.error("CHAT ROUTE ERROR:", err);
        return Response.json({ error: "Something went wrong." }, { status: 500 });
    }
}
