import {
    streamText,
    createUIMessageStream,
    toUIMessageStream,
    createUIMessageStreamResponse,
} from "ai";
import { openai } from "@ai-sdk/openai";

import { ChatMessage } from "@/lib/chat-types";
import { retrieve, buildSystemPrompt } from "@/lib/retrieve";
import { logQuery } from "@/lib/query-log";
import { parseChatRequest, BadRequestError } from "@/lib/chat-request";
import { checkRateLimit, clientKey } from "@/lib/rate-limit";

export async function POST(req: Request) {
    try {
        // ── 1. Rate limit BEFORE any paid work ───────────────────────
        // Ordering matters: embedding, reranking and generation all cost money, so the
        // limiter runs before the request is even parsed.
        const limit = await checkRateLimit(clientKey(req));
        if (!limit.allowed) {
            // Structured body: the client transport surfaces the response text as
            // error.message, so shipping JSON lets the UI explain WHICH limit was hit
            // instead of showing one generic failure.
            return Response.json(
                { code: "rate_limited", scope: limit.scope, retryAfter: limit.retryAfter },
                { status: 429, headers: { "Retry-After": String(limit.retryAfter) } }
            );
        }

        // ── 2. Parse and normalise — never trust the client's shape ──
        // Returns a message array we BUILT, not one we inspected: role + text only,
        // capped, assistant-authored parts stripped.
        const { question, messages } = parseChatRequest(await req.json());

        // ── 3. Retrieve — embed, vector search, rerank, threshold ────
        // Lives in lib/retrieve.ts so the eval harness runs this exact code.
        const startedAt = Date.now();
        const { relevant, mode } = await retrieve(question);
        const retrievalMs = Date.now() - startedAt;

        // ── 4. Generate, grounded ────────────────────────────────────
        const result = streamText({
            model: openai("gpt-4o-mini"),
            temperature: 0,
            system: buildSystemPrompt(relevant),
            messages,
            // Fires after the stream completes, so logging never delays a token.
            // Deliberately not awaited and it cannot throw — telemetry must not
            // be able to break the request it is observing.
            onFinish: ({ text }) => {
                void logQuery({ question, answer: text, relevant, mode, latencyMs: retrievalMs });
            },
        });

        const stream = createUIMessageStream<ChatMessage>({
            onError: (err) => {
                console.error("STREAM ERROR:", err);
                return "Stream failed";
            },
            execute: async ({ writer }) => {
                if (relevant.length > 0) {
                    writer.write({
                        type: "data-sources",
                        data: relevant.map((c, i) => ({
                            id: i + 1,
                            title: c.title,
                            url: c.source_url,
                            score: c.score,
                        })),
                    });
                }
                writer.merge(toUIMessageStream(result));
            },
        });

        return createUIMessageStreamResponse({ stream });
    } catch (err) {
        // A malformed request is the caller's fault and gets a 400 with a usable message.
        // Everything else is ours and stays opaque — internal errors must not leak
        // schema names, table names or stack traces to a public endpoint.
        if (err instanceof BadRequestError) {
            return Response.json({ error: err.message }, { status: 400 });
        }
        console.error("CHAT ROUTE ERROR:", err);
        return Response.json({ error: "Something went wrong." }, { status: 500 });
    }
}
