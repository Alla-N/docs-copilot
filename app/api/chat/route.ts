import {
    streamText,
    createUIMessageStream,
    toUIMessageStream,
    createUIMessageStreamResponse,
} from "ai";
import { openai } from "@ai-sdk/openai";

import { ChatMessage } from "@/lib/chat-types";
import { buildSystemPrompt } from "@/lib/retrieve";
import { plannedRetrieve, GREETING_MESSAGE } from "@/lib/plan";
import { logQuery } from "@/lib/query-log";
import { parseChatRequest, BadRequestError } from "@/lib/chat-request";
import { checkRateLimit, clientKey } from "@/lib/rate-limit";
import { visitorFrom } from "@/lib/visitor";
import { toSourcePills } from "@/lib/sources";

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
        const { question, messages } = parseChatRequest(await req.json());

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
        const startedAt = Date.now();
        const { greeting, relevant, mode, subQueries } = await plannedRetrieve(question, history);
        const retrievalMs = Date.now() - startedAt;

        // Plan-and-execute: generation ANSWERS the planner's resolved sub-queries, not the
        // raw message. The answerer anchors on the literal last turn, so a terse "What is
        // SDK?" gets refused against context its explicit twin answers from, and adversarial
        // noise primes it to drop legit intents. Swapping the final user turn for the resolved
        // queries fixes both: shorthand is already expanded and the noise never reaches
        // generation. Safe — off-topic/injection yield no usable sub-queries, so the planner
        // falls back to the raw question and rewrites nothing into an answerable one. Same
        // swap the eval performs, so eval and prod stay identical.
        const genMessages =
            !greeting && subQueries.length
                ? [...messages.slice(0, -1), { role: "user" as const, content: subQueries.join("\n") }]
                : messages;

        // ── 4. Generate ──────────────────────────────────────────────
        // A greeting short-circuits grounding: reply with the fixed scope message instead of
        // cold-refusing "hi". Everything else is grounded in the unioned retrieved chunks.
        const result = greeting
            ? streamText({
                model: openai("gpt-4o-mini"),
                temperature: 0,
                system: `Reply with EXACTLY the following text, verbatim, and nothing else:\n\n${GREETING_MESSAGE}`,
                messages,
            })
            : streamText({
                model: openai("gpt-4o-mini"),
                temperature: 0,
                system: buildSystemPrompt(relevant),
                messages: genMessages,
                onFinish: ({ text }) => {
                    // Attribution headers are read here, after the answer streamed, and are
                    // sanitised in lib/visitor.ts. They never touch retrieval or the prompt.
                    void logQuery({
                        question, answer: text, relevant, mode, latencyMs: retrievalMs,
                        visitor: visitorFrom(req),
                    });
                },
            });

        const stream = createUIMessageStream<ChatMessage>({
            onError: (err) => {
                console.error("STREAM ERROR:", err);
                return "Stream failed";
            },
            execute: async ({ writer }) => {
                if (!greeting && relevant.length > 0) {
                    // One pill per page, not per chunk; the prompt's "[Source N]" numbering is
                    // untouched and each pill lists the N's it stands for (lib/sources.ts).
                    writer.write({ type: "data-sources", data: toSourcePills(relevant) });
                }
                writer.merge(toUIMessageStream(result));
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
