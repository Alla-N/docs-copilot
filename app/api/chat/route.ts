import {
    streamText,
    createUIMessageStream,
    toUIMessageStream,
    createUIMessageStreamResponse,
} from "ai";
import { openai } from "@ai-sdk/openai";

import { ChatMessage } from "@/lib/chat-types";
import { buildSystemPrompt, REFUSAL_MESSAGE } from "@/lib/retrieve";
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
        const { intent, relevant, mode, subQueries } = await plannedRetrieve(question, history);
        const retrievalMs = Date.now() - startedAt;

        // ── 4a. Canned replies — no model call at all ────────────────
        // greeting  → the friendly scope message instead of cold-refusing "hi".
        // off-topic → the refusal sentence. The planner decided nothing here is about the
        //             SDK, so there is no context to ground on and nothing to generate; an
        //             injection classified off-topic never reaches the answering model.
        // Both used to be a streamText call told to "reply verbatim" — a paid call whose
        // output we already knew, and one the user's text could still argue with. Writing the
        // text parts straight to the UI stream is cheaper and cannot be talked out of.
        const canned =
            intent === "greeting" ? GREETING_MESSAGE : intent === "off-topic" ? REFUSAL_MESSAGE : null;

        if (canned) {
            if (intent === "off-topic") {
                // Logged like any refusal (chunk_count 0, mode "skipped") so real traffic
                // counts are complete; greetings are not questions and stay out of the log.
                void logQuery({
                    question, answer: canned, relevant, mode, latencyMs: retrievalMs,
                    visitor: visitorFrom(req),
                });
            }
            const stream = createUIMessageStream<ChatMessage>({
                execute: ({ writer }) => {
                    const id = "canned";
                    writer.write({ type: "start" });
                    writer.write({ type: "text-start", id });
                    writer.write({ type: "text-delta", id, delta: canned });
                    writer.write({ type: "text-end", id });
                    writer.write({ type: "finish" });
                },
            });
            return createUIMessageStreamResponse({ stream });
        }

        // Plan-and-execute: generation ANSWERS the planner's resolved sub-queries, not the
        // raw message. The answerer anchors on the literal last turn, so a terse "What is
        // SDK?" gets refused against context its explicit twin answers from, and adversarial
        // noise primes it to drop legit intents. Swapping the final user turn for the resolved
        // queries fixes both: shorthand is already expanded and the noise never reaches
        // generation. Safe — off-topic riders are dropped by the planner (a wholly off-topic
        // message never gets here), so nothing is rewritten into an answerable question. Same
        // swap the eval performs, so eval and prod stay identical.
        const genMessages = subQueries.length
            ? [...messages.slice(0, -1), { role: "user" as const, content: subQueries.join("\n") }]
            : messages;

        // ── 4b. Generate, grounded in the unioned retrieved chunks ───
        const result = streamText({
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
                if (relevant.length > 0) {
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
