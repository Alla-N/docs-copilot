import {
    streamText,
    createUIMessageStream,
    convertToModelMessages,
    toUIMessageStream,
    createUIMessageStreamResponse,
    UIMessage,
} from "ai";
import { openai } from "@ai-sdk/openai";

import { ChatMessage } from "@/lib/chat-types";
import { retrieve, buildSystemPrompt } from "@/lib/retrieve";
import { logQuery } from "@/lib/query-log";

export async function POST(req: Request) {
    try {
        const { messages }: { messages: UIMessage[] } = await req.json();

        // ── 1. Get the user's latest question as plain text ──────────
        const lastMessage = messages[messages.length - 1];
        const userQuestion = lastMessage.parts
            .filter((p) => p.type === "text")
            .map((p) => (p as { type: "text"; text: string }).text)
            .join(" ");

        // ── 2. Retrieve — embed, vector search, rerank, threshold ────
        // Lives in lib/retrieve.ts so the eval harness runs this exact code.
        const startedAt = Date.now();
        const { relevant, mode } = await retrieve(userQuestion);
        const retrievalMs = Date.now() - startedAt;

        // ── 3. Generate, grounded ────────────────────────────────────
        const result = streamText({
            model: openai("gpt-4o-mini"),
            temperature: 0,
            system: buildSystemPrompt(relevant),
            messages: await convertToModelMessages(messages),
            // Fires after the stream completes, so logging never delays a token.
            // Deliberately not awaited and it cannot throw — telemetry must not
            // be able to break the request it is observing.
            onFinish: ({ text }) => {
                void logQuery({ question: userQuestion, answer: text, relevant, mode, latencyMs: retrievalMs });
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
        console.error("CHAT ROUTE ERROR:", err);
        return new Response(JSON.stringify({ error: String(err) }), {
            status: 500,
        });
    }
}
