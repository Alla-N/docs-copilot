import {
    streamText,
    createUIMessageStream,
    convertToModelMessages,
    toUIMessageStream,
    createUIMessageStreamResponse,
    UIMessage,
    embed,
    rerank
} from "ai";
import { cohere } from "@ai-sdk/cohere";
import { openai } from "@ai-sdk/openai";
import { createClient } from "@supabase/supabase-js";

import { ChatMessage } from "@/lib/chat-types";

const RERANK_THRESHOLD = 0.3;

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_KEY!
);

export async function POST(req: Request) {
    try {
        const { messages }: { messages: UIMessage[] } = await req.json();

        // ── 1. Get the user's latest question as plain text ──────────
        const lastMessage = messages[messages.length - 1];
        const userQuestion = lastMessage.parts
            .filter((p) => p.type === "text")
            .map((p) => (p as { type: "text"; text: string }).text)
            .join(" ");

        // ── 2. Embed it (same model as ingestion — non-negotiable) ───
        const { embedding } = await embed({
            model: openai.embedding("text-embedding-3-small"),
            value: userQuestion,
        });

        const { data: candidates, error } = await supabase.rpc("match_documents", {
            query_embedding: embedding,
            match_count: 20,
        });
        if (error) throw new Error(`Retrieval failed: ${error.message}`);

        const docs = (candidates ?? []) as {
            content: string;
            title: string;
            source_url: string;
            similarity: number;
        }[];

        // 3. Rerank — cross-encoder narrows 20 → best 5 with real relevance scores
        let relevant: { content: string; title: string; source_url: string; score: number }[] = [];
        if (docs.length > 0) {
            const { ranking } = await rerank({
                model: cohere.reranking("rerank-v3.5"),
                query: userQuestion,
                documents: docs.map((d) => d.content),
                topN: 5,
            });

            // 4. Threshold gate on the RERANK score, map back to full chunk via originalIndex
            relevant = ranking
                .filter((r) => r.score >= RERANK_THRESHOLD)
                .map((r) => ({
                    content: docs[r.originalIndex].content,
                    title: docs[r.originalIndex].title,
                    source_url: docs[r.originalIndex].source_url,
                    score: r.score,
                }));
        }

        // ── 5. Build the grounded system prompt ──────────────────────
        const context =
            relevant.length > 0
                ? relevant
                    .map(
                        (c, i) =>
                            `[Source ${i + 1}] (relevance: ${c.score.toFixed(2)})\n${c.content}`
                    )
                    .join("\n\n---\n\n")
                : "NO RELEVANT DOCUMENTATION FOUND.";

        const system = `You are a documentation assistant for the Vercel AI SDK.

Answer ONLY using the documentation provided below. Rules:
- If the documentation below says "NO RELEVANT DOCUMENTATION FOUND", or does not contain the answer, say: "I don't have information about that in the documentation." Do not answer from general knowledge.
- When you answer, mention which source you used, e.g. (Source 1).
- Be concise and accurate.

DOCUMENTATION:
${context}`;

        // ── 6. Generate, grounded ────────────────────────────────────
        const result = streamText({
            model: openai("gpt-4o-mini"),
            temperature: 0,
            system,
            messages: await convertToModelMessages(messages),
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
