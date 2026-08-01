import { embedMany, embed, cosineSimilarity } from "ai";
import { openai } from "@ai-sdk/openai";

const model = openai.embedding("text-embedding-3-small");

export async function GET() {
    const chunks = [
        "streamText returns tokens incrementally as they are generated, rather than waiting for the full response.",
        "To configure billing, add a payment method in your account settings.",
        "Tool calling lets the model invoke functions you define, using Zod schemas for the inputs.",
    ];

    // const query = "why does my chat take so long to show text?";
    const query = "how do I deploy to production?";

    // Embed all chunks in one call
    const { embeddings } = await embedMany({ model, values: chunks });

    // Embed the user's question
    const { embedding: queryEmbedding } = await embed({ model, value: query });

    // Compare the query against each chunk
    const results = chunks.map((text, i) => ({
        text: text.slice(0, 50) + "...",
        similarity: cosineSimilarity(queryEmbedding, embeddings[i]),
    }));

    // Sort: most similar first
    results.sort((a, b) => b.similarity - a.similarity);

    return Response.json({ query, results });
}
