/**
 * The experiment behind the headline number: naive 0.546 -> structured 0.643.
 *
 *   npm run exp:chunking
 *
 * Same content, same query, same embedding model — only the chunk boundaries differ.
 * Re-run this whenever lib/chunker.ts changes.
 */
import { embedMany, embed, cosineSimilarity } from "ai";
import { openai } from "@ai-sdk/openai";

import { sampleDocs } from "../../lib/sample-docs";

const model = openai.embedding("text-embedding-3-small");
const QUERY = "how do I compare two embeddings for similarity?";

function naiveChunk(text: string, size = 500): string[] {
    const chunks: string[] = [];
    for (let i = 0; i < text.length; i += size) {
        chunks.push(text.slice(i, i + size));
    }
    return chunks;
}

function structuredChunk(text: string): string[] {
    // Split on markdown headings, keeping each heading with its content.
    return text
        .split(/\n(?=# )/)
        .map((c) => c.trim())
        .filter((c) => c.length > 0);
}

async function topScore(chunks: string[], queryEmbedding: number[]) {
    const { embeddings } = await embedMany({ model, values: chunks });
    const scored = chunks
        .map((text, i) => ({
            similarity: cosineSimilarity(queryEmbedding, embeddings[i]),
            preview: text.replace(/\n/g, " ").slice(0, 60) + "...",
        }))
        .sort((a, b) => b.similarity - a.similarity);
    return scored;
}

async function main() {
    const { embedding: queryEmbedding } = await embed({ model, value: QUERY });

    const naive = await topScore(naiveChunk(sampleDocs), queryEmbedding);
    const structured = await topScore(structuredChunk(sampleDocs), queryEmbedding);

    console.log("query:", QUERY);
    console.table([
        { strategy: "naive (fixed 500 chars)", chunks: naive.length, topScore: naive[0].similarity.toFixed(3) },
        { strategy: "structured (by heading)", chunks: structured.length, topScore: structured[0].similarity.toFixed(3) },
    ]);

    console.log("\nbest structured match:", structured[0].preview);
    console.log("best naive match:      ", naive[0].preview);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
