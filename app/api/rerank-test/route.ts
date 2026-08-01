import { embed, rerank } from "ai";
import { openai } from "@ai-sdk/openai";
import { cohere } from "@ai-sdk/cohere";
import { createClient } from "@supabase/supabase-js";

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_KEY!
);

export async function GET(req: Request) {
    const q = new URL(req.url).searchParams.get("q") ?? "how do I stream text";

    // 1. Vector search — get a WIDE net (top 20), not just 5
    const { embedding } = await embed({
        model: openai.embedding("text-embedding-3-small"),
        value: q,
    });
    const { data: candidates, error } = await supabase.rpc("match_documents", {
        query_embedding: embedding,
        match_count: 20,
    });
    if (error) return Response.json({ error: error.message }, { status: 500 });

    const docs = (candidates ?? []) as {
        content: string;
        title: string;
        similarity: number;
    }[];

    // 2. Rerank the 20 candidates — cross-encoder judgment
    const { ranking } = await rerank({
        model: cohere.reranking("rerank-v3.5"),
        query: q,
        documents: docs.map((d) => d.content),
        topN: 5,
    });

    // 3. Show BEFORE (vector order) vs AFTER (rerank order) side by side
    const before = docs.slice(0, 5).map((d, i) => ({
        rank: i + 1,
        vectorScore: Number(d.similarity.toFixed(3)),
        title: d.title,
        preview: d.content.slice(0, 55).replace(/\n/g, " "),
    }));

    const after = ranking.map((r, i) => ({
        rank: i + 1,
        rerankScore: Number(r.score.toFixed(3)),
        title: docs[r.originalIndex].title,
        preview: docs[r.originalIndex].content.slice(0, 55).replace(/\n/g, " "),
    }));

    return Response.json({ query: q, before, after });
}
