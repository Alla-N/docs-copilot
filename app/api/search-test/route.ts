import { embed } from "ai";
import { openai } from "@ai-sdk/openai";
import { createClient } from "@supabase/supabase-js";

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_KEY!
);

export async function GET(req: Request) {
    const q = new URL(req.url).searchParams.get("q");
    if (!q) return Response.json({ error: "add ?q=your+question" }, { status: 400 });

    const { embedding } = await embed({
        model: openai.embedding("text-embedding-3-small"),
        value: q,
    });

    const { data, error } = await supabase.rpc("match_documents", {
        query_embedding: embedding,
        match_count: 5,
    });
    if (error) return Response.json({ error: error.message }, { status: 500 });

    return Response.json({
        query: q,
        results: (data ?? []).map((r: { title: string; similarity: number; content: string }) => ({
            similarity: Number(r.similarity.toFixed(3)),
            title: r.title,
            preview: r.content.slice(0, 100).replace(/\n/g, " "),
        })),
    });
}
