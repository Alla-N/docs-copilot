/**
 * Chunking comparison on the REAL corpus, with the REAL chunker, against the eval queries.
 *
 *   npm run exp:chunking
 *
 * The first version of this experiment produced the README's "0.546 → 0.643" and was a toy:
 * three short paragraphs (lib/sample-docs.ts), one hand-picked query, n = 1, and a
 * "structured" chunker that split on `# ` only and was not lib/chunker.ts. It could not have
 * detected a change to the chunker it claimed to measure. This version can:
 *
 *   corpus     the 37 pages in lib/corpus.ts, fetched live and stripped like ingestion
 *   chunkers   fixed-500 (the old naive baseline), fixed-1500 (same cap as ours, no
 *              structure — the fair control), chunkPage (lib/chunker.ts, what is deployed)
 *   queries    every answerable eval case with an expected page (evals/dataset.ts)
 *   metric     pure vector search, cosine on text-embedding-3-small, raw question embedded —
 *              no HyDE, no reranker, so the ONLY variable is where the chunk boundaries fall.
 *              recall@5 and recall@40 (does the expected page make the top-N?), MRR over the
 *              expected page's best rank, and the expected page's best cosine.
 *
 * Re-run when lib/chunker.ts, the corpus or the embedding model changes. Cost: embeds every
 * chunk of every strategy once (~3–4k embeddings, a couple of cents).
 */
import { embedMany, cosineSimilarity } from "ai";
import { openai } from "@ai-sdk/openai";

import { chunkPage, stripBoilerplate } from "../../lib/chunker";
import { PAGES } from "../../lib/corpus";
import { CASES, type EvalCase } from "../../evals/dataset";

const model = openai.embedding("text-embedding-3-small");
const EMBED_BATCH = 100;

type Chunk = { content: string; url: string; title: string };
type Strategy = { name: string; chunk: (markdown: string) => string[] };

function fixed(size: number) {
    return (text: string): string[] => {
        const out: string[] = [];
        for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
        return out.filter((c) => c.trim().length > 80); // same floor as chunkPage
    };
}

const STRATEGIES: Strategy[] = [
    { name: "fixed 500 chars (old naive baseline)", chunk: fixed(500) },
    { name: "fixed 1500 chars (same cap, no structure)", chunk: fixed(1500) },
    { name: "chunkPage — lib/chunker.ts (deployed)", chunk: (md) => chunkPage(md) },
];

function matches(c: EvalCase, url: string): boolean {
    const wanted = Array.isArray(c.expectedSource) ? c.expectedSource : c.expectedSource ? [c.expectedSource] : [];
    return wanted.some((slug) => url.includes(slug));
}

async function embedAll(values: string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < values.length; i += EMBED_BATCH) {
        const { embeddings } = await embedMany({ model, values: values.slice(i, i + EMBED_BATCH) });
        out.push(...embeddings);
    }
    return out;
}

async function main() {
    // ── Corpus, exactly as ingestion sees it ──────────────────────
    const pages: { title: string; url: string; markdown: string }[] = [];
    for (const p of PAGES) {
        const res = await fetch(p.url);
        if (!res.ok) throw new Error(`Failed to fetch ${p.url}: ${res.status}`);
        pages.push({ ...p, markdown: stripBoilerplate(await res.text()) });
    }
    const totalChars = pages.reduce((n, p) => n + p.markdown.length, 0);
    console.log(`corpus: ${pages.length} pages, ${totalChars.toLocaleString()} chars\n`);

    // ── Queries: the answerable eval cases, raw text ──────────────
    const queries = CASES.filter((c) => c.shouldAnswer === true && !c.injection && c.expectedSource);
    const queryEmbeddings = await embedAll(queries.map((q) => q.query));
    console.log(`queries: ${queries.length} answerable eval cases (raw question embedded — no HyDE, no rerank)\n`);

    const rows: Record<string, string | number>[] = [];
    for (const s of STRATEGIES) {
        const chunks: Chunk[] = pages.flatMap((p) => s.chunk(p.markdown).map((content) => ({ content, url: p.url, title: p.title })));
        const embeddings = await embedAll(chunks.map((c) => c.content));

        let hit5 = 0, hit40 = 0, rr = 0, topSum = 0;
        const misses: string[] = [];
        queries.forEach((q, qi) => {
            const ranked = chunks
                .map((c, i) => ({ c, sim: cosineSimilarity(queryEmbeddings[qi], embeddings[i]) }))
                .sort((a, b) => b.sim - a.sim);
            const rank = ranked.findIndex((r) => matches(q, r.c.url)) + 1; // 0 → not found
            if (rank >= 1 && rank <= 5) hit5++;
            if (rank >= 1 && rank <= 40) hit40++;
            if (rank >= 1) { rr += 1 / rank; topSum += ranked[rank - 1].sim; } else misses.push(q.id);
        });

        const avgLen = Math.round(chunks.reduce((n, c) => n + c.content.length, 0) / chunks.length);
        rows.push({
            strategy: s.name,
            chunks: chunks.length,
            "avg chars": avgLen,
            "recall@5": `${hit5}/${queries.length}`,
            "recall@40": `${hit40}/${queries.length}`,
            MRR: (rr / queries.length).toFixed(3),
            "mean best cosine": (topSum / queries.length).toFixed(3),
        });
        if (misses.length) console.log(`  ${s.name}: expected page never surfaced for ${misses.join(", ")}`);
    }

    console.log();
    console.table(rows);
    console.log(
        "\nrecall@40 is the number that matters most in production: 40 is the candidate pool the reranker\n" +
        "sees, and the reranker can only re-order what vector search hands it."
    );
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
