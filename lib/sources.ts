// `import type` — fully erased at compile time, so this module never pulls retrieve.ts's
// Supabase/OpenAI/Cohere clients into whoever imports it.
import type { RetrievedChunk } from "./retrieve";

/** A source pill: one documentation PAGE, carrying the chunk numbers the answer may cite. */
export type SourcePill = {
    id: number;
    title: string;
    url: string;
    score: number;
    chunks: number[];
};

/**
 * Collapse retrieved chunks into one pill per page.
 *
 * Chunks are per page SECTION, and the prompt labels them "[Source 1]" … "[Source N]" in
 * retrieval order — that numbering is what the model cites and must not change. But a pill
 * per chunk showed "Core: Generating Structured Data" four times in one answer, which reads
 * as a bug rather than as four relevant sections. So the UI gets one pill per url, keeping
 * the best score and the list of chunk numbers it stands for; the citation renderer maps
 * "(Source 3)" back to the pill whose `chunks` contains 3. Order is preserved from `relevant`
 * (score-descending), so the first pill is still the best page.
 */
export function toSourcePills(relevant: RetrievedChunk[]): SourcePill[] {
    const byUrl = new Map<string, SourcePill>();
    relevant.forEach((c, i) => {
        const n = i + 1;
        const existing = byUrl.get(c.source_url);
        if (existing) {
            existing.chunks.push(n);
            existing.score = Math.max(existing.score, c.score);
            return;
        }
        byUrl.set(c.source_url, {
            id: byUrl.size + 1,
            title: c.title,
            url: c.source_url,
            score: c.score,
            chunks: [n],
        });
    });
    return [...byUrl.values()];
}
