/**
 * Logs every question and whether it was refused, so production traffic can be mined
 * for eval cases. A refusal with high-scoring chunks retrieved is the shape of a FALSE
 * refusal — see the `suspicious_refusals` view in db/002_query_log.sql.
 *
 * Logging must never affect the user's request: it runs after the response has streamed,
 * and any failure is swallowed into console.error. A telemetry bug should not become an
 * outage.
 *
 * Privacy: this stores the user's question text. Fine for a public docs demo; for anything
 * with real users it needs a retention policy and a note in a privacy statement.
 */
import { createClient } from "@supabase/supabase-js";

import { requireEnv } from "./env";
import { isRefusal } from "./refusal";
import { type RetrievalMode, type RetrievedChunk } from "./retrieve";
import { type TokenUsage } from "./plan";
import { type Visitor } from "./visitor";

/**
 * What a request cost and how long its stages took — MEASURED, from the providers' own
 * usage reports and the SDK's step timings, not estimated (db/005). The daily spend ceiling
 * in lib/rate-limit.ts was derived from a per-request estimate; `cost_daily` re-derives it
 * from these columns over real traffic.
 */
export type RequestMetrics = {
    planner: TokenUsage;
    /** Rerank calls that reached Cohere — the dominant per-request cost on a production key. */
    rerankCalls: number;
    generation: TokenUsage;
    /** Time from the generation call to its first output chunk, ms. Null for canned replies. */
    ttftMs: number | null;
    /** Whole generation call, ms. Null for canned replies. */
    generationMs: number | null;
};

const supabase = createClient(requireEnv("SUPABASE_URL"), requireEnv("SUPABASE_SERVICE_KEY"));

export async function logQuery(entry: {
    question: string;
    answer: string;
    relevant: RetrievedChunk[];
    mode: RetrievalMode;
    latencyMs: number;
    /** Attribution (db/003). Optional so the harness and older callers need not supply it. */
    visitor?: Visitor;
    /** Cost + timing (db/005). Optional for the same reason. */
    metrics?: RequestMetrics;
}): Promise<void> {
    try {
        const { error } = await supabase.from("query_log").insert({
            question: entry.question,
            refused: isRefusal(entry.answer),
            chunk_count: entry.relevant.length,
            top_score: entry.relevant[0]?.score ?? null,
            retrieval_mode: entry.mode,
            latency_ms: Math.round(entry.latencyMs),
            // Visitor attribution — already sanitised in lib/visitor.ts; all nullable.
            visitor_hash: entry.visitor?.visitorHash ?? null,
            landing_referrer: entry.visitor?.landingReferrer ?? null,
            utm_source: entry.visitor?.utmSource ?? null,
            country: entry.visitor?.country ?? null,
            device: entry.visitor?.device ?? null,
            // Cost + timing — db/005; all nullable.
            planner_input_tokens: entry.metrics?.planner.inputTokens ?? null,
            planner_output_tokens: entry.metrics?.planner.outputTokens ?? null,
            rerank_calls: entry.metrics?.rerankCalls ?? null,
            gen_input_tokens: entry.metrics?.generation.inputTokens ?? null,
            gen_output_tokens: entry.metrics?.generation.outputTokens ?? null,
            ttft_ms: entry.metrics?.ttftMs == null ? null : Math.round(entry.metrics.ttftMs),
            generation_ms: entry.metrics?.generationMs == null ? null : Math.round(entry.metrics.generationMs),
        } as never);
        if (error) console.error("QUERY LOG FAILED:", error.message);
    } catch (err) {
        console.error("QUERY LOG FAILED:", err);
    }
}
