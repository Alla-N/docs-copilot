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
import { type Visitor } from "./visitor";

const supabase = createClient(requireEnv("SUPABASE_URL"), requireEnv("SUPABASE_SERVICE_KEY"));

export async function logQuery(entry: {
    question: string;
    answer: string;
    relevant: RetrievedChunk[];
    mode: RetrievalMode;
    latencyMs: number;
    /** Attribution (db/003). Optional so the harness and older callers need not supply it. */
    visitor?: Visitor;
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
        } as never);
        if (error) console.error("QUERY LOG FAILED:", error.message);
    } catch (err) {
        console.error("QUERY LOG FAILED:", err);
    }
}
