/**
 * Reading a run back out of Langfuse (step 2.7).
 *
 * The service streams pages, not chunk texts, so the faithfulness judge had nothing to read on
 * the Python target (`EVAL_JUDGE=1` was refused there in 2.6). It does now: each answered turn
 * writes a `context` observation carrying the merged chunks the prompt was built from
 * (agent/src/copilot_agent/tracing.py), and this module fetches them back.
 *
 * By query, not one request per trace. A full run makes 81 traces and the free plan allows 30
 * observation requests a minute; one `name=context` query over the run's own time window,
 * paged, is two or three requests for all of them. The rows are then matched on trace id, which
 * the harness gets from the query_log rows it already reads for cost (db/007_trace_id.sql).
 *
 * Spans leave the service on a batch exporter (5 s) and Langfuse ingests them asynchronously, so
 * this waits for the traces it expects rather than reading once and finding half a run.
 */
import { requireEnv } from "../lib/env";

export type LangfuseApi = { baseUrl: string; auth: string };

/** The public API of the project the service traced to. Same three variables the service reads. */
export function langfuseApi(): LangfuseApi {
    const baseUrl = (process.env.LANGFUSE_BASE_URL ?? "https://cloud.langfuse.com").trim().replace(/\/+$/, "");
    const credentials = `${requireEnv("LANGFUSE_PUBLIC_KEY")}:${requireEnv("LANGFUSE_SECRET_KEY")}`;
    return { baseUrl, auth: Buffer.from(credentials).toString("base64") };
}

/** One chunk as the `context` span carries it, in the shape lib/retrieve.ts uses. */
export type ContextChunk = { title: string; content: string; source_url: string; score: number };

type ObservationRow = { traceId?: string; output?: unknown };

function asChunks(output: unknown): ContextChunk[] | null {
    // The v2 endpoint returns input and output as RAW STRINGS and rejects parseIoAsJson=true
    // with a 400, although the Python client still offers the parameter. Parse it here.
    const parsed = typeof output === "string" ? (JSON.parse(output) as unknown) : output;
    if (!Array.isArray(parsed)) return null;
    return parsed.map((item) => {
        const chunk = item as { title?: string; url?: string; score?: number; text?: string };
        return {
            title: chunk.title ?? "",
            content: chunk.text ?? "",
            source_url: chunk.url ?? "",
            score: chunk.score ?? 0,
        };
    });
}

async function fetchPage(api: LangfuseApi, since: Date, cursor: string | null): Promise<{ rows: ObservationRow[]; cursor: string | null }> {
    const url = new URL(`${api.baseUrl}/api/public/v2/observations`);
    url.searchParams.set("name", "context");
    url.searchParams.set("fields", "core,io");
    url.searchParams.set("fromStartTime", since.toISOString());
    url.searchParams.set("limit", "100");
    if (cursor) url.searchParams.set("cursor", cursor);

    const response = await fetch(url, { headers: { authorization: `Basic ${api.auth}` } });
    if (!response.ok) {
        throw new Error(`Langfuse answered ${response.status} for ${url.pathname}: ${(await response.text()).slice(0, 300)}`);
    }
    const body = (await response.json()) as { data?: ObservationRow[]; meta?: { cursor?: string | null } };
    return { rows: body.data ?? [], cursor: body.meta?.cursor ?? null };
}

/**
 * The grounding chunks of every traced turn since `since`, by trace id.
 *
 * Waits until `expected` of the wanted trace ids have arrived, or `waitMs` passes: a missing
 * trace is reported by the caller, never silently judged as an answer with no sources.
 */
export async function contextsByTrace(
    api: LangfuseApi,
    since: Date,
    wanted: Set<string>,
    waitMs = 60_000
): Promise<Map<string, ContextChunk[]>> {
    const deadline = Date.now() + waitMs;
    for (;;) {
        const found = new Map<string, ContextChunk[]>();
        let cursor: string | null = null;
        do {
            const page: { rows: ObservationRow[]; cursor: string | null } = await fetchPage(api, since, cursor);
            for (const row of page.rows) {
                if (!row.traceId || !wanted.has(row.traceId) || found.has(row.traceId)) continue;
                const chunks = asChunks(row.output);
                if (chunks) found.set(row.traceId, chunks);
            }
            cursor = page.cursor;
        } while (cursor);
        if (found.size >= wanted.size || Date.now() > deadline) return found;
        await new Promise((r) => setTimeout(r, 3000));
    }
}
