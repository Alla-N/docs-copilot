/**
 * Ingestion — run from the terminal, never deployed.
 *
 *   npm run ingest              diff against the DB and print the plan. No writes, no cost.
 *   npm run ingest -- --write   apply it: embed only what changed, delete what went stale.
 *
 * Idempotent by content hash. Re-running with unchanged docs is a no-op that
 * spends nothing. Writing is opt-in so a mistyped command can't corrupt the corpus.
 *
 * The deployed app only ever READS the vector store. Writing is an ops job,
 * so it lives here instead of behind a public URL.
 */
import { createHash } from "node:crypto";

import { embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import { createClient } from "@supabase/supabase-js";

// Relative, not "@/..." — tsx runs this outside Next.js, so the tsconfig alias isn't applied.
import { chunkPage, stripBoilerplate } from "../lib/chunker";

type DocumentRow = {
    content: string;
    source_url: string;
    title: string;
    content_hash: string;
    embedding: number[];
};

type Database = {
    public: {
        Tables: {
            documents: {
                Row: DocumentRow & { id: number };
                Insert: DocumentRow;
                Update: Partial<DocumentRow>;
            };
        };
        Views: Record<string, never>;
        Functions: Record<string, never>;
        Enums: Record<string, never>;
        CompositeTypes: Record<string, never>;
    };
};

function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(
            `Missing ${name}. Run via "npm run ingest" so --env-file=.env.local is applied.`
        );
    }
    return value;
}

const supabase = createClient<Database>(
    requireEnv("SUPABASE_URL"),
    requireEnv("SUPABASE_SERVICE_KEY")
);

/**
 * MUST stay byte-identical to db/001_content_hash.sql:
 *   encode(sha256(convert_to(source_url || E'\n' || content, 'UTF8')), 'hex')
 * source_url is in the hash so two pages may legitimately share identical text
 * without one of them being rejected by the unique index.
 */
function hashChunk(sourceUrl: string, content: string): string {
    return createHash("sha256").update(`${sourceUrl}\n${content}`, "utf8").digest("hex");
}

const PAGES = [
    // Foundations — conceptual, high-value
    { title: "Foundations: Overview", url: "https://ai-sdk.dev/docs/foundations/overview.md" },
    { title: "Foundations: Providers and Models", url: "https://ai-sdk.dev/docs/foundations/providers-and-models.md" },
    { title: "Foundations: Prompts", url: "https://ai-sdk.dev/docs/foundations/prompts.md" },
    { title: "Foundations: Tools", url: "https://ai-sdk.dev/docs/foundations/tools.md" },
    { title: "Foundations: Streaming", url: "https://ai-sdk.dev/docs/foundations/streaming.md" },

    // Getting started
    { title: "Getting Started: Next.js App Router", url: "https://ai-sdk.dev/docs/getting-started/nextjs-app-router.md" },
    { title: "Navigating the Library", url: "https://ai-sdk.dev/docs/getting-started/navigating-the-library.md" },

    // Core — the meat
    { title: "Core: Overview", url: "https://ai-sdk.dev/docs/ai-sdk-core/overview.md" },
    { title: "Core: Generating Text", url: "https://ai-sdk.dev/docs/ai-sdk-core/generating-text.md" },
    { title: "Core: Generating Structured Data", url: "https://ai-sdk.dev/docs/ai-sdk-core/generating-structured-data.md" },
    { title: "Core: Tool Calling", url: "https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling.md" },
    { title: "Core: MCP Tools", url: "https://ai-sdk.dev/docs/ai-sdk-core/mcp-tools.md" },
    { title: "Core: Prompt Engineering", url: "https://ai-sdk.dev/docs/ai-sdk-core/prompt-engineering.md" },
    { title: "Core: Settings", url: "https://ai-sdk.dev/docs/ai-sdk-core/settings.md" },
    { title: "Core: Embeddings", url: "https://ai-sdk.dev/docs/ai-sdk-core/embeddings.md" },
    { title: "Core: Reranking", url: "https://ai-sdk.dev/docs/ai-sdk-core/reranking.md" },
    { title: "Core: Error Handling", url: "https://ai-sdk.dev/docs/ai-sdk-core/error-handling.md" },
    { title: "Core: Middleware", url: "https://ai-sdk.dev/docs/ai-sdk-core/middleware.md" },
    { title: "Core: Telemetry", url: "https://ai-sdk.dev/docs/ai-sdk-core/telemetry.md" },

    // Agents
    { title: "Agents: Overview", url: "https://ai-sdk.dev/docs/agents/overview.md" },
    { title: "Agents: Building Agents", url: "https://ai-sdk.dev/docs/agents/building-agents.md" },
    { title: "Agents: Workflow Patterns", url: "https://ai-sdk.dev/docs/agents/workflows.md" },
    { title: "Agents: Loop Control", url: "https://ai-sdk.dev/docs/agents/loop-control.md" },
    { title: "Agents: Memory", url: "https://ai-sdk.dev/docs/agents/memory.md" },

    // UI
    { title: "UI: Overview", url: "https://ai-sdk.dev/docs/ai-sdk-ui/overview.md" },
    { title: "UI: Chatbot", url: "https://ai-sdk.dev/docs/ai-sdk-ui/chatbot.md" },
    { title: "UI: Chatbot Tool Usage", url: "https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage.md" },
    { title: "UI: Streaming Custom Data", url: "https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data.md" },
    { title: "UI: Error Handling", url: "https://ai-sdk.dev/docs/ai-sdk-ui/error-handling.md" },

    // Reference — the functions you actually use
    { title: "Reference: streamText", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text.md" },
    { title: "Reference: generateText", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/generate-text.md" },
    { title: "Reference: embed", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/embed.md" },
    { title: "Reference: embedMany", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/embed-many.md" },
    { title: "Reference: tool", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/tool.md" },
    { title: "Reference: useChat", url: "https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat.md" },
    { title: "Reference: convertToModelMessages", url: "https://ai-sdk.dev/docs/reference/ai-sdk-ui/convert-to-model-messages.md" },

    // Migration — explains the v7 changes you hit
    { title: "Migration: AI SDK 6 to 7", url: "https://ai-sdk.dev/docs/migration-guides/migration-guide-7-0.md" },
];

const WRITE = process.argv.includes("--write");
const EMBED_BATCH = 50;
const DELETE_BATCH = 200;
const PAGE_SIZE = 1000; // PostgREST caps a single select; page through it explicitly.

type FreshChunk = {
    content: string;
    source_url: string;
    title: string;
    content_hash: string;
};

/** Read every existing hash for the pages we manage. Paged — do not assume one request covers it. */
async function fetchExistingHashes(urls: string[]): Promise<Set<string>> {
    const hashes = new Set<string>();
    for (let from = 0; ; from += PAGE_SIZE) {
        const { data, error } = await supabase
            .from("documents")
            .select("content_hash")
            .in("source_url", urls)
            .range(from, from + PAGE_SIZE - 1);

        if (error) throw new Error(`Reading existing hashes failed: ${error.message}`);

        // Same generics gap as the insert path below: the hand-written Database type
        // isn't rich enough for PostgREST's inferred select shape, so narrow explicitly.
        const rows = (data ?? []) as unknown as { content_hash: string }[];
        if (rows.length === 0) break;

        for (const row of rows) hashes.add(row.content_hash);
        if (rows.length < PAGE_SIZE) break;
    }
    return hashes;
}

async function main() {
    // ── 1. Fetch + chunk every page ──────────────────────────────
    const fresh: FreshChunk[] = [];
    const perPage: { title: string; chunks: number }[] = [];

    for (const page of PAGES) {
        const res = await fetch(page.url);
        // A script should crash loudly. Half-ingested corpora are worse than none.
        if (!res.ok) throw new Error(`Failed to fetch ${page.url}: ${res.status}`);

        const markdown = stripBoilerplate(await res.text());
        const chunks = chunkPage(markdown);

        perPage.push({ title: page.title, chunks: chunks.length });
        for (const content of chunks) {
            fresh.push({
                content,
                source_url: page.url,
                title: page.title,
                content_hash: hashChunk(page.url, content),
            });
        }
    }

    // ── 2. Dedupe within this run ────────────────────────────────
    // Reference pages repeat parameter blocks verbatim. Identical text from the same
    // page hashes identically, and inserting both would violate the unique index.
    const byHash = new Map<string, FreshChunk>();
    for (const chunk of fresh) if (!byHash.has(chunk.content_hash)) byHash.set(chunk.content_hash, chunk);
    const dupesDropped = fresh.length - byHash.size;

    // ── 3. Diff against what is already stored ───────────────────
    const urls = PAGES.map((p) => p.url);
    const existing = await fetchExistingHashes(urls);

    const toInsert = [...byHash.values()].filter((c) => !existing.has(c.content_hash));
    const toDelete = [...existing].filter((h) => !byHash.has(h));
    const unchanged = byHash.size - toInsert.length;

    console.table(perPage);
    console.log(
        `pages ${PAGES.length}  |  chunks ${fresh.length}` +
        (dupesDropped ? ` (− ${dupesDropped} duplicate)` : "")
    );
    console.log(
        `unchanged ${unchanged}  ·  new ${toInsert.length}  ·  stale ${toDelete.length}`
    );

    // ── 4. Stop here unless writing was explicitly requested ─────
    if (!WRITE) {
        console.log(
            toInsert.length || toDelete.length
                ? "dry run — nothing written. Apply with: npm run ingest -- --write"
                : "dry run — corpus is already up to date. Nothing to do."
        );
        return;
    }

    if (!toInsert.length && !toDelete.length) {
        console.log("already up to date — no embeddings requested, no rows touched.");
        return;
    }

    // ── 5. Embed and insert only what is new ─────────────────────
    for (let i = 0; i < toInsert.length; i += EMBED_BATCH) {
        const batch = toInsert.slice(i, i + EMBED_BATCH);

        const { embeddings } = await embedMany({
            model: openai.embedding("text-embedding-3-small"),
            values: batch.map((r) => r.content),
        });

        const { error } = await supabase
            .from("documents")
            .insert(batch.map((r, j) => ({ ...r, embedding: embeddings[j] })) as never);

        if (error) throw new Error(`Insert failed at row ${i}: ${error.message}`);
        console.log(`  inserted ${Math.min(i + EMBED_BATCH, toInsert.length)}/${toInsert.length}`);
    }

    // ── 6. Remove rows whose source text no longer exists ────────
    // Deletion runs last: a crash mid-insert leaves the old chunks still serving queries.
    for (let i = 0; i < toDelete.length; i += DELETE_BATCH) {
        const batch = toDelete.slice(i, i + DELETE_BATCH);
        const { error } = await supabase.from("documents").delete().in("content_hash", batch);
        if (error) throw new Error(`Delete failed at row ${i}: ${error.message}`);
        console.log(`  deleted ${Math.min(i + DELETE_BATCH, toDelete.length)}/${toDelete.length}`);
    }

    console.log("done.");
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
