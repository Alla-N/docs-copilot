/**
 * Retrieval golden file: what the TypeScript retrieve() returns, frozen for the Python port.
 *
 *   npm run exp:golden
 *
 * Step 1e of the Windward plan. For each recall case of evals/dataset.ts (answerable, not an
 * injection, not parked: the same 12 the harness scores recall on), this runs the planner
 * ONCE, then retrieve() on every sub-query, and writes agent/tests/golden/retrieval-golden.json:
 *   - the planner's sub-queries and the text retrieve() EMBEDDED for each (the HyDE
 *     hypothetical, or the query when there is none). Frozen: the Python side retrieves for
 *     the SAME strings, so a difference means the port, not HyDE;
 *   - per sub-query: all vector candidates in cosine order, what survived rerank and the 0.30
 *     threshold, the mode, and how long retrieve() took.
 * agent/tests/test_golden_parity.py reads it:  cd agent && uv run pytest -m integration -s
 *
 * Once phase 2 deletes the TypeScript retrieve(), this file is the only record of what it
 * returned, which makes it the regression fixture for the Python retrieval from then on.
 *
 * Chunks are keyed by hashChunk() (lib/content-hash.ts, invariant 1) cut to 16 hex characters:
 * the Python side computes the same key, and the short form keeps the file small. A prefix
 * collision would silently merge two chunks, so the script checks for one and stops.
 *
 * Not the eval harness and not production: it calls planQuery + retrieve directly, as
 * threshold-sweep.ts does, and scores nothing. Sub-queries run one after another (production
 * runs them in parallel), so each retrieve() time is that call alone.
 *
 * Cost: 12 planner calls, one embedding and one rerank per sub-query (about 25 of each): a few
 * cents. On a Cohere TRIAL key set RERANK_INTERVAL_MS=6500 (10 calls/min).
 */
import { execSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { CASES } from "../../evals/dataset";
import { hashChunk } from "../../lib/content-hash";
import { planQuery } from "../../lib/plan";
import {
    COSINE_THRESHOLD,
    RERANK_THRESHOLD,
    RERANK_TOP_N,
    VECTOR_CANDIDATES,
    retrieve,
} from "../../lib/retrieve";

const OUT = "agent/tests/golden/retrieval-golden.json";
const KEY_LENGTH = 16;
const INTERVAL_MS = Number(process.env.RERANK_INTERVAL_MS ?? 250);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

type GoldenSubQuery = {
    /** The planner's standalone query. retrieve() RERANKS with this. */
    query: string;
    /** What retrieve() EMBEDDED: the HyDE hypothetical, or the query when there was none. */
    embedText: string;
    mode: string;
    /** Wall time of this retrieve() call: embed + vector search + rerank. */
    ms: number;
    /** Every vector candidate, cosine order: [key, similarity]. */
    candidates: [string, number][];
    /** What survived rerank + threshold, reranker order. What the model would see. */
    relevant: { key: string; score: number }[];
};

type GoldenCase = {
    id: string;
    question: string;
    expectedSource: string[];
    /** "fallback" when planQuery failed and fell back to the raw question (usage unknown). */
    planner: "ok" | "fallback";
    subQueries: GoldenSubQuery[];
};

function gitCommit(): { commit: string; dirty: boolean } {
    try {
        const run = (cmd: string) => execSync(cmd, { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
        return { commit: run("git rev-parse --short HEAD"), dirty: run("git status --porcelain") !== "" };
    } catch {
        return { commit: "unknown", dirty: true };
    }
}

async function main() {
    // Same filter as the recall line of evals/run.ts.
    const cases = CASES.filter((c) => c.shouldAnswer === true && !c.injection && !c.expectFail);
    console.log(`golden: ${cases.length} recall cases, ${VECTOR_CANDIDATES} candidates, top ${RERANK_TOP_N} over ${RERANK_THRESHOLD}\n`);

    const chunks: Record<string, { source_url: string; title: string }> = {};
    const fullHashOf = new Map<string, string>();
    const keyOf = (sourceUrl: string, content: string): string => {
        const full = hashChunk(sourceUrl, content);
        const key = full.slice(0, KEY_LENGTH);
        const seen = fullHashOf.get(key);
        if (seen !== undefined && seen !== full) {
            throw new Error(`key collision on ${key}: raise KEY_LENGTH (and the Python side with it)`);
        }
        fullHashOf.set(key, full);
        return key;
    };

    const out: GoldenCase[] = [];
    for (const c of cases) {
        if (!c.expectedSource) throw new Error(`${c.id} has no expectedSource, so recall cannot be scored`);

        const plan = await planQuery(c.query, c.history ?? []);
        if (plan.intent !== "search") {
            throw new Error(`${c.id}: the planner said ${plan.intent}; a recall case must retrieve`);
        }

        const subQueries: GoldenSubQuery[] = [];
        for (const q of plan.queries) {
            const embedText = q.hypothetical || q.query; // exactly what plannedRetrieve passes
            const t0 = performance.now();
            const r = await retrieve(q.query, embedText);
            const ms = Math.round((performance.now() - t0) * 10) / 10;

            // A fallback run is a different pipeline (cosine order, 0.45 gate): not an oracle.
            if (r.mode !== "reranked") {
                throw new Error(`${c.id}: retrieve() ran in ${r.mode} mode; fix the reranker and rerun`);
            }
            for (const cand of r.candidates) {
                chunks[keyOf(cand.source_url, cand.content)] = { source_url: cand.source_url, title: cand.title };
            }
            subQueries.push({
                query: q.query,
                embedText,
                mode: r.mode,
                ms,
                candidates: r.candidates.map((cand): [string, number] => [keyOf(cand.source_url, cand.content), cand.similarity]),
                relevant: r.relevant.map((chunk) => ({ key: keyOf(chunk.source_url, chunk.content), score: chunk.score })),
            });
            await sleep(INTERVAL_MS);
        }

        const planner = plan.usage.inputTokens === null ? "fallback" : "ok";
        out.push({
            id: c.id,
            question: c.query,
            expectedSource: Array.isArray(c.expectedSource) ? c.expectedSource : [c.expectedSource],
            planner,
            subQueries,
        });
        const kept = subQueries.map((s) => s.relevant.length).join("+");
        const note = planner === "fallback" ? "   (planner FELL BACK to the raw question)" : "";
        console.log(`  ${c.id.padEnd(24)} ${subQueries.length} sub-quer${subQueries.length === 1 ? "y  " : "ies"}  kept ${kept}${note}`);
    }

    const { commit, dirty } = gitCommit();
    const golden = {
        date: new Date().toISOString(),
        commit,
        dirty,
        knobs: {
            candidates: VECTOR_CANDIDATES,
            rerankTopN: RERANK_TOP_N,
            threshold: RERANK_THRESHOLD,
            cosineThreshold: COSINE_THRESHOLD,
            keyLength: KEY_LENGTH,
        },
        cases: out,
        chunks,
    };

    // Pretty JSON, but one [key, similarity] pair per line: about 100 lines per sub-query
    // instead of 400, and a diff of two golden files still shows which candidate moved.
    const text = JSON.stringify(golden, null, 2).replace(
        /\[\n\s+("[0-9a-f]+"),\n\s+(-?[0-9.e+-]+)\n\s+\]/g,
        "[$1, $2]"
    );
    mkdirSync(dirname(OUT), { recursive: true });
    writeFileSync(OUT, text + "\n");
    const nSub = out.reduce((n, c) => n + c.subQueries.length, 0);
    console.log(`\nwrote ${OUT}: ${out.length} cases, ${nSub} sub-queries, ${Object.keys(chunks).length} distinct chunks (commit ${commit}${dirty ? ", dirty tree" : ""})`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
