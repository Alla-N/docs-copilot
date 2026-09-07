/**
 * Threshold sweep — where does the numeric gate actually sit, before and after HyDE?
 *
 *   npm run exp:sweep
 *
 * Every number in the README's "Measured results" that concerns the threshold, the reranker
 * or the answerable/unanswerable gap comes from this script, so it can be re-run when the
 * corpus, the embedder, the reranker or the planner changes. (The first versions of those
 * numbers came from probe routes deleted in 50aeb99 — a number without a runnable source is
 * a rumour.)
 *
 * What it does, per eval case (12 answerable with an expected page, 6 must-refuse):
 *   A. PRE-HyDE   — embed the raw question, rerank with it. The planner is bypassed. This is
 *                   the pipeline the 0.30 threshold was calibrated on.
 *   B. POST-HyDE  — the real pipeline: planner → embed each sub-query's hypothetical → rerank
 *                   with the real sub-query. Off-topic messages are gated by the planner and
 *                   never retrieve; they are reported as such, not scored.
 * For both, the reranker is asked for ALL 40 candidates with no threshold, so the full score
 * distribution is visible — production only ever sees the top 5 above 0.30.
 *
 * Reported:
 *   - per case: best cosine and best rerank score of the EXPECTED page (answerable) or of ANY
 *     page (must-refuse), in A and in B;
 *   - answerable-min vs unanswerable-max, for cosine and rerank, in A and B — the gap the
 *     threshold has to sit in. If min ≤ max the threshold cannot separate them by itself;
 *   - "rerank rescued": answerable cases whose expected page is outside the cosine top-5
 *     but inside the rerank top-5;
 *   - recall@N of the expected page in rerank order (B) for N = 1, 3, 5, 10;
 *   - a threshold sweep (B, rerank): recall and guardrails-held-by-threshold at each cut.
 *
 * Cost: ~1 + (#sub-queries) rerank calls per case (~40 total ≈ $0.08 on a production Cohere
 * key) and a few cents of embeddings/planner calls. On a TRIAL key set RERANK_INTERVAL_MS=6500
 * (10 calls/min) and budget ~5 minutes — and remember the trial's 1,000 calls/MONTH cap.
 */
import { retrieve, VECTOR_CANDIDATES } from "../../lib/retrieve";
import { planQuery } from "../../lib/plan";
import { CASES, type EvalCase } from "../../evals/dataset";

const INTERVAL_MS = Number(process.env.RERANK_INTERVAL_MS ?? 250);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const ALL = { topN: VECTOR_CANDIDATES, threshold: 0 };

type Scored = { content: string; title: string; source_url: string; cosine: number; rerank: number };

/** Run one retrieval with the full ranking exposed, joined on chunk content. */
async function scoreAll(query: string, embedText: string): Promise<Scored[]> {
    const { candidates, relevant, mode } = await retrieve(query, embedText, ALL);
    if (mode !== "reranked") throw new Error("reranker unavailable — the sweep needs real rerank scores");
    const rerankByContent = new Map(relevant.map((r) => [r.content, r.score]));
    return candidates.map((c) => ({
        content: c.content,
        title: c.title,
        source_url: c.source_url,
        cosine: c.similarity,
        rerank: rerankByContent.get(c.content) ?? 0,
    }));
}

function matches(c: EvalCase, url: string): boolean {
    const wanted = Array.isArray(c.expectedSource) ? c.expectedSource : c.expectedSource ? [c.expectedSource] : [];
    return wanted.some((slug) => url.includes(slug));
}

/** Best score of the expected page (answerable) or of any page (must-refuse), plus ranks. */
function summarise(c: EvalCase, scored: Scored[]) {
    const pick = c.shouldAnswer === true ? scored.filter((s) => matches(c, s.source_url)) : scored;
    const best = (k: "cosine" | "rerank") => (pick.length ? Math.max(...pick.map((s) => s[k])) : NaN);
    const rank = (k: "cosine" | "rerank") => {
        const order = [...scored].sort((a, b) => b[k] - a[k]);
        const i = order.findIndex((s) => matches(c, s.source_url));
        return i < 0 ? Infinity : i + 1;
    };
    return { cosine: best("cosine"), rerank: best("rerank"), cosineRank: rank("cosine"), rerankRank: rank("rerank") };
}

type Row = {
    id: string;
    kind: "answerable" | "refuse";
    A: ReturnType<typeof summarise>;
    B: ReturnType<typeof summarise> | "gated";
};

async function main() {
    const answerable = CASES.filter((c) => c.shouldAnswer === true && !c.injection && c.expectedSource);
    const refuse = CASES.filter((c) => c.shouldAnswer === false && !c.injection);
    const cases = [...answerable, ...refuse];
    console.log(`sweep: ${answerable.length} answerable + ${refuse.length} must-refuse, ${VECTOR_CANDIDATES} candidates, all rerank scores\n`);

    const rows: Row[] = [];
    for (const c of cases) {
        const kind = c.shouldAnswer === true ? "answerable" : "refuse";

        // A — pre-HyDE: raw question embedded, planner bypassed.
        const A = summarise(c, await scoreAll(c.query, c.query));
        await sleep(INTERVAL_MS);

        // B — the real pipeline.
        const plan = await planQuery(c.query, c.history ?? []);
        let B: Row["B"];
        if (plan.intent !== "search") {
            B = "gated";
        } else {
            const perQuery: Scored[][] = [];
            for (const q of plan.queries) {
                perQuery.push(await scoreAll(q.query, q.hypothetical || q.query));
                await sleep(INTERVAL_MS);
            }
            // Union across sub-queries: keep each chunk's best scores, as plannedRetrieve does.
            const byKey = new Map<string, Scored>();
            for (const s of perQuery.flat()) {
                const key = `${s.source_url}::${s.content}`;
                const e = byKey.get(key);
                if (!e) byKey.set(key, { ...s });
                else { e.rerank = Math.max(e.rerank, s.rerank); e.cosine = Math.max(e.cosine, s.cosine); }
            }
            B = summarise(c, [...byKey.values()]);
        }
        rows.push({ id: c.id, kind, A, B });

        const fmt = (x: number) => (Number.isNaN(x) ? "  —  " : x.toFixed(3));
        const b = B === "gated" ? "gated by planner" : `cos ${fmt(B.cosine)}  rr ${fmt(B.rerank)}  (rank ${B.rerankRank})`;
        console.log(`  ${c.id.padEnd(24)} ${kind.padEnd(10)} A: cos ${fmt(A.cosine)}  rr ${fmt(A.rerank)}   B: ${b}`);
    }

    // ── Gap: answerable-min vs unanswerable-max ────────────────────────────
    const gap = (cond: "A" | "B", k: "cosine" | "rerank") => {
        const val = (r: Row) => (cond === "A" ? r.A : r.B === "gated" ? null : r.B);
        const ans = rows.filter((r) => r.kind === "answerable").map(val).filter((v): v is Row["A"] => v !== null).map((v) => v[k]);
        const ref = rows.filter((r) => r.kind === "refuse").map(val).filter((v): v is Row["A"] => v !== null).map((v) => v[k]);
        const min = Math.min(...ans);
        const max = Math.max(...ref);
        return { min, max, ratio: max > 0 ? min / max : Infinity, n: `${ans.length}/${ref.length}` };
    };
    console.log("\nanswerable-min vs unanswerable-max (the room the threshold has):");
    console.table(
        (["A", "B"] as const).flatMap((cond) =>
            (["cosine", "rerank"] as const).map((k) => {
                const g = gap(cond, k);
                return {
                    condition: cond === "A" ? "A pre-HyDE" : "B post-HyDE (real pipeline)",
                    score: k,
                    "answerable min": g.min.toFixed(3),
                    "unanswerable max": g.max.toFixed(3),
                    "gap (min/max)": `${g.ratio.toFixed(2)}×`,
                    separable: g.min > g.max ? "yes" : "NO",
                    "n (ans/ref)": g.n,
                };
            })
        )
    );
    const gatedIds = rows.filter((r) => r.B === "gated").map((r) => r.id);
    if (gatedIds.length)
        console.log(`(B excludes ${gatedIds.length} must-refuse case(s) the planner gated: ${gatedIds.join(", ")} — the threshold never sees them)`);

    // ── Rerank rescue: outside cosine top-5, inside rerank top-5 ───────────
    for (const cond of ["A", "B"] as const) {
        const rescued: string[] = [];
        const lost: string[] = [];
        for (const r of rows.filter((x) => x.kind === "answerable")) {
            const v = cond === "A" ? r.A : r.B;
            if (v === "gated") continue;
            if (v.cosineRank > 5 && v.rerankRank <= 5) rescued.push(r.id);
            if (v.cosineRank <= 5 && v.rerankRank > 5) lost.push(r.id);
        }
        const outside = rows.filter((x) => x.kind === "answerable" && (cond === "A" ? x.A : x.B) !== "gated" && (cond === "A" ? x.A : (x.B as Row["A"])).cosineRank > 5).length;
        console.log(`\nrerank rescue (${cond}): expected page outside cosine top-5 in ${outside} case(s); rerank pulled ${rescued.length} of them into its top-5${rescued.length ? ` (${rescued.join(", ")})` : ""}${lost.length ? `; pushed ${lost.length} OUT (${lost.join(", ")})` : ""}`);
    }

    // ── recall@N in rerank order, real pipeline ────────────────────────────
    const bRows = rows.filter((r) => r.kind === "answerable" && r.B !== "gated");
    console.log("\nrecall@N of the expected page, rerank order, real pipeline (B):");
    console.table(
        [1, 3, 5, 10].map((n) => ({
            N: n,
            "recall@N": `${bRows.filter((r) => (r.B as Row["A"]).rerankRank <= n).length}/${bRows.length}`,
        }))
    );

    // ── threshold sweep on rerank scores, real pipeline ────────────────────
    const refRows = rows.filter((r) => r.kind === "refuse");
    console.log("\nthreshold sweep, rerank score, real pipeline (B):");
    console.table(
        [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7].map((t) => ({
            threshold: t.toFixed(2),
            "answerable recall": `${bRows.filter((r) => (r.B as Row["A"]).rerank >= t).length}/${bRows.length}`,
            "refuse: held by threshold": `${refRows.filter((r) => r.B !== "gated" && r.B.rerank < t).length}/${refRows.filter((r) => r.B !== "gated").length}`,
            "refuse: gated by planner": `${refRows.filter((r) => r.B === "gated").length}/${refRows.length}`,
        }))
    );
    console.log("\nRead the sweep as: at threshold t, which answerable cases would lose their page, and which must-refuse cases would never reach the prompt.");
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
