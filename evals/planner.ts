/**
 * Planner-only eval — the check specs/query-planner.md promised ("planQuery(...) must not
 * return a sub-query that retrieves AI SDK chunks") and the main suite could not give.
 *
 *   npm run eval:planner
 *
 * The main harness sees the planner only through its consequences (which chunks came back,
 * whether the answer refused). That let the planner break invariant #7 for weeks without a
 * red line: "how do I deploy to AWS" became "deploy the Vercel AI SDK to AWS", HyDE embedded
 * the doc-shaped hypothetical, five chunks reached the model, and guard-aws still PASSED
 * because the prompt refused. The gate held; the wrong layer was holding it.
 *
 * This file asserts the planner's OUTPUT directly: intent, number of sub-queries, and strings
 * the sub-queries must / must not contain. No retrieval, no answer generation — one
 * gpt-4o-mini call per run, so the whole suite costs well under a cent and needs no throttle.
 *
 * Same verdict rules as evals/run.ts: N runs per case (planner output is model output),
 * PASS only if every run passes, FLAKY if some do, FAIL if none. 23 cases × 5 runs
 * (the cases live in evals/planner-cases.ts).
 */
import { planQuery } from "../lib/plan";
import { PLANNER_CASES, type PlannerCase } from "./planner-cases";

// 5, not 3: a planner call costs a fraction of a cent, and the failure mode this suite guards
// (the off-topic gate swallowing shorthand) showed up on a 4th sample after 3 had passed.
const RUNS = Number(process.env.PLANNER_RUNS ?? 5);
const ONLY = (process.env.EVAL_ONLY ?? "").split(",").map((x) => x.trim()).filter(Boolean);

type Verdict = "PASS" | "FAIL" | "FLAKY";

function check(c: PlannerCase, plan: Awaited<ReturnType<typeof planQuery>>): string[] {
    const problems: string[] = [];
    const wanted = Array.isArray(c.expect.intent) ? c.expect.intent : [c.expect.intent];
    if (!wanted.includes(plan.intent)) problems.push(`intent ${plan.intent}, expected ${wanted.join("|")}`);
    if (c.expect.count !== undefined && plan.queries.length !== c.expect.count)
        problems.push(`${plan.queries.length} queries, expected ${c.expect.count}`);
    const joined = plan.queries.map((q) => q.query.toLowerCase()).join(" | ");
    for (const s of c.expect.mustContain ?? []) if (!joined.includes(s.toLowerCase())) problems.push(`missing "${s}"`);
    for (const s of c.expect.mustNotContain ?? []) if (joined.includes(s.toLowerCase())) problems.push(`contains "${s}"`);
    return problems;
}

async function main() {
    const active = ONLY.length ? PLANNER_CASES.filter((c) => ONLY.includes(c.id)) : PLANNER_CASES;
    console.log(`planner eval: ${active.length} cases × ${RUNS} runs, temp 0\n`);

    let failures = 0;
    for (const c of active) {
        const plans = await Promise.all(Array.from({ length: RUNS }, () => planQuery(c.query, c.history ?? [])));
        const problems = plans.map((p) => check(c, p));
        const passes = problems.filter((p) => p.length === 0).length;
        const verdict: Verdict = passes === RUNS ? "PASS" : passes === 0 ? "FAIL" : "FLAKY";
        if (verdict !== "PASS") failures++;

        console.log(`  ${verdict.padEnd(5)} ${c.id.padEnd(24)} ${verdict === "PASS" ? "" : `${passes}/${RUNS}`}`);
        if (verdict !== "PASS") {
            // Show the run that broke, with what the planner actually produced. Look, don't guess.
            const i = problems.findIndex((p) => p.length > 0);
            console.log(`         ↳ ${problems[i].join("; ")}`);
            console.log(`         ↳ planner → ${plans[i].intent} ${JSON.stringify(plans[i].queries.map((q) => q.query))}`);
        }
    }

    console.log(`\n${failures ? `${failures} failing` : "all green"}.`);
    if (failures) process.exit(1);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
