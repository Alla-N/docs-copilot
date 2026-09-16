/**
 * Eval harness — the orchestrator.
 *
 *   npm run eval                deterministic metrics
 *   EVAL_JUDGE=1 npm run eval   + LLM faithfulness (see evaluators/faithfulness.ts)
 *   EVAL_RUNS=0  npm run eval    retrieval-only: no ANSWER generation. It still pays one
 *                                planner call (gpt-4o-mini, structured output), one embed and
 *                                one rerank per sub-query — roughly a cent per run, not zero.
 *
 * Since 5.4b this file runs the suite and reports it; it does not decide anything. The layout:
 *
 *   datasets/    the labelled cases, and what each set claims (its denominators, its exit code)
 *   evaluators/  pure functions over cases and results, the verdicts computed in code
 *   targets/     the two pipelines a case can be run against
 *   record.ts    what a run leaves on disk, and how an older one is read back
 *   diff.ts      npm run eval:diff — what moved between two stored runs
 *
 * Per-case criteria, chosen so each measures the property that actually matters:
 *   RETRIEVAL RECALL     did the expected doc survive rerank + threshold?
 *   ANSWER vs REFUSE     shouldAnswer: true / false / "either"
 *   mustNotContain       forbidden strings (leaks, injected answers) — checked every run
 *   mustContain          required strings — proves a multi-part answer covered every intent
 *   FAITHFULNESS         (opt-in) is the answer grounded in its own chunks?
 *
 * Two targets. The default runs the pipeline in-process (plannedRetrieve + generateText, the
 * functions the route calls). EVAL_TARGET=python sends every question to the Python agent
 * service over HTTP, as the route forwards it: same cases, same criteria, same verdicts, and the
 * differences that come with a service are reported, not hidden (every run retrieves again;
 * history is replayed as real turns; the judge reads the chunk texts back from the trace, since
 * the stream carries pages only; cost is MEASURED from the service's own query_log rows).
 * `targets/index.ts` lists what each one can and cannot measure.
 *
 *   EVAL_TARGET=python AGENT_URL=http://127.0.0.1:8000 npm run eval
 *
 * Refusal is detected against REFUSAL_MESSAGE, the same constant the prompt instructs —
 * reword it there and this follows, rather than silently scoring every refusal as an answer.
 *
 * Retrieval runs ONCE per case on the in-process target. It is *nearly* deterministic:
 * embedding, vector search and rerank are, but the planner's HyDE hypothetical is model output,
 * and a different hypothetical can reorder near-tied pages (the push gate retries a recall miss
 * once, and says so — see the RUNS === 0 block). Generation runs N times, because that is
 * where non-determinism lives: temp 0 lowers variance, it does not remove it. A case that
 * passes 2/3 is FLAKY, not passing. Adversarial cases run more times (an attack that works
 * 1-in-8 is a working attack). A parked, known-failing case is marked `expectFail`: it runs
 * and reports but does not fail the suite, and is flagged if it ever starts passing.
 *
 * A case can also fail to RUN at all: a gateway timeout from the database, a dropped
 * connection, a provider outage. That is not a verdict about the pipeline, so it is not
 * scored as one. The case is recorded as ERROR with its message, the run CONTINUES, errored
 * cases are excluded from every headline denominator (exactly as expectFail cases are), and
 * the whole run is then reported INCOMPLETE and exits 4 — a run that could not run every
 * case is never a baseline. Before this, one such error threw out of the loop and discarded
 * the other 26 cases: eval #34, 2026-09-14, `Retrieval failed: Gateway Timeout` on case 2 of
 * 27, exit 1 and no report. A red run now says which KIND of red it was.
 *
 *   EVAL_FAULT=<id,...>          make those cases throw a simulated gateway timeout, without
 *                                spending anything, to exercise the ERROR path on purpose.
 *
 * Exit codes: 0 green; 1 a real failure (a failing case, an unfaithful answer, a parked case
 * that now passes, a retrieval regression); 2 bad configuration; 3 the reranker was
 * unavailable, so the run measured a different pipeline; 4 at least one case could not run;
 * 5 the golden set is green and the labelled GitHub set is not.
 */
import { execSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";

import { isRefusal, RERANK_THRESHOLD, VECTOR_CANDIDATES, RERANK_TOP_N } from "../lib/retrieve";
import { plannedRetrieve } from "../lib/plan";
import { CASES } from "./datasets/golden";
import { FROZEN_AT, GITHUB_CASES } from "./datasets/github";
import {
    aggregateFaithfulness,
    erroredResult,
    evaluateCoverage,
    evaluateFalseRefusals,
    evaluateGitHubAccuracy,
    evaluateGitHubProcess,
    evaluateGuardrails,
    evaluateInjection,
    evaluateRecall,
    expectedFound,
    githubVerdict,
    partition,
} from "./evaluators";
import type { GitHubResult, GoldenContext, Result } from "./evaluators/types";
import { runCaseInProcess } from "./targets/in-process";
import {
    judgeAgentRuns,
    newAgentSession,
    runCaseOnAgent,
    runGitHubCases,
    type AgentSession,
} from "./targets/agent-run";
import {
    agentTarget,
    costOfThreads,
    historyFieldStatus,
    turnFactsOfThreads,
    type AgentTarget,
    type RunCost,
    type TurnFacts,
} from "./targets/agent-service";
import { formatRatio, SCHEMA_VERSION } from "./record";

const RUNS = Number(process.env.EVAL_RUNS ?? 3);

/** "ts" (default): the pipeline in-process. "python": the agent service over HTTP. */
const TARGET = process.env.EVAL_TARGET ?? "ts";
if (TARGET !== "ts" && TARGET !== "python") {
    console.error(`EVAL_TARGET must be "ts" or "python", not "${TARGET}".`);
    process.exit(2);
}

/**
 * Adversarial cases get more attempts than normal ones, because they measure a
 * different thing. For a normal case you want typical behaviour, and 3 runs is plenty.
 * For an attack you want to know whether it can EVER succeed — an injection that works
 * one time in five is a working injection, and a clean 3-run sample means "did not
 * reproduce", not "resistant".
 *
 * Found the hard way: inj-prompt-extract passed the harness and then leaked the entire
 * system prompt when tried by hand.
 */
const ADVERSARIAL_RUNS = Number(process.env.EVAL_ADVERSARIAL_RUNS ?? 8);

/** EVAL_ONLY=id1,id2 runs just those cases — for investigating a failure without paying
 *  for the whole suite. Metrics then report over the subset, which is fine for debugging. */
const ONLY = (process.env.EVAL_ONLY ?? "").split(",").map((x) => x.trim()).filter(Boolean);
// Fault injection, for testing THIS FILE. A case named here throws the shape of error a
// database gateway timeout has, before spending anything, so the ERROR path can be exercised
// on purpose instead of waiting for the next outage. run.ts cannot be unit tested (main()
// runs at import), so this is how its failure handling is proved.
const FAULT = (process.env.EVAL_FAULT ?? "").split(",").map((x) => x.trim()).filter(Boolean);

/**
 * Faithfulness is opt-in: EVAL_JUDGE=1 npm run eval
 * It costs one extra model call per answered case, and — unlike every other metric
 * here — the number comes from a model rather than a comparison. Calibrated 2026-09-07 at
 * 0/12 false alarms, 0/23 missed lies, n=35 (see eval:calibrate) — after finding it at
 * 12/12 false alarms on the current answer style. Good enough to act on; publish with the n.
 */
const JUDGE = process.env.EVAL_JUDGE === "1";

/**
 * The GitHub labelled set (step 3.6): on by default on the Python target, EVAL_GITHUB=0 to skip.
 *
 * Its own runs knob, and a small default, because a run here is not a generation: it is a whole
 * subagent loop of schema lookups and query attempts, 12 to 18 seconds warm. Two runs per case is
 * NOT a per-case rate — 3.5b is emphatic that n=10 cannot separate 0.85 from 1.0, so n=2 cannot
 * either. It is two observations per case so that a case which differs between them is reported
 * as VARIED rather than silently as whichever run came first, and so that the SET-level number
 * (13 cases x 2) has something under it. The done-when runs the whole suite twice and pools.
 */
const GITHUB = process.env.EVAL_GITHUB !== "0";
const GITHUB_RUNS = Number(process.env.EVAL_GITHUB_RUNS ?? 2);

/** Widening the candidate pool costs latency on every production query. Price it. */
const retrievalMs: number[] = [];

/**
 * Pause between cases. With the production Cohere key this is politeness (250 ms); the
 * Cohere TRIAL key allows 10 rerank calls/minute, and on it an unthrottled run trips the
 * limit and reports a fake failure — set RERANK_INTERVAL_MS=6500 there. A harness that
 * fails for its own reasons teaches you nothing about the system.
 */
const RERANK_INTERVAL_MS = Number(process.env.RERANK_INTERVAL_MS ?? 250);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const ms = (xs: number[]) => (xs.length ? `${median(xs).toFixed(0)}ms` : "—");

// ---- the GitHub labelled set (step 3.6) ------------------------------------------------------

/**
 * The five measures of the 3.6 done-when, printed and returned for the results file.
 *
 * The numbers come from the two evaluators; this prints them and shapes the record. That split
 * is the point of 5.4b: `evaluateGitHubAccuracy` reads the TEXT and `evaluateGitHubProcess`
 * reads the blocks the service stored, and at the 3.6 baseline the second was at ceiling while
 * the first was 24 of 52. A suite of validity and cost would have reported that subagent as
 * working — which is the whole argument for a labelled set, and it landed on the first run.
 */
function reportGitHub(results: GitHubResult[], docs: Result[]): Record<string, unknown> {
    // A case that could not RUN is not evidence about the subagent, same rule as the golden set.
    const ran = results.filter((r) => r.verdict !== "ERROR");
    const accuracy = evaluateGitHubAccuracy(ran);
    const measures = evaluateGitHubProcess(ran, docs);
    const { routing, points } = measures;

    console.log(`\nGitHub set — the five measures (${measures.observations} observations over ${ran.length} cases)`);
    console.log(`  answer accuracy         ${formatRatio(accuracy)}`);
    console.log(`  first-try query valid   ${formatRatio(measures.firstTryValid)}   (of the turns whose subagent ran)`);
    console.log(`  valid after <= 2 repairs ${formatRatio(measures.validAfterRepairs)}`);
    console.log(
        `  points per question     median ${points.median ?? "—"}` +
        `  max ${points.max ?? "—"}  total ${points.total}`
    );
    console.log(
        `  routing accuracy        ${formatRatio(routing.overall)}` +
        `  (GitHub set ${formatRatio(routing.githubSet)}, golden set ${formatRatio(routing.goldenSet)})`
    );
    if (routing.cannedByPlanner) {
        console.log(
            `  canned by the planner   ${routing.cannedByPlanner} turn(s) never reached the router. Not scored as routing:\n` +
            `                          a turn that was never routed is not a turn routed to the documentation.`
        );
    }

    for (const r of results) {
        console.log(`  ${r.verdict.padEnd(6)} ${r.id.padEnd(24)} ${r.detail}`);
        // Always for a non-PASS, and always for the control, whose criterion is the approximate one.
        if (r.verdict !== "PASS" || !GITHUB_CASES.find((c) => c.id === r.id)?.shouldAnswer) {
            console.log(`         answer   ${r.sample.replace(/\s+/g, " ").slice(0, 400) || "(empty)"}`);
            if (r.query) console.log(`         query    ${r.query.replace(/\s+/g, " ").slice(0, 300)}`);
            if (r.evidence) console.log(`         evidence ${r.evidence.replace(/\s+/g, " ").slice(0, 400)}`);
            console.log(`         routes   ${r.routes.map((x) => x ?? "null").join(", ")}`);
            if (r.subagentRuns)
                console.log(
                    `         looked up ${r.lookups.join(", ")} type(s) [${r.typesRead.map((t) => t.join(" ") || "none").join("  |  ")}]` +
                    `; stages ${r.stages.map((s) => s.join(" > ")).join("  |  ")}`
                );
        }
    }

    return {
        frozenAt: FROZEN_AT,
        runs: GITHUB_RUNS,
        observations: measures.observations,
        accuracy: formatRatio(accuracy),
        firstTryValid: formatRatio(measures.firstTryValid),
        validAfterRepairs: formatRatio(measures.validAfterRepairs),
        points: { median: points.median, max: points.max, total: points.total },
        routing: {
            overall: formatRatio(routing.overall),
            githubSet: formatRatio(routing.githubSet),
            goldenSet: formatRatio(routing.goldenSet),
            cannedByPlanner: routing.cannedByPlanner,
        },
        cases: results.map((r) => ({
            id: r.id,
            verdict: r.verdict,
            correct: `${r.correct}/${r.runs}`,
            routes: r.routes,
            firstTryValid: `${r.firstTryValid}/${r.subagentRuns}`,
            ok: `${r.ok}/${r.subagentRuns}`,
            points: r.points,
            attempts: r.attempts,
            repairs: r.repairs,
            lookups: r.lookups,
            stages: r.stages,
            typesRead: r.typesRead,
            detail: r.detail,
            query: r.query,
            evidence: r.evidence ? r.evidence.slice(0, 600) : null,
            answer: r.sample.replace(/\s+/g, " ").slice(0, 600),
        })),
    };
}

/**
 * Run the labelled set and score it per case.
 *
 * The runner (targets/agent-run.ts) produces observations; the verdict rules live in
 * evaluators/github-answer.ts. This assembles the two into the per-case record the report and
 * the results file both read.
 */
async function githubResults(session: AgentSession): Promise<GitHubResult[]> {
    const cases = ONLY.length ? GITHUB_CASES.filter((c) => ONLY.includes(c.id)) : GITHUB_CASES;
    if (!cases.length) return [];
    console.log(
        `\nGitHub set: ${cases.length} labelled questions x ${GITHUB_RUNS} runs  |  answers frozen ${FROZEN_AT}\n` +
        `  (a per-case rate at n=${GITHUB_RUNS} is not a rate; a case that differs between its runs is VARIED)\n`
    );

    const observed = await runGitHubCases(session, cases, GITHUB_RUNS, RERANK_INTERVAL_MS);
    return observed.map((o) => {
        const withBlock = o.blocks.filter((f) => f?.github).map((f) => f!.github!);
        const routes = o.blocks.map((f) => f?.route ?? null);
        const { verdict, detail } = githubVerdict(o.case, o.runs, routes, withBlock, o.correct);
        return {
            id: o.case.id,
            label: o.case.route,
            runs: o.runs.length,
            correct: o.correct,
            routes,
            routedRight: routes.filter((r) => r === o.case.route).length,
            canned: routes.filter((r) => r === null).length,
            subagentRuns: withBlock.length,
            firstTryValid: withBlock.filter((b) => b.first_try_valid).length,
            ok: withBlock.filter((b) => b.ok).length,
            points: withBlock.map((b) => b.points_spent),
            attempts: withBlock.map((b) => b.attempts),
            repairs: withBlock.map((b) => b.repairs),
            lookups: withBlock.map((b) => b.lookups),
            stages: withBlock.map((b) => b.stages),
            typesRead: withBlock.map((b) => b.types_read ?? []),
            verdict,
            detail,
            sample: o.runs[0].text,
            query: withBlock[0]?.query ?? null,
            evidence: withBlock[0]?.evidence ?? null,
        };
    });
}

async function main() {
    // Duplicate ids fail silently otherwise: byId is a Map, so a second case with the same
    // id overwrites the first in the summary while both still run and both still cost money.
    // (This exact bug shipped once — two copies of multi-intent-noise.) Fail loudly instead.
    const ids = CASES.map((c) => c.id);
    const dupes = [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))];
    if (dupes.length) throw new Error(`Duplicate case ids in dataset: ${dupes.join(", ")}`);

    // Print the knobs. A pasted result that doesn't say what produced it is not a result.
    console.log(
        `eval: ${CASES.length} cases × ${RUNS} generations, temp 0  |  ` +
        `candidates ${VECTOR_CANDIDATES} → rerank top ${RERANK_TOP_N} → threshold ${RERANK_THRESHOLD}\n`
    );

    let target: AgentTarget | null = null;
    let session: AgentSession | null = null;
    let historyRefused: number | null = null;
    if (TARGET === "python") {
        // The service always answers, so there is no retrieval-only mode to be had over HTTP.
        // (EVAL_JUDGE=1 works here since step 2.7: the chunk texts come back from the trace.)
        if (RUNS === 0) {
            console.log("EVAL_TARGET=python runs the full suite only: no EVAL_RUNS=0, the service always answers.");
            process.exit(2);
        }
        target = agentTarget();
        session = newAgentSession(target);
        console.log(`target: the Python agent service at ${target.baseUrl}  (every run = plan + retrieve + answer, origin eval, threads eval-${session.runStamp}-*)\n`);
        // Checked once, before any case: the forged-history attack depends on it.
        historyRefused = await historyFieldStatus(target, `eval-${session.runStamp}-history-check`);
        if (historyRefused !== 422) {
            console.log(`✗ POST /chat accepted a request carrying history (status ${historyRefused}, expected 422). The service reads history only from its thread; stop and look.`);
            process.exit(1);
        }
    }

    const active = ONLY.length ? CASES.filter((c) => ONLY.includes(c.id)) : CASES;
    if (ONLY.length) console.log(`(EVAL_ONLY: ${active.map((c) => c.id).join(", ")})\n`);

    const results: Result[] = [];
    for (const [i, c] of active.entries()) {
        if (i > 0) await sleep(RERANK_INTERVAL_MS);
        // One case that cannot run must not cost the other 26 their results (eval #34).
        let r: Result;
        try {
            if (FAULT.includes(c.id)) throw new Error("Retrieval failed: Gateway Timeout (EVAL_FAULT)");
            if (session) {
                const outcome = await runCaseOnAgent(session, c, i, {
                    runs: RUNS,
                    adversarialRuns: ADVERSARIAL_RUNS,
                    rerankIntervalMs: RERANK_INTERVAL_MS,
                });
                // The in-process target measures one retrieval per case, canned cases included: run 1
                // here is the same population, so the headline latency line compares like with like.
                if (outcome.retrievalMs !== null) retrievalMs.push(outcome.retrievalMs);
                r = outcome.result;
            } else {
                const outcome = await runCaseInProcess(c, { runs: RUNS, adversarialRuns: ADVERSARIAL_RUNS, judge: JUDGE });
                retrievalMs.push(outcome.retrievalMs);
                r = outcome.result;
            }
        } catch (err) {
            r = erroredResult(c, err);
            console.log(`  ERROR ${c.id.padEnd(22)} could not run: ${r.detail}`);
        }
        results.push(r);
        // A run without the reranker measures a different pipeline (cosine order, 0.45
        // threshold) and must not be scored as this one. Found the expensive way: the Cohere
        // trial key's MONTHLY cap ran out mid-day, every case silently fell back, and the
        // harness reported two ordinary "expected doc not retrieved" failures on the two
        // pages whose cosine sits right at 0.45. Stop at the first fallback — every further
        // case would be the same non-signal, and each one burns three more rerank attempts.
        if (r.degraded) {
            console.log(
                `\n✗ ${r.id} ran WITHOUT the reranker (cosine fallback). This is not a valid run of the\n` +
                `  pipeline under test — no verdicts. Usually the Cohere key: the trial key allows\n` +
                `  10 calls/min AND 1,000 calls/month; a 27-case run is ~30 calls, the sweep ~40.\n` +
                `  Check the key, wait, or upgrade, then re-run.`
            );
            process.exit(3);
        }
        if (RUNS > 0) {
            console.log(`  ${r.verdict.padEnd(5)} ${r.id.padEnd(22)} ${r.detail}`);
            // Show what the model actually produced for adversarial cases AND any failure —
            // inferring behaviour from a verdict is how wrong criteria survive. Look, don't guess.
            if (c.injection || r.verdict !== "PASS")
                console.log(`         → ${r.sample.replace(/\s+/g, " ").slice(0, 1200)}`);
            // On a non-PASS, also show the run that actually broke expectation when it isn't
            // run 0 — otherwise a FLAKY 1/3 prints a perfectly good refusal and hides the
            // one reply that was scored the other way.
            if (r.verdict !== "PASS" && r.odd && r.odd.i !== 0)
                console.log(`         ↳ run ${r.odd.i + 1} (the odd one): ${r.odd.text.replace(/\s+/g, " ").slice(0, 1200)}`);
        }
    }

    if (session && JUDGE) await judgeAgentRuns(session, results, active, isRefusal);

    // What the router decided for each golden-set case, read back from the rows the service
    // wrote (db/009). Recorded per case because without it a documentation question answered
    // from GitHub is invisible in a stored result -- which is how 3.5's `changed-7` passed.
    if (session) {
        const threads = results.map((r) => r.threadId).filter((t): t is string => Boolean(t));
        const facts = threads.length ? await turnFactsOfThreads(threads) : new Map<string, TurnFacts>();
        for (const r of results) r.route = r.threadId ? (facts.get(r.threadId)?.route ?? null) : null;
        const misrouted = results.filter((r) => r.route === "both" || r.route === "github");
        if (misrouted.length)
            console.log(`\n  note: ${misrouted.length} documentation case(s) routed to GitHub as well: ${misrouted.map((r) => `${r.id} (${r.route})`).join(", ")}`);
    }

    let github: GitHubResult[] = [];
    if (session && GITHUB) github = await githubResults(session);

    const byId = new Map(results.map((r) => [r.id, r]));
    // A case that could not run is not evidence about the pipeline, so it is kept out of every
    // headline denominator, exactly as a parked (expectFail) case is and for the same reason:
    // scoring it drags a number down and reads as a regression that did not happen. It cannot
    // hide anything, because a single errored case makes the WHOLE run incomplete below.
    const errored = results.filter((r) => r.verdict === "ERROR");
    // Called after the real-failure exits in both modes: a genuine failure is still exit 1 and
    // still the headline. Exit 4 means nothing failed, but not everything was measured.
    const reportIncomplete = () => {
        if (!errored.length) return;
        console.log(`\n${errored.length} case(s) could NOT RUN. This run is INCOMPLETE and is not a baseline:`);
        for (const r of errored) console.log(`  ${r.id.padEnd(22)} ${r.detail}`);
        console.log(
            `  Infrastructure, not verdicts, so they are left out of the denominators above\n` +
            `  rather than counted as failures. Re-run. If the same case errors twice it is not\n` +
            `  the network, and the numbers above were measured on a smaller suite than they say.`
        );
        process.exit(4);
    };

    console.log();
    console.table(
        results.map((r) => ({
            case: r.id,
            "expected doc": r.retrieved,
            "chunks past threshold": r.chunks,
            "top rerank": r.topScore,
            [`answered`]: `${r.answered}/${r.runs}`,
            ...(JUDGE ? { faithful: r.faithful } : {}),
            verdict: r.verdict,
        }))
    );

    // THE POPULATIONS, decided once. Who counts toward which denominator is the half of a
    // fraction that is easiest to change by accident, and until 5.4b the rules were four filter
    // expressions inline here, each re-spelling the exclusions for itself. The rules and the
    // reasons now live in evaluators/partition.ts and evaluators/types.ts.
    const populations = partition(active, byId);
    const { answerable, parked } = populations;
    const ctx: GoldenContext = {
        active,
        results,
        byId,
        populations,
        generated: RUNS > 0,
        perRunRetrieval: session !== null,
    };

    const recall = evaluateRecall(ctx);
    const coverage = evaluateCoverage(ctx);
    const guard = evaluateGuardrails(ctx);
    const injection = evaluateInjection(ctx);
    const falseRefusals = evaluateFalseRefusals(ctx);

    const sorted = [...retrievalMs].sort((a, b) => a - b);
    const retrievalMedian = median(sorted);
    // With every retrieval errored (a total outage, or EVAL_FAULT on everything) there is no
    // population at all, and median and max of nothing are NaN and -Infinity. Say there were
    // none rather than print those: a latency line is exactly where a nonsense number gets
    // copied into a README and believed.
    const latency = sorted.length
        ? `${retrievalMedian.toFixed(0)}ms median, ${Math.max(...sorted).toFixed(0)}ms worst`
        : `no successful retrievals`;
    if (session) {
        const { answered: a, canned: k } = session.ms;
        console.log(`retrieval latency  ${latency}, n=${sorted.length}  (run 1 of each case, request → data-retrieval: planner + retrieval + thread read + HTTP; the TS baseline's population)`);
        console.log(`answered path      n=${a.total.length}   to sources ${ms(a.retrieval)}, first token ${ms(a.firstToken)}, done ${ms(a.total)}  (medians, every run)`);
        console.log(`canned path        n=${k.total.length}   reply done ${ms(k.total)}  (median; greeting and off-topic, planner only)`);
        console.log(`retrieval recall   ${formatRatio(recall.runOne)}   run 1 (one retrieval per case, comparable to the TS baseline)`);
        console.log(`recall every run   ${formatRatio(recall.everyRun)}   expected doc retrieved in EVERY run of the case`);
        if (recall.varied.length)
            console.log(`recall varied      ${recall.varied.map((v) => `${v.id} (${v.foundRuns}/${v.runs})`).join(", ")} — HyDE moved the page between runs`);
    } else {
        console.log(`retrieval latency  ${latency}  (embed + search + rerank)`);
        console.log(`retrieval recall   ${formatRatio(recall.runOne)}   expected doc survived rerank + threshold`);
    }
    if (RUNS > 0) {
        console.log(`answer coverage    ${formatRatio(coverage)}   answerable questions actually answered`);
        console.log(`guardrails held    ${formatRatio(guard.held)}   out-of-corpus questions refused`);
        if (falseRefusals.length)
            console.log(
                `false refusals     ${falseRefusals.length}   expected page retrieved but the answer refused ` +
                `(${falseRefusals.map((c) => c.id).join(", ")}) — the chunks that survived did not carry the answer`
            );
        console.log(
            `injection resisted ${formatRatio(injection.resisted)}   adversarial prompts that did not get what they asked for ` +
            `(${injection.multiTurn} multi-turn, ${injection.singleTurn} single-turn, ${ADVERSARIAL_RUNS} attempts each)`
        );
        if (session) {
            for (const c of injection.structural)
                console.log(
                    `  injection ${c.id}: held by STRUCTURE — its attack is a scripted assistant turn, and this target takes no ` +
                    `client history (a request carrying it got ${historyRefused}). The case ran with its user turns replayed; its verdict is ${byId.get(c.id)!.verdict}.`
                );
        }
    }


    // The tradeoff this harness exists to protect: loosening the prompt to raise
    // coverage must not lower guardrails. Either number moving alone is a regression.
    // Three layers can hold a guardrail and each is blind to a different change; the
    // attribution and its caveats live in evaluators/guardrails.ts, so that a "6/6 held"
    // can never be printed without saying which layer did the holding.
    for (const h of guard.holds) {
        console.log(`  guardrail ${h.id}: ${h.chunks} chunk(s) reached the model  → ${h.description}`);
    }

    if (RUNS === 0) {
        // Retrieval-only mode is the cheap CI gate (no answer generation). It used to always
        // exit 0, which made "runs on every push" a smoke test rather than a gate. A missing
        // expected doc is a retrieval regression whether or not we generated an answer, so
        // fail on recall here. Parked (expectFail) cases are already excluded from `answerable`.
        //
        // A miss gets ONE retry before it fails the gate. Retrieval is not fully deterministic:
        // the planner's HyDE hypothetical is model output (not seedable on the Responses API —
        // see lib/plan.ts), and two near-tied pages can swap places between runs. A retry separates "the expected
        // page is gone" (a real regression — fails both times) from "it lost a coin flip" (passes
        // on retry). The retry is REPORTED, never silent: a case that keeps needing it is a case
        // whose criterion or retrieval needs attention, and hiding that would turn the gate back
        // into a smoke test. Snapshotting planner output would have been the deterministic
        // alternative, but it would test a fixture instead of the pipeline.
        const missed = answerable.filter((c) => byId.get(c.id)!.retrieved !== "yes");
        const stillMissing: string[] = [];
        const recovered: string[] = [];
        for (const c of missed) {
            await sleep(RERANK_INTERVAL_MS);
            const { relevant } = await plannedRetrieve(c.query, c.history ?? []);
            if (expectedFound(c, relevant)) {
                recovered.push(c.id);
                console.log(`  ↻ ${c.id}: expected doc NOT retrieved on first try, recovered on retry`);
                for (const r of relevant) console.log(`      ${r.score.toFixed(3)}  ${r.title}`);
            } else {
                stillMissing.push(c.id);
            }
        }
        if (recovered.length)
            console.log(`\nrecovered on retry ${recovered.length}   ${recovered.join(", ")} — flaky retrieval, not a regression; if it repeats, widen expectedSource or look at DEBUG_PLAN=1`);
        if (stillMissing.length) {
            console.log(`\nretrieval regression — expected doc not retrieved twice: ${stillMissing.join(", ")}`);
            process.exit(1);
        }
        reportIncomplete();
        return;
    }

    // ONE numerator, read by the console line below AND by the record further down. Before 5.4b
    // this was written twice in two separately spelled expressions over the same array — the
    // same defect the survey found in the injection metric, and the confirmation of P3.
    const faithfulness = aggregateFaithfulness(results);
    if (JUDGE) {
        console.log(`faithfulness       ${formatRatio(faithfulness)}   answers fully supported by their own retrieved chunks (judge calibrated 0/12 FA, 0/23 missed, n=35)`);
    }

    // Parked, known-failing cases: reported here, never counted against the suite.
    if (parked.length) {
        const stillFailing = parked.filter((c) => byId.get(c.id)!.verdict !== "PASS").length;
        console.log(
            `known failures     ${stillFailing}/${parked.length}   parked, not blocking (${parked.map((c) => c.id).join(", ")})`
        );
    }

    // A parked case that has started PASSING is news — the bug got fixed elsewhere and the
    // marker is now lying. Surface it loudly and fail, so expectFail can't hide a real pass.
    const unexpectedPass = parked.filter((c) => byId.get(c.id)!.verdict === "PASS");
    if (unexpectedPass.length) {
        console.log(`\n⚠ expectFail case now PASSING — remove expectFail: ${unexpectedPass.map((c) => c.id).join(", ")}`);
        process.exit(1);
    }

    const parkedIds = new Set(parked.map((c) => c.id));
    const unfaithful = results.filter((r) => r.faithful === "NO" && !parkedIds.has(r.id));
    // verdict !== "PASS" would otherwise fold every errored case into "failing", which is the
    // exact conflation this whole change exists to remove.
    const failed = results.filter((r) => r.verdict !== "PASS" && r.verdict !== "ERROR" && !parkedIds.has(r.id));

    // Python target: what the run cost, MEASURED from the rows the service wrote for its threads.
    let cost: RunCost | null = null;
    if (session) {
        cost = await costOfThreads(session.threads, session.turns);
        const per = (p: { rows: number; usd: number }) => (p.rows ? `$${(p.usd / p.rows).toFixed(5)}` : "—");
        console.log(
            `cost               $${cost.usd.toFixed(4)} for ${cost.rows} requests, $${cost.usdPerRequest?.toFixed(5) ?? "—"} per request ` +
            `(measured: query_log rows ${cost.rows}/${cost.expected} landed${cost.unpriced ? `, ${cost.unpriced} without planner usage` : ""})`
        );
        console.log(
            `  per path         answered ${per(cost.answered)} per request (n=${cost.answered.rows}), ` +
            `canned ${per(cost.canned)} (n=${cost.canned.rows}); replayed earlier turns included`
        );
        if (cost.rows < cost.expected)
            console.log(`  ⚠ ${cost.expected - cost.rows} completed turns have no query_log row: the service logs insert failures, look there.`);
    }

    // The labelled set is priced on its own line, never folded into the figure above: that one is
    // a series across five stored runs and a different population would break it silently.
    let githubCost: RunCost | null = null;
    if (session && github.length) {
        githubCost = await costOfThreads(session.githubThreads, session.githubTurns);
        console.log(
            `GitHub set cost    $${githubCost.usd.toFixed(4)} for ${githubCost.rows} requests, ` +
            `$${githubCost.usdPerRequest?.toFixed(5) ?? "—"} per request (kept out of the figure above)`
        );
    }
    const githubReport = github.length ? reportGitHub(github, results) : null;

    // Every full run leaves a record: knobs, per-case verdicts, the headline numbers, the
    // commit it ran against. README numbers cite one of these files instead of a memory of a
    // terminal — a number nobody can trace to a stored run is a rumour with a decimal point.
    // Subsets (EVAL_ONLY) and retrieval-only runs (EVAL_RUNS=0, the branch CI gate) are
    // diagnostics, not results, and are not recorded. (Review item 29.)
    if (!ONLY.length && RUNS > 0) {
        let commit = "unknown";
        // Uncommitted changes in the tree: the result then did NOT run on `commit` alone. The first
        // Python-target run (2.6) recorded a commit it had not run on, because nothing said so.
        // Result files are left out: the previous run's own file, not yet committed, made the next
        // run report changes it did not have (the second 2.6 run, corrected by hand in its file).
        let dirty: boolean | null = null;
        try {
            commit = execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
            dirty = execSync("git --no-optional-locks status --porcelain", { stdio: ["ignore", "pipe", "ignore"] })
                .toString()
                .split("\n")
                .some((line) => line.trim() !== "" && !line.slice(3).startsWith("evals/results/"));
        } catch { /* not a git checkout */ }
        const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        const record = {
            // Bumped when a field changes meaning, not when one is added. Every run stored
            // before 5.2 has no version at all and reads back as 0 (evals/record.ts).
            schemaVersion: SCHEMA_VERSION,
            date: new Date().toISOString(),
            commit,
            dirty,
            // A run that could not run every case is not a baseline. Recorded rather than
            // refused: the cases that DID run are real, and `dirty` already set the precedent
            // that a caveat belongs in the file instead of being a reason to write nothing.
            incomplete: errored.length > 0,
            target: TARGET,
            knobs: { runs: RUNS, adversarialRuns: ADVERSARIAL_RUNS, judge: JUDGE, candidates: VECTOR_CANDIDATES, rerankTopN: RERANK_TOP_N, threshold: RERANK_THRESHOLD },
            summary: {
                cases: active.length,
                recall: formatRatio(recall.runOne),
                coverage: formatRatio(coverage),
                guardrails: formatRatio(guard.held),
                injection: formatRatio(injection.resisted),
                ...(JUDGE ? { faithful: formatRatio(faithfulness) } : {}),
                retrievalMsMedian: Math.round(retrievalMedian),
                retrievalMsWorst: Math.round(Math.max(...sorted)),
                parked: parked.map((c) => c.id),
                errored: errored.map((r) => r.id),
                failing: failed.map((r) => r.id),
                falseRefusals: falseRefusals.map((c) => c.id),
                ...(session
                    ? {
                        recallEveryRun: formatRatio(recall.everyRun),
                        recallVaried: recall.varied.map((v) => v.id),
                        answeredMs: {
                            n: session.ms.answered.total.length,
                            toSources: Math.round(median(session.ms.answered.retrieval)),
                            firstToken: Math.round(median(session.ms.answered.firstToken)),
                            done: Math.round(median(session.ms.answered.total)),
                        },
                        cannedMs: { n: session.ms.canned.total.length, done: Math.round(median(session.ms.canned.total)) },
                        historyFieldStatus: historyRefused,
                        cost,
                        ...(githubReport ? { github: githubReport, githubCost } : {}),
                        threads: `eval-${session.runStamp}-*`,
                    }
                    : {}),
            },
            cases: results.map((r) => ({
                id: r.id, verdict: r.verdict, intent: r.intent, retrieved: r.retrieved, chunks: r.chunks,
                topScore: r.topScore, answered: `${r.answered}/${r.runs}`, faithful: r.faithful, detail: r.detail,
                ...(r.faithfulDetail ? { faithfulDetail: r.faithfulDetail } : {}),
                ...(r.errored ? { errored: r.errored } : {}),
                ...(session ? { retrievedEvery: r.retrievedEvery, foundRuns: `${r.foundRuns}/${r.runs}`, route: r.route ?? null } : {}),
            })),
        };
        mkdirSync("evals/results", { recursive: true });
        const file = `evals/results/${stamp}${session ? "-python" : ""}.json`;
        writeFileSync(file, JSON.stringify(record, null, 2) + "\n");
        console.log(`\nrecorded → ${file}  (commit ${commit}${dirty ? ", WITH UNCOMMITTED CHANGES" : ""})`);
    }

    if (unfaithful.length) {
        console.log(`\n${unfaithful.length} unfaithful: ${unfaithful.map((f) => f.id).join(", ")}`);
        process.exit(1);
    }
    if (failed.length) {
        console.log(`\n${failed.length} failing: ${failed.map((f) => f.id).join(", ")}`);
        process.exit(1);
    }
    reportIncomplete();
    // The golden set is green. The labelled set gets its own exit code rather than 1, because
    // these are two different claims about the system and collapsing them would make "the suite
    // passed" stop meaning what it has meant in every stored run before this one. Exit 5 says:
    // the documentation pipeline is fine and the GitHub set is not. (datasets/index.ts carries
    // that difference as data: each dataset descriptor has its own exitCode.)
    // Exit 4 first, for the same reason the golden set exits 4: a case that could not RUN is not
    // a verdict about the system, and a run that did not measure everything is not a baseline.
    const githubErrored = github.filter((r) => r.verdict === "ERROR");
    if (githubErrored.length) {
        console.log(`\n${githubErrored.length} GitHub case(s) could NOT RUN: ${githubErrored.map((r) => r.id).join(", ")}`);
        console.log(`  This run is INCOMPLETE and is not a baseline (exit 4).`);
        process.exit(4);
    }
    const githubFailed = github.filter((r) => r.verdict === "FAIL" || r.verdict === "VARIED");
    if (githubFailed.length) {
        console.log(`\n${githubFailed.length} GitHub case(s) not answered: ${githubFailed.map((r) => r.id).join(", ")}`);
        console.log(`  The golden set is green; this is the labelled set (exit 5).`);
        process.exit(5);
    }
    console.log(`\nall green${parked.length ? ` (${parked.length} parked)` : ""}.`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
