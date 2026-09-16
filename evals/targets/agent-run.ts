/**
 * The Python target's case loop (EVAL_TARGET=python).
 *
 * `agent-service.ts` is the CLIENT — HTTP, thread ids, and the readers for query_log and
 * Langfuse. This is the target built on it: the run's session state, one case against the
 * service, the post-run judging, and the labelled GitHub set's loop.
 *
 * Two files rather than one, and the reason is a cycle. `evals/evaluators/types.ts` type-imports
 * `TurnFacts` from the client, because a `Result` records the route the service chose. A runner
 * that imports the evaluators therefore cannot live in the file the evaluators point at without
 * making the two mutually dependent — type-only today, which TypeScript erases, and a real
 * runtime cycle the first time either side needs a value from the other. Splitting the client
 * from the loop that drives it costs one file and removes the question.
 *
 * ── the session ────────────────────────────────────────────────────────────────────────────
 * Until 5.4b this was five module-level mutable globals in run.ts: two thread lists, two turn
 * counters and the timing populations. They are a session — one run's worth of state — and the
 * two thread lists are kept apart ON PURPOSE (see `githubThreads`), which is exactly the kind of
 * invariant that survives being written down and does not survive being a pair of file-scope
 * `let`s that anything can push to.
 */
import type { EvalCase } from "../datasets/types";
import { expectedFound, scoreRuns, verdictOf } from "../evaluators/case-verdict";
import { judgeFaithfulness, reportUnfaithful } from "../evaluators/faithfulness";
import { answeredCorrectly, type GitHubRun } from "../evaluators/github-answer";
import type { Result } from "../evaluators/types";
import type { GitHubCase } from "../datasets/github";
import { contextsByTrace, langfuseApi } from "./langfuse-api";
import {
    askAgent,
    chunkCount,
    evalThreadId,
    pagesAsChunks,
    topScore as pagesTopScore,
    traceIdsOfThreads,
    turnFactsOfThreads,
    type AgentReply,
    type AgentTarget,
    type TurnFacts,
} from "./agent-service";

export type AgentKnobs = {
    runs: number;
    adversarialRuns: number;
    /** Pause between cases: politeness on the production Cohere key, necessary on the trial one. */
    rerankIntervalMs: number;
};

/**
 * One run's state against the service.
 *
 * `threads` and `githubThreads` are separate lists and must stay separate. `costOfThreads` over
 * the first is the suite's cost per request, and that figure is a SERIES across stored runs:
 * $0.1972 local, $0.1972 on AWS, $0.2022 with the router, $0.2042 with the planner paragraph.
 * Folding thirteen subagent turns into it would break the series at the exact moment it is most
 * worth reading, and would price two different populations as one. The labelled set gets its
 * own line instead.
 */
export type AgentSession = {
    target: AgentTarget;
    /** Identifies this run's threads in query_log: a timestamp plus a little randomness. */
    runStamp: string;
    /** When the run started, minus a minute of clock skew: the window the judge asks Langfuse for. */
    startedAt: Date;
    threads: string[];
    turns: number;
    githubThreads: string[];
    githubTurns: number;
    /**
     * Timings per request for the case's own question (not the replayed turns), kept apart by
     * path. A greeting or off-topic reply streams everything at once right after the planner, so
     * its "first token" is its retrieval time; mixed into one median with answered requests (63
     * of the 121 requests in a full run are canned) the median described the canned path.
     */
    ms: {
        answered: { retrieval: number[]; firstToken: number[]; total: number[] };
        canned: { total: number[] };
    };
};

export function newAgentSession(target: AgentTarget): AgentSession {
    return {
        target,
        runStamp: new Date().toISOString().replace(/[-:]/g, "").slice(0, 15) + Math.random().toString(16).slice(2, 6),
        startedAt: new Date(Date.now() - 60_000),
        threads: [],
        turns: 0,
        githubThreads: [],
        githubTurns: 0,
        ms: {
            answered: { retrieval: [], firstToken: [], total: [] },
            canned: { total: [] },
        },
    };
}

export type AgentCaseOutcome = {
    result: Result;
    /**
     * Run 1's retrieval time, or null if that run errored or was canned without one. The TS
     * target measures one retrieval per case, canned cases included, so run 1 here is the same
     * population and the headline latency line compares like with like.
     */
    retrievalMs: number | null;
};

/**
 * One case against the service. Each run is a fresh thread: the case's earlier user turns are
 * sent first as real turns (the service takes no history, a `history` field is a 422), then the
 * query. Each run plans and retrieves again, so recall is recorded per run.
 */
export async function runCaseOnAgent(
    session: AgentSession,
    c: EvalCase,
    index: number,
    knobs: AgentKnobs,
): Promise<AgentCaseOutcome> {
    const runs = c.injection ? knobs.adversarialRuns : knobs.runs;
    const replies: AgentReply[] = [];
    let replayed: AgentReply[] = [];
    let retrievalMs: number | null = null;
    for (let i = 0; i < runs; i++) {
        const threadId = evalThreadId(session.runStamp, index, i);
        session.threads.push(threadId);
        const earlier: AgentReply[] = [];
        for (const turn of c.history ?? []) {
            if (turn.role !== "user") continue;
            const reply = await askAgent(session.target, threadId, turn.text);
            if (reply.error) throw new Error(`${c.id}: replaying an earlier turn failed on thread ${threadId}`);
            session.turns++;
            earlier.push(reply);
        }
        if (i === 0) replayed = earlier;
        const reply = await askAgent(session.target, threadId, c.query);
        if (!reply.error) session.turns++;
        replies.push(reply);
        if (i === 0 && reply.ms.retrieval !== null) retrievalMs = reply.ms.retrieval;
        if (reply.error) continue;
        if (reply.intent === "search") {
            if (reply.ms.retrieval !== null) session.ms.answered.retrieval.push(reply.ms.retrieval);
            if (reply.ms.firstToken !== null) session.ms.answered.firstToken.push(reply.ms.firstToken);
            session.ms.answered.total.push(reply.ms.total);
        } else {
            session.ms.canned.total.push(reply.ms.total);
        }
    }

    const first = replies[0];
    const degraded = [...replayed, ...replies].some((r) => r.mode === "cosine-fallback");
    if (degraded) console.log(`  !! ${c.id}: reranker unavailable, cosine fallback`);
    const foundPerRun = replies.map((r) => expectedFound(c, pagesAsChunks(r.pages)));
    const found = foundPerRun[0];
    const foundRuns = foundPerRun.filter((f) => f === true).length;
    const retrievedEvery =
        found === null ? "—" : foundRuns === runs ? "yes" : foundRuns === 0 ? "NO" : "varied";

    // Look, don't guess: what the conversation actually was, since it is not the dataset's text.
    if (replayed.length) {
        console.log(`  ${c.id}: earlier turns replayed on the service (run 1), its own replies:`);
        for (const r of replayed) console.log(`      ↳ ${r.text.replace(/\s+/g, " ").slice(0, 110)}`);
    }

    const scored = scoreRuns(c, replies.map((r) => r.text));
    let { verdict, detail } = verdictOf(c, runs, scored, found);
    // A stream that failed produced no answer to score; it is a failure of the run, not a refusal.
    const errors = replies.filter((r) => r.error).length;
    if (errors) {
        verdict = "FAIL";
        detail = `STREAM ERROR in ${errors}/${runs} runs (the service logged the cause)`;
    }
    if (verdict !== "PASS" && first.pages.length) {
        console.log(`      [pages, run 1] ${chunkCount(first.pages)} chunks:`);
        for (const p of first.pages) console.log(`        ${p.score.toFixed(3)} ${p.title}  (chunks ${p.chunks.join(", ")})`);
    }

    const top = pagesTopScore(first.pages);
    return {
        result: {
            id: c.id,
            threadId: evalThreadId(session.runStamp, index, 0),
            intent: first.intent,
            degraded,
            retrieved: found === null ? "—" : found ? "yes" : "NO",
            retrievedEvery,
            foundRuns,
            chunks: chunkCount(first.pages),
            topScore: top === null ? "—" : top.toFixed(3),
            answered: scored.answered,
            runs,
            sample: scored.firstAnswer,
            ...(scored.oddRun ? { odd: scored.oddRun } : {}),
            faithful: "—",
            verdict,
            detail,
        },
        retrievalMs,
    };
}

/**
 * Faithfulness on the Python target, judged AFTER the run from the traces (step 2.7).
 *
 * On the in-process target the chunks are in hand while the case runs, so the judge is called
 * there. Over HTTP the harness never sees a chunk text: the stream carries pages. So run 1 of
 * every case that answered is looked up in two hops - thread id to trace id on the query_log row
 * (db/007_trace_id.sql), trace id to the `context` observation the service wrote - and judged
 * with the chunks the prompt was actually built from.
 *
 * One judgement per case, run 1, exactly as on the other target: generation varies between runs,
 * but not usually in whether it stayed inside its sources, and this keeps the cost linear.
 *
 * A case whose trace never arrives is REPORTED, never judged as an answer with no sources: an
 * observability gap must not become a faithfulness failure.
 */
export async function judgeAgentRuns(
    session: AgentSession,
    results: Result[],
    active: EvalCase[],
    isRefusalFn: (text: string) => boolean,
): Promise<void> {
    const byId = new Map(results.map((r) => [r.id, r]));
    const threads = new Map<string, EvalCase>();
    active.forEach((c, i) => {
        const r = byId.get(c.id);
        // Same gate as the in-process path: a refusal has no claims to be unfaithful about, and a
        // greeting or off-topic reply retrieved nothing, so it has no `context` span either.
        if (!r || r.intent !== "search" || r.answered === 0 || isRefusalFn(r.sample)) return;
        threads.set(evalThreadId(session.runStamp, i, 0), c);
    });
    if (!threads.size) return;

    const traces = await traceIdsOfThreads([...threads.keys()]);
    const judged = new Map<string, EvalCase>();
    for (const [threadId, c] of threads) {
        const traceId = traces.get(threadId);
        if (traceId) judged.set(traceId, c);
    }
    if (!judged.size) {
        console.log("\nfaithfulness skipped: the run's rows carry no trace id — were the service's Langfuse keys set?");
        return;
    }

    const contexts = await contextsByTrace(langfuseApi(), session.startedAt, new Set(judged.keys()));
    let missing = 0;
    for (const [traceId, c] of judged) {
        const chunks = contexts.get(traceId);
        if (!chunks) {
            missing++;
            continue;
        }
        const r = byId.get(c.id)!;
        const v = await judgeFaithfulness(c.query, chunks, r.sample);
        r.faithful = v.supported ? "yes" : "NO";
        if (!v.supported) r.faithfulDetail = reportUnfaithful(c.id, v);
    }
    if (missing) {
        console.log(`\nfaithfulness: ${missing}/${judged.size} traces had no context observation in time — not judged, and not counted`);
    }
}

/** Case indices for the GitHub set, kept clear of the golden set so thread ids stay unique and
 *  a GitHub turn is recognisable in query_log by its thread id alone. */
export const GITHUB_INDEX_BASE = 1000;

/** One labelled case's observations, and what the service recorded for each of them. */
export type GitHubObservations = {
    case: GitHubCase;
    runs: GitHubRun[];
    /** The per-run facts, in the same order as `runs`. null where no row came back. */
    blocks: (TurnFacts | null)[];
    correct: number;
};

/**
 * Run the labelled set, then read back what the service recorded for each turn.
 *
 * The facts come from query_log and not from the stream: decision 12, and the reason is that the
 * stream is a product surface. Every measure except answer accuracy is computed from the block
 * db/010 stores, and answer accuracy is computed from the text, which is the whole point of
 * having it — 3.5b's turn had four perfect counters and no answer in it.
 */
export async function runGitHubCases(
    session: AgentSession,
    cases: GitHubCase[],
    githubRuns: number,
    rerankIntervalMs: number,
): Promise<GitHubObservations[]> {
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
    const runsByCase = new Map<string, GitHubRun[]>();
    for (const [i, c] of cases.entries()) {
        const runs: GitHubRun[] = [];
        for (let run = 0; run < githubRuns; run++) {
            const threadId = evalThreadId(session.runStamp, GITHUB_INDEX_BASE + i, run);
            session.githubThreads.push(threadId);
            let reply: AgentReply;
            try {
                reply = await askAgent(session.target, threadId, c.query);
            } catch (err) {
                runs.push({ threadId, text: "", pageFound: null, error: String(err) });
                continue;
            }
            if (!reply.error) session.githubTurns++;
            runs.push({
                threadId,
                text: reply.text,
                pageFound: c.expectedSource ? expectedFound(c, pagesAsChunks(reply.pages)) : null,
                error: reply.error,
            });
            await sleep(rerankIntervalMs);
        }
        runsByCase.set(c.id, runs);
    }

    // The rows land in the background after each turn, same as the cost and trace rows.
    const facts = await turnFactsOfThreads(session.githubThreads);

    return cases.map((c) => {
        const runs = runsByCase.get(c.id)!;
        return {
            case: c,
            runs,
            blocks: runs.map((r) => facts.get(r.threadId) ?? null),
            correct: runs.filter((r) => answeredCorrectly(c, r)).length,
        };
    });
}
