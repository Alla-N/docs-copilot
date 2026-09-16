/**
 * GITHUB ANSWER ACCURACY — did the turn contain the frozen answer?
 *
 * The one measure in the labelled set that reads the TEXT. Every other measure in
 * ./github-process.ts is computed from the block the service stored, and that separation is the
 * entire finding of phase 3.6: at the baseline, first-try validity was 41 of 42, points were at
 * median 1, routing was 74 of 74 — and accuracy was 24 of 52. A suite of validity and cost
 * would have reported that subagent as working.
 *
 * Process metrics say how much happened. This one says whether the question got answered.
 */
import { containsAnswer, type GitHubCase } from "../datasets/github";
import { isRefusal } from "../../lib/retrieve";
import type { Ratio } from "../record";

/** One observation of one labelled case. */
export type GitHubRun = {
    threadId: string;
    text: string;
    /** For the compound case: was the documentation page there too? null when the case has none. */
    pageFound: boolean | null;
    error: string | null;
};

/**
 * Did the answer decline, rather than assert something?
 *
 * Only the control case is scored with this, and it is the one criterion in the set that is
 * openly approximate. `isRefusal` alone is not enough here: it is positional (invariant 5) and
 * 3.5 watched it read an answered turn as refused when a GitHub answer opened with the refusal
 * sentence. A model that declines in its own words is also correct, so both shapes count, and
 * the control's sample is printed on every run whatever the verdict — the criterion is the
 * thing most likely to be wrong, so it is the thing kept under the eye.
 */
export function declined(text: string): boolean {
    return (
        isRefusal(text) ||
        /\b(cannot|can't|can not|unable to|do(es)? not (provide|have|record)|no (historical|record|data))\b/i.test(text)
    );
}

export function answeredCorrectly(c: GitHubCase, run: GitHubRun): boolean {
    if (run.error) return false;
    if (!c.shouldAnswer) return declined(run.text);
    if (!containsAnswer(run.text, c.answerContains ?? [])) return false;
    // The compound case is the only test of the claim that `both` means both, so a right date
    // with the documentation half missing is not a pass.
    return run.pageFound !== false;
}

/**
 * The set-level fraction, over observations rather than over cases.
 *
 * 13 cases x 2 runs is 26 observations in one stored run, and the headline in the state doc
 * pools a PAIR of runs into 52. A stored run holds one, and the diff compares one file against
 * one file — so this number is out of 26 and the pooled figure is a thing a human does with
 * two of them, never a thing this function returns.
 */
export function evaluateGitHubAccuracy(scored: { correct: number; runs: number }[]): Ratio {
    return {
        num: scored.reduce((n, r) => n + r.correct, 0),
        den: scored.reduce((n, r) => n + r.runs, 0),
    };
}

/**
 * The per-case verdict for a labelled question — the GitHub set's `verdictOf`.
 *
 * Four outcomes, not three. VARIED exists because two observations per case is NOT a per-case
 * rate: 3.5b is emphatic that n=10 cannot separate 0.85 from 1.0, so n=2 certainly cannot. Two
 * runs are there so that a case which differs between them is REPORTED as differing, rather
 * than silently as whichever run came first. Collapsing VARIED into PASS or FAIL would throw
 * away the only thing n=2 is good for.
 *
 * The FAIL detail lines are a small diagnostic ladder, and the order matters: never routed is a
 * different defect from routed-with-nothing-back, which is different again from a query that
 * ran and did not contain the answer. 3.5 found the planner canning three of three repository
 * questions, and a FAIL line that just said "wrong" would have hidden it.
 */
export function githubVerdict(
    c: GitHubCase,
    runs: GitHubRun[],
    routes: (string | null)[],
    blocks: { ok: boolean }[],
    correct: number,
): { verdict: "PASS" | "FAIL" | "VARIED" | "ERROR"; detail: string } {
    const errored = runs.filter((r) => r.error).length;
    if (errored === runs.length) return { verdict: "ERROR", detail: `could not run: ${runs[0].error}` };
    if (correct === runs.length)
        return {
            verdict: "PASS",
            detail: c.shouldAnswer ? "the frozen answer is in every run" : "declined in every run",
        };
    if (correct === 0)
        return {
            verdict: "FAIL",
            detail: routes.every((r) => r === null)
                ? "the planner canned it: never routed, never asked"
                : blocks.length === 0
                  ? "routed, but no GitHub evidence came back"
                  : blocks.every((b) => b.ok)
                    ? "the query ran and the answer is not in it"
                    : "the subagent could not get a query through",
        };
    return { verdict: "VARIED", detail: `${correct}/${runs.length} runs carried the frozen answer` };
}
