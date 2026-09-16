/**
 * GITHUB PROCESS METRICS — first-try validity, validity after repairs, points, routing.
 *
 * Every number here is computed from the block db/010 stores, never from the stream, because
 * the stream is a product surface (decision 12). And every number here was at or near ceiling
 * on the run where answer accuracy was 24 of 52 — which is why they live in a different module
 * from ./github-answer.ts and are printed on different lines. They measure how much happened,
 * not whether it was right.
 *
 * ── two denominators that are easy to get wrong ────────────────────────────────────────────
 *
 * 1. Validity is over the turns whose SUBAGENT RAN, not over observations. A turn the planner
 *    canned never wrote a query, and scoring it as an invalid query would be inventing a
 *    failure. This is also why first-try validity reads 20/20 at 3.6 and 21/21 at 3.6c: the
 *    same rate over a different number of attempts. A naive diff calls that unchanged; it is a
 *    changed population, not a flat line.
 *
 *    Cases that could not RUN are the caller's to exclude, and run.ts excludes them before
 *    calling — same rule as the golden set's ERROR verdict, for the same reason: a case the
 *    infrastructure prevented from running is not evidence about the subagent.
 *
 * 2. Routing accuracy is over the turns that WERE ROUTED, pooled from both datasets — the
 *    golden set supplies the `docs` label and the labelled set the `both` label. Turns the
 *    planner canned get their own line rather than a score, because db/009 is explicit that a
 *    null route is not a route to the documentation, and because 3.5 found the planner throwing
 *    away three of three repository questions before the router existed for them. That is a
 *    different defect with a different fix, and folding it into a routing percentage would
 *    bury it.
 */
import type { Ratio } from "../record";
import type { GitHubResult, Result } from "./types";

/**
 * What this evaluator needs from a scored case. `GitHubResult` satisfies it structurally, so
 * run.ts hands its list straight over — but the narrower shape is written out because it says
 * which six of that type's twenty fields are load-bearing here.
 */
export type GitHubCaseFacts = Pick<
    GitHubResult,
    "id" | "runs" | "routes" | "routedRight" | "canned" | "subagentRuns" | "firstTryValid" | "ok" | "points"
>;

export type GitHubProcessScore = {
    firstTryValid: Ratio;
    validAfterRepairs: Ratio;
    points: { median: number | null; max: number | null; total: number };
    routing: {
        overall: Ratio;
        githubSet: Ratio;
        goldenSet: Ratio;
        cannedByPlanner: number;
    };
    observations: number;
};

const sum = (ns: number[]) => ns.reduce((a, b) => a + b, 0);
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

/**
 * `docs` is the golden set's label for every case in it, so a golden case recorded as `both` or
 * `github` is a routing miss and a golden case with no route at all was never routed. Only the
 * ones that WERE routed are in the denominator, same rule as the labelled set.
 */
export function evaluateGitHubProcess(cases: GitHubCaseFacts[], goldenResults: Result[]): GitHubProcessScore {
    const observations = sum(cases.map((r) => r.runs));
    const subagentRuns = sum(cases.map((r) => r.subagentRuns));
    const points = cases.flatMap((r) => r.points);

    const docsRouted = goldenResults.filter((r) => r.route === "docs" || r.route === "both" || r.route === "github");
    const docsRight = docsRouted.filter((r) => r.route === "docs").length;
    const githubRouted = sum(cases.map((r) => r.runs - r.canned));
    const githubRight = sum(cases.map((r) => r.routedRight));

    return {
        firstTryValid: { num: sum(cases.map((r) => r.firstTryValid)), den: subagentRuns },
        validAfterRepairs: { num: sum(cases.map((r) => r.ok)), den: subagentRuns },
        points: {
            median: points.length ? median(points) : null,
            max: points.length ? Math.max(...points) : null,
            total: sum(points),
        },
        routing: {
            overall: { num: githubRight + docsRight, den: githubRouted + docsRouted.length },
            githubSet: { num: githubRight, den: githubRouted },
            goldenSet: { num: docsRight, den: docsRouted.length },
            cannedByPlanner: sum(cases.map((r) => r.canned)),
        },
        observations,
    };
}
