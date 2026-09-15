/**
 * The GitHub labelled set (step 3.6). Twelve questions with frozen answers, plus one control.
 *
 * Its own file, next to planner-cases.ts, for the same reason that one is not in dataset.ts: it
 * is a different kind of set, scored on different criteria, with its own denominators. The
 * 27-case golden set measures a retrieval pipeline; this one measures whether a subagent that
 * writes its own GraphQL fetched the fact it was asked for.
 *
 * ## Where the answers come from
 *
 * All of them from one run of `agent/experiments/freeze_github_answers.py` against the live API
 * at FROZEN_AT, cost 1 point. That file is the provenance and `frozen` on each case names the
 * alias in it that produced the literal. Nothing here was written from memory, mine or the
 * model's, which is the point: a literal recalled rather than fetched would make this metric
 * agree with whatever the model already believes about vercel/ai.
 *
 * ## What makes a question usable here, and why so few qualify
 *
 * 1. **Frozen.** The answer cannot change. Publication dates, merge dates, the login that opened
 *    an issue, the commit a tag points at. Star counts, open-issue counts and "the latest
 *    release" are not frozen, and a set built on them fails for the wrong reason within a week.
 * 2. **Not obtainable by accident.** 3.5b watched the subagent answer "when was ai 5.0.0
 *    released" by listing the ten most recently created releases: valid, well formed, and with
 *    no answer in it, while `ok`, `attempts`, `first_try_valid` and `points_spent` all reported
 *    success. If a generic listing could contain the literal, this metric would score that run
 *    as correct and the whole set would be decorative. Dates from 2023 and 2024, a commit oid
 *    and a login cannot turn up in a listing of this week's releases by accident.
 * 3. **Not answerable without a query.** "What licence is this repository under" and "what is
 *    the default branch" were both frozen by the same run and are both left out: NOASSERTION and
 *    `main` are exactly what a model would say having queried nothing at all.
 *
 * `gh-oldest-release` is the deliberate inverse of failure 2: it can only be answered by ordering
 * ascending, so a model reaching for the default listing lands on the wrong end of the repository.
 *
 * ## Two of the anchors came back null, and that is worth knowing before reading a result
 *
 * The freeze run asked for `issue(number: 50)` and `pullRequest(number: 100)` and GitHub answered
 * "Could not resolve to an Issue with the number of 50" and the same for the pull request, as
 * field-level errors next to data that was otherwise fine. Neither number is missing: issues and
 * pull requests share ONE number space in a repository, so #50 is a pull request and #100 is an
 * issue, and asking for either through the wrong field is a type error rather than a lookup that
 * found nothing. The subagent will meet this, and when it does the failure will name a number
 * that plainly exists. The cases below use numbers confirmed to resolve through the field they
 * are asked through.
 */

/** When every literal below was read from the API. */
export const FROZEN_AT = "2026-09-15T14:21:43Z";

export type GitHubCase = {
    id: string;
    query: string;
    /**
     * The routing label. Two, not three: after the 3.5 reversal GitHub is additive and never
     * exclusive, so a repository question is `both` and there is no `github` to label.
     */
    route: "docs" | "both";
    /**
     * true — the answer must contain one of `answerContains`. false — GitHub genuinely cannot
     * answer and a refusal is the correct outcome.
     */
    shouldAnswer: boolean;
    /**
     * Any ONE of these, after normalisation (see `containsAnswer`), must appear in the answer.
     * An array because a date has renderings, not because the answer has alternatives: the fact
     * is single, and every entry is the same fact written the way a model writes prose.
     */
    answerContains?: string[];
    /**
     * For the compound case: a documentation page that must also be in the retrieved set, so
     * that "both" means both rather than "GitHub answered and retrieval was ignored".
     */
    expectedSource?: string | string[];
    /** The alias in experiments/freeze_github_answers.py this literal came out of, and its value. */
    frozen: string;
    note?: string;
};

/**
 * Compare an answer against a frozen literal.
 *
 * Lowercased, punctuation that only ever separates removed, ordinal suffixes dropped and runs of
 * whitespace collapsed — so "July 31st, 2025", "July 31 2025" and "july 31  2025" are one string
 * and the `answerContains` list stays a list of renderings rather than a list of typographies.
 * Deliberately NOT a fuzzy match: it normalises how the fact is written, never which fact it is.
 */
export function normaliseAnswer(text: string): string {
    return text
        .toLowerCase()
        .replace(/(\d+)(st|nd|rd|th)\b/g, "$1")
        .replace(/[,()"'`*]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

export function containsAnswer(text: string, literals: string[]): boolean {
    const answer = normaliseAnswer(text);
    return literals.some((literal) => answer.includes(normaliseAnswer(literal)));
}

/** One date, in the renderings a model actually writes. */
const asDate = (iso: string, long: string, day: string, month: string): string[] => [
    iso,
    `${long} ${day} ${iso.slice(0, 4)}`,
    `${day} ${long} ${iso.slice(0, 4)}`,
    `${iso.slice(0, 4)}/${month}/${day}`,
];

export const GITHUB_CASES: GitHubCase[] = [
    // ── Releases: the shape 3.5b caught failing ──────────────────────────────
    {
        id: "gh-ai5-release",
        query: "when was version 5.0.0 of the ai package released?",
        route: "both",
        shouldAnswer: true,
        answerContains: asDate("2025-07-31", "july", "31", "07"),
        frozen: "ai5.publishedAt = 2025-07-31T15:38:27Z",
        note:
            "The question 3.5b asked by hand, where the subagent listed this week's releases " +
            "instead. It is here because its failure mode is the only one already observed.",
    },
    {
        id: "gh-ai4-release",
        query: "what date did ai 4.0.0 come out?",
        route: "both",
        shouldAnswer: true,
        answerContains: asDate("2024-11-18", "november", "18", "11"),
        frozen: "ai4.publishedAt = 2024-11-18T17:39:37Z",
    },
    {
        id: "gh-oldest-release",
        query: "what was the very first release ever published in this repository?",
        route: "both",
        shouldAnswer: true,
        answerContains: ["v0.0.2"],
        frozen: "oldestReleases.nodes[0] = v0.0.2, published 2023-05-23T22:16:36Z",
        note:
            "Only answerable by ordering CREATED_AT ascending. The default listing gives the " +
            "newest releases, which is the exact reach 3.5b caught, so this case fails " +
            "whenever that habit wins.",
    },
    {
        id: "gh-tag-commit",
        query: "which commit does the ai@5.0.0 tag point at?",
        route: "both",
        shouldAnswer: true,
        answerContains: ["a5e92fe"],
        frozen: "ai5Tag.oid = a5e92fe3c692f8e152fe1eb9ef50a4e734ad0ffb",
        note:
            "The freeze run also settled the SHAPE: __typename came back Commit, not Tag, so " +
            "ai@5.0.0 is a lightweight tag and a query insisting on Tag.target gets null.",
    },

    // ── Issues and pull requests: the other half of the schema ───────────────
    {
        id: "gh-issue-1-author",
        query: "who opened issue 1 in the vercel ai repository?",
        route: "both",
        shouldAnswer: true,
        answerContains: ["jaredpalmer"],
        frozen: "issue1.author.login = jaredpalmer",
    },
    {
        id: "gh-issue-1-title",
        query: "what is issue 1 in this repository about?",
        route: "both",
        shouldAnswer: true,
        answerContains: ["anthropic"],
        frozen: "issue1.title = Add support for Anthropic",
        note:
            "The literal is the proper noun, not the title. A title case scored on its exact " +
            "wording punishes a correct paraphrase -- checked by hand: \"it asks for Anthropic " +
            "support\" is right and fails an exact-title match. The noun is what cannot be " +
            "there without the lookup, which is the property the set is selected for.",
    },
    {
        id: "gh-issue-2-closed",
        query: "when was issue 2 closed?",
        route: "both",
        shouldAnswer: true,
        answerContains: asDate("2023-06-14", "june", "14", "06"),
        frozen: "issue2.closedAt = 2023-06-14T16:47:27Z",
        note:
            "A field that is null on an open issue, so it also asks the model to notice the " +
            "issue is closed.",
    },
    {
        id: "gh-pr-500-author",
        query: "who opened pull request 500?",
        route: "both",
        shouldAnswer: true,
        answerContains: ["devjiwonchoi"],
        frozen: "pr500.author.login = devjiwonchoi",
        note:
            "A login no model would produce without fetching it, which is the whole reason " +
            "this case is here rather than a prettier one.",
    },
    {
        id: "gh-pr-500-title",
        query: "what is pull request 500 about?",
        route: "both",
        shouldAnswer: true,
        answerContains: ["handling errors", "error handling"],
        frozen: "pr500.title = docs: add handling errors for OpenAI provider",
        note:
            "Both orderings, for the same reason as issue 1: a model summarising this title " +
            "writes error handling at least as often as the title\u2019s own word order, and " +
            "scoring that as a miss would measure prose style rather than the lookup.",
    },
    {
        id: "gh-pr-1000-merged",
        query: "was pull request 1000 merged, and when?",
        route: "both",
        shouldAnswer: true,
        answerContains: asDate("2024-02-27", "february", "27", "02"),
        frozen: "pr1000.merged = true, mergedAt = 2024-02-27T16:21:44Z",
    },
    {
        id: "gh-repo-created",
        query: "when was the vercel ai repository created?",
        route: "both",
        shouldAnswer: true,
        answerContains: asDate("2023-05-23", "may", "23", "05"),
        frozen: "repository.createdAt = 2023-05-23T15:04:08Z",
    },

    // ── The additive case: the only one that checks the route called `both` ──
    {
        id: "gh-both-compound",
        query: "what does streamText do, and when was ai 5.0.0 released?",
        route: "both",
        shouldAnswer: true,
        answerContains: asDate("2025-07-31", "july", "31", "07"),
        expectedSource: ["generating-text", "stream-text"],
        frozen: "ai5.publishedAt = 2025-07-31T15:38:27Z",
        note:
            "Decision 18 says GitHub is additive and never exclusive; across four green suite " +
            "runs the route `both` was chosen zero times, so nothing has ever tested that " +
            "claim end to end. This case fails if either half goes missing: the date, or the " +
            "documentation page behind the first clause.",
    },

    // ── The control ─────────────────────────────────────────────────────────
    {
        id: "gh-control-unanswerable",
        query: "how many stars did the vercel ai repository have on 1 January 2024?",
        route: "both",
        shouldAnswer: false,
        frozen: "nothing: the API serves the current stargazer count and no history of it",
        note:
            "The set needs one question the subagent cannot answer however well it writes " +
            "GraphQL. Routing it is still correct — it IS a repository question — so a wrong " +
            "answer here is a fabricated number reported as fetched, which is the failure the " +
            "other twelve cannot catch.",
    },
];
