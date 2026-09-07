/**
 * Faithfulness judge — is every claim in the answer supported by the retrieved chunks?
 *
 * This is a MODEL, not a measurement. It hallucinates, it has opinions, and it is
 * biased toward approving output from its own family. Nothing it says is worth
 * anything until it has been calibrated against answers whose faithfulness is known.
 * See evals/calibrate-judge.ts — run that before trusting any number from here.
 *
 * Scope note: faithfulness asks whether the answer is grounded in the provided text.
 * It does NOT ask whether the answer is true. If the docs are wrong, a perfectly
 * faithful answer repeats the error and this judge approves it.
 */
import { generateText, Output } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";

import type { RetrievedChunk } from "../lib/retrieve";

/**
 * Structured output, not free text. The model is forced into this shape by the SDK,
 * so there is no parsing step that can silently mis-read a verdict.
 *
 * NOTE: generateObject() is deprecated in ai@7 — the type definition says
 * "Use `generateText` with an `output` setting instead". So this uses
 * generateText + Output.object(), and the result arrives on `.output`.
 */
const ClaimSchema = z.object({
    claim: z.string().describe("one atomic claim, restated from the answer"),
    kind: z
        .enum(["factual", "scaffolding", "scope"])
        .describe(
            "factual: states something about the SDK that a source could confirm (APIs, options, " +
            "behaviour, defaults, code). scaffolding: a sentence with no checkable content — an " +
            "intro ('Here is an example:'), a heading, a closing remark ('These changes improve…'). " +
            "scope: a statement about what the documentation does or does not cover."
        ),
    quote: z
        .string()
        .describe(
            "For factual claims: VERBATIM text copied character-for-character from a source that " +
            "states this claim, at least 20 characters. If you must skip words, write … between " +
            "two verbatim spans. Empty string if no source states it. Never paraphrase — the quote " +
            "is checked against the source text automatically. Empty for scaffolding and scope."
        ),
});

/**
 * Claim-level with VERIFIABLE evidence. Two failed designs got here, both caught by
 * calibration rather than reasoning:
 *
 *   v1  one global "supported" boolean over the whole answer.
 *       -> missed 2/10. A single verdict lets the model average across claims: it saw the
 *          word "streamText" in a Tool Calling chunk and approved an entire code example.
 *
 *   v2  per-claim `supportedBy: number | null`.
 *       -> missed 5/10, WORSE. Models are very reluctant to emit null; given five sources
 *          and a claim, it picks the nearest one. Five chances to gesture at a source is
 *          more room to hand-wave, not less.
 *
 *   v3  per-claim VERBATIM QUOTE, verified in code against the actual chunk text.
 *       The model proposes evidence; deterministic string matching disposes. A claim whose
 *       quote does not literally appear is unsupported regardless of what the model claimed.
 *
 *   v4  per-claim KIND (factual / scaffolding / scope) + verbatim quote, with the kind
 *       trusted only where the code can't see checkable content (mustBeFactual).
 *       -> v3 recalibrated on Day 15 at 12/12 FALSE ALARMS (0 missed lies). Not the
 *          matcher: the answers had changed. The format contract (Day 14) made every answer
 *          open with "Here's an example:" and close with "The documentation doesn't cover
 *          X" — sentences with nothing to quote — and v3 required a quote for every
 *          sentence. A judge calibrated on one answer style is not calibrated on another;
 *          that is why the calibration runs again in CI every week.
 *
 * The lesson generalises past judges: when you need to trust a model inside a pipeline,
 * ask it for evidence you can check, not for a conclusion you have to believe.
 */
export const VerdictSchema = z.object({
    claims: z.array(ClaimSchema).describe("every distinct factual claim made in the answer"),
    reasoning: z.string().describe("one or two sentences summarising the verdict"),
});

/** Exactly what the model returns — this is what the schema is pinned to. */
export type RawVerdict = z.infer<typeof VerdictSchema>;

export type Verdict = RawVerdict & {
    /** Derived in code, not by the model: every claim must be supported. */
    supported: boolean;
    unsupportedClaims: string[];
    /** Per-claim verification detail, so a failure can be inspected instead of guessed at.
     *  found: true/false for factual claims; "skipped" for scaffolding/scope, which carry no
     *  checkable content and do not count either way. */
    checked: { claim: string; kind: "factual" | "scaffolding" | "scope"; quote: string; found: boolean | "skipped" }[];
};

/**
 * The model labels claim kinds, and a label is an escape hatch: a fabrication tagged
 * "scaffolding" would never be checked. So the label is only trusted for sentences that
 * COULD be scaffolding. Anything carrying an identifier (backticks, camelCase), a number, a
 * date or a quoted value is factual whatever the model called it — which is exactly the
 * shape every planted fabrication takes (`streamTextSync`, `maxRetries` is 5, 12 March
 * 2026, { cache: 'aggressive' }). Calibration measures whether that guard is enough.
 */
/** Does the ANSWER itself say the documentation doesn't cover something? Only then may a
 *  claim be `scope`. A judge that finds no support is tempted to invent a scope claim
 *  ("the documentation does not cover changes in version 7") for an answer that never said
 *  so — observed on a swapped calibration sample. The answer's own words are the check. */
export function answerHasScopeStatement(answer: string): boolean {
    return /\b(?:documentation|docs?|sources?) (?:do(?:es)?n[o']t|does not|do not) (?:cover|mention|include|describe|discuss|provide|address|explain|say)\b/i.test(answer);
}

/** Is this CLAIM itself a plain scope statement — one clause, nothing asserted about the
 *  topic? "The documentation doesn't cover additional options for `streamText`." is scope
 *  even though it names an API; "…doesn't cover X, but `fooSync` does it" is not one clause. */
export function isScopeClaim(claim: string): boolean {
    return /^(?:the )?(?:documentation|docs?|sources?) (?:do(?:es)?n[o']t|does not|do not) (?:cover|mention|include|describe|discuss|provide|address|explain|say)\b[^.,;:]*\.?\s*$/i.test(claim.trim());
}

/**
 * Identifiers a factual sentence carries: backticked tokens, camelCase names, quoted values,
 * braces. If a sentence in the answer carries one and NO listed claim mentions it, the judge
 * skipped that sentence — and a skipped sentence is never checked. Observed: a planted
 * "{ cache: 'aggressive' }" sentence at the end of a long two-part answer was simply not
 * listed, and the answer was approved. Coverage is checked on prose only; code blocks are
 * excluded because their lines are legitimately summarised into claims.
 */
/**
 * The checkable tokens a sentence carries, lower-cased: backticked spans, camelCase names,
 * dotted names (Node.js, result.textStream), quoted values, and numbers of two or more
 * digits once the product version is stripped. Numbers were left out at first ("too common
 * to anchor on") — and the one planted fabrication with no identifier in it, "released on
 * 12 March 2026 and requires Node.js 24", walked straight past the coverage check. A date
 * or a version requirement IS the claim; it anchors as well as an API name does.
 */
function identifiers(sentence: string): string[] {
    const withoutVersion = sentence.replace(/\b(?:AI SDK|SDK|version|v)\s?\d+(?:\.\d+)*\b/gi, "");
    return [
        ...(sentence.match(/`([^`]+)`/g) ?? []).map((m) => m.slice(1, -1)),
        ...(sentence.match(/\b[a-z]+[A-Z][A-Za-z]*\b/g) ?? []),
        ...(sentence.match(/\b[A-Za-z]+\.[A-Za-z]{2,}\b/g) ?? []),
        ...(sentence.match(/['"]([^'"]{2,40})['"]/g) ?? []).map((m) => m.slice(1, -1)),
        ...(withoutVersion.match(/\b\d{2,}(?:\.\d+)*\b/g) ?? []),
    ].map((x) => x.toLowerCase());
}

export function unlistedFactualSentences(answer: string, claims: string[]): string[] {
    const prose = answer.replace(/```[\s\S]*?```/g, " ");
    const sentences = prose
        .split(/\n+|(?<=[.!?])\s+/)
        .map((x) => x.replace(/^[\s#>*-]+/, "").trim())
        .filter((x) => x.length > 15);
    const claimText = claims.join(" \n ").toLowerCase();
    const missing: string[] = [];
    for (const sentence of sentences) {
        if (!mustBeFactual(sentence)) continue;
        const ids = identifiers(sentence);
        if (ids.length === 0) continue; // numbers alone are too common to anchor on
        if (!ids.some((id) => claimText.includes(id))) missing.push(sentence);
    }
    return missing;
}

export function mustBeFactual(claim: string): boolean {
    // The product version alone ("In AI SDK 7, several changes were made") is how an intro
    // sentence names the topic, not a checkable number — strip it before looking for digits.
    const withoutVersion = claim.replace(/\b(?:AI SDK|SDK|version|v)\s?\d+(?:\.\d+)*\b/gi, "");
    return /`|[a-z][A-Z]|\d|[{}]|(?:^|[\s(:,])['"][^'"]{1,40}['"]/.test(withoutVersion);
}

/**
 * Judging with the same model that generated the answer is a known weakness —
 * models favour their own output. Overridable so calibration can test a stronger
 * judge and compare, which is the honest way to find out whether it matters here.
 */
const JUDGE_MODEL = process.env.JUDGE_MODEL ?? "gpt-4o-mini";

/**
 * Markdown syntax is invisible to a reader, so a model quoting "[tools](/docs/x) are
 * objects" will write "tools are objects" — verbatim in meaning, not in bytes. v3 asked
 * for verbatim quotes from markdown and then matched against markdown: 0 missed lies,
 * but 5/5 FALSE ALARMS, because no legitimate quote could ever match.
 *
 * So flatten once, show the judge the flattened text, and verify against the same string.
 * The verifier has to compare like with like.
 */
export function flatten(md: string): string {
    return md
        .replace(/```/g, " ")                        // code fences, keep the code itself
        .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")      // [text](url) -> text
        // Inline delimiters WRAP a word — delete them. Replacing them with a space invents
        // whitespace that was never in the sentence: `customProvider`. becomes
        // "customProvider ." which no human or model would ever reproduce. This cost two
        // calibration rounds of false alarms.
        .replace(/[`*]/g, "")
        // Line-level markers separate things — a space is correct here.
        .replace(/[_#>|]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

/** Tokens with edge punctuation stripped: "customprovider." and "customprovider" are one word. */
function tokens(s: string): string[] {
    // Hyphens split like spaces: "provider-agnostic" and "provider agnostic" are the same
    // two words — a judge normalises hyphens without noticing.
    return s
        .toLowerCase()
        .split(/[\s-]+/)
        .map((w) => w.replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, ""))
        .filter(Boolean);
}

const MIN_CONTIGUOUS_WORDS = 5;

/**
 * Exact match on flattened text, with a narrow allowance for edge punctuation and one
 * clipped edge word — and NOTHING looser.
 *
 * The first fallback was bag-of-words: "90% of the significant words present anywhere in
 * the haystack". That accepts a sentence assembled from corpus vocabulary — exactly what a
 * fabricating model produces — because word presence is not evidence of a quote. A quote is
 * CONTIGUOUS. So the fallback demands a contiguous run of all but one of the quote's words
 * (and at least 5), verbatim and in order, in the haystack. Why only one: an 80% run was
 * tried first and accepted "The generateText function streams text generations…" against a
 * source that says streamText — the substituted identifier sat at the edge and the run
 * skipped it. A quote that lost one word to clipping is rescued; one with a swapped word
 * is not. (Review item 20.)
 */
/** Letters and digits only. "provider-agnostic" and "provider agnostic" become one string;
 *  so do `activeTools: ['weather'],` and `activeTools: ['weather'].` */
const alnum = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

/** Is `run` (a word array) present contiguously in the tokenised haystack? Returns its index. */
function runIndex(run: string[], hayWords: string[], from = 0): number {
    outer: for (let i = from; i + run.length <= hayWords.length; i++) {
        for (let j = 0; j < run.length; j++) if (hayWords[i + j] !== run[j]) continue outer;
        return i;
    }
    return -1;
}

export function isQuoteFound(quote: string, haystack: string): boolean {
    const qRaw = flatten(quote);
    const q = qRaw.toLowerCase();
    if (q.length < 20) return false;              // "streamText" proves nothing
    if (haystack.includes(q)) return true;

    // An explicitly elided quote ("first span … second span") is several quotes. Fragments
    // under 20 chars are ignored rather than failed — a code quote like `values: [...]`
    // carries the source's OWN ellipsis, and its tail `] });` proves nothing either way.
    if (/\.{3}|…/.test(q)) {
        const spans = q.split(/\.{3}|…/).map((x) => x.trim()).filter((x) => x.length >= 20);
        return spans.length > 0 && spans.every((span) => isQuoteFound(span, haystack));
    }

    // Punctuation-insensitive exact match: same letters and digits, in the same order, with
    // nothing skipped. Rescues a normalised hyphen or a trailing comma-for-period; cannot
    // rescue a recombination, because it is still contiguous.
    const qa = alnum(q);
    if (qa.length >= 15 && alnum(haystack).includes(qa)) return true;

    // Word-level: one contiguous run of all but one word (a clipped edge word). The clipped
    // word may NOT be an identifier or a number: "streamText is ideal for non-interactive…"
    // against a source that says generateText must fail, and it is the first word that
    // differs. Identifiers are checked on the original-case tokens (camelCase, digits).
    const words = tokens(q);
    if (words.length < MIN_CONTIGUOUS_WORDS) return false;
    const rawWords = qRaw.split(/\s+/).map((w) => w.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "")).filter(Boolean);
    const isIdentifier = (i: number) => /[A-Z]|\d/.test(rawWords[i] ?? "");
    const hayWords = tokens(haystack);
    if (runIndex(words, hayWords) >= 0) return true;
    if (words.length - 1 >= MIN_CONTIGUOUS_WORDS) {
        if (!isIdentifier(0) && runIndex(words.slice(1), hayWords) >= 0) return true;
        if (!isIdentifier(words.length - 1) && runIndex(words.slice(0, -1), hayWords) >= 0) return true;
    }

    // …or TWO contiguous runs in source order — an elision the judge forgot to mark ("ideal
    // for non-interactive use cases such as automation tasks ⟨where you need to write text⟩
    // (e.g. drafting email…)"). Nothing from the quote is dropped, so no identifier can hide;
    // a recombined fabrication would need both halves verbatim and in order. The short half
    // must still be 3+ words and the long one 7+, on a quote of 10+ words.
    if (words.length >= 10) {
        for (let k = 3; k <= words.length - 3; k++) {
            if (Math.max(k, words.length - k) < 7) continue;
            const first = runIndex(words.slice(0, k), hayWords);
            if (first < 0) continue;
            if (runIndex(words.slice(k), hayWords, first + k) >= 0) return true;
        }
    }
    return false;
}

/** Ask for verbatim evidence for specific claims only. Returns claim → quote ("" = none). */
async function secondLook(claims: string[], sources: string): Promise<Map<string, string>> {
    const Schema = z.object({
        evidence: z.array(z.object({ claim: z.string(), quote: z.string() })),
    });
    const { output } = await generateText({
        model: openai(JUDGE_MODEL),
        temperature: 0,
        output: Output.object<z.infer<typeof Schema>>({ schema: Schema, name: "second_look" }),
        system: `For each claim below, search the SOURCES carefully and copy one VERBATIM span (at least
20 characters, character for character, … between two spans if you must skip words) that states
the claim. Return the claim text unchanged. If — after reading every source — nothing states it,
return an empty quote. Do not paraphrase; the quote is checked against the source automatically.`,
        prompt: `SOURCES\n${sources}\n\nCLAIMS\n${claims.map((c) => `- ${c}`).join("\n")}`,
    });
    const map = new Map<string, string>();
    for (const e of output.evidence) map.set(e.claim, e.quote);
    // The model may lightly reword a claim key; fall back to positional matching.
    if (map.size === 0 || claims.some((c) => !map.has(c))) {
        output.evidence.forEach((e, i) => { if (claims[i] && !map.has(claims[i])) map.set(claims[i], e.quote); });
    }
    return map;
}

export async function judgeFaithfulness(
    question: string,
    chunks: RetrievedChunk[],
    answer: string
): Promise<Verdict> {
    // Flattened for the judge AND for verification — the same bytes on both sides.
    const sources = chunks
        .map((c, i) => `[Source ${i + 1}] ${c.title}\n${flatten(c.content)}`)
        .join("\n\n---\n\n");
    const haystack = flatten(chunks.map((c) => c.content).join(" ")).toLowerCase();

    const { output } = await generateText({
        model: openai(JUDGE_MODEL),
        temperature: 0,
        // Type parameter pinned explicitly: left to inference, FlexibleSchema<OBJECT>
        // can fail to resolve through zod v4's generics and leaks `core.output<output>`
        // instead of Verdict. Naming the type makes it deterministic.
        output: Output.object<RawVerdict>({
            schema: VerdictSchema,
            name: "faithfulness_verdict",
            description: "Whether every factual claim in the answer is supported by the sources.",
        }),
        system: `You break an answer into its atomic factual claims and check each one
against the sources, individually.

WORK THROUGH THE ANSWER IN ORDER, SENTENCE BY SENTENCE. Every sentence produces at least
one claim, including the LAST sentence, and each claim gets a KIND:
- factual: states something about the SDK a source could confirm — an API name, an option,
  a default, a behaviour, a use case, a line of code. Nearly every claim is factual.
- scaffolding: no checkable content at all — "Here's a basic example of using \`streamText\`:",
  a heading, "To configure \`streamText\`, follow this example:", "Tool calling involves
  several core elements:", a closing "These changes improve the SDK". A sentence that only
  INTRODUCES a list or a code block is scaffolding even if it names the API the list is
  about — the API is checked in the claims that follow. A sentence that states an option,
  a value, a default, a behaviour or a number is NEVER scaffolding.
- scope: a statement about what the documentation covers or does not cover ("The
  documentation doesn't cover error handling"). Not checkable by quote.
A claim you do not list is never checked, so omitting one silently passes it. Do not
summarise and do not skip sentences that look like asides.

Break the answer down completely. Code examples, import paths, API names, described
behaviours and use cases are each separate claims — a code block is not one claim.

For each factual claim, copy a VERBATIM span FROM THE SOURCES that states it — character
for character, at least 20 characters. Do not paraphrase, do not reconstruct from memory,
do not silently drop words from the middle of a span: if you need to skip words, write …
between two verbatim spans, each at least 20 characters.

The quote must come from the SOURCES section. Copying the answer's own wording back is
not evidence — it is circular, and it is automatically rejected. If the source says the
same thing in different words, quote the SOURCE's words.
The quote is checked automatically against the source text; an inexact quote counts as
no evidence at all.

If no source states the claim, return an empty string. That is the correct and expected
answer for many claims — an empty quote is not a failure on your part.

A source does NOT support a claim merely by mentioning the same terms. If the answer
shows \`import { streamText } from 'ai'\` and a source only says the word "streamText"
in passing, that claim is UNSUPPORTED. The source must contain the claim's content.

Judge against the sources ONLY, never your own knowledge — a claim can be true in
general and still unsupported here.

Ignore citation markers like "(Source 1)"; they are formatting, not claims.`,
        prompt: `QUESTION
${question}

SOURCES
${sources}

ANSWER TO CHECK
${answer}`,
    });

    // The verdict is computed here, not asked for. A model given a global "supported"
    // field will average across claims; a model forced to fill one row per claim cannot.
    // Verify every quote against the source text. This is the part the model cannot fake.
    const scopeAllowed = answerHasScopeStatement(answer);
    const factualClaimText = output.claims
        .filter((c) => c.kind === "factual")
        .map((c) => c.claim.toLowerCase())
        .join(" \n ");
    // "Here's a basic example of using `streamText`:" names an API and asserts nothing. The
    // guard would force it factual for the backtick; it stands down only when the sentence
    // introduces something (ends with ":") AND every identifier in it is checked in some
    // factual claim anyway — the API is not escaping, it is being verified elsewhere.
    const coveredElsewhere = (claim: string) =>
        /:\s*$/.test(claim.trim()) &&
        identifiers(claim).every((id) => factualClaimText.includes(id));
    const checked = output.claims.map((c) => {
        // The label is the model's; these are the code's overrides of it.
        const plainScope = scopeAllowed && isScopeClaim(c.claim);
        const factual =
            !plainScope &&
            (c.kind === "factual" ||
                (mustBeFactual(c.claim) && !(c.kind === "scaffolding" && coveredElsewhere(c.claim))) ||
                (c.kind === "scope" && !scopeAllowed)); // invented scope claim → must show evidence
        return {
            claim: c.claim,
            kind: factual ? ("factual" as const) : plainScope ? ("scope" as const) : c.kind,
            quote: c.quote,
            found: factual ? isQuoteFound(c.quote, haystack) : ("skipped" as const),
        };
    });

    // Sentences the judge never listed cannot have been checked. Each counts as unsupported.
    const unlisted = unlistedFactualSentences(answer, checked.map((c) => c.claim));
    for (const sentence of unlisted) {
        checked.push({ claim: `[not listed by the judge] ${sentence}`, kind: "factual", quote: "", found: false });
    }

    // SECOND LOOK at exactly the claims that failed. The judge stops looking too early —
    // observed: one listed claim for a fourteen-claim answer, and empty quotes for sentences
    // the source states almost verbatim. Asking again for only the failed claims turns that
    // laziness into evidence when evidence exists. It cannot turn it into a false pass: the
    // second-pass quotes go through the same verifier, and a claim with no real support still
    // has no quote that matches. (Calibration: false alarms 3/12 → measured after this.)
    const failed = checked.filter((c) => c.found === false);
    if (failed.length > 0) {
        const retried = await secondLook(failed.map((c) => c.claim.replace(/^\[not listed by the judge\] /, "")), sources);
        for (const c of checked) {
            if (c.found !== false) continue;
            const again = retried.get(c.claim.replace(/^\[not listed by the judge\] /, ""));
            if (again && isQuoteFound(again, haystack)) {
                c.quote = again;
                c.found = true;
            }
        }
    }

    // An answer with content but zero factual claims is the collapse-everything escape hatch.
    const factualCount = checked.filter((c) => c.kind === "factual").length;
    if (factualCount === 0 && answer.trim().length > 80) {
        checked.push({ claim: "[judge listed no factual claims for a substantive answer]", kind: "factual", quote: "", found: false });
    }

    const unsupportedClaims = checked.filter((c) => c.found === false).map((c) => c.claim);

    return {
        ...output,
        supported: unsupportedClaims.length === 0,
        unsupportedClaims,
        checked,
    };
}
