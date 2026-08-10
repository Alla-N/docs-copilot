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
    claim: z.string().describe("one atomic factual claim, restated from the answer"),
    quote: z
        .string()
        .describe(
            "VERBATIM text copied character-for-character from a source that states this claim. " +
            "At least 20 characters. Empty string if no source states it. Never paraphrase — " +
            "the quote is checked against the source text automatically."
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
    /** Per-claim verification detail, so a failure can be inspected instead of guessed at. */
    checked: { claim: string; quote: string; found: boolean }[];
};

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
function flatten(md: string): string {
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

/** Exact match on flattened text, with a narrow allowance for minor re-spacing. */
function isQuoteFound(quote: string, haystack: string): boolean {
    const q = flatten(quote).toLowerCase();
    if (q.length < 20) return false;              // "streamText" proves nothing
    if (haystack.includes(q)) return true;

    // Fallback: nearly all significant words present. Deliberately strict — 90% and at
    // least 5 words — so it rescues re-spacing, not invention.
    // Strip trailing punctuation from tokens: "customprovider." failed to match
    // "customprovider" and dragged a real quote to 8/9 = 0.889, just under the bar.
    const words = q
        .split(" ")
        .map((w) => w.replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, ""))
        .filter((w) => w.length > 3);
    if (words.length < 5) return false;
    const hits = words.filter((w) => haystack.includes(w)).length;
    return hits / words.length >= 0.9;
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

WORK THROUGH THE ANSWER IN ORDER, SENTENCE BY SENTENCE. Every sentence that states
something factual must produce at least one claim, including the LAST sentence. A claim
you do not list is never checked, so omitting one silently passes it. Do not summarise
and do not skip sentences that look like asides.

Break the answer down completely. Code examples, import paths, API names, described
behaviours and use cases are each separate claims — a code block is not one claim.

For each claim, copy a VERBATIM span FROM THE SOURCES that states it — character for
character, at least 20 characters. Do not paraphrase, do not reconstruct from memory.

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
    const checked = output.claims.map((c) => ({
        claim: c.claim,
        quote: c.quote,
        found: isQuoteFound(c.quote, haystack),
    }));
    const unsupportedClaims = checked.filter((c) => !c.found).map((c) => c.claim);

    return {
        ...output,
        supported: unsupportedClaims.length === 0,
        unsupportedClaims,
        checked,
    };
}
