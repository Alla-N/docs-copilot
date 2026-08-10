/**
 * The golden set. Small on purpose — every case is here because it was OBSERVED
 * failing or succeeding during development, not invented to pad a number.
 *
 * `expectedSource` is the doc page that should survive reranking for this query.
 * `shouldAnswer: false` means the corpus genuinely does not cover it and the
 * correct behaviour is refusal.
 */
export type EvalCase = {
    id: string;
    query: string;
    shouldAnswer: boolean;
    /** Substring of the source_url that must appear in the retrieved set. */
    expectedSource?: string;
    note?: string;
};

export const CASES: EvalCase[] = [
    {
        id: "new-7",
        query: "what is new in AI SDK 7",
        shouldAnswer: true,
        expectedSource: "migration-guide-7-0",
        note: "TARGET: refuses 5/5 despite retrieval scoring 0.768. Generation-side bug.",
    },
    {
        id: "changed-7",
        query: "what was changed in AI SDK 7",
        shouldAnswer: true,
        expectedSource: "migration-guide-7-0",
        note: "Near-synonym of new-7 that passes. The pair is the whole diagnosis.",
    },
    {
        id: "stream-text",
        query: "how do I stream text",
        shouldAnswer: true,
        expectedSource: "stream-text",
        note: "Vector search ranked Agents:Building Agents (0.594) above streamText (0.562).",
    },
    {
        id: "tool-calling",
        query: "what is tool calling",
        shouldAnswer: true,
        expectedSource: "tools-and-tool-calling",
    },
    {
        id: "embeddings",
        query: "how do I use embeddings",
        shouldAnswer: true,
        expectedSource: "embeddings",
    },
    {
        id: "guard-aws",
        query: "how do I deploy to AWS",
        shouldAnswer: false,
        note: "GUARDRAIL: the dangerous near-miss. Scored 0.349 on cosine — adjacent, not covered.",
    },
    {
        id: "guard-france",
        query: "what is the capital of France",
        shouldAnswer: false,
        note: "GUARDRAIL: trivially out of corpus. If this ever answers, grounding is bypassed.",
    },

    // ── Guardrails that actually reach the model ─────────────────────────────
    // aws/france are held by the THRESHOLD alone (0 chunks survive) — they cannot
    // detect the prompt being loosened. These two put plausible-looking context in
    // front of the model and make the PROMPT do the refusing. Measured, not assumed:
    // 'how much does gpt-4o-mini cost per token' was also tried and dropped — it
    // retrieved nothing past 0.3, so it was a third copy of the same blind test.
    {
        id: "guard-finetune",
        query: "how do I fine-tune a model with the AI SDK",
        shouldAnswer: false,
        note: "Reaches the model: 5 chunks, top 0.561. Tests the PROMPT, not the threshold.",
    },
    {
        id: "guard-ratelimit",
        query: "what is the rate limit for the OpenAI provider",
        shouldAnswer: false,
        note: "Reaches the model at 0.301 — one chunk, just over the line. The most sensitive guardrail.",
    },
];
