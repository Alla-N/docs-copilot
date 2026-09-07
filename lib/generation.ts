/**
 * The generation step's settings, shared by the route (streamText) and the evals
 * (generateText). Invariants #3 and #4 in CLAUDE.md say production and the harness run the
 * same pipeline and both answer the planner's RESOLVED query — this module is what makes
 * that structural rather than a matter of keeping two call sites in sync by hand. The
 * calibration script had drifted (raw `retrieve` + raw query) precisely because it had its
 * own copy.
 */
import type { ModelMessage } from "ai";
import { openai } from "@ai-sdk/openai";

import { buildSystemPrompt, type RetrievedChunk } from "./retrieve";

export const GENERATION_MODEL = process.env.GENERATION_MODEL ?? "gpt-4o-mini";

/**
 * Hard cap on answer length. A grounded docs answer with one code block is 300–600 tokens;
 * the longest observed eval answer is under 800. The cap bounds a runaway answer's cost
 * (and its latency) without touching a normal one. Priced in the README's cost section.
 */
export const MAX_OUTPUT_TOKENS = Number(process.env.MAX_OUTPUT_TOKENS ?? 1024);

/** Everything but `messages` for a grounded generation call. Spread into streamText/generateText. */
export function generationSettings(relevant: RetrievedChunk[]) {
    return {
        model: openai(GENERATION_MODEL),
        temperature: 0,
        maxOutputTokens: MAX_OUTPUT_TOKENS,
        system: buildSystemPrompt(relevant),
    };
}

/**
 * Plan-and-execute, execute half: the final user turn the model answers is the planner's
 * resolved sub-queries, not the raw message. The answerer anchors on the literal last turn,
 * so a terse "What is SDK?" gets refused against context its explicit twin answers from, and
 * adversarial noise primes it to drop legitimate intents. Swapping the turn fixes both:
 * shorthand is already expanded and dropped riders never reach generation. Safe — a wholly
 * off-topic message is gated before this point, so nothing is rewritten into something
 * answerable. `history` is everything before the current question.
 */
export function generationMessages(
    history: ModelMessage[],
    rawQuestion: string,
    subQueries: string[]
): ModelMessage[] {
    const content = subQueries.length ? subQueries.join("\n") : rawQuestion;
    return [...history, { role: "user", content }];
}
