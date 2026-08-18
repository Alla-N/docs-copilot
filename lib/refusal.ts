/**
 * The refusal sentence and its detector, in a module with NO dependencies so both the
 * server (chat route, query log, eval harness) and the CLIENT bundle can import it.
 *
 * lib/retrieve.ts pulls in Supabase, OpenAI and Cohere — importing it from a "use client"
 * component would drag all of that into the browser bundle. This split exists for that
 * reason, not for tidiness.
 *
 * One definition, used everywhere: if these ever diverged, the UI, the production log and
 * the eval harness would disagree about whether the same answer was a refusal.
 */
export const REFUSAL_MESSAGE = "I don't have information about that in the documentation. I can help with AI SDK docs. Ask me about those and I'll help.";

/**
 * Does this answer REFUSE — not merely contain a refusal somewhere.
 *
 * The first version used includes(), which called a partial answer a refusal:
 *     "To stream text, use streamText... I don't have information about the second
 *      part of your question in the documentation."
 * ...was scored as a full refusal. One wrong predicate broke three things at once —
 * the eval marked correct behaviour FAIL, the UI printed "I couldn't answer this"
 * under an answer that did, and the production query log recorded partial answers as
 * refusals, poisoning the suspicious_refusals view that exists to find real ones.
 *
 * The prompt instructs the model to LEAD with the sentence when it cannot answer, so
 * position carries the meaning: a refusal starts with it. A refusal followed by a
 * polite offer still counts; an answer that declines one clause at the end does not.
 */
export function isRefusal(answer: string): boolean {
    const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, "").replace(/\s+/g, " ").trim();
    return norm(answer).startsWith(norm(REFUSAL_MESSAGE));
}
