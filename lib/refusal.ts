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
export const REFUSAL_MESSAGE = "I don't have information about that in the documentation.";

export function isRefusal(answer: string): boolean {
    const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, "").replace(/\s+/g, " ").trim();
    return norm(answer).includes(norm(REFUSAL_MESSAGE));
}
