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
 *
 * The marker is the CORE sentence — "I don't have information about that in the
 * documentation" — not the whole message. The polite tail ("I can help with AI SDK docs…") is
 * an offer the model drops or garbles at will, and requiring it made a must-refuse case flaky:
 * the model refused every time, but one run in three said it without the tail and the exact
 * 100-character match missed. Matching the core keeps the meaning and drops the noise. It is
 * still positional (startsWith / "the documentation" prefix), so a partial answer that answers
 * first and declines a clause at the end still does not count — the Day-12 includes() bug
 * cannot come back through here.
 *
 * For a specific out-of-scope topic the model names what it can't cover, in one of two shapes,
 * both complete refusals with nothing substantive said:
 *   a) "The documentation doesn't cover fine-tuning. <core refusal>[tail]" — prefix, then the
 *      core sentence; caught by requiring the core to be present after the prefix.
 *   b) "The documentation doesn't cover fine-tuning a model with the Vercel AI SDK." — the
 *      naming clause ALONE. Caught only when the WHOLE reply is that single sentence, matched on
 *      RAW text so the anchor holds: an injection leak ("…doesn't cover X, but here's how: …")
 *      puts content after the first period and fails it.
 * Neither shape can pass a real answer.
 */

/** The load-bearing first sentence of REFUSAL_MESSAGE. Derived, so the two can't drift. */
const REFUSAL_CORE = REFUSAL_MESSAGE.split(". ")[0];

export function isRefusal(answer: string): boolean {
    const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, "").replace(/\s+/g, " ").trim();
    const n = norm(answer);
    const core = norm(REFUSAL_CORE);
    if (n.startsWith(core) || (n.startsWith("the documentation") && n.includes(core))) return true;
    // Shape (b): the reply is nothing but a single "The documentation doesn't cover <topic>."
    // sentence. Anchored on RAW text (^…$) so any substantive continuation disqualifies it.
    return /^the documentation does(n't| not) cover [^.]+\.?\s*$/i.test(answer.trim());
}
