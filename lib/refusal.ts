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
 * For a specific out-of-scope topic the model names what it can't cover. Observed shapes:
 *   a) "The documentation doesn't cover fine-tuning. <core refusal>[tail]"
 *   b) "The documentation doesn't cover fine-tuning a model with the Vercel AI SDK."
 *   c) "The documentation doesn't cover fine-tuning …. I can help with AI SDK docs. Ask me
 *      about those and I'll help."  — prefix + polite tail, NO core sentence.
 * The first detector listed (a) and (b) one by one, and then (c) showed up (1 run in 3 on
 * guard-finetune) and slipped through. Enumerating shapes is a losing game, so the rule is now
 * COMPOSITIONAL: a reply is a refusal when its first sentence is a negative statement about the
 * docs ("The documentation doesn't cover/mention/… X") and EVERY sentence after it is one of the
 * canonical refusal sentences from REFUSAL_MESSAGE — or there are none. That covers (a), (b),
 * (c) and any recombination of them, and nothing else: a leak ("…doesn't cover X. The capital
 * of France is Paris.") has a non-canonical sentence and fails; the topic clause may not carry
 * clause punctuation (",;:") so "…doesn't cover X, but here's how: …" cannot hide inside it.
 */

/** The load-bearing first sentence of REFUSAL_MESSAGE. Derived, so the two can't drift. */
const REFUSAL_CORE = REFUSAL_MESSAGE.split(". ")[0];

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, "").replace(/\s+/g, " ").trim();

/** Sentence boundary: terminal punctuation followed by whitespace. norm() strips the punctuation. */
const SENTENCE_BOUNDARY = /[.!?]\s+/;

/** Every sentence of REFUSAL_MESSAGE, normalised. Derived — reword the message and this follows. */
const CANONICAL_SENTENCES = new Set(REFUSAL_MESSAGE.split(SENTENCE_BOUNDARY).map(norm).filter(Boolean));

/**
 * "The documentation doesn't cover <topic>." as an opening sentence. The verb list is closed on
 * purpose — "does not require…" or "does not throw…" can open a real answer; these verbs cannot.
 * The topic may not contain ",;:" (see above). The final period is optional only at the very
 * end of the reply, so "…doesn't cover X" alone still counts.
 */
const NEGATIVE_OPENER =
    /^the documentation (?:does(?:n't|n’t| not) (?:cover|mention|include|discuss|describe|contain|provide|address|explain)|has no (?:information|details?|guidance))\b[^.,;:]*(?:[.!]\s*|$)/i;

export function isRefusal(answer: string): boolean {
    const raw = answer.trim();
    const n = norm(raw);
    const core = norm(REFUSAL_CORE);
    // The canonical refusal, or a "The documentation…" opener followed by the core sentence
    // anywhere. The second form is the one place includes() is safe: the opener already rules
    // out an answer-first reply, and the core sentence contradicts any answer before it.
    if (n.startsWith(core) || (n.startsWith("the documentation") && n.includes(core))) return true;

    // Compositional: negative opener + nothing but canonical refusal sentences after it.
    const opener = NEGATIVE_OPENER.exec(raw);
    if (!opener) return false;
    const rest = raw.slice(opener[0].length).trim();
    if (rest === "") return true;
    return rest.split(SENTENCE_BOUNDARY).map(norm).filter(Boolean).every((s) => CANONICAL_SENTENCES.has(s));
}
