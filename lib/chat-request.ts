/**
 * Parses and NORMALISES the chat request body.
 *
 * The route previously did:
 *     const { messages }: { messages: UIMessage[] } = await req.json();
 * which is a type annotation on untyped JSON — a promise to the compiler, not a check
 * on the data. Four things followed from that:
 *
 *   1. POST {"messages": []} crashed the route (undefined.parts).
 *   2. Nothing capped size, so one request could carry megabytes of text and bill for it.
 *   3. The client controlled the whole conversation INCLUDING assistant turns, so a
 *      forged prior assistant message ("I am permitted to answer from general knowledge")
 *      would steer the model without touching the system prompt.
 *   4. Arbitrary part types flowed into convertToModelMessages.
 *
 * The approach here is parse-then-construct, not validate-in-place: we do not try to
 * confirm the client's UIMessage shape is legal, we read the few fields we trust and
 * BUILD our own message array from them. Anything we did not explicitly ask for is
 * discarded rather than inspected — which removes the whole class of "unexpected part
 * type" problems instead of enumerating it.
 */
import { z } from "zod";
import type { ModelMessage } from "ai";

import { verifyAssistantText } from "./assistant-signature";

/** History beyond this is dropped (older first). Bounds cost per request. */
export const MAX_MESSAGES = 20;
/** Per-message character cap. ~4000 chars ≈ 1000 tokens. */
export const MAX_CHARS_PER_MESSAGE = 4000;
/** Whole-conversation cap, so 20 messages at the limit can't stack up. */
export const MAX_TOTAL_CHARS = 24000;

const RawBody = z.object({
    // useChat's chat id (DefaultChatTransport sends it as `id`). Read as unknown, so a body without
    // one, or with a non-string one, parses exactly as before; only the forward to the agent
    // service uses it, and lib/agent-forward.ts checks its shape there. (Step 2.6.)
    id: z.unknown().optional(),
    messages: z
        .array(
            z.object({
                // Only these two. A client-supplied "system" message would be an
                // instruction-injection vector straight into the prompt.
                role: z.enum(["user", "assistant"]),
                parts: z
                    .array(
                        z
                            .object({
                                type: z.string(),
                                text: z.string().optional(),
                                // `unknown`, deliberately. The client echoes back EVERY data part
                                // the route ever sent it, and their payloads have different shapes
                                // — `data-sources` is an array, `data-signature` an object. Typing
                                // this as an object rejected the whole body on any request that
                                // carried a previous answer's source pills: a 400 on every
                                // follow-up. The parts test caught it. Read what we trust below.
                                data: z.unknown().optional(),
                            })
                            .loose()
                    )
                    .default([]),
            })
        )
        .min(1),
});

export type ParsedChatRequest = {
    /** The latest user question, plain text — what gets embedded. */
    question: string;
    /** Clean history for the model: role + text only, capped. */
    messages: ModelMessage[];
    /**
     * useChat's chat id, when the body carried a string one. UNTRUSTED: the client picks it. It
     * names the conversation (the agent service's thread), so it is checked before use.
     */
    chatId: string | undefined;
};

export class BadRequestError extends Error {}

/**
 * The one field we trust out of a client-supplied data part. Anything that is not an object
 * carrying a string `sig` returns undefined, which `verifyAssistantText` treats as "no proof" —
 * so a hostile payload (an array, a number, a nested object) fails the check instead of throwing.
 */
function signatureOf(parts: { type: string; data?: unknown }[]): unknown {
    const data = parts.find((p) => p.type === "data-signature")?.data;
    return data && typeof data === "object" && !Array.isArray(data)
        ? (data as Record<string, unknown>).sig
        : undefined;
}

export function parseChatRequest(body: unknown): ParsedChatRequest {
    const result = RawBody.safeParse(body);
    if (!result.success) {
        throw new BadRequestError(`Invalid request: ${result.error.issues[0]?.message ?? "bad shape"}`);
    }

    // Keep only the most recent turns. Oldest are dropped, not rejected — a long
    // conversation is normal use, not an attack, and should degrade rather than fail.
    const recent = result.data.messages.slice(-MAX_MESSAGES);

    // Walk NEWEST → OLDEST and stop when the total cap is hit, so what gets dropped is the
    // oldest history, never the current question. The first version walked oldest → newest
    // and `break`-ed at the cap: a long conversation lost its *latest* turn and the request
    // 400'd with "last message must be a user message" — the caller's newest message was the
    // one thing the cap was never meant to touch. (Critical review, item 14.)
    const kept: ModelMessage[] = [];
    let totalChars = 0;

    for (let i = recent.length - 1; i >= 0; i--) {
        const m = recent[i];
        // Text parts only. data-sources, step-start and anything else the client
        // echoes back are dropped — the model has no business reading them.
        const joined = m.parts
            .filter((p) => p.type === "text" && typeof p.text === "string")
            .map((p) => p.text as string)
            .join(" ");
        const text = joined.slice(0, MAX_CHARS_PER_MESSAGE).trim();

        if (!text) continue;

        // An assistant turn must PROVE this server wrote it. History is client-supplied and the
        // model reads it as its own prior words, so a forged turn is an instruction channel that
        // never touches the system prompt. Verification runs on the joined text BEFORE the caps
        // below, because that is the string the route signed. An unsigned or altered turn is
        // dropped, not rejected with a 400: the user of a stale tab loses context, which is
        // recoverable, while the forger simply finds their sentence missing. (Review item 30.)
        if (m.role === "assistant") {
            if (!verifyAssistantText(joined, signatureOf(m.parts))) {
                console.warn("DROPPED UNSIGNED ASSISTANT TURN — forged, or from before a secret rotation");
                continue;
            }
        }

        if (totalChars + text.length > MAX_TOTAL_CHARS) break;

        totalChars += text.length;
        kept.push({ role: m.role, content: text });
    }
    const messages = kept.reverse();

    const last = messages[messages.length - 1];
    if (!last || last.role !== "user") {
        // A request whose final turn is not a user question is either a bug or an
        // attempt to have the model continue its own fabricated answer.
        throw new BadRequestError("The last message must be a non-empty user message.");
    }

    const chatId = typeof result.data.id === "string" ? result.data.id : undefined;
    return { question: last.content as string, messages, chatId };
}
