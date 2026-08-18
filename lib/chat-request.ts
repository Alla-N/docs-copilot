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

/** History beyond this is dropped (older first). Bounds cost per request. */
export const MAX_MESSAGES = 20;
/** Per-message character cap. ~4000 chars ≈ 1000 tokens. */
export const MAX_CHARS_PER_MESSAGE = 4000;
/** Whole-conversation cap, so 20 messages at the limit can't stack up. */
export const MAX_TOTAL_CHARS = 24000;

const RawBody = z.object({
    messages: z
        .array(
            z.object({
                // Only these two. A client-supplied "system" message would be an
                // instruction-injection vector straight into the prompt.
                role: z.enum(["user", "assistant"]),
                parts: z
                    .array(z.object({ type: z.string(), text: z.string().optional() }).loose())
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
};

export class BadRequestError extends Error {}

export function parseChatRequest(body: unknown): ParsedChatRequest {
    const result = RawBody.safeParse(body);
    if (!result.success) {
        throw new BadRequestError(`Invalid request: ${result.error.issues[0]?.message ?? "bad shape"}`);
    }

    // Keep only the most recent turns. Oldest are dropped, not rejected — a long
    // conversation is normal use, not an attack, and should degrade rather than fail.
    const recent = result.data.messages.slice(-MAX_MESSAGES);

    const messages: ModelMessage[] = [];
    let totalChars = 0;

    for (const m of recent) {
        // Text parts only. data-sources, step-start and anything else the client
        // echoes back are dropped — the model has no business reading them.
        const text = m.parts
            .filter((p) => p.type === "text" && typeof p.text === "string")
            .map((p) => p.text as string)
            .join(" ")
            .slice(0, MAX_CHARS_PER_MESSAGE)
            .trim();

        if (!text) continue;
        if (totalChars + text.length > MAX_TOTAL_CHARS) break;

        totalChars += text.length;
        messages.push({ role: m.role, content: text });
    }

    const last = messages[messages.length - 1];
    if (!last || last.role !== "user") {
        // A request whose final turn is not a user question is either a bug or an
        // attempt to have the model continue its own fabricated answer.
        throw new BadRequestError("The last message must be a non-empty user message.");
    }

    return { question: last.content as string, messages };
}
