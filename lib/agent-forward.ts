/**
 * Forwarding a chat request to the Python agent service (agent/). Step 2.6 of the Windward plan.
 *
 * With AGENT_URL set, the route still rate-limits and parses the useChat body (invariant 9, the
 * signature check included), then hands the question to the service and returns the service's
 * stream as it is: a byte pipe, `new Response(upstream.body)`. The service already speaks the AI
 * SDK UI message stream protocol (agent/src/copilot_agent/ui_stream.py), and
 * tests/python-stream-contract.test.ts shows the real client builds the same message from its bytes
 * as from this route. Parsing and re-emitting the stream here would be a second implementation of
 * the protocol sitting in the path.
 *
 * What goes over, and what does not:
 *   - thread_id: useChat's chat id. The service keeps the conversation under it (LangGraph
 *     checkpointer, step 2.5), so NO history is forwarded: the service reads the turns it
 *     recorded itself, and there is no client-supplied assistant text left to forge or replay.
 *   - question: the parsed last user message.
 *   - origin "web", and the visitor attribution (lib/visitor.ts): the service writes the query_log
 *     row now (agent/src/copilot_agent/query_log.py), because only it sees the tokens, the rerank
 *     calls and the timings.
 *   - the request's abort signal: a closed tab aborts this fetch, the socket to the service
 *     closes, and the service cancels the run (planner, searches or the model's stream).
 *
 * Failures before the stream starts come back as a 502 with the route's generic message; the
 * service's own error text never reaches the browser (invariant 9: never String(err) to the
 * client). A failure inside the stream is already an error chunk in a 200 and passes through.
 * The route cannot see that one: the service's log (and, in 2.7, Langfuse) has to.
 */
import { UI_MESSAGE_STREAM_HEADERS } from "ai";

import { BadRequestError } from "./chat-request";
import { requireEnv } from "./env";
import type { Visitor } from "./visitor";

/**
 * What the service accepts as a thread id (THREAD_ID_PATTERN in agent/src/copilot_agent/api.py;
 * agent/tests/test_ts_parity.py pins the two). useChat's default chat id is 16 characters of
 * 0-9A-Za-z, and app/page.tsx gives it a crypto UUID (36 characters with hyphens): both fit.
 */
export const CHAT_ID_PATTERN = /^[A-Za-z0-9_-]{16,64}$/;

/** The service's base URL, or null when the route should answer by itself (the default). */
export function agentUrl(): string | null {
    const value = process.env.AGENT_URL?.trim();
    return value ? value.replace(/\/+$/, "") : null;
}

/** The body POST /chat takes (ChatRequest in api.py; extra fields are a 422 there). */
export type AgentChatRequest = {
    thread_id: string;
    question: string;
    origin: "web";
    visitor: {
        visitor_hash: string;
        landing_referrer: string | null;
        utm_source: string | null;
        country: string | null;
        device: "mobile" | "desktop";
    };
};

export function agentChatRequest(chatId: string, question: string, visitor: Visitor): AgentChatRequest {
    return {
        thread_id: chatId,
        question,
        origin: "web",
        visitor: {
            visitor_hash: visitor.visitorHash,
            landing_referrer: visitor.landingReferrer,
            utm_source: visitor.utmSource,
            country: visitor.country,
            device: visitor.device,
        },
    };
}

const gatewayError = () => Response.json({ error: "Something went wrong." }, { status: 502 });

export async function forwardToAgent(options: {
    baseUrl: string;
    chatId: string | undefined;
    question: string;
    visitor: Visitor;
    signal: AbortSignal;
}): Promise<Response> {
    const { baseUrl, chatId, question, visitor, signal } = options;
    // A malformed id is the caller's error, and it is caught here, before anything is paid for.
    if (chatId === undefined || !CHAT_ID_PATTERN.test(chatId)) {
        throw new BadRequestError("Invalid chat id.");
    }

    // Outside the try: a missing key is a deployment error, and the route's own 500 names it.
    const key = requireEnv("AGENT_API_KEY");

    let upstream: Response;
    try {
        upstream = await fetch(`${baseUrl}/chat`, {
            method: "POST",
            headers: {
                authorization: `Bearer ${key}`,
                "content-type": "application/json",
            },
            body: JSON.stringify(agentChatRequest(chatId, question, visitor)),
            signal,
        });
    } catch (err) {
        // A client that left while we waited for headers is not an error of ours.
        if (!signal.aborted) console.error("AGENT UNREACHABLE:", err);
        return gatewayError();
    }

    if (!upstream.ok || !upstream.body) {
        // The status is enough to diagnose (401: key mismatch, 422: a contract change, 5xx: the
        // service); the body is the service's text and stays out of the response.
        console.error("AGENT ERROR:", upstream.status);
        await upstream.body?.cancel();
        return gatewayError();
    }

    // The byte pipe. Headers are the AI SDK's own, set here rather than copied from upstream: the
    // response's headers stay this route's decision, and hop-by-hop ones never ride along.
    return new Response(upstream.body, { status: 200, headers: UI_MESSAGE_STREAM_HEADERS });
}
