/**
 * The route with AGENT_URL set: it forwards to the Python agent service and pipes the stream back
 * (lib/agent-forward.ts, step 2.6).
 *
 * What must hold, each asserted below:
 *   - the rate limit and the parse still run first, and nothing is forwarded without them;
 *   - the service gets exactly {thread_id, question, origin, visitor}: the chat id the REAL
 *     useChat transport sends, the parsed question, and NO history, signed or not;
 *   - the service's bytes reach the client unchanged, under the AI SDK's stream headers, and the
 *     real client builds the same message from them as from the service directly;
 *   - the request's abort signal reaches the upstream fetch (a closed tab cancels the run);
 *   - a failure before the stream is a 502 with the route's generic text, never the service's;
 *   - the TypeScript pipeline, the TypeScript query log and the TypeScript model are not touched.
 *
 * The upstream is global fetch, stubbed; the golden Python streams (agent/tests/golden/chat-stream)
 * stand in for what the service sends.
 */
import { readFileSync } from "node:fs";

import { Chat } from "@ai-sdk/react";
import { DefaultChatTransport, UI_MESSAGE_STREAM_HEADERS } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CHAT_ID_PATTERN, agentUrl } from "@/lib/agent-forward";
import { signAssistantText } from "@/lib/assistant-signature";
import type { ChatMessage } from "@/lib/chat-types";

const GOLDEN = new URL("../agent/tests/golden/chat-stream/", import.meta.url);
const pythonBytes = (id: string) => readFileSync(new URL(`${id}.sse`, GOLDEN));

const mocks = vi.hoisted(() => ({
    plannedRetrieve: vi.fn(),
    logQuery: vi.fn(),
    checkRateLimit: vi.fn(),
}));

vi.mock("@/lib/plan", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@/lib/plan")>()),
    plannedRetrieve: mocks.plannedRetrieve,
}));
vi.mock("@/lib/query-log", () => ({ logQuery: mocks.logQuery }));
vi.mock("next/server", () => ({ after: () => undefined }));
vi.mock("@/lib/rate-limit", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@/lib/rate-limit")>()),
    checkRateLimit: mocks.checkRateLimit,
}));

const { POST } = await import("@/app/api/chat/route");

const AGENT = "http://agent.test:8000";
const KEY = "k".repeat(64);
const CHAT_ID = "7f1c3a52-9e0b-4c1d-8f6e-2a4b5c6d7e8f"; // crypto.randomUUID()'s shape

const user = (text: string) => ({ role: "user", parts: [{ type: "text", text }] });
const assistant = (text: string) => ({
    role: "assistant",
    parts: [{ type: "text", text }, { type: "data-signature", data: { sig: signAssistantText(text) } }],
});

let upstream: ReturnType<typeof vi.fn>;

function sse(id: string, init: ResponseInit = {}): Response {
    return new Response(pythonBytes(id), { headers: { "content-type": "text/event-stream" }, ...init });
}

function post(body: unknown, init: RequestInit = {}): Promise<Response> {
    return POST(
        new Request("http://localhost/api/chat", {
            method: "POST",
            headers: { "content-type": "application/json", "x-landing-referrer": "linkedin.com" },
            body: JSON.stringify(body),
            ...init,
        })
    );
}

/** What the route sent upstream: [url, init] of the one fetch call. */
function forwarded(): { url: string; init: RequestInit; body: Record<string, unknown> } {
    expect(upstream).toHaveBeenCalledTimes(1);
    const [url, init] = upstream.mock.calls[0] as [string, RequestInit];
    return { url, init, body: JSON.parse(init.body as string) };
}

beforeEach(() => {
    process.env.AGENT_URL = AGENT;
    process.env.AGENT_API_KEY = KEY;
    mocks.plannedRetrieve.mockReset();
    mocks.logQuery.mockReset();
    mocks.checkRateLimit.mockReset().mockResolvedValue({ allowed: true, retryAfter: 0, scope: "none" });
    upstream = vi.fn(async () => sse("answered"));
    vi.stubGlobal("fetch", upstream);
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
});

afterEach(() => {
    delete process.env.AGENT_URL;
    delete process.env.AGENT_API_KEY;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

describe("forwarding to the agent service", () => {
    it("sends the chat id, the parsed question, origin web and the visitor, and no history", async () => {
        const res = await post({
            id: CHAT_ID,
            messages: [user("how do I stream text"), assistant("Use streamText."), user("  and configure it?  ")],
        });
        expect(res.status).toBe(200);

        const { url, init, body } = forwarded();
        expect(url).toBe(`${AGENT}/chat`);
        expect(init.method).toBe("POST");
        expect(init.headers).toEqual({ authorization: `Bearer ${KEY}`, "content-type": "application/json" });
        expect(body).toEqual({
            thread_id: CHAT_ID,
            question: "and configure it?",
            origin: "web",
            visitor: {
                visitor_hash: expect.stringMatching(/^[0-9a-f]{32}$/),
                landing_referrer: "linkedin.com",
                utm_source: null,
                country: null,
                device: "desktop",
            },
        });
        // The signed assistant turn was verified by the parse, and still not sent: the service
        // reads the conversation from its own thread.
        expect(JSON.stringify(body)).not.toContain("Use streamText.");
    });

    it("pipes the service's bytes through unchanged, under the AI SDK's stream headers", async () => {
        const res = await post({ id: CHAT_ID, messages: [user("q")] });
        expect(Buffer.from(await res.arrayBuffer())).toEqual(pythonBytes("answered"));
        for (const [name, value] of Object.entries(UI_MESSAGE_STREAM_HEADERS)) {
            expect(res.headers.get(name)).toBe(value);
        }
    });

    it("the real client, through the route, builds the message it builds from the service", async () => {
        async function converse(respond: (init: RequestInit) => Promise<Response>) {
            const chat = new Chat<ChatMessage>({
                id: CHAT_ID,
                transport: new DefaultChatTransport<ChatMessage>({
                    api: "http://localhost/api/chat",
                    fetch: async (_url, init) => respond(init ?? {}),
                }),
            });
            await chat.sendMessage({ text: "how do I stream text" });
            return { status: chat.status, messages: chat.messages.map((m) => ({ role: m.role, parts: m.parts })) };
        }

        const direct = await converse(async () => sse("answered"));
        const viaRoute = await converse((init) => POST(new Request("http://localhost/api/chat", init)));
        expect(viaRoute).toEqual(direct);
        expect(viaRoute.status).toBe("ready");
        // The id DefaultChatTransport put in the body is the thread the service was asked for.
        expect(forwarded().body.thread_id).toBe(CHAT_ID);
    });

    it("hands the request's abort signal to the upstream fetch", async () => {
        const controller = new AbortController();
        await post({ id: CHAT_ID, messages: [user("q")] }, { signal: controller.signal });
        const { signal } = forwarded().init;
        expect(signal).toBeInstanceOf(AbortSignal);
        expect(signal!.aborted).toBe(false);
        controller.abort();
        expect(signal!.aborted).toBe(true);
    });

    it("rate-limits before forwarding: a 429 costs the service nothing", async () => {
        mocks.checkRateLimit.mockResolvedValue({ allowed: false, retryAfter: 30, scope: "burst" });
        const res = await post({ id: CHAT_ID, messages: [user("q")] });
        expect(res.status).toBe(429);
        expect(upstream).not.toHaveBeenCalled();
    });

    it.each([
        ["missing", undefined],
        ["too short", "abc"],
        ["with a slash", "thread/../0000001"],
        ["not a string", 1234567890123456],
        ["with a trailing newline", "PyTestThread0001\n"],
    ])("a chat id that is %s is a 400, and nothing is forwarded", async (_name, id) => {
        const res = await post({ id, messages: [user("q")] });
        expect(res.status).toBe(400);
        expect(await res.json()).toEqual({ error: "Invalid chat id." });
        expect(upstream).not.toHaveBeenCalled();
    });

    it("a malformed body is still the parse's 400, before the chat id is looked at", async () => {
        const res = await post({ id: CHAT_ID, messages: [] });
        expect(res.status).toBe(400);
        expect(upstream).not.toHaveBeenCalled();
    });

    it.each([401, 422, 500, 503])("an upstream %i is a 502 with the route's own text", async (status) => {
        upstream.mockResolvedValue(new Response("upstream said: sk-must-never-reach-the-client", { status }));
        const res = await post({ id: CHAT_ID, messages: [user("q")] });
        expect(res.status).toBe(502);
        const text = await res.text();
        expect(JSON.parse(text)).toEqual({ error: "Something went wrong." });
        expect(text).not.toContain("sk-must-never");
    });

    it("an unreachable service is a 502 with the route's own text", async () => {
        upstream.mockRejectedValue(new TypeError("fetch failed: connect ECONNREFUSED"));
        const res = await post({ id: CHAT_ID, messages: [user("q")] });
        expect(res.status).toBe(502);
        expect(await res.json()).toEqual({ error: "Something went wrong." });
    });

    it("never runs the TypeScript pipeline or writes the TypeScript query log", async () => {
        await (await post({ id: CHAT_ID, messages: [user("q")] })).text();
        expect(mocks.plannedRetrieve).not.toHaveBeenCalled();
        expect(mocks.logQuery).not.toHaveBeenCalled();
    });

    it("without AGENT_API_KEY it refuses with a 500 and forwards nothing", async () => {
        delete process.env.AGENT_API_KEY;
        const res = await post({ id: CHAT_ID, messages: [user("q")] });
        expect(res.status).toBe(500);
        expect(upstream).not.toHaveBeenCalled();
    });
});

describe("AGENT_URL", () => {
    it("unset or blank means the route answers by itself", async () => {
        for (const value of [undefined, "", "   "]) {
            if (value === undefined) delete process.env.AGENT_URL;
            else process.env.AGENT_URL = value;
            expect(agentUrl()).toBeNull();
        }
        mocks.plannedRetrieve.mockResolvedValue({
            intent: "greeting",
            relevant: [],
            subQueries: [],
            mode: "skipped",
            plannerUsage: { inputTokens: 1, outputTokens: 1 },
            rerankCalls: 0,
        });
        const res = await post({ id: CHAT_ID, messages: [user("hi")] });
        await res.text();
        expect(mocks.plannedRetrieve).toHaveBeenCalledTimes(1);
        expect(upstream).not.toHaveBeenCalled();
    });

    it("loses trailing slashes, so the path is always <base>/chat", async () => {
        process.env.AGENT_URL = `${AGENT}//`;
        await post({ id: CHAT_ID, messages: [user("q")] });
        expect(forwarded().url).toBe(`${AGENT}/chat`);
    });
});

describe("CHAT_ID_PATTERN", () => {
    it("accepts useChat's default ids and crypto UUIDs", () => {
        expect(CHAT_ID_PATTERN.test("aB3dE5gH7jK9mN1p")).toBe(true); // generateId(): 16 of 0-9A-Za-z
        expect(CHAT_ID_PATTERN.test(crypto.randomUUID())).toBe(true);
    });
});
