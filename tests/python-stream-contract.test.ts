/**
 * The Python service's stream, read by the REAL AI SDK client, must build the same message as
 * the TypeScript route does for the same run.
 *
 * The request-parity goldens (agent/tests/golden/*-requests.json) go TypeScript -> Python: the
 * TS side records, the Python side must match. This one goes the other way. The Python side
 * writes the exact bytes POST /chat sends for six scenarios (agent/tests/test_chat_api.py, which
 * also fails while the files and the code disagree), and this test feeds them to the class
 * `useChat` wraps: `Chat` from @ai-sdk/react, through `DefaultChatTransport`, which parses every
 * chunk with the AI SDK's strict schema. Then it replays each scenario through the TypeScript
 * route itself, with plannedRetrieve and the model mocked but the real streamText,
 * toUIMessageStream and createUIMessageStream, through the same client, and compares.
 *
 * So the question is never "does Python emit what I believe the protocol is", but "does the
 * real client end up with the same message". One assistant message per answer (a second one is
 * the Day 15 duplicate bubble, invariant 12), the same parts, the same pills from the real
 * toSourcePills, and a signature the real verifyAssistantText accepts (invariant 8).
 *
 * One difference is expected and named: when retrieval fails, the TS route answers with a 500
 * before streaming, while Python has already sent a 200 and reports it as an error chunk. The
 * reader sees the same screen (no bubble, the error box), which is asserted; the error text
 * differs, which is asserted too.
 */
import { readFileSync } from "node:fs";

import { Chat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatMessage } from "@/lib/chat-types";
import type { RetrievedChunk } from "@/lib/retrieve";
import { toSourcePills } from "@/lib/sources";
import { verifyAssistantText } from "@/lib/assistant-signature";

type Scenario = {
    id: string;
    question: string;
    intent: "search" | "greeting" | "off-topic";
    subQueries: string[];
    relevant: RetrievedChunk[];
    deltas: string[];
    failure: "none" | "model" | "retrieval";
};

const GOLDEN = new URL("../agent/tests/golden/chat-stream/", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("scenarios.json", GOLDEN), "utf8")) as {
    signingSecret: string;
    messageId: string;
    scenarios: Scenario[];
};
const pythonBytes = (id: string) => readFileSync(new URL(`${id}.sse`, GOLDEN));

const mocks = vi.hoisted(() => ({
    plannedRetrieve: vi.fn(),
    /** The scenario the mocked model is replaying. */
    scenario: undefined as Scenario | undefined,
}));

vi.mock("@/lib/plan", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@/lib/plan")>()),
    plannedRetrieve: mocks.plannedRetrieve,
}));
vi.mock("@/lib/query-log", () => ({ logQuery: vi.fn() }));
vi.mock("next/server", () => ({ after: () => undefined }));

// The real streamText, handed a scripted model instead of openai(...). Everything between the
// model and the bytes (streamText, toUIMessageStream, the route's createUIMessageStream) is real.
vi.mock("ai", async (importOriginal) => {
    const actual = await importOriginal<typeof import("ai")>();
    type StreamTextOptions = Parameters<typeof actual.streamText>[0];

    function scriptedModel(scenario: Scenario) {
        const parts: unknown[] = [
            { type: "stream-start", warnings: [] },
            { type: "text-start", id: "answer" },
            ...scenario.deltas.map((delta) => ({ type: "text-delta", id: "answer", delta })),
        ];
        if (scenario.failure !== "model") {
            parts.push(
                { type: "text-end", id: "answer" },
                {
                    type: "finish",
                    finishReason: { unified: "stop", raw: "stop" },
                    usage: {
                        inputTokens: { total: 1500, noCache: 1500, cacheRead: 0, cacheWrite: 0 },
                        outputTokens: { total: 60, text: 60, reasoning: 0 },
                    },
                }
            );
        }
        return {
            specificationVersion: "v4" as const,
            provider: "contract-test",
            modelId: "scripted",
            supportedUrls: {},
            doGenerate: async () => {
                throw new Error("the route streams");
            },
            // One part per pull, then (for a model failure) the stream errors the way a broken
            // network read does: after the client has already received the deltas.
            doStream: async () => ({
                stream: new ReadableStream({
                    pull(controller) {
                        const next = parts.shift();
                        if (next !== undefined) controller.enqueue(next);
                        else if (scenario.failure === "model") controller.error(new Error("upstream said: no"));
                        else controller.close();
                    },
                }),
            }),
        };
    }

    return {
        ...actual,
        streamText: (options: StreamTextOptions) =>
            actual.streamText({ ...options, model: scriptedModel(mocks.scenario!) as never }),
    };
});

const { POST } = await import("@/app/api/chat/route");

type Outcome = {
    status: string;
    error: string | undefined;
    messages: { role: string; parts: unknown[] }[];
    /** The raw chunks the client was sent, [DONE] dropped; null for a non-stream response. */
    chunks: { type: string; [k: string]: unknown }[] | null;
};

/** Send the scenario's question through a fresh Chat, over the given fetch, like useChat does. */
async function converse(scenario: Scenario, respond: (init: RequestInit) => Promise<Response>): Promise<Outcome> {
    let raw: Promise<string> | undefined;
    const chat = new Chat<ChatMessage>({
        transport: new DefaultChatTransport<ChatMessage>({
            api: "http://localhost/api/chat",
            fetch: async (_url, init) => {
                const response = await respond(init ?? {});
                raw = response.clone().text();
                return response;
            },
        }),
    });
    await chat.sendMessage({ text: scenario.question });
    const body = await raw!;
    const isStream = body.startsWith("data: ");
    return {
        status: chat.status,
        error: chat.error?.message,
        // Message ids are random on both sides; everything else must match.
        messages: chat.messages.map((m) => ({ role: m.role, parts: m.parts })),
        chunks: isStream
            ? body
                  .split("\n")
                  .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
                  .map((l) => JSON.parse(l.slice("data: ".length)))
            : null,
    };
}

function fromPython(scenario: Scenario): Promise<Outcome> {
    return converse(scenario, async () =>
        new Response(pythonBytes(scenario.id), {
            headers: { "content-type": "text/event-stream", "x-vercel-ai-ui-message-stream": "v1" },
        })
    );
}

function fromTypeScript(scenario: Scenario): Promise<Outcome> {
    mocks.scenario = scenario;
    if (scenario.failure === "retrieval") {
        mocks.plannedRetrieve.mockRejectedValue(new Error("db said: no"));
    } else {
        mocks.plannedRetrieve.mockResolvedValue({
            intent: scenario.intent,
            relevant: scenario.relevant,
            subQueries: scenario.subQueries,
            mode: scenario.intent === "search" ? "reranked" : "skipped",
            plannerUsage: { inputTokens: 1500, outputTokens: 60 },
            rerankCalls: scenario.intent === "search" ? scenario.subQueries.length : 0,
        });
    }
    return converse(scenario, (init) => POST(new Request("http://localhost/api/chat", init)));
}

/** Chunk types up to the first error (the client stops reading there), signature dropped:
 *  its place in the TS stream is a race the client does not care about. */
function framing(chunks: Outcome["chunks"]): string[] {
    const types = (chunks ?? []).map((c) => c.type);
    const firstError = types.indexOf("error");
    return (firstError === -1 ? types : types.slice(0, firstError + 1)).filter((t) => t !== "data-signature");
}

beforeEach(() => {
    mocks.plannedRetrieve.mockReset();
    // The route logs failures on purpose; the model-failure and retrieval-failure scenarios
    // would print them on every run.
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
});

describe("the Python stream contract", () => {
    it("the golden streams were signed with the secret this test process verifies with", () => {
        // tests/setup.ts sets it; the Python test signs with the same value.
        expect(process.env.ASSISTANT_SIGNING_SECRET).toBe(meta.signingSecret);
        expect(meta.scenarios.map((s) => s.id)).toEqual([
            "answered", "greeting", "off-topic", "no-context", "error-mid-answer", "retrieval-fails",
        ]);
    });

    describe.each(meta.scenarios.map((s) => [s.id, s] as const))("%s", (_id, scenario) => {
        it("builds the same message in the real client as the TypeScript route", async () => {
            const python = await fromPython(scenario);
            const ts = await fromTypeScript(scenario);

            expect(python.status).toBe(ts.status);
            expect(python.messages).toEqual(ts.messages);
            if (scenario.failure !== "retrieval") {
                expect(framing(python.chunks)).toEqual(framing(ts.chunks));
                expect(python.chunks!.find((c) => c.type === "finish")).toEqual(
                    ts.chunks!.find((c) => c.type === "finish")
                );
            }
        });

        it("opens at most one message, with `start` first (invariant 12)", async () => {
            const python = await fromPython(scenario);
            const starts = python.chunks!.filter((c) => c.type === "start");
            if (scenario.failure === "retrieval") {
                // Nothing to show yet, so no message: a `start` here would leave an empty bubble.
                expect(starts).toHaveLength(0);
                expect(python.messages.map((m) => m.role)).toEqual(["user"]);
            } else {
                expect(starts).toEqual([{ type: "start", messageId: meta.messageId }]);
                expect(python.chunks![0]).toBe(starts[0]);
                expect(python.messages.map((m) => m.role)).toEqual(["user", "assistant"]);
            }
        });

        if (scenario.failure === "none") {
            it("carries the pills toSourcePills makes and a signature verifyAssistantText accepts", async () => {
                const python = await fromPython(scenario);
                expect(python.status).toBe("ready");
                const parts = python.messages[1].parts as { type: string; text?: string; data?: unknown }[];
                const text = parts.filter((p) => p.type === "text").map((p) => p.text).join(" ");
                const sources = parts.find((p) => p.type === "data-sources")?.data;
                const sig = (parts.find((p) => p.type === "data-signature")?.data as { sig: string }).sig;

                expect(sources).toEqual(scenario.relevant.length ? toSourcePills(scenario.relevant) : undefined);
                // The same join lib/chat-request.ts does before verifying the next request's history.
                expect(verifyAssistantText(text, sig)).toBe(true);
            });
        } else {
            it("ends in the error state, without a signature for the partial answer", async () => {
                const python = await fromPython(scenario);
                const ts = await fromTypeScript(scenario);
                expect(python.status).toBe("error");
                expect(python.error).toBe("Stream failed");
                // The named difference: TS answers a retrieval failure with a 500 before streaming.
                expect(ts.error).toBe(scenario.failure === "retrieval" ? '{"error":"Something went wrong."}' : "Stream failed");
                const parts = python.messages[1]?.parts ?? [];
                expect(parts.some((p) => (p as { type: string }).type === "data-signature")).toBe(false);
            });
        }
    });
});
