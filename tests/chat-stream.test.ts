/**
 * The chat route's STREAM SHAPE — not what it says, but how it frames it.
 *
 * The bug this exists for: the route wrote its `data-retrieval` / `data-sources` parts
 * before anything opened the assistant message, and let the merged generation stream send
 * its own `start`. The client keys the streaming message by id and either replaces the last
 * message (same id) or pushes a new one; `start` is what assigns that id. So the data parts
 * pushed a message under the client's provisional id, the later `start` renamed the
 * in-flight message, and the next update pushed it a SECOND time — the UI showed an empty
 * bubble with the source pills, then the real answer with the same pills underneath.
 *
 * Nothing caught it: the evals read the model's text, never the transport, and the canned
 * path (which writes `start` first) looked fine. Hence these tests, which assert the one
 * rule that was broken — EXACTLY ONE `start` per response, before any other chunk.
 *
 * The planner, the query log and the model are mocked at the module boundary: the subject
 * here is the chunk sequence the route emits, and a test that needed an OpenAI key to check
 * message framing would be a test nobody runs. (Review item 35.)
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import { GREETING_MESSAGE } from "@/lib/plan";
import { REFUSAL_MESSAGE } from "@/lib/refusal";
import { verifyAssistantText } from "@/lib/assistant-signature";

const mocks = vi.hoisted(() => ({
    plannedRetrieve: vi.fn(),
    /** What the mocked model "streams". The route signs exactly this string. */
    answerText: "Tool calling lets the model call functions.",
    /** Options the route passed to toUIMessageStream — `sendStart` is the one that matters. */
    uiStreamOptions: [] as Record<string, unknown>[],
}));

vi.mock("@/lib/plan", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@/lib/plan")>()),
    plannedRetrieve: mocks.plannedRetrieve,
}));

vi.mock("@/lib/query-log", () => ({ logQuery: vi.fn() }));

// `after()` defers work until the response has flushed; here it must not run at all — the
// only thing behind it is the query log, which is mocked out anyway.
vi.mock("next/server", () => ({ after: () => undefined }));

vi.mock("ai", async (importOriginal) => {
    const actual = await importOriginal<typeof import("ai")>();
    return {
        ...actual,
        // No model call. `stream` is what toUIMessageStream reads, and that is mocked too.
        streamText: () => ({ stream: new ReadableStream({ start: (c) => c.close() }), text: Promise.resolve(mocks.answerText) }),
        // Stands in for the generation stream — faithfully, including the part that made the
        // bug: the SDK sends its OWN `start` unless told not to. Keeping that default here is
        // what makes this test fail against the old route (whose first chunk was a data part,
        // with the generation's `start` arriving third and renaming the message).
        toUIMessageStream: (options: Record<string, unknown>) => {
            mocks.uiStreamOptions.push(options);
            return new ReadableStream({
                start(c) {
                    if (options.sendStart !== false) c.enqueue({ type: "start" });
                    c.enqueue({ type: "text-start", id: "t" });
                    c.enqueue({ type: "text-delta", id: "t", delta: mocks.answerText });
                    c.enqueue({ type: "text-end", id: "t" });
                    c.enqueue({ type: "finish" });
                    c.close();
                },
            });
        },
    };
});

const { POST } = await import("@/app/api/chat/route");

type Chunk = { type: string; [k: string]: unknown };

async function ask(text: string): Promise<Chunk[]> {
    const res = await POST(
        new Request("http://localhost/api/chat", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ messages: [{ role: "user", parts: [{ type: "text", text }] }] }),
        })
    );
    const body = await res.text();
    return body
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice("data: ".length))
        .filter((payload) => payload !== "[DONE]")
        .map((payload) => JSON.parse(payload) as Chunk);
}

const ANSWERED = {
    intent: "search" as const,
    relevant: [
        { content: "Tools are…", title: "Core: Tool Calling", source_url: "https://x/tools", score: 0.85 },
        { content: "More tools…", title: "Core: Tool Calling", source_url: "https://x/tools", score: 0.8 },
    ],
    subQueries: ["What is tool calling in the Vercel AI SDK?"],
    mode: "reranked" as const,
    plannerUsage: { inputTokens: 320, outputTokens: 44 },
    rerankCalls: 1,
};

beforeEach(() => {
    mocks.uiStreamOptions.length = 0;
    mocks.plannedRetrieve.mockReset();
});

describe("chat route stream framing", () => {
    it("opens the message with exactly one `start`, before any data part", async () => {
        mocks.plannedRetrieve.mockResolvedValue(ANSWERED);

        const chunks = await ask("What is tool calling?");
        const types = chunks.map((c) => c.type);

        // The bug, precisely: a second `start` mid-stream renamed the message and split it in two.
        expect(types.filter((t) => t === "start")).toHaveLength(1);
        expect(types[0]).toBe("start");
        expect(types.indexOf("data-sources")).toBeGreaterThan(0);
    });

    it("signs the answer it streamed, so the next turn can prove the server wrote it", async () => {
        mocks.plannedRetrieve.mockResolvedValue(ANSWERED);

        const chunks = await ask("What is tool calling?");
        const streamed = chunks.filter((c) => c.type === "text-delta").map((c) => c.delta).join("");
        const sig = (chunks.find((c) => c.type === "data-signature")?.data as { sig: string } | undefined)?.sig;

        expect(verifyAssistantText(streamed, sig)).toBe(true);
    });

    it("signs canned replies too — an unsigned greeting would vanish from the next request", async () => {
        mocks.plannedRetrieve.mockResolvedValue({
            intent: "greeting", relevant: [], subQueries: [], mode: "skipped",
            plannerUsage: { inputTokens: 210, outputTokens: 12 }, rerankCalls: 0,
        });

        const chunks = await ask("hi");
        const sig = (chunks.find((c) => c.type === "data-signature")?.data as { sig: string } | undefined)?.sig;

        expect(verifyAssistantText(GREETING_MESSAGE, sig)).toBe(true);
    });

    it("suppresses the generation stream's own `start`", async () => {
        mocks.plannedRetrieve.mockResolvedValue(ANSWERED);

        await ask("What is tool calling?");

        // Belt and braces: the assertion above only holds because of this option, and the
        // SDK's default is true — so a refactor that drops it fails here by name.
        expect(mocks.uiStreamOptions).toHaveLength(1);
        expect(mocks.uiStreamOptions[0]?.sendStart).toBe(false);
    });

    it("sends one pill per page and the retrieval mode, after the start", async () => {
        mocks.plannedRetrieve.mockResolvedValue(ANSWERED);

        const chunks = await ask("What is tool calling?");
        const sources = chunks.find((c) => c.type === "data-sources")?.data;
        const retrieval = chunks.find((c) => c.type === "data-retrieval")?.data;

        // Two chunks from one page → one pill carrying both [Source N] numbers.
        expect(sources).toEqual([
            { id: 1, title: "Core: Tool Calling", url: "https://x/tools", score: 0.85, chunks: [1, 2] },
        ]);
        expect(retrieval).toEqual({ mode: "reranked", intent: "search" });
    });

    it("frames a canned greeting the same way — one start, then the text", async () => {
        mocks.plannedRetrieve.mockResolvedValue({
            intent: "greeting",
            relevant: [],
            subQueries: [],
            mode: "skipped",
            plannerUsage: { inputTokens: 210, outputTokens: 12 },
            rerankCalls: 0,
        });

        const chunks = await ask("hi");
        const types = chunks.map((c) => c.type);

        expect(types.filter((t) => t === "start")).toHaveLength(1);
        expect(types[0]).toBe("start");
        expect(chunks.filter((c) => c.type === "text-delta").map((c) => c.delta).join("")).toBe(GREETING_MESSAGE);
        // No pills on a canned reply: retrieval never ran, so there is nothing to attribute.
        expect(types).not.toContain("data-sources");
    });

    it("refuses an off-topic message without a model call, still as one message", async () => {
        mocks.plannedRetrieve.mockResolvedValue({
            intent: "off-topic",
            relevant: [],
            subQueries: [],
            mode: "skipped",
            plannerUsage: { inputTokens: 214, outputTokens: 10 },
            rerankCalls: 0,
        });

        const chunks = await ask("How do I deploy to AWS Lambda?");

        expect(chunks.map((c) => c.type).filter((t) => t === "start")).toHaveLength(1);
        expect(chunks.filter((c) => c.type === "text-delta").map((c) => c.delta).join("")).toBe(REFUSAL_MESSAGE);
        // The off-topic path must never reach the answering model.
        expect(mocks.uiStreamOptions).toHaveLength(0);
    });

    it("rejects a body that is not JSON with a 400, not a 500", async () => {
        const res = await POST(
            new Request("http://localhost/api/chat", { method: "POST", body: "not json" })
        );
        expect(res.status).toBe(400);
    });
});
