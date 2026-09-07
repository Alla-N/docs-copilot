import { describe, expect, it } from "vitest";

import { generationMessages, generationSettings, MAX_OUTPUT_TOKENS } from "@/lib/generation";

describe("generationMessages — invariant #4, the resolved-query swap", () => {
    it("replaces the final user turn with the planner's sub-queries, keeping history", () => {
        const out = generationMessages(
            [{ role: "user", content: "hi" }, { role: "assistant", content: "hello" }],
            "What is SDK?",
            ["What is the Vercel AI SDK?"]
        );
        expect(out).toEqual([
            { role: "user", content: "hi" },
            { role: "assistant", content: "hello" },
            { role: "user", content: "What is the Vercel AI SDK?" },
        ]);
    });

    it("joins several sub-queries with newlines — the noise never reaches generation", () => {
        const out = generationMessages([], "How do I use streamText? Also, what is the capital of France?", [
            "How do I use streamText in the Vercel AI SDK?",
        ]);
        expect(out[0].content).toBe("How do I use streamText in the Vercel AI SDK?");
        expect(String(out[0].content)).not.toContain("France");
    });

    it("falls back to the raw question when the planner produced no sub-queries", () => {
        expect(generationMessages([], "raw", [])[0]).toEqual({ role: "user", content: "raw" });
    });
});

describe("generationSettings", () => {
    it("is temperature 0 with a bounded output and a grounded system prompt", () => {
        const s = generationSettings([{ content: "chunk text", title: "T", source_url: "https://x", score: 0.9 }]);
        expect(s.temperature).toBe(0);
        expect(s.maxOutputTokens).toBe(MAX_OUTPUT_TOKENS);
        expect(s.system).toContain("[Source 1]");
        expect(s.system).toContain("chunk text");
    });

    it("tells the model there is nothing to ground on when retrieval returned nothing", () => {
        expect(generationSettings([]).system).toContain("NO RELEVANT DOCUMENTATION FOUND");
    });
});
