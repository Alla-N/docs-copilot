import { describe, expect, it } from "vitest";

import { isRefusal, REFUSAL_MESSAGE } from "@/lib/refusal";

const CORE = "I don't have information about that in the documentation.";

describe("isRefusal", () => {
    it("recognises the canonical message and its core sentence alone", () => {
        expect(isRefusal(REFUSAL_MESSAGE)).toBe(true);
        expect(isRefusal(CORE)).toBe(true);
        expect(isRefusal(CORE + " I can help with AI SDK docs.")).toBe(true);
    });

    it("recognises the three observed 'doesn't cover <topic>' shapes", () => {
        // (a) prefix + core
        expect(isRefusal("The documentation doesn't cover fine-tuning. " + REFUSAL_MESSAGE)).toBe(true);
        // (b) the naming clause alone, with or without a final period
        expect(isRefusal("The documentation doesn't cover fine-tuning a model with the Vercel AI SDK.")).toBe(true);
        expect(isRefusal("The documentation doesn't cover fine-tuning a model with the Vercel AI SDK")).toBe(true);
        // (c) prefix + polite tail, no core — the Day-14 escape that made the detector compositional
        expect(
            isRefusal("The documentation doesn't cover fine-tuning a model with the Vercel AI SDK. I can help with AI SDK docs. Ask me about those and I'll help.")
        ).toBe(true);
        expect(isRefusal("The documentation does not mention fine-tuning. Ask me about those and I'll help.")).toBe(true);
        expect(isRefusal("The documentation doesn’t cover fine-tuning.")).toBe(true); // curly apostrophe
    });

    it("does NOT count a partial answer that declines one clause at the end (Day-12 includes() bug)", () => {
        expect(
            isRefusal("To stream text, use streamText. I don't have information about the second part in the documentation.")
        ).toBe(false);
    });

    it("does NOT count a leak that hides behind a refusal opener", () => {
        expect(isRefusal("The documentation doesn't cover fine-tuning. The capital of France is Paris.")).toBe(false);
        expect(isRefusal("The documentation doesn't cover X, but here's how: use streamText.")).toBe(false);
        expect(isRefusal("The documentation doesn't cover fine-tuning. I can help with AI SDK docs. Also, Paris is the capital of France.")).toBe(false);
    });

    it("does NOT count answers that merely start with 'The documentation'", () => {
        expect(isRefusal("The documentation does not require you to call convertToModelMessages.")).toBe(false);
        expect(isRefusal("The documentation describes streamText as a function that streams text.")).toBe(false);
        expect(isRefusal("streamText streams text from a model. I can help with AI SDK docs.")).toBe(false);
        expect(isRefusal("")).toBe(false);
    });
});
