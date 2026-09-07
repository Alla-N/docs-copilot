import { describe, expect, it } from "vitest";

import { CITE_PREFIX, linkCitations } from "@/lib/citations";

describe("linkCitations", () => {
    it("turns (Source N) into a markdown link per number", () => {
        expect(linkCitations("Use streamText (Source 1).")).toBe(`Use streamText [1](${CITE_PREFIX}1).`);
    });

    it("handles plural and list forms", () => {
        expect(linkCitations("Both work (Sources 1 and 3).")).toBe("Both work [1](#source-1)[3](#source-3).");
        expect(linkCitations("See (Source 1, Source 4) and (Source 2, 5).")).toBe(
            "See [1](#source-1)[4](#source-4) and [2](#source-2)[5](#source-5)."
        );
    });

    it("leaves fenced code alone — closed or still streaming", () => {
        expect(linkCitations("```ts\nconst x = 1; // (Source 1)\n```\nAfter (Source 2)")).toBe(
            "```ts\nconst x = 1; // (Source 1)\n```\nAfter [2](#source-2)"
        );
        expect(linkCitations("Unterminated ```ts\ncode (Source 1)")).toBe("Unterminated ```ts\ncode (Source 1)");
    });

    it("ignores anything that is not the parenthesised numbered form", () => {
        const s = "Nothing to see (source). Also (Source). Source 1 without parens.";
        expect(linkCitations(s)).toBe(s);
    });
});
