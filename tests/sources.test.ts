import { describe, expect, it } from "vitest";

import { toSourcePills } from "@/lib/sources";

describe("toSourcePills", () => {
    it("collapses chunks into one pill per page, keeping chunk numbers and the best score", () => {
        const pills = toSourcePills([
            { content: "a", title: "Structured", source_url: "https://x/structured", score: 0.9 },
            { content: "b", title: "Structured", source_url: "https://x/structured", score: 0.8 },
            { content: "c", title: "Overview", source_url: "https://x/overview", score: 0.7 },
            { content: "d", title: "Structured", source_url: "https://x/structured", score: 0.95 },
        ]);
        expect(pills).toEqual([
            { id: 1, title: "Structured", url: "https://x/structured", score: 0.95, chunks: [1, 2, 4] },
            { id: 2, title: "Overview", url: "https://x/overview", score: 0.7, chunks: [3] },
        ]);
    });

    it("preserves the prompt's [Source N] numbering — chunk 3 is still 3 after dedupe", () => {
        const pills = toSourcePills([
            { content: "a", title: "A", source_url: "https://x/a", score: 0.9 },
            { content: "b", title: "A", source_url: "https://x/a", score: 0.8 },
            { content: "c", title: "B", source_url: "https://x/b", score: 0.7 },
        ]);
        expect(pills.find((p) => p.chunks.includes(3))?.title).toBe("B");
    });

    it("returns an empty list for no chunks", () => {
        expect(toSourcePills([])).toEqual([]);
    });
});
