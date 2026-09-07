import { describe, expect, it } from "vitest";

import { chunkPage, stripBoilerplate } from "@/lib/chunker";

const para = (n: number, ch = "a") => ch.repeat(n);

describe("stripBoilerplate", () => {
    it("removes the site-wide '## Navigation' section up to the next h2", () => {
        const md = "# Page\n\nintro\n\n## Navigation\n\n- [Home](/)\n- [Docs](/docs)\n\n## Real section\n\ncontent";
        const out = stripBoilerplate(md);
        expect(out).not.toContain("Navigation");
        expect(out).not.toContain("[Home]");
        expect(out).toContain("## Real section");
        expect(out).toContain("intro");
    });

    it("leaves a page without the section untouched", () => {
        const md = "# Page\n\n## A\n\ntext";
        expect(stripBoilerplate(md)).toBe(md);
    });
});

describe("chunkPage", () => {
    it("splits on h1–h3 headings and keeps each heading with its content", () => {
        const md = `# Title\n\n${para(100)}\n\n## Second\n\n${para(100, "b")}\n\n### Third\n\n${para(100, "c")}`;
        const chunks = chunkPage(md);
        expect(chunks).toHaveLength(3);
        expect(chunks[0].startsWith("# Title")).toBe(true);
        expect(chunks[1].startsWith("## Second")).toBe(true);
        expect(chunks[2].startsWith("### Third")).toBe(true);
    });

    it("does not split on h4+ headings", () => {
        const md = `## A\n\n${para(100)}\n\n#### Deep\n\n${para(100, "b")}`;
        expect(chunkPage(md)).toHaveLength(1);
    });

    it("never emits a chunk longer than maxLen", () => {
        const md = `## Long\n\n${Array.from({ length: 12 }, (_, i) => para(400, String.fromCharCode(97 + i))).join("\n\n")}`;
        for (const c of chunkPage(md, 1500)) expect(c.length).toBeLessThanOrEqual(1500);
    });

    it("carries the heading into every overflow chunk of a long section", () => {
        const md = `## Long\n\n${Array.from({ length: 6 }, (_, i) => para(600, String.fromCharCode(97 + i))).join("\n\n")}`;
        const chunks = chunkPage(md, 1500);
        expect(chunks.length).toBeGreaterThan(1);
        for (const c of chunks) expect(c.startsWith("## Long")).toBe(true);
    });

    it("hard-splits a single paragraph that exceeds maxLen", () => {
        const md = `## P\n\n${para(4000)}`;
        const chunks = chunkPage(md, 1500);
        expect(chunks.length).toBeGreaterThanOrEqual(3);
        for (const c of chunks) expect(c.length).toBeLessThanOrEqual(1500);
    });

    it("drops near-empty fragments (≤ 80 chars) — they are noise vectors", () => {
        const md = `## Tiny\n\nshort\n\n## Real\n\n${para(200)}`;
        const chunks = chunkPage(md);
        expect(chunks).toHaveLength(1);
        expect(chunks[0].startsWith("## Real")).toBe(true);
    });
});
