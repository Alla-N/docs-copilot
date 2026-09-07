import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import { hashChunk } from "@/lib/content-hash";

describe("hashChunk — parity with db/001_content_hash.sql", () => {
    // encode(sha256(convert_to(source_url || E'\n' || content, 'UTF8')), 'hex')
    // The fixture is the sha256 of exactly those bytes; a change to the separator, the
    // order, or the encoding on the TypeScript side moves this hex and silently re-embeds
    // the whole corpus on the next ingest (invariant #1).
    it("hashes source_url + LF + content as UTF-8, hex-encoded", () => {
        expect(hashChunk("https://ai-sdk.dev/docs/x.md", "hello")).toBe(
            createHash("sha256").update("https://ai-sdk.dev/docs/x.md\nhello", "utf8").digest("hex")
        );
    });

    it("pins a known value (regenerate deliberately if the SQL changes)", () => {
        expect(hashChunk("u", "c")).toBe("f83e30bab19727a0baa8f3310891dcd0f7fdb9cbea7154c6905cc4aac96e0c9d");
    });

    it("distinguishes identical content on different pages", () => {
        expect(hashChunk("https://a", "same")).not.toBe(hashChunk("https://b", "same"));
    });

    it("is sensitive to the separator — no LF, no match", () => {
        expect(hashChunk("u", "c")).not.toBe(createHash("sha256").update("uc").digest("hex"));
    });
});
