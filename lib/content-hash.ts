import { createHash } from "node:crypto";

/**
 * The content hash that makes ingestion idempotent (invariant #1 in CLAUDE.md).
 *
 * MUST stay byte-identical to db/001_content_hash.sql:
 *   encode(sha256(convert_to(source_url || E'\n' || content, 'UTF8')), 'hex')
 *
 * source_url is in the hash so two pages may legitimately share identical text without one
 * of them being rejected by the unique index. Change the separator, the order or the encoding
 * on either side and the diff step sees every stored chunk as stale: a full, silent
 * re-embed of the corpus. tests/content-hash.test.ts pins the exact output.
 */
export function hashChunk(sourceUrl: string, content: string): string {
    return createHash("sha256").update(`${sourceUrl}\n${content}`, "utf8").digest("hex");
}
