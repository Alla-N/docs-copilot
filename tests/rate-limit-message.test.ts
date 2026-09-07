import { describe, expect, it } from "vitest";

import { parseRateLimit } from "@/lib/rate-limit-message";

const err = (payload: unknown) => new Error(typeof payload === "string" ? payload : JSON.stringify(payload));

describe("parseRateLimit", () => {
    it("returns null for no error, non-JSON, and non-rate-limit payloads", () => {
        expect(parseRateLimit(undefined)).toBeNull();
        expect(parseRateLimit(err("Stream failed"))).toBeNull();
        expect(parseRateLimit(err({ error: "Something went wrong." }))).toBeNull();
    });

    it("maps each scope to its own copy and keeps retryAfter", () => {
        expect(parseRateLimit(err({ code: "rate_limited", scope: "global-daily", retryAfter: 3600 }))).toMatchObject({
            title: "Today's demo budget is used up",
            retryAfter: 3600,
        });
        expect(parseRateLimit(err({ code: "rate_limited", scope: "visitor-daily", retryAfter: 120 }))?.title).toBe(
            "You've reached the per-visitor limit"
        );
        expect(parseRateLimit(err({ code: "rate_limited", scope: "burst", retryAfter: 30 }))?.title).toBe("Slow down a moment");
    });

    it("humanises the wait: moment / minutes / an hour / hours", () => {
        const body = (s: number) => parseRateLimit(err({ code: "rate_limited", scope: "burst", retryAfter: s }))!.body;
        expect(body(30)).toContain("in a moment");
        expect(body(600)).toContain("in about 10 minutes");
        expect(body(5000)).toContain("in about an hour");
        expect(body(10800)).toContain("in about 3 hours");
    });

    it("defaults retryAfter to 60 when the payload omits it", () => {
        expect(parseRateLimit(err({ code: "rate_limited", scope: "burst" }))?.retryAfter).toBe(60);
    });
});
