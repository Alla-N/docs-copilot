import { describe, expect, it } from "vitest";

import { visitorFrom } from "@/lib/visitor";

const req = (headers: Record<string, string>) => new Request("http://localhost/api/chat", { method: "POST", headers });

describe("visitorFrom", () => {
    it("keeps well-formed attribution and normalises case", () => {
        const v = visitorFrom(
            req({
                "x-landing-referrer": "LinkedIn.com",
                "x-utm-source": "GitHub",
                "x-vercel-ip-country": "gr",
                "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
                "x-forwarded-for": "203.0.113.7, 10.0.0.1",
            })
        );
        expect(v.landingReferrer).toBe("linkedin.com");
        expect(v.utmSource).toBe("github");
        expect(v.country).toBe("GR");
        expect(v.device).toBe("mobile");
    });

    it("drops anything that does not match the tight shapes — headers are client input", () => {
        const v = visitorFrom(
            req({
                "x-landing-referrer": "https://evil.example/?q=<script>",
                "x-utm-source": "a b; drop table",
                "x-vercel-ip-country": "GRC",
            })
        );
        expect(v.landingReferrer).toBeNull();
        expect(v.utmSource).toBeNull();
        expect(v.country).toBeNull();
        expect(v.device).toBe("desktop");
    });

    it("hashes the first x-forwarded-for hop to a stable 32-hex pseudonym, never the IP", () => {
        const a = visitorFrom(req({ "x-forwarded-for": "203.0.113.7, 10.0.0.1" }));
        const b = visitorFrom(req({ "x-forwarded-for": "203.0.113.7" }));
        const c = visitorFrom(req({ "x-forwarded-for": "203.0.113.8" }));
        expect(a.visitorHash).toMatch(/^[0-9a-f]{32}$/);
        expect(a.visitorHash).toBe(b.visitorHash);
        expect(a.visitorHash).not.toBe(c.visitorHash);
        expect(a.visitorHash).not.toContain("203");
    });

    it("all attribution fields are nullable — a bare request still yields a row", () => {
        const v = visitorFrom(req({}));
        expect(v).toEqual({ visitorHash: expect.any(String), landingReferrer: null, utmSource: null, country: null, device: "desktop" });
    });
});
