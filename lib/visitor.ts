/**
 * Visitor attribution for the query log — see specs/visitor-analytics.md.
 *
 * Every value here comes from a header the CLIENT sent (referrer, UTM) or the edge added
 * (country, user-agent). That makes it untrusted input in exactly the way the request body
 * is: each field is validated against a tight shape and dropped to null otherwise. None of
 * it reaches the model or the prompt — it goes only to the log row.
 *
 * What is deliberately not collected: identity, raw IPs, full referrer URLs (query strings
 * carry personal data), cities, user-agent strings. `visitorHash` is the same salted sha256
 * the rate limiter keys on, so one anonymisation rule covers both uses.
 */
import { clientKey } from "./rate-limit";

export type Visitor = {
    /** sha256(ip + IP_HASH_SALT)[0:32] — stable per visitor, never reversible to the IP. */
    visitorHash: string;
    /** Hostname of the page that linked here on the FIRST load of the session, e.g. "linkedin.com". */
    landingReferrer: string | null;
    /** `?utm_source=` on the landing URL — the reliable attribution; referrers get stripped. */
    utmSource: string | null;
    /** ISO-2 from Vercel's edge (`x-vercel-ip-country`). Null in local dev. */
    country: string | null;
    /** Class only, derived server-side from the UA. The UA string itself is not stored. */
    device: "mobile" | "desktop";
};

const REFERRER_HOST = /^[a-z0-9.-]{1,100}$/;
const UTM_SOURCE = /^[a-z0-9_-]{1,40}$/;
const COUNTRY_ISO2 = /^[A-Z]{2}$/;

export function visitorFrom(req: Request): Visitor {
    const h = req.headers;
    const referrer = (h.get("x-landing-referrer") ?? "").trim().toLowerCase();
    const utm = (h.get("x-utm-source") ?? "").trim().toLowerCase();
    const country = (h.get("x-vercel-ip-country") ?? "").trim().toUpperCase();
    const ua = h.get("user-agent") ?? "";

    return {
        visitorHash: clientKey(req),
        landingReferrer: REFERRER_HOST.test(referrer) ? referrer : null,
        utmSource: UTM_SOURCE.test(utm) ? utm : null,
        country: COUNTRY_ISO2.test(country) ? country : null,
        device: /Mobi|Android|iPhone|iPad/i.test(ua) ? "mobile" : "desktop",
    };
}
