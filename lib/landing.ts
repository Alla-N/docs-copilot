/**
 * Landing attribution, client side — see specs/visitor-analytics.md.
 *
 * The API call's own Referer is always our origin; the page that *linked* here is only
 * visible on the first page load. So we read `document.referrer` and `?utm_source=` once
 * per browser session, keep them in sessionStorage, and send them as headers with every
 * chat request. "Once" matters: a later in-app load would otherwise overwrite the real
 * source with our own hostname.
 *
 * Only the referrer's hostname is kept — a full URL can carry query strings with personal
 * data. The server re-validates both values before writing them (lib/visitor.ts).
 *
 * Safe under SSR: every access to window/sessionStorage is guarded and try/catch'd
 * (private mode, blocked storage), and a failure simply means "no attribution".
 */

const KEY = "docs-copilot:landing";

type Landing = { referrer: string | null; utmSource: string | null };

/**
 * Call on mount. Two different rules, on purpose:
 *   - referrer: FIRST load of the session wins. A later in-app load's referrer is our own
 *     origin, and letting it overwrite the real source would erase the attribution.
 *   - utm_source: a tag on the CURRENT url always wins. It is an explicit statement of
 *     source — someone who visited direct earlier and now clicks the LinkedIn link in the
 *     same tab should be attributed to LinkedIn, not to the first visit.
 * The first version was "first load wins" for both, which silently ignored a UTM on any
 * tab that had already opened the app once. Found while testing; would have hidden real
 * LinkedIn traffic too.
 */
export function captureLanding(): void {
    if (typeof window === "undefined") return;
    try {
        const utm = new URLSearchParams(window.location.search).get("utm_source");
        const utmSource = utm ? utm.trim().toLowerCase().slice(0, 40) : null;

        const raw = sessionStorage.getItem(KEY);
        const existing: Landing | null = raw ? (JSON.parse(raw) as Landing) : null;

        // Already captured and no new tag to apply → nothing to do.
        if (existing && !utmSource) return;

        let referrer: string | null = existing?.referrer ?? null;
        if (!existing && document.referrer) {
            const host = new URL(document.referrer).hostname.toLowerCase();
            // Our own origin is not a source — it's an in-app navigation.
            if (host && host !== window.location.hostname) referrer = host;
        }

        const landing: Landing = { referrer, utmSource: utmSource ?? existing?.utmSource ?? null };
        sessionStorage.setItem(KEY, JSON.stringify(landing));
    } catch {
        /* storage unavailable — fine, we just won't attribute this session */
    }
}

/** Resolved per request by the chat transport. Returns only the headers that have a value. */
export function landingHeaders(): Record<string, string> {
    if (typeof window === "undefined") return {};
    try {
        const raw = sessionStorage.getItem(KEY);
        if (!raw) return {};
        const { referrer, utmSource } = JSON.parse(raw) as Landing;
        const headers: Record<string, string> = {};
        if (referrer) headers["x-landing-referrer"] = referrer;
        if (utmSource) headers["x-utm-source"] = utmSource;
        return headers;
    } catch {
        return {};
    }
}
