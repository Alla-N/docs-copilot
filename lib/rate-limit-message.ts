/**
 * Turns a 429 body into something a visitor should read.
 *
 * The client transport throws with the raw response text as error.message, so the
 * structured payload arrives here as a string. Parsing is best-effort: any error that
 * is not a recognised rate-limit payload falls through to the generic banner.
 *
 * The copy matters more than it looks. If a recruiter hits this wall, the wall is the
 * only thing they see — so it explains the constraint as a deliberate choice and points
 * at the repo. A blocked visitor who reads "personal project, real API keys, here are
 * the measured results" is not a lost visitor.
 */
export const REPO_URL = "https://github.com/Alla-N/docs-copilot";

export type RateLimitNotice = {
    title: string;
    body: string;
    retryAfter: number;
};

function humanise(seconds: number): string {
    if (seconds < 90) return "in a moment";
    if (seconds < 3600) return `in about ${Math.ceil(seconds / 60)} minutes`;
    if (seconds < 7200) return "in about an hour";
    return `in about ${Math.round(seconds / 3600)} hours`;
}

export function parseRateLimit(error: Error | undefined): RateLimitNotice | null {
    if (!error?.message) return null;

    let payload: { code?: string; scope?: string; retryAfter?: number };
    try {
        payload = JSON.parse(error.message);
    } catch {
        return null;
    }
    if (payload.code !== "rate_limited") return null;

    const retryAfter = payload.retryAfter ?? 60;
    const when = humanise(retryAfter);

    if (payload.scope === "global-daily") {
        return {
            title: "Today's demo budget is used up",
            body:
                "This is a personal project running on my own OpenAI and Cohere keys, so it has a daily " +
                "spending cap. It resets tomorrow. In the meantime the code, the architecture and the " +
                "measured results are on GitHub — including the eval harness that keeps it honest.",
            retryAfter,
        };
    }

    if (payload.scope === "visitor-daily") {
        return {
            title: "You've reached the per-visitor limit",
            body:
                `Thanks for trying it properly. The cap keeps one visitor from using up the whole day's ` +
                `budget — it resets ${when}. The code and the measured results are on GitHub if you want ` +
                `to see how it works under the hood.`,
            retryAfter,
        };
    }

    return {
        title: "Slow down a moment",
        body: `That's a few too many questions at once — try again ${when}.`,
        retryAfter,
    };
}
