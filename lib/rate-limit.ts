/**
 * Rate limiting for /api/chat — the endpoint sits in front of paid OpenAI and Cohere
 * keys, so an unthrottled loop is a bill, not just load.
 *
 * Redis rather than Postgres because a limiter must be ATOMIC. Counting rows and then
 * inserting is a read-modify-write race: two requests arriving together both read a
 * count under the limit and both pass. That race is worst exactly when you are being
 * hit hardest, which is the only moment a limiter matters.
 *
 * In-memory (a Map) is not an option: serverless instances do not share memory, are
 * created on demand and reclaimed when idle, so the effective limit is
 * "10 × however many instances the load happened to create" — a number that rises
 * with the attack it is meant to stop, and resets on every deploy.
 */
import { createHash } from "node:crypto";

import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

const configured = Boolean(
    process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN
);

/**
 * Three limits, because they protect three different things. Env-configurable so the
 * budget can be raised before an interview without editing code.
 *
 *   BURST_PER_MIN   responsiveness. 10/min was originally also Cohere's TRIAL ceiling
 *                   (past it the pipeline degraded to cosine fallback). The key is a
 *                   production one now (Day 15), so this is purely a responsiveness
 *                   choice — kept at 10 because nobody types faster than that.
 *
 *   DAILY_PER_IP    stops one visitor eating the whole budget (~6% cap each).
 *
 *   DAILY_GLOBAL    THE ACTUAL COST CEILING. A per-IP limit bounds abuse but not
 *                   spend: a public link gets hundreds of distinct IPs, each with
 *                   its own allowance. Only a global counter bounds the bill.
 *                   RE-DERIVED on Day 15 when the Cohere key moved off the trial (its
 *                   1,000 calls/month ran out mid-eval). Rerank is now the dominant
 *                   per-request cost, as the previous version of this comment
 *                   predicted: ~$0.002 per rerank call at $2/1k searches, and the
 *                   planner issues one call per sub-query (1–2 typical, 4 max), plus
 *                   ~€0.0005 of gpt-4o-mini. Call it ~€0.004 per request worst-ish
 *                   case. 200/day ≈ €0.80/day, ≈ €25/month if someone saturates it —
 *                   was 800 ≈ €0.50/day when rerank was free. Raise via env before
 *                   an interview, not in code.
 */
const BURST_PER_MIN = Number(process.env.RATE_BURST_PER_MIN ?? 10);
const DAILY_PER_IP = Number(process.env.RATE_DAILY_PER_IP ?? 50);
const DAILY_GLOBAL = Number(process.env.RATE_DAILY_GLOBAL ?? 200);

const redis = configured ? Redis.fromEnv() : null;

/**
 * The salt is what makes the IP hash a pseudonym rather than a lookup table: the IPv4 space
 * is 4 billion values, and sha256 of each of them keyed on a constant that is in a public
 * repo is an afternoon's brute force. The first version fell back to "docs-copilot" when
 * IP_HASH_SALT was unset — silently, in production. Now: Upstash configured (= this is a
 * real deployment) and no salt is a misconfiguration that must be visible on the first
 * request, not a quiet downgrade. Local dev (no Upstash) still works without one.
 * (Review item 19.)
 */
const IP_HASH_SALT = process.env.IP_HASH_SALT;
if (configured && !IP_HASH_SALT) {
    throw new Error(
        "IP_HASH_SALT is not set but the rate limiter is configured. Refusing to hash visitor IPs with a public constant — set IP_HASH_SALT in the deployment's environment."
    );
}

const burst = redis
    ? new Ratelimit({
        redis,
        limiter: Ratelimit.slidingWindow(BURST_PER_MIN, "1 m"),
        prefix: "dc:burst",
        analytics: true,
    })
    : null;

const perIpDaily = redis
    ? new Ratelimit({ redis, limiter: Ratelimit.slidingWindow(DAILY_PER_IP, "1 d"), prefix: "dc:ip-day" })
    : null;

const globalDaily = redis
    ? new Ratelimit({ redis, limiter: Ratelimit.slidingWindow(DAILY_GLOBAL, "1 d"), prefix: "dc:global-day" })
    : null;

/**
 * Identify the caller by a SALTED HASH of their IP, never the raw address.
 * An IP is personal data under GDPR, this is an EU-facing demo, and a limiter only
 * needs to tell callers apart — it never needs to know who they are.
 *
 * Known imprecision, accepted: one person on wifi then mobile gets two allowances;
 * a whole office behind one NAT shares one. You can only rate limit as precisely as
 * you can identify, and an anonymous public endpoint has nothing better than IP.
 */
export function clientKey(req: Request): string {
    const forwarded = req.headers.get("x-forwarded-for") ?? "";
    const ip = forwarded.split(",")[0]?.trim() || req.headers.get("x-real-ip") || "unknown";
    // "dev" only ever applies with no limiter configured (see the check above).
    return createHash("sha256")
        .update(`${ip}:${IP_HASH_SALT ?? "dev"}`)
        .digest("hex")
        .slice(0, 32);
}

export type RateLimitScope = "burst" | "visitor-daily" | "global-daily" | "none";

export type RateLimitResult = {
    allowed: boolean;
    /** Seconds until the caller may retry. Only meaningful when allowed === false. */
    retryAfter: number;
    scope: RateLimitScope;
};

export async function checkRateLimit(key: string): Promise<RateLimitResult> {
    // Unconfigured means local development — failing open keeps `npm run dev` working
    // without Redis. Production has the env vars, so this branch is never taken there.
    if (!burst || !perIpDaily || !globalDaily) {
        return { allowed: true, retryAfter: 0, scope: "none" };
    }

    try {
        // One round-trip for all three. Tradeoff: a request blocked by one limit still
        // consumes a token from the other two. At these volumes the over-count is
        // irrelevant, and sequential checks would triple the latency on the happy path.
        const [g, ip, b] = await Promise.all([
            globalDaily.limit("global"),
            perIpDaily.limit(key),
            burst.limit(key),
        ]);

        // Most-restrictive first, so the message explains the real reason.
        if (!g.success) return { allowed: false, retryAfter: secondsUntil(g.reset), scope: "global-daily" };
        if (!ip.success) return { allowed: false, retryAfter: secondsUntil(ip.reset), scope: "visitor-daily" };
        if (!b.success) return { allowed: false, retryAfter: secondsUntil(b.reset), scope: "burst" };

        return { allowed: true, retryAfter: 0, scope: "none" };
    } catch (err) {
        // Fail OPEN, loudly. A Redis blip taking down the demo is worse than a few
        // unmetered minutes. This is a real tradeoff, not an obvious one: fail-open
        // means anyone who can disrupt Redis also disables the limiter. For a demo with
        // provider-side caps behind it, that is acceptable; where real money is at
        // stake, fail closed and alert.
        console.error("RATE LIMIT CHECK FAILED — failing open:", err);
        return { allowed: true, retryAfter: 0, scope: "none" };
    }
}

function secondsUntil(resetMs: number): number {
    return Math.max(1, Math.ceil((resetMs - Date.now()) / 1000));
}
