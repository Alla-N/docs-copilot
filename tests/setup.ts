/**
 * Dummy environment for modules that construct clients at import time (lib/retrieve.ts,
 * lib/query-log.ts). `createClient` does not connect on construction, so a placeholder URL
 * is enough to import them; no test here makes a network call.
 *
 * UPSTASH_* is deliberately left unset: the rate limiter must be unconfigured, or
 * lib/rate-limit.ts refuses to load without IP_HASH_SALT (which is the behaviour under test).
 */
process.env.SUPABASE_URL ??= "http://localhost:54321";
process.env.SUPABASE_SERVICE_KEY ??= "test-service-key";
process.env.OPENAI_API_KEY ??= "test-openai-key";
process.env.COHERE_API_KEY ??= "test-cohere-key";
delete process.env.UPSTASH_REDIS_REST_URL;
delete process.env.UPSTASH_REDIS_REST_TOKEN;
