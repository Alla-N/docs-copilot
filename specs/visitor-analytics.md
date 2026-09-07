# Spec — Visitor analytics (who comes, from where, what they ask)

**Status:** SHIPPED 2026-09-05 (db/003, lib/visitor.ts, lib/landing.ts, `<Analytics />`). Spec kept as the design record.
**Why now:** the app is on LinkedIn and in the CV. The question is not "does it work" but
"did anyone come, from which channel, and what did they do" — per channel, so the next
post/application can be judged by evidence.

## What "who" means here — and doesn't

No identity. LinkedIn passes no name or profile; we do not fingerprint. We learn, per
visitor: a stable anonymous id, the channel that sent them, country, device, and the
questions they asked. That is the job-search signal ("the post drove N visitors; M asked;
three asked about the eval harness"). Anything more is both impossible and inappropriate
for an EU-hosted public demo.

## Two layers

| Layer | Answers | Mechanism | Cost |
|---|---|---|---|
| **Page** | did anyone open it; referrer, country, device; opened-but-never-asked | Vercel Web Analytics (`@vercel/analytics`), cookieless, beacon to Vercel — **not** to our API | free on Hobby |
| **Question** | what each visitor asked, from which channel | enrich the EXISTING `query_log` write in the chat route | one migration, five columns |

**No new endpoint.** `CLAUDE.md` invariant #2: `app/api` holds exactly one route and there is
no mutating public endpoint. A `/api/track` beacon anyone could spam is the wrong shape;
the chat route already writes `query_log` behind the rate limiter, so it carries the extra
fields. Page views come from Vercel's own beacon, which never touches our surface.

## Columns added to `query_log` (db/003)

| column | value | source | why not more |
|---|---|---|---|
| `visitor_hash` | `sha256(ip + IP_HASH_SALT)[0:32]` | `clientKey(req)` — already computed for rate limiting | same anonymisation the limiter uses; the raw IP is never stored |
| `landing_referrer` | hostname only, e.g. `linkedin.com` | client: `document.referrer` on FIRST load of the session, sent as `x-landing-referrer` | full URLs can carry query strings with personal data; hostname is enough |
| `utm_source` | e.g. `linkedin`, `cv` | client: `?utm_source=` on the landing URL, sent as `x-utm-source` | referrers are stripped by many mobile apps; UTM is the reliable attribution |
| `country` | ISO-2, e.g. `DE` | Vercel header `x-vercel-ip-country` | coarse by design; no city, no IP |
| `device` | `mobile` \| `desktop` | user-agent, one regex | class only; no UA string stored |

All nullable: local dev has no Vercel headers; a direct visit has no referrer or UTM.

**Why the client must send referrer/UTM:** the `Referer` on the API call is always our own
origin — the LinkedIn referrer exists only on the initial page load. So the client captures
`document.referrer` + `utm_source` once per browser session into `sessionStorage` and
attaches them as headers to every chat request via the transport's `headers` resolver.
"Once per session" matters: a later in-app load would otherwise overwrite the real source
with our own hostname.

## Server-side sanitisation (route, before logQuery)

Headers are client-supplied and therefore untrusted, exactly like the request body.
- `landing_referrer`: lowercase, must match `^[a-z0-9.-]{1,100}$`, else `null`.
- `utm_source`: lowercase, must match `^[a-z0-9_-]{1,40}$`, else `null`.
- `country`: must match `^[A-Z]{2}$`, else `null`.
- `device`: derived server-side from the UA; never taken from a client header.
Nothing here can reach the model or the prompt — it goes only to the log row.

## Views (db/003, `security_invoker = true` like `suspicious_refusals`)

- `visits_by_source` — per `coalesce(utm_source, landing_referrer, 'direct')`: distinct
  visitors, questions, refusals, first/last seen. **The LinkedIn question.**
- `recent_visitors` — per visitor (last 30 days): source, country, device, question count,
  and the questions in order (first 80 chars). **"What did they ask?"**

## Privacy rules (so this stays a feature)

- No raw IP anywhere (existing rule). Referrer hostname only. Country, not city.
- Retention: rows older than 90 days are deleted — documented as a manual SQL statement in
  db/003 for now (pg_cron is the follow-up if it ever matters).
- One sentence in the README saying what is logged. Vercel Analytics is cookieless → no
  consent banner.

## Attribution hygiene (no code)

Tag the links: LinkedIn post/profile → `?utm_source=linkedin`; CV → `?utm_source=cv`;
GitHub README → `?utm_source=github`. Untagged = `direct`.

## Done when

- `npm run dev`, ask a question → row in `query_log` has `visitor_hash`, `device`, and
  (when opened via `/?utm_source=test`) `utm_source = 'test'`; `country` null locally.
- `select * from visits_by_source;` returns the row.
- Deployed: Vercel Analytics shows the visit; `country` populated on the logged row.
- `tsc` clean; full eval unchanged (this touches nothing on the retrieval/generation path).
