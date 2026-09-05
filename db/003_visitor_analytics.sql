-- 003 — visitor attribution on the query log. See specs/visitor-analytics.md.
--
-- The app is public and linked from LinkedIn and a CV. The question this migration
-- answers is "which channel sent people, and what did they ask" — per channel, so the
-- next post or application can be judged by evidence rather than feel.
--
-- What is deliberately NOT here: identity, raw IPs, full referrer URLs, cities, user-agent
-- strings. visitor_hash is the same salted sha256 the rate limiter already uses; the IP it
-- came from is never written anywhere.

alter table query_log
    add column if not exists visitor_hash     text,   -- sha256(ip + IP_HASH_SALT)[0:32]; same as the limiter key
    add column if not exists landing_referrer text,   -- hostname only, e.g. 'linkedin.com'; from the FIRST load of the session
    add column if not exists utm_source       text,   -- '?utm_source=' on the landing URL; the reliable attribution
    add column if not exists country          text,   -- ISO-2 from Vercel's x-vercel-ip-country; null locally
    add column if not exists device           text;   -- 'mobile' | 'desktop', derived server-side from the UA

-- Per-visitor lookups and the views below.
create index if not exists query_log_visitor_idx on query_log (visitor_hash, created_at desc);

-- ── The LinkedIn question ───────────────────────────────────────────────────
-- Distinct visitors and questions per channel. UTM wins over referrer because mobile apps
-- (LinkedIn's included) often strip the referrer; a tagged link is the only reliable signal.
-- security_invoker = true: a view over an RLS-protected table must not hand out the rows
-- RLS withholds (same reasoning as suspicious_refusals in 002).
create or replace view visits_by_source
with (security_invoker = true) as
select
    coalesce(utm_source, landing_referrer, 'direct')   as source,
    count(distinct visitor_hash)                       as visitors,
    count(*)                                           as questions,
    count(*) filter (where refused)                    as refused,
    min(created_at)                                    as first_seen,
    max(created_at)                                    as last_seen
from query_log
where created_at > now() - interval '90 days'
  and visitor_hash is not null   -- rows from before 003 carry no attribution; counting them
                                 -- under 'direct' made "1 visitor, 28 questions" — keep them out
group by 1
order by visitors desc, questions desc;

-- ── "What did they ask?" ────────────────────────────────────────────────────
-- One row per visitor in the last 30 days, with their questions in order (truncated —
-- this is for scanning, not reading). A visitor from the post who asked three questions
-- about the eval harness is a different signal from one who asked "hi" and left.
create or replace view recent_visitors
with (security_invoker = true) as
select
    visitor_hash,
    coalesce(utm_source, landing_referrer, 'direct')   as source,
    country,
    device,
    count(*)                                           as questions,
    count(*) filter (where refused)                    as refused,
    min(created_at)                                    as first_seen,
    max(created_at)                                    as last_seen,
    array_agg(left(question, 80) order by created_at)  as questions_asked
from query_log
where created_at > now() - interval '30 days'
  and visitor_hash is not null
group by 1, 2, 3, 4
order by last_seen desc;

-- ── Retention ───────────────────────────────────────────────────────────────
-- The rows hold user question text. 90 days is the stated retention; run this by hand
-- (or wire it to pg_cron if it ever matters):
--   delete from query_log where created_at < now() - interval '90 days';

-- Sanity checks after deploying:
--   select * from visits_by_source;
--   select source, country, device, questions, questions_asked from recent_visitors limit 20;
