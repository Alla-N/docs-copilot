-- 006 — who wrote the row, and which conversation it belongs to. Step 2.6 of the Windward plan.
--
-- Until now one writer filled query_log: the Next.js route (lib/query-log.ts), for the requests
-- visitors send. With AGENT_URL set the route forwards to the Python agent service and becomes a
-- byte pipe, so the service writes the row instead (agent/src/copilot_agent/query_log.py) — and
-- the eval harness can now send its questions through that same service. Two columns follow:
--
--   origin     'web' for a visitor's request (either writer), 'eval' for the eval harness. Every
--              view below counts web rows only: an eval run must not look like traffic, must not
--              be priced into the daily ceiling, and must not mine its own refusals as eval
--              cases. The harness reads its eval rows back (query_cost) for a MEASURED cost per
--              request, which until now was an estimate.
--   thread_id  the conversation (useChat's chat id, the LangGraph thread). Null for rows the
--              TypeScript route writes. One key per conversation, like a session id: it is only
--              readable here, where RLS already limits query_log to the service role (db/002).
--
-- Existing rows get origin 'web' (the default): every one of them came from the route.
-- The Python service refuses to start until this has run (query_log.readiness_problems).

alter table query_log
    add column if not exists origin    text not null default 'web',
    add column if not exists thread_id text;

alter table query_log drop constraint if exists query_log_origin_check;
alter table query_log add constraint query_log_origin_check check (origin in ('web', 'eval'));

create index if not exists query_log_thread_idx on query_log (thread_id) where thread_id is not null;

-- ── Per-row cost ─────────────────────────────────────────────────────────────
-- The prices of db/005 moved here, once, so cost_daily and the eval harness read one formula.
-- Update prices HERE when they move; every view recomputes history.
-- gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output. Cohere rerank-v3.5: $2.00 / 1,000 searches.
create or replace view query_cost
with (security_invoker = true) as
select
    id,
    created_at,
    origin,
    thread_id,
    retrieval_mode,
    (coalesce(planner_input_tokens, 0) + coalesce(gen_input_tokens, 0))   * 0.15 / 1e6
  + (coalesce(planner_output_tokens, 0) + coalesce(gen_output_tokens, 0)) * 0.60 / 1e6
  + coalesce(rerank_calls, 0) * 2.00 / 1000                                   as usd,
    planner_input_tokens is not null                                          as priced,
    latency_ms, ttft_ms, generation_ms
from query_log;

-- ── The views of 002–005, web rows only ─────────────────────────────────────
-- Same columns as before (create or replace keeps them); each gains `origin = 'web'`.

create or replace view suspicious_refusals
with (security_invoker = true) as
select question, top_score, chunk_count, retrieval_mode, created_at
from query_log
where refused
  and chunk_count > 0
  and origin = 'web'
order by top_score desc nulls last;

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
  and visitor_hash is not null
  and origin = 'web'
group by 1
order by visitors desc, questions desc;

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
  and origin = 'web'
group by 1, 2, 3, 4
order by last_seen desc;

create or replace view retrieval_health
with (security_invoker = true) as
select
    date_trunc('day', created_at)::date                               as day,
    count(*)                                                          as questions,
    count(*) filter (where retrieval_mode = 'reranked')               as reranked,
    count(*) filter (where retrieval_mode = 'cosine-fallback')        as cosine_fallback,
    count(*) filter (where retrieval_mode = 'skipped')                as skipped_by_planner,
    round(100.0 * count(*) filter (where retrieval_mode = 'cosine-fallback') / nullif(count(*), 0), 1)
                                                                      as fallback_pct,
    count(*) filter (where refused)                                   as refused,
    round(avg(latency_ms))                                            as avg_retrieval_ms
from query_log
where origin = 'web'
group by 1
order by 1 desc;

create or replace view cost_daily
with (security_invoker = true) as
select
    date_trunc('day', created_at)::date                         as day,
    count(*)                                                    as requests,
    count(*) filter (where priced)                              as priced_requests,
    round(sum(usd)::numeric, 4)                                 as usd_total,
    round((avg(usd) filter (where priced))::numeric, 5)         as usd_per_request,
    round((max(usd) filter (where priced))::numeric, 5)         as usd_max_request,
    round(avg(latency_ms))                                      as avg_retrieval_ms,
    round(avg(ttft_ms))                                         as avg_ttft_ms,
    round(avg(latency_ms + coalesce(ttft_ms, 0)))               as avg_time_to_first_token_ms,
    round(avg(generation_ms))                                   as avg_generation_ms
from query_cost
where origin = 'web'
group by 1
order by 1 desc;

-- Sanity checks after running it:
--   select origin, count(*) from query_log group by 1;                 -- every old row is 'web'
--   select * from cost_daily limit 3;                                  -- unchanged numbers
--   select count(*), round(avg(usd)::numeric, 5) from query_cost where origin = 'eval';
