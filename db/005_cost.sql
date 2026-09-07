-- 005 — cost and timing, measured per request.
--
-- The daily spend ceiling (RATE_DAILY_GLOBAL, lib/rate-limit.ts) has always been derived from
-- an ESTIMATE of per-request cost: first "~€0.0005" with rerank free on the trial key, then
-- "~€0.004" after the key went to production. Every provider reports what a call actually
-- used; nothing was writing it down. These columns do. `cost_daily` then prices real traffic
-- with the list prices below, so the next ceiling is derived from data, and a price change is
-- a one-line edit here rather than a re-estimate. (Critical review, item 25.)
--
-- Timing: ttft_ms is the generation call's time to first output chunk (the SDK's
-- timeToFirstOutputMs); generation_ms the whole call. latency_ms (db/002) is planner +
-- embed + rerank. A visitor's time-to-first-token is latency_ms + ttft_ms.

alter table query_log
    add column if not exists planner_input_tokens  int,
    add column if not exists planner_output_tokens int,
    add column if not exists rerank_calls          int,   -- calls that reached Cohere (per sub-query)
    add column if not exists gen_input_tokens      int,
    add column if not exists gen_output_tokens     int,
    add column if not exists ttft_ms               int,   -- generation: time to first output chunk
    add column if not exists generation_ms         int;   -- generation: whole call

-- List prices, USD. gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output. Cohere rerank-v3.5:
-- $2.00 / 1,000 searches (one search = one rerank call of up to 100 documents).
-- Embeddings (text-embedding-3-small, $0.02 / 1M tokens) are not tracked: one short string
-- per sub-query, well under a hundredth of a cent — noise next to the rerank line.
-- Update here when prices move; the view recomputes history.
create or replace view cost_daily
with (security_invoker = true) as
with priced as (
    select
        created_at,
        retrieval_mode,
        (coalesce(planner_input_tokens, 0) + coalesce(gen_input_tokens, 0))   * 0.15 / 1e6
      + (coalesce(planner_output_tokens, 0) + coalesce(gen_output_tokens, 0)) * 0.60 / 1e6
      + coalesce(rerank_calls, 0) * 2.00 / 1000                                   as usd,
        planner_input_tokens is not null                                          as priced,
        latency_ms, ttft_ms, generation_ms
    from query_log
)
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
from priced
group by 1
order by 1 desc;

-- Re-deriving the ceiling: RATE_DAILY_GLOBAL × usd_per_request is the worst-case daily bill.
-- With usd_per_request measured, pick the ceiling from the budget, not the other way round:
--   select round(0.80 / usd_per_request) from cost_daily order by day desc limit 1;   -- €0.80/day
