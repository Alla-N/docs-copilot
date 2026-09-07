-- 004 — retrieval health over real traffic.
--
-- retrieval_mode already lands on every query_log row ('reranked' | 'cosine-fallback' |
-- 'skipped'). What was missing was a way to SEE it: on Day 15 the Cohere key's monthly cap
-- ran out and every production request silently took the cosine fallback — stricter cut,
-- more refusals — for hours, with nothing in the app or the data pointing at it. The UI now
-- says so per reply; this view says so per day, so the question "how often did the
-- reranker fail last week?" has an answer instead of a guess. (Critical review, item 22.)
--
-- security_invoker for the same reason as suspicious_refusals (db/002): a view over an
-- RLS-protected table must not run with its owner's privileges.
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
group by 1
order by 1 desc;

-- Anything above 0 in fallback_pct on a normal day means the reranker key or quota needs a
-- look before the eval numbers do — a degraded pipeline refuses questions the docs answer.
