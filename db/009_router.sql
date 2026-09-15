-- 009 — what the router decided, and what deciding cost. Step 3.5 of the Windward plan.
--
-- Phase 3 puts a router node in front of retrieval (specs/github-subagent.md, decision 1): a
-- separate model call, on every search turn, that chooses docs / github / both. It is a model
-- call like the planner's and the answer's, and until this migration ran it was the only one of
-- the three that no column recorded.
--
-- That matters more than a missing number usually does. The eval harness reports a MEASURED cost
-- per request by reading these rows back (query_cost, db/006), and the last stored figure is
-- $0.1972 per run. Without these columns the router's tokens are spent and not priced, so that
-- figure would have come back from the phase 3 runs unchanged — not because the cost had not
-- moved, but because nothing was watching the part that moved. A number that holds still because
-- it is no longer measuring anything is the failure this phase keeps finding, and the eval suite
-- is the last place it belongs.
--
--   router_input_tokens   the router call's prompt tokens; null on any row written without a
--   router_output_tokens  router in the path — the TypeScript route, and any deployment with no
--                         GITHUB_TOKEN, where the router node is not built at all (decision 11).
--   route                 'docs' | 'github' | 'both', the decision itself. Null means no router
--                         ran, which is NOT the same as 'docs' and must not be read as it: a
--                         turn that was never routed and a turn routed to the documentation are
--                         different events, and 3.6 measures routing accuracy off this column.
--
-- No index. query_log is read by the eval harness and by hand, at a few thousand rows; an index
-- here would be a habit rather than a measurement (db/006's thread_id index exists because the
-- service looks rows up by it, and this column is only ever aggregated).
--
-- The Python service refuses to start until this has run (query_log.readiness_problems).

alter table query_log
    add column if not exists router_input_tokens  integer,
    add column if not exists router_output_tokens integer,
    add column if not exists route                text;

alter table query_log drop constraint if exists query_log_route_check;
alter table query_log add constraint query_log_route_check
    check (route is null or route in ('docs', 'github', 'both'));

-- ── query_cost: the router's tokens join the formula ────────────────────────
-- Prices live here and only here (db/006 says so), so this is the one place the third call gets
-- priced. gpt-4o-mini, the same model as the planner and the answer (spec decision 8), so the
-- same two rates: $0.15 / 1M input, $0.60 / 1M output. Cohere rerank-v3.5: $2.00 / 1,000.
--
-- `priced` still keys off planner_input_tokens, unchanged: it means "this row came from a writer
-- that records tokens at all", and every routed row has a planner row behind it. Keying it off
-- the router instead would quietly re-define priced to mean "phase 3 or later" and silently drop
-- every earlier row out of the averages it is used to filter.
--
-- New columns go at the END of the select list: create or replace can add columns to a view,
-- never reorder or rename them (db/007 learned this the same way).
create or replace view query_cost
with (security_invoker = true) as
select
    id,
    created_at,
    origin,
    thread_id,
    retrieval_mode,
    (coalesce(planner_input_tokens, 0) + coalesce(gen_input_tokens, 0)
       + coalesce(router_input_tokens, 0))                                    * 0.15 / 1e6
  + (coalesce(planner_output_tokens, 0) + coalesce(gen_output_tokens, 0)
       + coalesce(router_output_tokens, 0))                                   * 0.60 / 1e6
  + coalesce(rerank_calls, 0) * 2.00 / 1000                                   as usd,
    planner_input_tokens is not null                                          as priced,
    latency_ms, ttft_ms, generation_ms,
    trace_id,
    route,
    router_input_tokens,
    router_output_tokens
from query_log;

-- cost_daily is NOT recreated: it selects usd from query_cost, and a view is recomputed at query
-- time, so it prices the router from the moment this file runs. Its history changes with it —
-- which is correct, because none of those rows had a router and coalesce prices them at zero.

-- Sanity checks after running it:
--   select count(*) from query_log where route is not null;      -- 0 before the first phase 3 run
--   select * from cost_daily limit 3;                            -- unchanged: no router rows yet
--   select route, count(*), round(avg(usd)::numeric, 5)
--     from query_cost where origin = 'eval' group by 1;          -- after the 3.5 eval runs
