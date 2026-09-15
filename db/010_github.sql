-- 010 - what the GitHub subagent did, and what it cost. Step 3.6 of the Windward plan.
--
-- 3.5 wired the subagent in and deliberately deferred this column: the router's decision was the
-- thing that had just changed, and one migration per change keeps a stored run readable. 3.6 is
-- the step that cannot be done without it. Its done-when reports five measures, and four of them
-- (first-try validity, validity after repairs, points per question, routing accuracy) live in
-- fields the service already computes per turn and then dropped on the floor: the eval harness
-- talks to the service over HTTP and cannot see inside it (spec decision 12).
--
--   github                  the whole GitHubEvidence block for the turn, as jsonb. Null on every
--                           turn with no subagent in the path, which is most of them: a docs-only
--                           route, the TypeScript route, and any deployment without GITHUB_TOKEN,
--                           where the node is not built at all.
--   github_input_tokens     the subagent's own model calls. Columns rather than keys inside the
--   github_output_tokens    jsonb for one reason, and it is db/009's reason: query_cost prices
--                           what it can name. A cost computed out of `github->>'input_tokens'`
--                           would silently become zero the day that key is renamed, and a cost
--                           that holds still because it stopped measuring anything is the exact
--                           failure db/009 was written to prevent. A renamed COLUMN breaks the
--                           view loudly instead.
--
-- Why the evidence text is in here. The block carries the query the subagent wrote and the
-- evidence it handed to generation, not only the counters. 3.5b is why: asked when ai@5.0.0 was
-- released, the subagent listed the ten newest releases, and `ok`, `attempts`, `first_try_valid`
-- and `points_spent` all reported success on a turn that answered nothing. When answer accuracy
-- fails in 3.6, the only thing that separates "the subagent fetched the wrong facts" from
-- "generation had the facts and did not use them" is reading the evidence next to the answer.
-- Both are already in this row. The subagent caps its evidence at 4,000 bytes before it ever
-- reaches here, so the size of this column is bounded by that cap and not by the repository.
--
-- The `route` check constraint from db/009 is deliberately NOT tightened to ('docs', 'both'),
-- even though the code can no longer emit 'github' since the 3.5 reversal. Six rows from the
-- three-route run have it, and they are the evidence for the reversal. A constraint describes
-- what the table is allowed to hold, and this table holds a measurement that happened.
--
-- The Python service refuses to start until this has run (query_log.readiness_problems).

alter table query_log
    add column if not exists github               jsonb,
    add column if not exists github_input_tokens  integer,
    add column if not exists github_output_tokens integer;

-- Any row with a GitHub block came from a turn that ran the subagent, and a turn that ran the
-- subagent was routed 'both'. Stated as a constraint because it is the one cross-column claim
-- 3.6's routing accuracy rests on: if it ever fails, the routing number is being computed over
-- rows that did not route the way the column says. Validated, not `not valid`: every existing row
-- has a null block, so there is nothing to grandfather and nothing to leave half-checked.
alter table query_log drop constraint if exists query_log_github_route_check;
alter table query_log add constraint query_log_github_route_check
    check (github is null or route = 'both');

-- ---- query_cost: the subagent's tokens join the formula --------------------
-- Prices live here and only here (db/006 says so). The subagent is gpt-4o-mini like the other
-- three calls (spec decision 8), so the same two rates: $0.15 / 1M in, $0.60 / 1M out. Without
-- this the harness's measured cost per request would under-report every 'both' turn by exactly
-- the work the phase exists to add, and would do it quietly.
--
-- `priced` still keys off planner_input_tokens, unchanged, for db/009's reason: it means "this
-- row came from a writer that records tokens at all", and re-keying it would silently drop every
-- earlier row out of the averages it is used to filter.
--
-- New columns go at the END of the select list: create or replace can add columns to a view,
-- never reorder or rename them (db/007 learned this the same way, db/009 repeated it).
create or replace view query_cost
with (security_invoker = true) as
select
    id,
    created_at,
    origin,
    thread_id,
    retrieval_mode,
    (coalesce(planner_input_tokens, 0) + coalesce(gen_input_tokens, 0)
       + coalesce(router_input_tokens, 0) + coalesce(github_input_tokens, 0))  * 0.15 / 1e6
  + (coalesce(planner_output_tokens, 0) + coalesce(gen_output_tokens, 0)
       + coalesce(router_output_tokens, 0) + coalesce(github_output_tokens, 0)) * 0.60 / 1e6
  + coalesce(rerank_calls, 0) * 2.00 / 1000                                    as usd,
    planner_input_tokens is not null                                           as priced,
    latency_ms, ttft_ms, generation_ms,
    trace_id,
    route,
    router_input_tokens,
    router_output_tokens,
    github,
    github_input_tokens,
    github_output_tokens
from query_log;

-- cost_daily is NOT recreated: it selects usd from query_cost, and a view is recomputed at query
-- time, so it prices the subagent from the moment this file runs.

-- Sanity checks after running it:
--   select count(*) from query_log where github is not null;     -- 0 before the first 3.6 run
--   select route, count(*) filter (where github is not null), count(*)
--     from query_log where origin = 'eval' group by 1;           -- after a 3.6 run
--   select github->>'ok', github->>'attempts', github->>'points_spent', github->>'query'
--     from query_log where github is not null order by created_at desc limit 5;
