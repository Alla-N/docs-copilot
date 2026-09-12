-- 007: the trace id on every row the agent service writes (step 2.7).
--
-- The service traces each /chat turn in Langfuse (agent/src/copilot_agent/tracing.py). The trace
-- id is made before the run starts, so the row can carry it: given a slow answer, a refusal that
-- looks wrong, or an eval case that missed its page, this column is the way from "what it cost"
-- to "what it did". Null for rows the TypeScript route writes, and null whenever tracing is off,
-- which is the default (no keys, no trace).
--
-- 32 lowercase hex characters (the W3C trace id), not the thread id: one conversation is one
-- Langfuse session and many traces. thread_id (db/006) is the session key, this is the turn key.
--
-- The eval harness reads it back with the cost of its run, and fetches the retrieved chunk texts
-- from the trace to score faithfulness, which the stream does not carry.
--
-- The Python service refuses to start until this has run (query_log.readiness_problems).

alter table query_log add column if not exists trace_id text;

create index if not exists query_log_trace_idx on query_log (trace_id) where trace_id is not null;

-- query_cost gains trace_id at the END of the select list: create or replace can add columns,
-- never reorder or rename them, so anything new goes last.
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
    latency_ms, ttft_ms, generation_ms,
    trace_id
from query_log;
