-- 002 — query log. The point of this table is NOT analytics; it is eval-case mining.
--
-- The eval set has a few dozen cases I invented (9 when this table was written, 27 now).
-- Production has questions I never imagined, and the only way a false refusal enters the
-- eval set is if something notices it. This is that something.

create table if not exists query_log (
    id             bigint generated always as identity primary key,
    created_at     timestamptz not null default now(),
    question       text        not null,
    refused        boolean     not null,
    chunk_count    int         not null,   -- how many chunks cleared the threshold
    top_score      double precision,       -- best rerank score, null when nothing survived
    retrieval_mode text        not null,   -- 'reranked' | 'cosine-fallback' | 'skipped' (planner: off-topic)
    latency_ms     int
);

-- RLS on, no policies: the app connects with the service role key (which bypasses RLS),
-- so the server keeps working and nobody else can read anything. This table holds USER
-- QUESTION TEXT, and the anon key is designed to be publishable — without RLS, anyone
-- holding it could read every question ever asked. Stricter than `documents`, which only
-- contains public docs, and deliberately so.
alter table query_log enable row level security;

create index if not exists query_log_created_idx on query_log (created_at desc);
create index if not exists query_log_refused_idx on query_log (refused, top_score desc nulls last);

-- ── The query this table exists for ──────────────────────────────────────────
-- A refusal with NO chunks past threshold is almost certainly correct: the corpus
-- genuinely doesn't cover it. A refusal WITH high-scoring chunks is the new-7 shape —
-- good material was in front of the model and it said no anyway. Those are the
-- candidates for the eval set.
-- security_invoker = true is load-bearing. By default a Postgres view runs with its
-- OWNER's privileges, so a view over an RLS-protected table hands out exactly the rows RLS
-- was meant to withhold — and PostgREST exposes views in the public schema. Without this,
-- enabling RLS on query_log above would be undone by the view sitting on top of it.
create or replace view suspicious_refusals
with (security_invoker = true) as
select question, top_score, chunk_count, retrieval_mode, created_at
from query_log
where refused
  and chunk_count > 0
order by top_score desc nulls last;

-- Sanity check on the health of the whole system, over real traffic rather than eval cases:
--   select refused, count(*), round(avg(top_score)::numeric, 3) as avg_top
--   from query_log group by refused;
