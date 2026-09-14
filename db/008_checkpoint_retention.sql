-- 008: retention for the LangGraph checkpoint tables (before phase 2b).
--
-- The four tables the checkpointer creates (agent/src/copilot_agent/checkpoint.py) grow without
-- bound. LangGraph ships no TTL and no prune: adelete_thread(thread_id) is the only delete in
-- langgraph/checkpoint/postgres/aio.py, and it takes a thread id, never an age. Every line below
-- is a decision, not a default that happened to be on.
--
-- WHY THE UNIT IS A THREAD, NOT A CHECKPOINT
-- checkpoint_blobs is keyed by (thread_id, checkpoint_ns, channel, version) and carries no
-- checkpoint_id: which blob version a checkpoint uses is written inside that checkpoint's own
-- channel_versions map (see SELECT_SQL in langgraph/checkpoint/postgres/base.py). Ageing out the
-- old checkpoints of a LIVE thread would either orphan blob rows that nothing references, or, if
-- the blobs went with them, remove a version a surviving checkpoint still points at, and that
-- thread would load with nulls where its state should be. Whole threads is the only unit the
-- three tables agree on without a join, which is also the only unit adelete_thread offers.
--
-- WHY 30 DAYS
-- A thread id is useChat's chat id, and app/page.tsx makes it with
-- useState(() => crypto.randomUUID()): component state, not localStorage, not a cookie. A tab
-- that is reloaded or closed can never address its thread again, so nothing older than one
-- browsing session is reachable by anybody. 30 days is therefore not what a conversation needs;
-- it is Langfuse Cloud free-tier retention (decision 10). Pairing the two means a trace and the
-- state that produced it expire together: for as long as you can see a turn in Langfuse, the
-- thread behind it is still here to load. At the measured 9.3 KiB per turn, volume argues for
-- nothing either way; having a bound at all is the point.
--
-- WHY NOT THE public SCHEMA
-- Supabase's Data API serves every function in public as POST /rest/v1/rpc/<name> to the
-- publishable anon key, and PostgreSQL grants EXECUTE to PUBLIC on a new function by default. A
-- prune function in public would be an unauthenticated HTTP endpoint that deletes conversations.
-- The Data API is not configured to expose the maintenance schema, so nothing here has a URL.
-- Same threat the Row Level Security in checkpoint.py and db/002_query_log.sql answers: the anon
-- key is designed to be published.
--
-- WHY query_log IS NOT PRUNED HERE
-- It is the measurement record, not conversation state. query_cost, cost_daily and the retrieval
-- health views read it, the eval harness reads its own rows back for a measured cost per run,
-- and every latency and cost figure in the README rests on its history. One row per turn is a
-- rounding error next to the checkpoints. It keeps everything, on purpose.
--
-- Run this once, in the Supabase SQL editor. Re-running it is safe: cron.schedule upserts by job
-- name, and everything else is if-not-exists or create-or-replace.
-- To stop it:  select cron.unschedule('prune-checkpoint-threads');

-- pg_cron. If this statement errors, enable the extension in the Dashboard (Database >
-- Extensions > pg_cron) and re-run from the next statement: on Supabase the extension itself may
-- only live in pg_catalog, while its own tables (cron.job, cron.job_run_details) are always in
-- the cron schema.
create extension if not exists pg_cron;
grant usage on schema cron to postgres;

-- Not public: see WHY NOT THE public SCHEMA above. The revoke is a no-op on a fresh schema (only
-- the owner gets usage), and is here so the intent survives someone granting later.
create schema if not exists maintenance;
revoke all on schema maintenance from anon, authenticated;

create table if not exists maintenance.checkpoint_retention_log (
    id                  bigint generated always as identity primary key,
    ran_at              timestamptz not null default now(),
    retain_days         integer     not null,
    cutoff              timestamptz not null,
    threads_deleted     integer     not null,
    checkpoints_deleted integer     not null,
    blobs_deleted       integer     not null,
    writes_deleted      integer     not null
);

comment on table maintenance.checkpoint_retention_log is
    'One row per prune run, including runs that deleted nothing. Counts and timestamps only, '
    'never a thread id and never any text: this answers "is retention actually running", it is '
    'not a record of what was deleted. Read by: python -m copilot_agent.checkpoint check';

create or replace function maintenance.prune_checkpoint_threads(retain_days integer default 30)
returns integer
language plpgsql
security invoker
-- An empty search_path (pg_catalog is always searched) so the function cannot be redirected at
-- another schema's tables by whatever path the caller happens to have. Every name below is
-- therefore qualified.
set search_path = ''
as $$
declare
    cutoff        timestamptz := now() - make_interval(days => retain_days);
    stale         text[];
    n_threads     integer;
    n_writes      integer;
    n_blobs       integer;
    n_checkpoints integer;
begin
    -- A typo in the schedule must not empty the tables. One day is already far longer than any
    -- conversation that is still reachable, so anything under it is a mistake, not a policy.
    if retain_days is null or retain_days < 1 then
        raise exception 'retain_days must be at least 1, got %', retain_days;
    end if;

    -- A thread's age is its NEWEST checkpoint. The tables have no timestamp column: the ISO
    -- string LangGraph writes into the checkpoint JSONB is the only readable one. (checkpoint_id
    -- is time-ordered too, but it is a UUIDv6 -- 100-nanosecond intervals since 1582, split over
    -- three fields with the version nibble in the middle -- and nobody rereading this would
    -- believe the arithmetic.)
    --
    -- The regex is a guard, not a parser. A row whose ts does not look like a timestamp is
    -- skipped, so a thread whose rows are all unreadable never reaches this list and is KEPT:
    -- retention fails towards keeping data.
    select coalesce(array_agg(thread_id), '{}'::text[])
      into stale
      from (
            select c.thread_id
              from public.checkpoints c
             where c.checkpoint ->> 'ts' ~ '^\d{4}-\d{2}-\d{2}T'
             group by c.thread_id
            having max((c.checkpoint ->> 'ts')::timestamptz) < cutoff
           ) aged_out;

    n_threads := coalesce(cardinality(stale), 0);

    -- One statement per table, all in one transaction (a plpgsql function is one). No foreign
    -- keys tie the three together, so nothing cascades and nothing here can half-happen. A live
    -- thread is never in `stale`: its newest checkpoint is minutes old, not 30 days.
    with gone as (delete from public.checkpoint_writes where thread_id = any(stale) returning 1)
    select count(*) into n_writes from gone;

    with gone as (delete from public.checkpoint_blobs where thread_id = any(stale) returning 1)
    select count(*) into n_blobs from gone;

    with gone as (delete from public.checkpoints where thread_id = any(stale) returning 1)
    select count(*) into n_checkpoints from gone;

    insert into maintenance.checkpoint_retention_log
        (retain_days, cutoff, threads_deleted, checkpoints_deleted, blobs_deleted, writes_deleted)
    values
        (retain_days, cutoff, n_threads, n_checkpoints, n_blobs, n_writes);

    return n_threads;
end;
$$;

-- Belt and braces next to the schema: PostgreSQL grants EXECUTE to PUBLIC on every new function.
revoke all on function maintenance.prune_checkpoint_threads(integer) from public;

-- 03:17 UTC daily. Off-peak, and an odd minute so it does not share a slot with everything else
-- in the world that runs on the hour. pg_cron replaces a job with the same name, so re-running
-- this file changes the schedule instead of adding a second job.
select cron.schedule(
    'prune-checkpoint-threads',
    '17 3 * * *',
    $$select maintenance.prune_checkpoint_threads(30)$$
);

-- What it would delete right now, without deleting it:
--
--   select count(*) as threads
--     from (select thread_id
--             from public.checkpoints
--            where checkpoint ->> 'ts' ~ '^\d{4}-\d{2}-\d{2}T'
--            group by thread_id
--           having max((checkpoint ->> 'ts')::timestamptz) < now() - interval '30 days') s;
--
-- That the job exists, and what it has done:
--
--   select jobid, jobname, schedule, command, active from cron.job
--    where jobname = 'prune-checkpoint-threads';
--
--   select d.status, d.return_message, d.start_time, d.end_time
--     from cron.job_run_details d join cron.job j using (jobid)
--    where j.jobname = 'prune-checkpoint-threads'
--    order by d.start_time desc limit 5;
--
--   select * from maintenance.checkpoint_retention_log order by ran_at desc limit 5;
--
-- The last two are also the two lines that
--   cd agent && uv run python -m copilot_agent.checkpoint check
-- prints, so a deploy check does not need the SQL editor.
