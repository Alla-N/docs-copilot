-- 000 — base schema. Run this first on a fresh Supabase project, then 001.
--
-- Embedding dimension 1536 is not arbitrary: it is text-embedding-3-small's output.
-- The ingestion script and the query path must use the SAME model, or the vectors
-- share a dimension while meaning nothing to each other — a failure that produces
-- plausible-looking garbage rather than an error.

create extension if not exists vector;

create table if not exists documents (
    id          bigint generated always as identity primary key,
    content     text        not null,
    source_url  text        not null,
    title       text        not null,
    embedding   vector(1536) not null
);

-- Read path. Cosine distance (<=>) ascending = most similar first;
-- returned as 1 - distance so callers get a similarity where higher is better.
create or replace function public.match_documents(
    query_embedding vector,
    match_count integer default 5
)
returns table (
    id bigint,
    content text,
    source_url text,
    title text,
    similarity double precision
)
language sql
stable
as $function$
    select
        documents.id,
        documents.content,
        documents.source_url,
        documents.title,
        1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    order by documents.embedding <=> query_embedding
    limit match_count;
$function$;

-- Row Level Security is enabled on this table; the app connects with the service
-- role key from server-side code only, which bypasses RLS. The key is never sent
-- to the browser — every query goes through /api/chat.
alter table documents enable row level security;

-- NOTE: no ANN index on `embedding`. At this corpus size the planner does an exact
-- scan, which is both faster and perfectly recalled. An hnsw/ivfflat index trades
-- recall for speed and only pays off at a much larger row count. Measure before adding:
--
--   explain analyze select * from match_documents(
--     (select embedding from documents limit 1), 5
--   );
