-- 001 — content hashing, so ingestion becomes idempotent.
--
-- The hash must match hashChunk() in scripts/ingest.ts exactly:
--   sha256(source_url + "\n" + content), hex encoded.
-- If the two ever disagree, every row looks new and the whole corpus re-embeds.
-- That fails expensively, not destructively — but check the verification query at the bottom.

-- 1. Add the column, nullable for now so existing rows survive.
alter table documents add column if not exists content_hash text;

-- 2. Backfill. convert_to(...,'UTF8') matters: sha256 takes bytea, and the byte
--    encoding has to match what Node hashes, or nothing will ever line up.
update documents
set content_hash = encode(sha256(convert_to(source_url || E'\n' || content, 'UTF8')), 'hex')
where content_hash is null;

-- 3. Drop pre-existing duplicates (earlier non-idempotent re-runs may have left some).
--    Keeps the lowest id of each identical (source_url, content) pair.
delete from documents a
using documents b
where a.id > b.id
  and a.content_hash = b.content_hash;

-- 4. Now the constraints can be enforced.
alter table documents alter column content_hash set not null;
create unique index if not exists documents_content_hash_key on documents (content_hash);

-- 5. Verify: expect duplicates = 0, unhashed = 0, and total = your row count.
select
  count(*)                                        as total_rows,
  count(*) filter (where content_hash is null)    as unhashed,
  count(*) - count(distinct content_hash)         as duplicates
from documents;
