"""hash_chunk: a chunk's identity, byte-identical to lib/content-hash.ts and the SQL.

    sha256(source_url + LF + content), UTF-8, hex
    (db/001_content_hash.sql: encode(sha256(convert_to(source_url || E'\\n' || content,
    'UTF8')), 'hex'))

CLAUDE.md invariant 1: ingestion is idempotent because all sides agree on these bytes. Here it
keys chunks in the golden-file parity test; if ingestion is ever ported, it is the upsert key.
tests/test_content_hash.py pins the same value as tests/content-hash.test.ts.
"""

import hashlib


def hash_chunk(source_url: str, content: str) -> str:
    return hashlib.sha256(f"{source_url}\n{content}".encode()).hexdigest()
