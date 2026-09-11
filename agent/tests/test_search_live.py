"""Smoke test of search_docs against the REAL services: OpenAI, Supabase, Cohere.

Marked integration, so the pre-commit hook skips it (network, secrets, about $0.002 per
rerank). Run it on purpose:

    uv run pytest -m integration

Step 1e replaces this with the full parity check against the TypeScript retrieve().
"""

import pytest

from copilot_agent.retrieval import RetrievalResult, open_search
from copilot_agent.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

# From evals/dataset.ts: the stream-text case and its accepted pages.
QUERY = "how do I stream text"
STREAM_TEXT_PAGES = ("generating-text", "stream-text")


async def test_live_search_finds_the_stream_text_page() -> None:
    async with open_search(get_settings()) as search:
        result: RetrievalResult = await search(QUERY)

    assert result.mode == "reranked", "cosine fallback means the reranker call failed"
    assert len(result.candidates) == 100
    assert 1 <= len(result.relevant) <= 5
    assert all(chunk.score >= 0.30 for chunk in result.relevant)
    assert any(page in c.source_url for c in result.relevant for page in STREAM_TEXT_PAGES)
