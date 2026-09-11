"""Smoke test of search_docs against the REAL services: OpenAI, Supabase, Cohere.

Marked integration, so the pre-commit hook skips it (network, secrets, about $0.002 per
rerank). Run it on purpose:

    uv run pytest -m integration

Step 1e replaces this with the full parity check against the TypeScript retrieve().
"""

import asyncio

import pytest

from copilot_agent.retrieval import RetrievalResult, open_search
from copilot_agent.settings import get_settings

pytestmark = pytest.mark.integration

# From evals/dataset.ts: the stream-text case and its accepted pages.
QUERY = "how do I stream text"
STREAM_TEXT_PAGES = ("generating-text", "stream-text")


async def _search(query: str) -> RetrievalResult:
    async with open_search(get_settings()) as search:
        return await search(query)


def test_live_search_finds_the_stream_text_page() -> None:
    # asyncio.run for now; step 1d introduces async tests properly.
    result = asyncio.run(_search(QUERY))

    assert result.mode == "reranked", "cosine fallback means the reranker call failed"
    assert len(result.candidates) == 100
    assert 1 <= len(result.relevant) <= 5
    assert all(chunk.score >= 0.30 for chunk in result.relevant)
    assert any(page in c.source_url for c in result.relevant for page in STREAM_TEXT_PAGES)
