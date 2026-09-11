"""search_docs with fake I/O: what goes where, and what happens when the reranker fails.

search_docs takes its three I/O steps as arguments, so these tests pass small fakes that
record every call. No network, no secrets, no cost: they run in the pre-commit hook.
"""

import logging
from collections.abc import Callable

import httpx
import pytest

from copilot_agent import retrieval_config
from copilot_agent.retrieval import (
    Candidate,
    Ranking,
    RetrievalResult,
    search_docs,
)

# Every test in this file is async and runs under the anyio plugin (backend pinned in conftest).
pytestmark = pytest.mark.anyio

MakeCandidate = Callable[..., Candidate]  # the make_candidate fixture, from conftest.py

QUERY = "how do I stream text"
HYPOTHETICAL = "Call streamText with a model and a prompt, then iterate over textStream."
EMBEDDING = [0.1, 0.2, 0.3]


class FakeServices:
    """Stand-ins for embed, vector_search and rerank. Each records what it was sent."""

    def __init__(
        self,
        found: list[Candidate],
        ranking: list[Ranking] | None = None,
        rerank_error: Exception | None = None,
    ) -> None:
        self.found = found
        self.ranking = ranking or []
        self.rerank_error = rerank_error
        self.embedded: list[str] = []
        self.searched: list[tuple[list[float], int]] = []
        self.reranked: list[tuple[str, list[str], int]] = []

    async def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return EMBEDDING

    async def vector_search(self, embedding: list[float], count: int) -> list[Candidate]:
        self.searched.append((embedding, count))
        return self.found

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[Ranking]:
        self.reranked.append((query, documents, top_n))
        if self.rerank_error is not None:
            raise self.rerank_error
        return self.ranking

    async def search(self, query: str, embed_text: str | None = None, **knobs) -> RetrievalResult:
        return await search_docs(
            query,
            embed_text,
            embed=self.embed,
            vector_search=self.vector_search,
            rerank=self.rerank,
            **knobs,
        )


@pytest.fixture
def found(make_candidate: MakeCandidate) -> list[Candidate]:
    """Four candidates in cosine order, as match_documents() returns them."""
    return [make_candidate(s) for s in (0.62, 0.58, 0.47, 0.40)]


# ---- routing: which text goes to which step -------------------------------------------


async def test_hyde_text_is_embedded_and_the_real_query_is_reranked(
    found: list[Candidate],
) -> None:
    services = FakeServices(found, ranking=[Ranking(index=0, score=0.8)])

    await services.search(QUERY, HYPOTHETICAL)

    # The hypothetical answer finds the neighbourhood; the cross-encoder judges the question.
    assert services.embedded == [HYPOTHETICAL]
    [(reranked_query, documents, top_n)] = services.reranked
    assert reranked_query == QUERY
    assert documents == [c.content for c in found]  # all of them, in cosine order
    assert top_n == retrieval_config.RERANK_TOP_N


async def test_without_a_hypothetical_the_query_itself_is_embedded(
    found: list[Candidate],
) -> None:
    services = FakeServices(found)

    await services.search(QUERY)

    assert services.embedded == [QUERY]
    assert services.reranked[0][0] == QUERY


async def test_the_embedding_and_candidate_depth_reach_vector_search(
    found: list[Candidate],
) -> None:
    services = FakeServices(found)

    await services.search(QUERY)

    assert services.searched == [(EMBEDDING, retrieval_config.VECTOR_CANDIDATES)]


async def test_knobs_override_the_calibrated_defaults(found: list[Candidate]) -> None:
    services = FakeServices(found, ranking=[Ranking(index=0, score=0.2)])

    result = await services.search(QUERY, candidates=40, top_n=3, threshold=0.1)

    assert services.searched[0][1] == 40
    assert services.reranked[0][2] == 3
    assert [c.score for c in result.relevant] == [0.2]  # would be dropped at 0.30


# ---- the reranked path ----------------------------------------------------------------


async def test_reranked_result_is_the_gated_ranking(found: list[Candidate]) -> None:
    ranking = [
        Ranking(index=2, score=0.74),
        Ranking(index=0, score=0.51),
        Ranking(index=1, score=0.12),
    ]
    services = FakeServices(found, ranking=ranking)

    result = await services.search(QUERY)

    assert result.mode == "reranked"
    assert result.candidates == found
    assert [c.score for c in result.relevant] == [0.74, 0.51]  # 0.12 is under 0.30
    assert [c.content for c in result.relevant] == [found[2].content, found[0].content]
    assert set(result.timings_ms) == {"embed", "vector", "rerank"}


async def test_no_candidates_means_no_rerank_call() -> None:
    services = FakeServices(found=[])

    result = await services.search(QUERY)

    assert services.reranked == []  # nothing to rerank, so no paid call
    # Same shape as lib/retrieve.ts: an empty search is still mode "reranked".
    assert result == RetrievalResult(
        candidates=[], relevant=[], mode="reranked", timings_ms=result.timings_ms
    )
    assert set(result.timings_ms) == {"embed", "vector"}


# ---- the fallback path ----------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(httpx.ConnectError("connection refused"), id="network"),
        pytest.param(
            httpx.HTTPStatusError(
                "401 Unauthorized",
                request=httpx.Request("POST", "https://api.cohere.com/v2/rerank"),
                response=httpx.Response(401),
            ),
            id="bad-key",
        ),
        pytest.param(RuntimeError("anything else"), id="unexpected"),
    ],
)
async def test_rerank_failure_falls_back_to_cosine(
    found: list[Candidate], error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    services = FakeServices(found, rerank_error=error)

    with caplog.at_level(logging.WARNING, logger="copilot_agent.retrieval"):
        result = await services.search(QUERY)

    assert result.mode == "cosine-fallback"
    assert result.candidates == found
    # Cosine order, the stricter 0.45 gate: 0.40 is dropped.
    assert [c.score for c in result.relevant] == [0.62, 0.58, 0.47]
    assert "rerank" in result.timings_ms  # the failed call still took time
    assert "falling back to cosine" in caplog.text


async def test_a_bug_after_the_rerank_call_raises_instead_of_falling_back(
    found: list[Candidate],
) -> None:
    # The reranker answers, but with an index outside the candidate list: gate_reranked hits
    # an IndexError. That is a bug on OUR side, and it must surface as one. lib/retrieve.ts
    # wraps the gating in the same try, so there it would pass as a Cohere outage.
    services = FakeServices(found, ranking=[Ranking(index=len(found), score=0.9)])

    with pytest.raises(IndexError):
        await services.search(QUERY)
