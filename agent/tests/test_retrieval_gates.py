"""The pure parts of retrieval: the two score gates and the pgvector text format.

No I/O and no event loop. These are the lines that decide what the model gets to see, so the
boundaries are tested exactly: both gates are INCLUSIVE, like the >= in lib/retrieve.ts.
"""

from collections.abc import Callable

import pytest

from copilot_agent import retrieval_config
from copilot_agent.retrieval import (
    Candidate,
    Ranking,
    RetrievedChunk,
    cosine_fallback,
    gate_reranked,
    to_vector_literal,
)

THRESHOLD = retrieval_config.RERANK_THRESHOLD
MakeCandidate = Callable[..., Candidate]  # the make_candidate fixture, from conftest.py


# ---- gate_reranked --------------------------------------------------------------------


def test_gate_keeps_the_rerankers_order_not_the_cosine_order(make_candidate: MakeCandidate) -> None:
    first, second, third = make_candidate(0.9), make_candidate(0.8), make_candidate(0.7)
    ranking = [Ranking(index=2, score=0.91), Ranking(index=0, score=0.55)]

    relevant = gate_reranked([first, second, third], ranking, THRESHOLD)

    assert relevant == [
        RetrievedChunk(
            content=third.content, title=third.title, source_url=third.source_url, score=0.91
        ),
        RetrievedChunk(
            content=first.content, title=first.title, source_url=first.source_url, score=0.55
        ),
    ]


@pytest.mark.parametrize(
    ("score", "kept"),
    [
        pytest.param(0.30, True, id="exactly-0.30-kept"),
        pytest.param(0.2999, False, id="just-below-dropped"),
        pytest.param(0.3001, True, id="just-above-kept"),
    ],
)
def test_rerank_threshold_is_inclusive(
    make_candidate: MakeCandidate, score: float, kept: bool
) -> None:
    relevant = gate_reranked([make_candidate(0.5)], [Ranking(index=0, score=score)], THRESHOLD)
    assert (len(relevant) == 1) is kept


def test_gate_scores_are_rerank_scores_not_cosine(make_candidate: MakeCandidate) -> None:
    # A chunk with a high cosine but a low rerank score is dropped: the gate reads the
    # cross-encoder's judgement, never the vector similarity.
    high_cosine = make_candidate(0.95)
    assert gate_reranked([high_cosine], [Ranking(index=0, score=0.1)], THRESHOLD) == []


def test_gate_with_nothing_ranked_returns_nothing(make_candidate: MakeCandidate) -> None:
    assert gate_reranked([make_candidate(0.9)], [], THRESHOLD) == []


# ---- cosine_fallback ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("similarity", "kept"),
    [
        pytest.param(0.45, True, id="exactly-0.45-kept"),
        pytest.param(0.4499, False, id="just-below-dropped"),
        pytest.param(0.6 - 0.15, False, id="computed-0.45-is-below"),
    ],
)
def test_cosine_gate_is_inclusive(
    make_candidate: MakeCandidate, similarity: float, kept: bool
) -> None:
    relevant = cosine_fallback([make_candidate(similarity)], top_n=5)
    assert (len(relevant) == 1) is kept


def test_fallback_keeps_cosine_order_and_caps_at_top_n(make_candidate: MakeCandidate) -> None:
    candidates = [make_candidate(s) for s in (0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)]

    relevant = cosine_fallback(candidates, top_n=5)

    assert [c.score for c in relevant] == [0.9, 0.85, 0.8, 0.75, 0.7]
    assert [c.content for c in relevant] == [c.content for c in candidates[:5]]


def test_fallback_gates_before_it_caps(make_candidate: MakeCandidate) -> None:
    # Same order as the TypeScript .filter().slice(): a weak chunk never takes one of the
    # top_n places, so the next strong one moves up.
    candidates = [make_candidate(s) for s in (0.9, 0.2, 0.8, 0.7)]

    relevant = cosine_fallback(candidates, top_n=2)

    assert [c.score for c in relevant] == [0.9, 0.8]


# ---- to_vector_literal ----------------------------------------------------------------


def test_vector_literal_is_pgvector_text_format() -> None:
    assert to_vector_literal([0.5, -1.0, 0.25]) == "[0.5,-1.0,0.25]"


def test_vector_literal_round_trips_every_float_exactly() -> None:
    # Values whose shortest text form is unusual: a binary rounding artefact, exponent
    # notation, negative zero, and 1.0 (the largest component a unit-length embedding has).
    embedding = [0.1 + 0.2, 1e-7, -3.5e-05, -0.0, 0.123456789012345678, 1.0]

    literal = to_vector_literal(embedding)

    assert literal.startswith("[") and literal.endswith("]")
    assert " " not in literal
    assert [float(part) for part in literal[1:-1].split(",")] == embedding
