"""union_relevant: the port of plannedRetrieve's union in lib/plan.ts.

Recall is scored on this union, so the parity test's recall is only as right as this is.
"""

from copilot_agent import retrieval_config
from copilot_agent.retrieval import RetrievedChunk, union_relevant


def chunk(name: str, score: float, page: str = "page") -> RetrievedChunk:
    return RetrievedChunk(
        content=f"{name} text",
        title=name,
        source_url=f"https://ai-sdk.dev/docs/{page}",
        score=score,
    )


def test_a_chunk_found_twice_is_kept_once_with_its_best_score() -> None:
    merged = union_relevant([[chunk("a", 0.5)], [chunk("a", 0.9)], [chunk("a", 0.7)]])
    assert merged == [chunk("a", 0.9)]


def test_same_text_on_another_page_is_another_chunk() -> None:
    merged = union_relevant([[chunk("a", 0.5, page="one")], [chunk("a", 0.4, page="two")]])
    assert [c.source_url for c in merged] == [
        "https://ai-sdk.dev/docs/one",
        "https://ai-sdk.dev/docs/two",
    ]


def test_best_score_first_across_sub_queries() -> None:
    merged = union_relevant([[chunk("a", 0.5), chunk("b", 0.4)], [chunk("c", 0.8)]])
    assert [c.title for c in merged] == ["c", "a", "b"]


def test_equal_scores_keep_first_seen_order() -> None:
    # Like the stable JS sort over Map insertion order. A later, better copy of "b" moves its
    # score but not its first-seen position, so b stays ahead of c on the tie.
    merged = union_relevant(
        [[chunk("a", 0.6), chunk("b", 0.3)], [chunk("c", 0.6), chunk("b", 0.6)]]
    )
    assert [c.title for c in merged] == ["a", "b", "c"]


def test_caps_at_eight_like_the_typescript() -> None:
    per_query = [[chunk(f"q{q}-{i}", 0.9 - q * 0.1 - i * 0.01) for i in range(5)] for q in range(3)]

    merged = union_relevant(per_query)

    assert len(merged) == retrieval_config.UNION_CAP == 8
    assert [c.score for c in merged] == sorted(
        (c.score for q in per_query for c in q), reverse=True
    )[:8]


def test_nothing_in_nothing_out() -> None:
    assert union_relevant([]) == []
    assert union_relevant([[], []]) == []
