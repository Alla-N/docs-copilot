"""The chat graph with fake services: routing, fan-out, merge, generation, stream events.

No network. The planner is a fake Runnable (as in test_planner.py), the search is a recorder
that returns scripted results, and the chat model is LangChain's GenericFakeChatModel, which
streams its reply token by token through the real callbacks, so LangGraph's "messages" mode sees
it exactly as it would see ChatOpenAI. test_graph_wired.py runs the real ChatOpenAI instead.
"""

import asyncio
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from copilot_agent.generation import REFUSAL_MESSAGE
from copilot_agent.graph import SubQueryRetrieval, build_graph, merge_retrievals
from copilot_agent.planner import GREETING_MESSAGE, HistoryTurn, TokenUsage
from copilot_agent.retrieval import Candidate, RetrievalResult, RetrievedChunk, union_relevant

pytestmark = pytest.mark.anyio

ANSWER = "Use streamText to stream text (Source 1)."
USAGE = {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280}


def planner_replying(parsed: dict[str, Any] | None = None, error: Exception | None = None):
    calls: list[list[BaseMessage]] = []

    async def reply(messages: list[BaseMessage]) -> dict[str, Any]:
        calls.append(messages)
        if error is not None:
            raise error
        return {
            "raw": AIMessage(content="", usage_metadata=USAGE),
            "parsed": parsed,
            "parsing_error": None,
        }

    return RunnableLambda(reply), calls


def search_plan(*queries: tuple[str, str]) -> dict[str, Any]:
    return {"intent": "search", "queries": [{"query": q, "hypothetical": h} for q, h in queries]}


def chunk(n: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"chunk {n}",
        title=f"Page {n}",
        source_url=f"https://ai-sdk.dev/docs/{n}",
        score=score,
    )


def result(
    *chunks: RetrievedChunk, mode: str = "reranked", candidates: int = 100
) -> RetrievalResult:
    cands = [
        Candidate(content=f"c{i}", title="T", source_url="https://x", similarity=0.5)
        for i in range(candidates)
    ]
    return RetrievalResult(candidates=cands, relevant=list(chunks), mode=mode, timings_ms={})


class FakeSearch:
    """Records (query, embed_text) calls; answers from a script, optionally after a delay."""

    def __init__(self, script: dict[str, RetrievalResult], delays: dict[str, float] | None = None):
        self.script = script
        self.delays = delays or {}
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(self, query: str, embed_text: str | None = None) -> RetrievalResult:
        self.calls.append((query, embed_text))
        await asyncio.sleep(self.delays.get(query, 0))
        return self.script[query]


class RecordingFakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel that also keeps the messages it was called with."""

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            yield chunk


def model(*replies: str) -> RecordingFakeChatModel:
    return RecordingFakeChatModel(messages=iter(replies))


async def run(
    graph: Any, question: str, history: list[HistoryTurn] | None = None
) -> dict[str, Any]:
    return await graph.ainvoke({"question": question, "history": history or []})


# ---- the gate: greeting and off-topic cost one planner call and nothing else ----


@pytest.mark.parametrize(
    ("intent", "reply"),
    [("greeting", GREETING_MESSAGE), ("off-topic", REFUSAL_MESSAGE)],
)
async def test_canned_replies_skip_retrieval_and_generation(intent: str, reply: str) -> None:
    planner, planner_calls = planner_replying({"intent": intent, "queries": []})
    search = FakeSearch({})
    chat = model()  # no replies: a model call would raise StopIteration
    state = await run(build_graph(planner=planner, search=search, model=chat), "hi")
    assert state["answer"] == reply
    assert (state["mode"], state["relevant"], state["rerank_calls"]) == ("skipped", [], 0)
    assert len(planner_calls) == 1
    assert search.calls == []
    assert chat.seen == []
    assert "generation" not in state


# ---- the search path ----------------------------------------------------------------------------


async def test_every_sub_query_is_retrieved_with_its_hypothetical() -> None:
    planner, _ = planner_replying(search_plan(("q0", "h0"), ("q1", ""), ("q2", "h2")))
    search = FakeSearch({"q0": result(chunk(1, 0.9)), "q1": result(), "q2": result()})
    await run(build_graph(planner=planner, search=search, model=model(ANSWER)), "three things")
    # An empty hypothetical embeds the query itself: q.hypothetical || q.query in lib/plan.ts.
    assert sorted(search.calls) == [("q0", "h0"), ("q1", "q1"), ("q2", "h2")]


async def test_retrievals_run_in_parallel() -> None:
    planner, _ = planner_replying(search_plan(*((f"q{i}", "h") for i in range(4))))
    search = FakeSearch({f"q{i}": result() for i in range(4)}, {f"q{i}": 0.2 for i in range(4)})
    graph = build_graph(planner=planner, search=search, model=model(ANSWER))
    started = asyncio.get_running_loop().time()
    await run(graph, "four things")
    # Four 0.2 s searches one after another would take 0.8 s.
    assert asyncio.get_running_loop().time() - started < 0.5


async def test_the_union_follows_plan_order_not_finishing_order() -> None:
    # Equal scores keep first-seen order in the union, so the order the sub-queries are merged
    # in is visible. q0 finishes last here; its chunk must still come first.
    tie_a, tie_b = chunk(1, 0.8), chunk(2, 0.8)
    planner, _ = planner_replying(search_plan(("q0", "h"), ("q1", "h")))
    search = FakeSearch({"q0": result(tie_a), "q1": result(tie_b)}, {"q0": 0.1})
    state = await run(build_graph(planner=planner, search=search, model=model(ANSWER)), "two")
    assert state["relevant"] == [tie_a, tie_b] == union_relevant([[tie_a], [tie_b]])


async def test_answer_is_generated_for_the_resolved_sub_queries_with_the_context() -> None:
    planner, planner_calls = planner_replying(search_plan(("What is A?", "h"), ("What is B?", "h")))
    search = FakeSearch({"What is A?": result(chunk(1, 0.9)), "What is B?": result(chunk(2, 0.7))})
    chat = model(ANSWER)
    history = [HistoryTurn("user", "earlier"), HistoryTurn("assistant", "reply")]
    state = await run(build_graph(planner=planner, search=search, model=chat), "A and B?", history)

    assert state["answer"] == ANSWER
    [messages] = chat.seen
    assert [m.type for m in messages] == ["system", "human", "ai", "human"]
    assert "[Source 1] (relevance: 0.90)\nchunk 1" in messages[0].content
    assert "[Source 2] (relevance: 0.70)\nchunk 2" in messages[0].content
    assert messages[-1].content == [{"type": "text", "text": "What is A?\nWhat is B?"}]
    # The planner saw the same history, folded into its prompt.
    assert "user: earlier\nassistant: reply" in planner_calls[0][1].content[0]["text"]


async def test_generation_metrics_are_recorded() -> None:
    planner, _ = planner_replying(search_plan(("q", "h")))
    search = FakeSearch({"q": result(chunk(1, 0.9))})
    state = await run(build_graph(planner=planner, search=search, model=model(ANSWER)), "q")
    metrics = state["generation"]
    assert metrics.ttft_ms is not None and 0 <= metrics.ttft_ms <= metrics.generation_ms
    # GenericFakeChatModel reports no usage; ChatOpenAI's is checked in test_graph_wired.py.
    assert metrics.usage == TokenUsage(None, None)


async def test_a_planner_failure_still_searches_the_raw_question() -> None:
    planner, _ = planner_replying(error=ConnectionError("down"))
    search = FakeSearch({"What is SDK?": result(chunk(1, 0.9))})
    chat = model(ANSWER)
    state = await run(build_graph(planner=planner, search=search, model=chat), "What is SDK?")
    assert search.calls == [("What is SDK?", "What is SDK?")]
    assert state["answer"] == ANSWER
    assert chat.seen[0][-1].content == [{"type": "text", "text": "What is SDK?"}]


# ---- merge: mode and rerank count, as plannedRetrieve computes them ----


def retrievals(*results: RetrievalResult) -> list[SubQueryRetrieval]:
    return [SubQueryRetrieval(f"q{i}", r) for i, r in enumerate(results)]


def test_one_fallback_makes_the_whole_answer_a_fallback() -> None:
    merged = merge_retrievals(retrievals(result(), result(mode="cosine-fallback")))
    assert merged["mode"] == "cosine-fallback"


def test_rerank_calls_count_only_reranked_searches_with_candidates() -> None:
    merged = merge_retrievals(
        retrievals(result(), result(candidates=0), result(mode="cosine-fallback"), result())
    )
    assert merged["rerank_calls"] == 2
    assert merged["mode"] == "cosine-fallback"


# ---- what streams out ---------------------------------------------------------------------------


async def test_stream_events_arrive_in_pipeline_order() -> None:
    planner, _ = planner_replying(search_plan(("q0", "h"), ("q1", "h")))
    search = FakeSearch({"q0": result(chunk(1, 0.9)), "q1": result(chunk(2, 0.8))})
    graph = build_graph(planner=planner, search=search, model=model(ANSWER))

    events: list[tuple[str, str]] = []
    tokens: list[str] = []
    async for part in graph.astream(
        {"question": "q", "history": []}, stream_mode=["updates", "messages"], version="v2"
    ):
        if part["type"] == "updates":
            events += [("updates", node) for node in part["data"]]
        else:
            message, metadata = part["data"]
            events.append(("messages", metadata["langgraph_node"]))
            tokens.append(message.text)

    updates = [node for kind, node in events if kind == "updates"]
    assert updates == ["plan", "retrieve", "retrieve", "merge", "generate"]
    first_token = events.index(("messages", "generate"))
    assert events.index(("updates", "merge")) < first_token < events.index(("updates", "generate"))
    # Only the answer streams, token by token, each token once.
    assert {node for kind, node in events if kind == "messages"} == {"generate"}
    assert len(tokens) > 1 and "".join(tokens) == ANSWER


# ---- cancellation: the client went away ---------------------------------------------------------


async def test_cancelling_a_run_cancels_its_searches() -> None:
    started = asyncio.Event()
    cancelled: list[str] = []

    async def hanging_search(query: str, embed_text: str | None = None) -> RetrievalResult:
        started.set()
        try:
            await asyncio.Event().wait()  # never set
        except asyncio.CancelledError:
            cancelled.append(query)
            raise
        raise AssertionError("unreachable")

    planner, _ = planner_replying(search_plan(("q0", "h"), ("q1", "h")))
    graph = build_graph(planner=planner, search=hanging_search, model=model())
    task = asyncio.create_task(run(graph, "q"))
    await started.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(cancelled) == ["q0", "q1"]
