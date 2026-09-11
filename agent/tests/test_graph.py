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
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ConfigDict, Field

from copilot_agent.checkpoint import serializer
from copilot_agent.generation import REFUSAL_MESSAGE
from copilot_agent.graph import SubQueryRetrieval, build_graph, merge_retrievals
from copilot_agent.history import MAX_STORED_TURNS
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


async def run(graph: Any, question: str) -> dict[str, Any]:
    return await graph.ainvoke({"question": question})


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
    assert state["generation"] is None


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
    planner, _ = planner_replying(search_plan(("What is A?", "h"), ("What is B?", "h")))
    search = FakeSearch({"What is A?": result(chunk(1, 0.9)), "What is B?": result(chunk(2, 0.7))})
    chat = model(ANSWER)
    state = await run(build_graph(planner=planner, search=search, model=chat), "A and B?")

    assert state["answer"] == ANSWER
    [messages] = chat.seen
    assert [m.type for m in messages] == ["system", "human"]
    assert "[Source 1] (relevance: 0.90)\nchunk 1" in messages[0].content
    assert "[Source 2] (relevance: 0.70)\nchunk 2" in messages[0].content
    assert messages[-1].content == [{"type": "text", "text": "What is A?\nWhat is B?"}]


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
    return [SubQueryRetrieval.of(f"q{i}", r) for i, r in enumerate(results)]


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
        {"question": "q"}, stream_mode=["updates", "messages"], version="v2"
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


# ---- a conversation: the thread's turns, kept by the checkpointer --------------------------------
#
# InMemorySaver is LangGraph's in-process checkpointer: the same saving and loading as the
# Postgres one (serialisation included), without a database. Each test below runs several turns
# on one thread, the way a conversation arrives at /chat.

DURABILITIES = ["sync", "async", "exit"]


def planner_script(*parsed: dict[str, Any]) -> tuple[Any, list[list[BaseMessage]]]:
    """A planner that returns the given plans, one per turn."""
    plans = iter(parsed)
    calls: list[list[BaseMessage]] = []

    async def reply(messages: list[BaseMessage]) -> dict[str, Any]:
        calls.append(messages)
        return {"raw": AIMessage(content="", usage_metadata=USAGE), "parsed": next(plans)}

    return RunnableLambda(reply), calls


class TurnModel(RecordingFakeChatModel):
    """Answers turn after turn; can fail on one call, or hang on one until cancelled."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fail_on_call: int | None = None
    hang_on_call: int | None = None
    hanging: asyncio.Event = Field(default_factory=asyncio.Event)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        call = len(self.seen)
        if call in (self.fail_on_call, self.hang_on_call):
            self.seen.append(messages)
            if call == self.fail_on_call:
                raise RuntimeError("model down")
            self.hanging.set()
            await asyncio.Event().wait()  # never set
        async for piece in super()._astream(messages, stop, run_manager, **kwargs):
            yield piece


def thread(thread_id: str = "thread-1") -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


async def turn(
    graph: Any, question: str, thread_id: str = "thread-1", durability: str = "async"
) -> dict[str, Any]:
    return await graph.ainvoke({"question": question}, thread(thread_id), durability=durability)


async def stored_turns(graph: Any, thread_id: str = "thread-1") -> list[HistoryTurn]:
    return (await graph.aget_state(thread(thread_id))).values.get("turns", [])


def conversation(*turns: tuple[str, str]) -> list[HistoryTurn]:
    return [t for q, a in turns for t in (HistoryTurn("user", q), HistoryTurn("assistant", a))]


async def test_the_next_turn_reads_the_last_one_from_the_thread() -> None:
    planner, planner_calls = planner_script(
        search_plan(("stream text", "h")), search_plan(("x", "h"))
    )
    search = FakeSearch({"stream text": result(chunk(1, 0.9)), "x": result(chunk(2, 0.8))})
    chat = TurnModel(messages=iter(["Use streamText.", "Set options."]))
    graph = build_graph(planner=planner, search=search, model=chat, checkpointer=InMemorySaver())

    await turn(graph, "how do I stream text")
    await turn(graph, "and configure it?")

    # The first turn had no history; the second reads the first, from the thread.
    assert [m.type for m in chat.seen[0]] == ["system", "human"]
    assert [m.type for m in chat.seen[1]] == ["system", "human", "ai", "human"]
    assert chat.seen[1][1].content == [{"type": "text", "text": "how do I stream text"}]
    assert chat.seen[1][2].content == [{"type": "text", "text": "Use streamText."}]
    prompt = planner_calls[1][1].content[0]["text"]
    assert "user: how do I stream text\nassistant: Use streamText." in prompt
    assert await stored_turns(graph) == conversation(
        ("how do I stream text", "Use streamText."), ("and configure it?", "Set options.")
    )


async def test_a_turn_is_answered_from_its_own_retrievals_only() -> None:
    # The bug a checkpointer brings to an append reducer, reproduced before the fix: the second
    # answer was grounded on the first turn's chunk as well, and showed it as a source.
    planner, _ = planner_script(
        search_plan(("stream text", "h")), search_plan(("define a tool", "h"))
    )
    search = FakeSearch(
        {"stream text": result(chunk(1, 0.9)), "define a tool": result(chunk(2, 0.8))}
    )
    chat = TurnModel(messages=iter(["one", "two"]))
    graph = build_graph(planner=planner, search=search, model=chat, checkpointer=InMemorySaver())

    await turn(graph, "how do I stream text")
    state = await turn(graph, "how do I define a tool")

    assert [r.query for r in state["retrievals"]] == ["define a tool"]
    assert state["relevant"] == [chunk(2, 0.8)]
    context = chat.seen[1][0].content
    assert "chunk 2" in context and "chunk 1" not in context


async def test_a_canned_turn_is_recorded_and_reports_no_generation() -> None:
    planner, _ = planner_script(search_plan(("q", "h")), {"intent": "greeting", "queries": []})
    search = FakeSearch({"q": result(chunk(1, 0.9))})
    graph = build_graph(
        planner=planner,
        search=search,
        model=TurnModel(messages=iter(["answer"])),
        checkpointer=InMemorySaver(),
    )
    await turn(graph, "how do I stream text")
    state = await turn(graph, "thanks, hi")

    # Not the last answer's metrics: this turn called no model.
    assert state["generation"] is None
    assert (state["relevant"], state["mode"]) == ([], "skipped")
    assert await stored_turns(graph) == conversation(
        ("how do I stream text", "answer"), ("thanks, hi", GREETING_MESSAGE)
    )


@pytest.mark.parametrize("durability", DURABILITIES)
async def test_a_failed_answer_leaves_the_thread_as_it_was(durability: str) -> None:
    planner, _ = planner_script(*(search_plan(("q", "h")) for _ in range(3)))
    search = FakeSearch({"q": result(chunk(1, 0.9))})
    chat = TurnModel(messages=iter(["first", "third"]), fail_on_call=1)
    graph = build_graph(planner=planner, search=search, model=chat, checkpointer=InMemorySaver())

    await turn(graph, "one", durability=durability)
    with pytest.raises(RuntimeError, match="model down"):
        await turn(graph, "two", durability=durability)
    assert await stored_turns(graph) == conversation(("one", "first"))

    # The next turn reads the conversation without the failed one: no question without its answer.
    await turn(graph, "three", durability=durability)
    assert [m.content[0]["text"] for m in chat.seen[2][1:]] == ["one", "first", "q"]
    assert await stored_turns(graph) == conversation(("one", "first"), ("three", "third"))


@pytest.mark.parametrize("durability", DURABILITIES)
async def test_a_cancelled_answer_leaves_the_thread_as_it_was(durability: str) -> None:
    # Stop in the UI, or a closed tab: the run is cancelled while the model streams.
    planner, _ = planner_script(*(search_plan(("q", "h")) for _ in range(2)))
    search = FakeSearch({"q": result(chunk(1, 0.9))})
    chat = TurnModel(messages=iter(["first"]), hang_on_call=1)
    graph = build_graph(planner=planner, search=search, model=chat, checkpointer=InMemorySaver())

    await turn(graph, "one", durability=durability)
    task = asyncio.create_task(turn(graph, "two", durability=durability))
    await chat.hanging.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await stored_turns(graph) == conversation(("one", "first"))


async def test_threads_do_not_share_turns() -> None:
    planner, planner_calls = planner_script(*(search_plan(("q", "h")) for _ in range(2)))
    search = FakeSearch({"q": result(chunk(1, 0.9))})
    chat = TurnModel(messages=iter(["for A", "for B"]))
    graph = build_graph(planner=planner, search=search, model=chat, checkpointer=InMemorySaver())

    await turn(graph, "question A", thread_id="thread-A")
    await turn(graph, "question B", thread_id="thread-B")

    assert [m.type for m in chat.seen[1]] == ["system", "human"]
    assert "question A" not in planner_calls[1][1].content[0]["text"]
    assert await stored_turns(graph, "thread-B") == conversation(("question B", "for B"))


async def test_the_thread_keeps_only_what_a_request_can_read() -> None:
    n = 12  # 24 messages, more than MAX_STORED_TURNS
    planner, _ = planner_script(*({"intent": "greeting", "queries": []} for _ in range(n)))
    graph = build_graph(
        planner=planner, search=FakeSearch({}), model=model(), checkpointer=InMemorySaver()
    )
    for i in range(n):
        await turn(graph, f"hi {i}")
    stored = await stored_turns(graph)
    assert len(stored) == MAX_STORED_TURNS
    assert stored[-2:] == conversation((f"hi {n - 1}", GREETING_MESSAGE))


async def test_what_a_turn_saves_does_not_carry_the_candidates() -> None:
    # 100 candidates of ~1 KB per sub-query are what the reranker reads, and nothing after it.
    # Saved, they were 213 of the 216 KiB a turn wrote (experiments/checkpoint_overhead.py).
    big = [
        # Distinct texts: pickle (the cloud stand-in's serializer) stores a repeated one once.
        Candidate(
            content=f"{i:04}" + "x" * 996, title="T", source_url=f"https://x/{i}", similarity=0.5
        )
        for i in range(100)
    ]
    kept = [chunk(1, 0.9)]
    found = RetrievalResult(candidates=big, relevant=kept, mode="reranked", timings_ms={})
    planner, _ = planner_script(search_plan(("q0", "h"), ("q1", "h")))
    saver = InMemorySaver(serde=serializer())
    graph = build_graph(
        planner=planner,
        search=FakeSearch({"q0": found, "q1": found}),
        model=TurnModel(messages=iter(["answer"])),
        checkpointer=saver,
    )
    state = await turn(graph, "two things")

    assert state["rerank_calls"] == 2  # still counted, from what was kept
    saved = (await saver.aget_tuple(thread())).checkpoint["channel_values"]
    size = sum(len(serializer().dumps_typed(v)[1]) for v in saved.values())
    assert size < 20_000, f"a turn saved {size} bytes"
