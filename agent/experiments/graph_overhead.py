"""What does LangGraph cost per request? The same pipeline as plain async code and as the graph.

    cd agent && uv run python experiments/graph_overhead.py

Every service is an instant fake (planner, two searches, a chat model streaming a 200-word
answer), so what is left is framework time: LangChain's callbacks in all three variants, and
LangGraph's scheduling, state channels and stream plumbing in the two graph variants.
  direct    plan_query, asyncio.gather over the searches, union, generation model astream
  invoke    graph.ainvoke
  stream    graph.astream(stream_mode=["updates", "messages"], version="v2"), fully consumed:
            what the HTTP route in step 2.4 will do
Free, no network. Prints, for each of 3 repeats (one repeat is not a measurement), the median
time per run and the median time to the first answer token. The second is the latency a user
feels; the first also includes per-token work that, on a real model streaming a token every
10 to 20 ms, overlaps with waiting for the next token. Compare both with a real request: about
1 s of retrieval, then the model.
"""

import asyncio
import itertools
import statistics
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from copilot_agent.generation import generation_messages
from copilot_agent.graph import build_graph
from copilot_agent.planner import plan_query
from copilot_agent.retrieval import RetrievalResult, RetrievedChunk, union_relevant

RUNS = 300
WARMUP = 20
REPEATS = 3
ANSWER = " ".join(["word"] * 200)
PLAN = {
    "intent": "search",
    "queries": [
        {"query": "How do I stream text?", "hypothetical": "h"},
        {"query": "What is generateText?", "hypothetical": "h"},
    ],
}
CHUNKS = [
    RetrievedChunk(content=f"chunk {i}", title="T", source_url=f"https://x/{i}", score=0.9 - i / 10)
    for i in range(5)
]


async def planner_reply(_: Any) -> dict[str, Any]:
    return {
        "raw": AIMessage(
            content="", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        ),
        "parsed": PLAN,
        "parsing_error": None,
    }


PLANNER = RunnableLambda(planner_reply)


async def search(query: str, embed_text: str | None = None) -> RetrievalResult:
    return RetrievalResult(candidates=[], relevant=CHUNKS, mode="reranked", timings_ms={})


FIRST_TOKEN: list[float] = []


def mark_first_token() -> None:
    if not FIRST_TOKEN:
        FIRST_TOKEN.append(time.perf_counter())


def fresh_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.cycle([ANSWER]))


async def direct(model: GenericFakeChatModel) -> str:
    plan = await plan_query("q", [], planner=PLANNER)
    results = await asyncio.gather(*(search(q.query, q.hypothetical) for q in plan.queries))
    relevant = union_relevant([r.relevant for r in results])
    messages = generation_messages(relevant, [], "q", [q.query for q in plan.queries])
    text = ""
    async for chunk in model.astream(messages):
        mark_first_token()
        text += chunk.text
    return text


async def main() -> None:
    model = fresh_model()
    graph = build_graph(planner=PLANNER, search=search, model=model)
    state = {"question": "q"}

    async def invoke() -> str:
        return (await graph.ainvoke(state))["answer"]

    async def stream() -> str:
        text = ""
        async for part in graph.astream(state, stream_mode=["updates", "messages"], version="v2"):
            if part["type"] == "messages":
                mark_first_token()
                text += part["data"][0].text
        return text

    variants: dict[str, Callable[[], Awaitable[str]]] = {
        "direct": lambda: direct(model),
        "invoke": invoke,
        "stream": stream,
    }
    for fn in variants.values():  # same answer from all three, or the comparison is void
        assert (await fn()) == ANSWER
    print(f"{RUNS} runs per repeat, {WARMUP} warm-up runs, median ms per run\n")
    print(f"{'':8}" + "".join(f"repeat {i + 1:<5}" for i in range(REPEATS)))
    medians: dict[str, list[float]] = {name: [] for name in variants}
    first: dict[str, list[float]] = {"direct": [], "stream": []}
    for _ in range(REPEATS):
        for name, fn in variants.items():
            for _ in range(WARMUP):
                await fn()
            times, ttfts = [], []
            for _ in range(RUNS):
                FIRST_TOKEN.clear()
                started = time.perf_counter()
                await fn()
                times.append((time.perf_counter() - started) * 1000)
                if FIRST_TOKEN:
                    ttfts.append((FIRST_TOKEN[0] - started) * 1000)
            medians[name].append(statistics.median(times))
            if ttfts:
                first[name].append(statistics.median(ttfts))
    for name, values in medians.items():
        print(f"{name:8}" + "".join(f"{v:<12.2f}" for v in values))
    print("\nmedian ms to the first answer token (what the user waits for):\n")
    for name, values in first.items():
        print(f"{name:8}" + "".join(f"{v:<12.2f}" for v in values))
    base = statistics.median(medians["direct"])
    print()
    for name in ("invoke", "stream"):
        print(f"{name} - direct: {statistics.median(medians[name]) - base:+.2f} ms per request")
    extra = statistics.median(first["stream"]) - statistics.median(first["direct"])
    print(f"stream - direct, to the first token: {extra:+.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
