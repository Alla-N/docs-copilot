"""ui_stream.py without HTTP: encoding, the source pills, and the graph-to-chunks adapter.

The adapter is fed hand-written graph stream parts (the shape graph.astream yields with
stream_mode=["updates", "messages"], version="v2"), so each rule can be pinned on its own. The
real graph behind the real HTTP route is in test_chat_api.py, and what the real AI SDK client
makes of the result is in the Vitest contract test.
"""

import asyncio
import json
import math
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from copilot_agent import ui_stream
from copilot_agent.graph import GenerationMetrics
from copilot_agent.planner import NO_USAGE, Plan, SubQuery
from copilot_agent.retrieval import RetrievedChunk

pytestmark = pytest.mark.anyio


def sign(text: str) -> str:
    return f"sig({text})"


def chunk(n: int, score: float, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"c{n}", title=f"Page {page}", source_url=f"https://x/{page}", score=score
    )


def plan(intent: str, *queries: str) -> dict[str, Any]:
    subs = tuple(SubQuery(query=q, hypothetical="") for q in queries)
    update = {"plan": Plan(intent=intent, queries=subs, usage=NO_USAGE)}  # type: ignore[arg-type]
    return {"type": "updates", "ns": (), "data": {"plan": update}}


def update(node: str, **values: Any) -> dict[str, Any]:
    return {"type": "updates", "ns": (), "data": {node: values}}


def token(text: str, node: str = "generate") -> dict[str, Any]:
    return {
        "type": "messages",
        "ns": (),
        "data": (AIMessageChunk(content=text), {"langgraph_node": node}),
    }


def generated(finish_reason: str = "stop") -> dict[str, Any]:
    metrics = GenerationMetrics(NO_USAGE, 1.0, 2.0, finish_reason)  # type: ignore[arg-type]
    return update("generate", answer="ignored", generation=metrics)


async def parts_of(*items: dict[str, Any], error: Exception | None = None) -> AsyncIterator[Any]:
    for item in items:
        yield item
    if error is not None:
        raise error


async def collect(parts: AsyncIterator[Any]) -> list[dict[str, Any]]:
    return [c async for c in ui_stream.ui_message_chunks(parts, message_id="m1", sign=sign)]


# ---- encoding ---------------------------------------------------------------------------------


def test_encode_is_compact_and_keeps_non_ascii() -> None:
    text = "caf" + chr(0xE9) + chr(0x2028)
    assert ui_stream.encode({"type": "text-delta", "id": "a", "delta": text}) == (
        '{"type":"text-delta","id":"a","delta":"' + text + '"}'
    )
    assert ui_stream.encode(ui_stream.DONE) == "[DONE]"


def test_encode_refuses_nan_rather_than_send_invalid_json() -> None:
    with pytest.raises(ValueError):
        ui_stream.encode(ui_stream.data("sources", [{"score": math.nan}]))


def test_message_ids_look_like_the_ai_sdk_ones() -> None:
    ids = {ui_stream.new_message_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(i) == 16 and i.isascii() and i.isalnum() for i in ids)


def test_no_chunk_constructor_writes_a_null() -> None:
    # uiMessageChunkSchema is strict: null in an optional field fails the whole stream.
    chunks = [
        ui_stream.start("m"),
        ui_stream.start_step(),
        ui_stream.text_start("a"),
        ui_stream.text_delta("a", ""),
        ui_stream.text_end("a"),
        ui_stream.finish_step(),
        ui_stream.finish(),
        ui_stream.finish("stop"),
        ui_stream.data("x", {}),
        ui_stream.error("e"),
    ]
    assert all(None not in c.values() for c in chunks)
    assert ui_stream.finish() == {"type": "finish"}


# ---- source pills: toSourcePills --------------------------------------------------------------


def test_pills_are_one_per_page_with_chunk_numbers_and_best_score() -> None:
    relevant = [chunk(1, 0.9, 1), chunk(2, 0.8, 2), chunk(3, 0.95, 1), chunk(4, 0.4, 3)]
    assert ui_stream.source_pills(relevant) == [
        {"id": 1, "title": "Page 1", "url": "https://x/1", "score": 0.95, "chunks": [1, 3]},
        {"id": 2, "title": "Page 2", "url": "https://x/2", "score": 0.8, "chunks": [2]},
        {"id": 3, "title": "Page 3", "url": "https://x/3", "score": 0.4, "chunks": [4]},
    ]
    assert ui_stream.source_pills([]) == []


# ---- the adapter ------------------------------------------------------------------------------


async def test_canned_reply_is_the_route_sequence() -> None:
    chunks = await collect(
        parts_of(plan("greeting"), update("canned", answer="Hi!", relevant=[], mode="skipped"))
    )
    assert chunks == [
        {"type": "start", "messageId": "m1"},
        {"type": "data-retrieval", "data": {"mode": "skipped", "intent": "greeting"}},
        {"type": "text-start", "id": "canned"},
        {"type": "text-delta", "id": "canned", "delta": "Hi!"},
        {"type": "text-end", "id": "canned"},
        {"type": "data-signature", "data": {"sig": "sig(Hi!)"}},
        {"type": "finish"},
    ]


async def test_answer_is_the_route_sequence() -> None:
    relevant = [chunk(1, 0.9, 1)]
    chunks = await collect(
        parts_of(
            plan("search", "q"),
            update("retrieve", retrievals=[]),
            update("merge", relevant=relevant, mode="reranked", rerank_calls=1),
            token(""),  # the stream's opening chunk carries no text
            token("Use "),
            token("streamText."),
            generated("length"),
        )
    )
    assert chunks == [
        {"type": "start", "messageId": "m1"},
        {"type": "data-retrieval", "data": {"mode": "reranked", "intent": "search"}},
        {"type": "data-sources", "data": ui_stream.source_pills(relevant)},
        {"type": "start-step"},
        {"type": "text-start", "id": "answer"},
        {"type": "text-delta", "id": "answer", "delta": "Use "},
        {"type": "text-delta", "id": "answer", "delta": "streamText."},
        {"type": "text-end", "id": "answer"},
        {"type": "finish-step"},
        {"type": "data-signature", "data": {"sig": "sig(Use streamText.)"}},
        {"type": "finish", "finishReason": "length"},
    ]


async def test_no_sources_part_when_nothing_was_kept() -> None:
    chunks = await collect(
        parts_of(
            plan("search", "q"),
            update("merge", relevant=[], mode="reranked", rerank_calls=1),
            token("No."),
            generated(),
        )
    )
    assert "data-sources" not in [c["type"] for c in chunks]


async def test_an_empty_answer_still_gets_its_text_part() -> None:
    chunks = await collect(
        parts_of(
            plan("search", "q"),
            update("merge", relevant=[], mode="reranked", rerank_calls=1),
            generated(),
        )
    )
    types = [c["type"] for c in chunks]
    assert types[-6:] == [
        "start-step",
        "text-start",
        "text-end",
        "finish-step",
        "data-signature",
        "finish",
    ]
    assert chunks[-2]["data"] == {"sig": "sig()"}


async def test_only_the_answer_node_streams() -> None:
    chunks = await collect(
        parts_of(
            plan("search", "q"),
            token('{"intent":', node="plan"),
            update("merge", relevant=[], mode="reranked", rerank_calls=1),
            token("Answer."),
            generated(),
        )
    )
    deltas = [c["delta"] for c in chunks if c["type"] == "text-delta"]
    assert deltas == ["Answer."]


async def test_nothing_is_sent_before_retrieval_is_done() -> None:
    # `start` waits for the first data part, so a failure before it opens no message.
    chunks = await collect(
        parts_of(plan("search", "q"), update("retrieve", retrievals=[]), error=OSError("db"))
    )
    assert chunks == [{"type": "error", "errorText": "Stream failed"}]


async def test_a_failure_mid_answer_keeps_what_streamed_and_ends_with_one_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunks = await collect(
        parts_of(
            plan("search", "q"),
            update("merge", relevant=[], mode="reranked", rerank_calls=1),
            token("Use "),
            error=RuntimeError("upstream said: sk-secret"),
        )
    )
    assert [c["type"] for c in chunks][-4:] == ["start-step", "text-start", "text-delta", "error"]
    assert chunks[-1] == {"type": "error", "errorText": "Stream failed"}
    assert "sk-secret" not in json.dumps(chunks)
    assert "sk-secret" in caplog.text  # the reason is in the log


async def test_cancellation_is_not_turned_into_an_error_chunk() -> None:
    with pytest.raises(asyncio.CancelledError):
        await collect(parts_of(plan("search", "q"), error=asyncio.CancelledError()))  # type: ignore[arg-type]


async def test_closing_the_adapter_closes_the_graph_stream() -> None:
    closed = asyncio.Event()

    async def parts() -> AsyncIterator[Any]:
        try:
            yield plan("search", "q")
            yield update("merge", relevant=[], mode="reranked", rerank_calls=1)
            await asyncio.Event().wait()  # never set
        finally:
            closed.set()

    chunks = ui_stream.ui_message_chunks(parts(), message_id="m1", sign=sign)
    assert (await anext(chunks))["type"] == "start"
    await chunks.aclose()
    assert closed.is_set()
