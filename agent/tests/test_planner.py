"""plan_query: prompt building, reply normalisation and the fallback, with a fake planner.

No network. The fake stands in for the structured-output Runnable (build_planner's result): it
records the messages it was given and returns what with_structured_output(include_raw=True)
returns, {"raw": AIMessage, "parsed": dict | None, "parsing_error": Exception | None}. The real
ChatOpenAI wiring is tested against the TypeScript request in test_planner_request_parity.py.
"""

import asyncio
import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from copilot_agent.planner import (
    MAX_SUB_QUERIES,
    NO_USAGE,
    PLAN_JSON_SCHEMA,
    SYSTEM_PROMPT,
    HistoryTurn,
    PlanOutput,
    SubQuery,
    TokenUsage,
    _js_trim,
    plan_query,
    planner_messages,
)

pytestmark = pytest.mark.anyio

USAGE = {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280}


class FakePlanner:
    """Returns one scripted reply (or raises one exception) and records every call."""

    def __init__(self, parsed: Any = None, *, error: BaseException | None = None) -> None:
        self.parsed = parsed
        self.error = error
        self.calls: list[list[BaseMessage]] = []
        self.runnable = RunnableLambda(self._reply)

    async def _reply(self, messages: list[BaseMessage]) -> dict[str, Any]:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return {
            "raw": AIMessage(content="", usage_metadata=USAGE),
            "parsed": self.parsed,
            "parsing_error": None,
        }


def search(*queries: tuple[str, str]) -> dict[str, Any]:
    return {"intent": "search", "queries": [{"query": q, "hypothetical": h} for q, h in queries]}


# ---- the request side: what the model is sent -----------------------------------------------


def user_text(messages: list[BaseMessage]) -> str:
    content = messages[1].content
    assert isinstance(content, list) and len(content) == 1
    return content[0]["text"]


def test_messages_without_history_are_the_system_prompt_and_the_question() -> None:
    messages = planner_messages("How do I stream text?")
    assert [m.type for m in messages] == ["system", "human"]
    assert messages[0].content == SYSTEM_PROMPT
    assert user_text(messages) == "How do I stream text?"


def test_history_is_folded_into_the_user_turn_and_cut_to_the_last_four() -> None:
    history = [HistoryTurn("user" if i % 2 == 0 else "assistant", f"turn {i}") for i in range(6)]
    text = user_text(planner_messages("And the other one?", history))
    assert text == (
        "CONVERSATION SO FAR:\n"
        "user: turn 2\nassistant: turn 3\nuser: turn 4\nassistant: turn 5\n\n"
        "LATEST USER MESSAGE:\nAnd the other one?"
    )


# ---- the reply side: normalisation ------------------------------------------------------------


async def test_search_plan_is_trimmed_and_keeps_the_usage() -> None:
    fake = FakePlanner(search(("  How do I stream text?\n", " A streamText call ... ")))
    plan = await plan_query("stream?", planner=fake.runnable)
    assert plan.intent == "search"
    assert plan.queries == (
        SubQuery(query="How do I stream text?", hypothetical="A streamText call ..."),
    )
    assert plan.usage == TokenUsage(input_tokens=1200, output_tokens=80)
    assert len(fake.calls) == 1


async def test_empty_queries_are_dropped_and_the_rest_capped_at_four() -> None:
    fake = FakePlanner(search(("   ", "h0"), *((f"q{i}", f"h{i}") for i in range(1, 7))))
    plan = await plan_query("many", planner=fake.runnable)
    assert [q.query for q in plan.queries] == ["q1", "q2", "q3", "q4"]
    assert len(plan.queries) == MAX_SUB_QUERIES


async def test_search_with_no_usable_query_searches_the_raw_question_without_hyde() -> None:
    fake = FakePlanner(search((" ", "a hypothetical for nothing")))
    plan = await plan_query("What is SDK?", planner=fake.runnable)
    assert plan.intent == "search"
    assert plan.queries == (SubQuery(query="What is SDK?", hypothetical=""),)
    # The call succeeded, so what it cost is still logged (TypeScript does the same).
    assert plan.usage == TokenUsage(input_tokens=1200, output_tokens=80)


async def test_greeting_has_no_queries_even_when_the_model_wrote_one() -> None:
    fake = FakePlanner(search(("What can the assistant do?", "The assistant answers ...")))
    fake.parsed["intent"] = "greeting"
    plan = await plan_query("hi", planner=fake.runnable)
    assert (plan.intent, plan.queries) == ("greeting", ())
    assert plan.usage.input_tokens == 1200


async def test_off_topic_is_trusted_even_when_the_model_also_wrote_queries() -> None:
    # The queries are exactly the SDK-shaped rewrite of an off-topic message that invariant 7
    # forbids; the intent is the decision.
    fake = FakePlanner(
        {
            "intent": "off-topic",
            "queries": [{"query": "deploy the Vercel AI SDK to AWS", "hypothetical": "..."}],
        }
    )
    plan = await plan_query("how do I deploy to AWS", planner=fake.runnable)
    assert (plan.intent, plan.queries) == ("off-topic", ())


# ---- failures: always the fallback, never an exception ------------------------------------------


def assert_fallback(plan: Any, question: str) -> None:
    assert plan.intent == "search"
    assert plan.queries == (SubQuery(query=question, hypothetical=""),)
    assert plan.usage == NO_USAGE


async def test_a_failed_call_falls_back_to_the_raw_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakePlanner(error=ConnectionError("network down"))
    with caplog.at_level(logging.WARNING, logger="copilot_agent.planner"):
        plan = await plan_query("How do I stream text?", planner=fake.runnable)
    assert_fallback(plan, "How do I stream text?")
    assert "falling back" in caplog.text


async def test_a_reply_that_did_not_parse_falls_back() -> None:
    async def unparsable(_: list[BaseMessage]) -> dict[str, Any]:
        return {
            "raw": AIMessage(content="not json", usage_metadata=USAGE),
            "parsed": None,
            "parsing_error": ValueError("Invalid json output"),
        }

    plan = await plan_query("q", planner=RunnableLambda(unparsable))
    assert_fallback(plan, "q")


@pytest.mark.parametrize(
    "parsed",
    [
        pytest.param({"intent": "banana", "queries": []}, id="unknown-intent"),
        pytest.param({"intent": "search"}, id="missing-queries"),
        pytest.param({"intent": "search", "queries": [{"query": "q"}]}, id="missing-hypothetical"),
        pytest.param(search(("q", "h")) | {"extra": 1}, id="extra-field"),
    ],
)
async def test_a_reply_of_the_wrong_shape_falls_back(parsed: dict[str, Any]) -> None:
    plan = await plan_query("q", planner=FakePlanner(parsed).runnable)
    assert_fallback(plan, "q")


async def test_cancellation_is_not_a_failure() -> None:
    # The TypeScript planner gets the request's abort signal; here the task is cancelled. A
    # fallback would go on to spend an embedding and a rerank on a closed tab.
    fake = FakePlanner(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await plan_query("q", planner=fake.runnable)


# ---- small parts ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "trimmed"),
    [
        pytest.param("  q \n", "q", id="ascii-whitespace"),
        pytest.param("\ufeffq\u00a0", "q", id="bom-and-nbsp-trimmed-like-js"),
        pytest.param("q\x1f", "q\x1f", id="unit-separator-kept-like-js"),
    ],
)
def test_trim_matches_javascript(raw: str, trimmed: str) -> None:
    assert _js_trim(raw) == trimmed


def test_the_validation_model_matches_the_wire_schema() -> None:
    """PLAN_JSON_SCHEMA is what the model is told; PlanOutput is what the reply is checked
    against. They are written separately (see build_planner), so this keeps them in step."""
    ours = PlanOutput.model_json_schema()
    sub = ours["$defs"]["SubQuery"]
    wire = PLAN_JSON_SCHEMA
    wire_sub = wire["properties"]["queries"]["items"]

    assert ours["properties"].keys() == wire["properties"].keys()
    assert ours["required"] == wire["required"]
    assert ours["properties"]["intent"]["enum"] == wire["properties"]["intent"]["enum"]
    assert sub["properties"].keys() == wire_sub["properties"].keys()
    assert sub["required"] == wire_sub["required"]
    assert ours["additionalProperties"] is wire["additionalProperties"] is False
    assert sub["additionalProperties"] is wire_sub["additionalProperties"] is False
