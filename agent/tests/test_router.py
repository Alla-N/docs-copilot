"""The router (step 3.5) with a fake Runnable: what it returns, and what it does when it fails.

No network. The same shape as test_planner.py, because the router is deliberately the planner's
call with a different prompt: one system message, one user turn with the recent history folded
into its text, strict structured output, and a failure that degrades instead of raising.
"""

import asyncio
from typing import Any, get_args

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from copilot_agent.planner import HISTORY_TURNS, NO_USAGE, HistoryTurn, TokenUsage
from copilot_agent.router import (
    ROUTE_JSON_SCHEMA,
    SYSTEM_PROMPT,
    Route,
    RouterOutput,
    Routing,
    route_question,
    router_messages,
)

pytestmark = pytest.mark.anyio

USAGE = {"input_tokens": 300, "output_tokens": 4, "total_tokens": 304}


def router_replying(parsed: dict[str, Any] | None = None, error: Exception | None = None):
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


# ---- the schema the model reads and the schema the reply is checked against --------------------


def test_the_json_schema_and_the_parsed_model_allow_the_same_routes() -> None:
    # Two declarations of one fact: the enum the model is shown, and the Literal the reply is
    # validated against. test_planner.py pins the planner's pair for the same reason -- they
    # drift apart silently, and the symptom is a route that validates and was never offered.
    assert set(ROUTE_JSON_SCHEMA["properties"]["route"]["enum"]) == set(get_args(Route))
    assert ROUTE_JSON_SCHEMA["required"] == ["route"]
    assert ROUTE_JSON_SCHEMA["additionalProperties"] is False


# ---- what it returns -------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["docs", "github", "both"])
async def test_each_route_comes_back_with_the_call_s_tokens(route: str) -> None:
    router, _ = router_replying({"route": route})
    routing = await route_question("when was v7 released", router=router)
    assert routing.route == route
    assert routing.usage == TokenUsage(input_tokens=300, output_tokens=4)


@pytest.mark.parametrize(
    ("route", "docs", "github"),
    [("docs", True, False), ("github", False, True), ("both", True, True)],
)
def test_wants_docs_and_wants_github_cover_the_three_routes(
    route: Route, docs: bool, github: bool
) -> None:
    routing = Routing(route=route, usage=NO_USAGE)
    assert (routing.wants_docs, routing.wants_github) == (docs, github)


# ---- what it does when it fails ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("parsed", "error", "why"),
    [
        (None, RuntimeError("openai is down"), "the call raised"),
        (None, None, "the reply was not valid JSON (parsed is None)"),
        ({"route": "gitlab"}, None, "the reply named a route that does not exist"),
        ({}, None, "the reply had no route at all"),
        ({"route": "docs", "reason": "x"}, None, "the reply carried an extra field"),
    ],
)
async def test_every_failure_routes_to_docs_with_no_usage(
    parsed: dict[str, Any] | None, error: Exception | None, why: str
) -> None:
    # docs, not both: `both` would spend a subagent loop and GitHub points on a question nobody
    # classified. The fallback is the cheapest path that phase 2 already measured.
    router, _ = router_replying(parsed, error)
    routing = await route_question("anything", router=router)
    assert routing.route == "docs", why
    assert routing.usage == NO_USAGE, why


async def test_cancellation_is_not_swallowed() -> None:
    # CancelledError is a BaseException, so `except Exception` lets it through and the client
    # going away stops the turn instead of quietly routing it to docs. Same rule as plan_query,
    # and the same reason: a cancelled turn must not look like a turn that decided something.
    async def cancelling(_: list[BaseMessage]) -> dict[str, Any]:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await route_question("q", router=RunnableLambda(cancelling))


# ---- the messages ----------------------------------------------------------------------------


def test_with_no_history_the_user_turn_is_the_question() -> None:
    messages = router_messages("when was v7 released")
    assert len(messages) == 2
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1].content == [{"type": "text", "text": "when was v7 released"}]


def test_history_is_folded_into_the_user_turn_and_capped() -> None:
    history = [HistoryTurn("user" if i % 2 == 0 else "assistant", f"t{i}") for i in range(10)]
    text = router_messages("and when was that released?", history)[1].content[0]["text"]
    # The same cap as the planner's, so the two calls see the same conversation.
    assert "t9" in text
    assert f"t{10 - HISTORY_TURNS}" in text
    assert f"t{10 - HISTORY_TURNS - 1}" not in text
    assert text.endswith("LATEST USER MESSAGE:\nand when was that released?")


async def test_the_router_is_given_the_history_because_a_follow_up_needs_it() -> None:
    # Not a style preference: "and when was that released?" cannot be routed from its own text,
    # and a router structurally unable to be right about follow-ups would be measuring something
    # other than routing. The assertion is that the previous turn reaches the prompt.
    router, calls = router_replying({"route": "github"})
    await route_question(
        "and when was that released?",
        [HistoryTurn("user", "what is streamText"), HistoryTurn("assistant", "It streams text.")],
        router=router,
    )
    assert "what is streamText" in calls[0][1].content[0]["text"]


def test_the_reply_model_refuses_an_unknown_route() -> None:
    with pytest.raises(ValueError, match="route"):
        RouterOutput.model_validate({"route": "gitlab"})
