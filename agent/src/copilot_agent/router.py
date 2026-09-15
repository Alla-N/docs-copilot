"""The router: which source can answer this question - the documentation, GitHub, or both.

Spec decision 1, and the reason it is a node of its own rather than a new field on the planner:
extending the planner's structured output would change its prompt, which forks it from the
TypeScript baseline, invalidates tests/test_planner_request_parity.py and the 23x5 planner eval,
and makes phase 2's retrieval numbers stop being a clean comparison - all to save one small call.
A node of its own also makes routing accuracy a number that can be measured on its own, which is
what the phase 3 done-when asks for.

It runs AFTER plan, so greetings and off-topic messages never reach it (they are already at the
canned reply) and the question it sees is one the planner called a search.

Three things here are deliberate:

  - **It sees the conversation, not the sub-queries.** The spec's open question was whether to
    give the router the planner's resolved sub-queries; the answer for now is no, so that a
    routing failure is a routing failure and not a plan it inherited. History is a different
    matter: "and when was that released?" cannot be routed from its own text, and a router that
    is structurally unable to be right about follow-ups would be measuring the wrong thing. So
    the history is folded into the user turn exactly as planner_messages folds it, same cap.
  - **It returns a route and nothing else.** No reason field. Routing accuracy in 3.6 is measured
    against a label, and a model's own account of why it chose is not evidence for whether it
    chose correctly - phase 3 finding 5 is what an asserted aside inside a measurement is worth.
    It also costs output tokens on every turn, which is exactly what P6 is measuring.
  - **A failure routes to docs, not to both.** The router is an enhancement, like the planner
    (plan_query's fallback rule). `both` on a failed router call would spend a subagent loop and
    GitHub points on a question nobody classified; `docs` degrades to the pipeline phase 2 already
    measured. The cheapest known-good answer is the right fallback. Since the routes became two,
    this is also the only remaining way a turn can lose GitHub -- and losing a supplement is a
    strictly smaller failure than losing the documentation, which is what the third route could
    do.

The model is the planner's (decision 8: every phase 3 call is gpt-4o-mini, so phase 6's Bedrock
A/B moves one variable), and it is configured through the same explicit switches, for the same
reasons - Responses API, temperature 0, streaming=False.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict

from copilot_agent.planner import (
    HISTORY_TURNS,
    NO_USAGE,
    HistoryTurn,
    TokenUsage,
)
from copilot_agent.settings import Settings

logger = logging.getLogger(__name__)

# Two routes, not three. There WAS a third, `github`, which answered from GitHub alone and
# skipped retrieval, and it was removed by its first measurement rather than by an argument.
#
# In the 3.5 eval run (2026-09-15, 27 documentation questions) the router chose it twice, for
# "what is new in AI SDK 7" and "what was changed in AI SDK 7" -- the near-synonym pair this
# project already keeps as its retrieval diagnosis. Both turns retrieved nothing, and the same
# defect came out wearing two faces: one answered from GitHub after opening with the
# documentation refusal sentence, so isRefusal (positional, invariant 5) logged an answered turn
# as refused and the harness scored it 0/3; the other answered cleanly from GitHub with no
# documentation behind it and PASSED. The suite also got CHEAPER, $0.1892 against $0.1972, because
# the skipped rerank calls cost more than the router saved -- a cost improvement that was a
# correctness regression.
#
# Making GitHub additive kills all of it by construction: a mis-route now costs latency and
# GitHub points and can never cost an answer, and no case can go green with no documentation
# because of a routing decision. What it costs is retrieval on every GitHub question (about 2 s
# and $0.004), and routing accuracy in 3.6 becomes a two-label measure.
#
# db/009's check constraint still allows 'github'. It has run against the database, so it is not
# edited; a constraint that permits a value this code cannot emit is harmless, and narrowing it
# would be a migration whose only effect is tidiness.
Route = Literal["docs", "both"]

DOCS: Route = "docs"

# The reply is {"route": "docs"} and nothing else, so the cap is far above what a correct answer
# needs. It is not a budget: a truncated reply fails validation and falls back silently to docs,
# so a cap set tight enough to bite would be a routing bug wearing a cost saving.
ROUTER_MAX_OUTPUT_TOKENS = 64

# The planner's, and for the same reason (planner.PLANNER_MAX_RETRIES).
ROUTER_MAX_RETRIES = 2

SYSTEM_PROMPT = """\
You decide whether a question about the Vercel AI SDK needs the GitHub repository as well as the
documentation. You do NOT answer.

The documentation is always searched. It covers what the SDK is, what its functions and hooks do,
how to use them, settings, providers, migration guides, error handling and examples.

The vercel/ai repository on GitHub can be searched as well. It covers releases and their dates,
version tags, issues and their state, pull requests and who wrote or merged them, commits,
contributors, and the contents of files in the repository.

Answer with one of:
- "docs" when the documentation is enough on its own.
- "both" when the question asks for something only the repository can say: a release date, a
  version tag, an issue, a pull request, a commit, a contributor, or a file in the repository.

Rules:
- Prefer "docs". Choosing "both" when the repository is not needed costs a slow lookup for
  nothing, so ask whether a repository fact is genuinely being requested.
- A question naming a version number is not by itself a repository question. "how do I migrate to
  v7", "what is new in v7" and "what changed in v7" are answered by the migration guide, which is
  documentation. "when was v7 released" and "which release first carried this tag" are repository
  questions.
- Use the conversation so far to resolve a follow-up. "and when was that released?" needs the
  repository, about whatever the previous turn was about."""

# The reply schema, sent the way the planner sends its own (a ready-made {name, schema, strict}
# dict rather than a Pydantic class, so what the model reads is written here and not derived by
# a library). RouterOutput below validates the reply against the same structure.
ROUTE_SCHEMA_NAME = "route_choice"
ROUTE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["docs", "both"],
            "description": (
                "'docs' when the Vercel AI SDK documentation is enough on its own, 'both' when "
                "the question also asks for something only the vercel/ai repository can say: a "
                "release date, a version tag, an issue, a pull request, a commit, a contributor, "
                "or a file in the repository."
            ),
        }
    },
    "required": ["route"],
    "additionalProperties": False,
}


class RouterOutput(BaseModel):
    """The reply's shape. Strict structured output should guarantee it; a model reply is still
    input from outside, so it is parsed, not trusted (the same rule as planner.PlanOutput)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Route


@dataclass(frozen=True)
class Routing:
    """Where the turn goes, and what deciding cost. NO_USAGE on the fallback, like Plan.

    There is no `wants_docs`. Every route wants the documentation now, so the property would
    return True always -- a branch nobody can reach, which phase 3 finding 9 says is worse than
    no branch at all. The fact that retrieval always runs is expressed where it is enforced, in
    graph.route_after_router.
    """

    route: Route
    usage: TokenUsage

    @property
    def wants_github(self) -> bool:
        return self.route == "both"


# Same shape as planner.Planner: messages in, {"raw", "parsed", "parsing_error"} out.
Router = Runnable[list[BaseMessage], dict[str, Any]]


def openai_router_model(settings: Settings, **client_options: Any) -> ChatOpenAI:
    """The router's chat model. The planner's model and the planner's explicit switches.

    streaming=False is not a default being restated: LangChain streams any call made while a
    streaming callback handler is attached, and LangGraph's "messages" stream mode attaches one
    to every model call inside the graph. test_graph_wired.py found that with the planner.
    """
    return ChatOpenAI(
        model=settings.planner_model,
        temperature=0,
        max_tokens=ROUTER_MAX_OUTPUT_TOKENS,
        max_retries=ROUTER_MAX_RETRIES,
        use_responses_api=True,
        streaming=False,
        api_key=settings.openai_api_key,
        **client_options,
    )


def build_router(model: BaseChatModel) -> Router:
    """Bind the route schema to a chat model, the way build_planner binds the plan schema."""
    return model.with_structured_output(
        {"name": ROUTE_SCHEMA_NAME, "schema": ROUTE_JSON_SCHEMA, "strict": True},
        method="json_schema",
        include_raw=True,
    )


def router_messages(question: str, history: Sequence[HistoryTurn] = ()) -> list[BaseMessage]:
    """The system prompt and one user turn, with the recent history folded into its text.

    planner_messages' shape, deliberately: the same fold, the same HISTORY_TURNS cap, and the
    same one-text-block content, so the two calls differ in their prompt and in nothing else.
    """
    history_text = "\n".join(f"{t.role}: {t.text}" for t in history[-HISTORY_TURNS:])
    prompt = (
        f"CONVERSATION SO FAR:\n{history_text}\n\nLATEST USER MESSAGE:\n{question}"
        if history_text
        else question
    )
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(content=[{"type": "text", "text": prompt}])]


async def route_question(
    question: str, history: Sequence[HistoryTurn] = (), *, router: Router
) -> Routing:
    """Route one question. Never raises except on cancellation: any failure routes to docs.

    Cancellation (the client went away) is a BaseException, so `except Exception` lets it
    through and the task stops, exactly as in plan_query.
    """
    try:
        result = await router.ainvoke(router_messages(question, history))
        output = RouterOutput.model_validate(result["parsed"])
        usage_metadata = result["raw"].usage_metadata or {}
        return Routing(
            route=output.route,
            usage=TokenUsage(
                input_tokens=usage_metadata.get("input_tokens"),
                output_tokens=usage_metadata.get("output_tokens"),
            ),
        )
    except Exception:
        logger.warning("router failed, falling back to the documentation", exc_info=True)
        return Routing(route=DOCS, usage=NO_USAGE)
