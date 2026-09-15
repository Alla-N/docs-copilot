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
    measured. The cheapest known-good answer is the right fallback.

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

Route = Literal["docs", "github", "both"]

DOCS: Route = "docs"

# The reply is {"route": "docs"} and nothing else, so the cap is far above what a correct answer
# needs. It is not a budget: a truncated reply fails validation and falls back silently to docs,
# so a cap set tight enough to bite would be a routing bug wearing a cost saving.
ROUTER_MAX_OUTPUT_TOKENS = 64

# The planner's, and for the same reason (planner.PLANNER_MAX_RETRIES).
ROUTER_MAX_RETRIES = 2

SYSTEM_PROMPT = """\
You decide which source can answer a question about the Vercel AI SDK. You do NOT answer.

There are two sources:
- DOCS: the Vercel AI SDK documentation. What the SDK is, what its functions and hooks do, how to
  use them, settings, providers, migration guides, error handling, examples. Anything about how
  the library works or how to write code with it.
- GITHUB: the vercel/ai repository on GitHub. Releases and their dates, version tags, what a
  release contained, issues and their state, pull requests and who wrote or merged them, commits,
  contributors, and the contents of files in the repository.

Answer with one of:
- "docs" when the documentation alone can answer it.
- "github" when only the repository can: the question is about a release, a version's date, an
  issue, a pull request, a commit, a contributor, or the repository's own files.
- "both" when the question needs each of them, for example when it asks how something works AND
  when it shipped, or whether a documented behaviour has an open issue against it.

Rules:
- Prefer "docs". A question about how to do something is a docs question even when the answer may
  have changed between versions.
- A question naming a version number is not automatically GitHub: "how do I migrate to v7" is
  documentation. "when was v7 released" is GitHub.
- Choose "both" only when an answer that left one source out would be incomplete, not when the
  other source might add colour.
- Use the conversation so far to resolve a follow-up. "and when was that released?" is a GITHUB
  question about whatever the previous turn was about."""

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
            "enum": ["docs", "github", "both"],
            "description": (
                "'docs' when the Vercel AI SDK documentation alone can answer the question, "
                "'github' when only the vercel/ai repository can (releases, issues, pull "
                "requests, commits, contributors, files), 'both' when an answer that left "
                "either out would be incomplete."
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
    """Where the turn goes, and what deciding cost. NO_USAGE on the fallback, like Plan."""

    route: Route
    usage: TokenUsage

    @property
    def wants_docs(self) -> bool:
        return self.route in ("docs", "both")

    @property
    def wants_github(self) -> bool:
        return self.route in ("github", "both")


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
