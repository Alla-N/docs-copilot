"""The subagent loop, offline: the repair that fires, the ones that must not, and the shape.

Nothing here reaches a network. The model is a script (ScriptedModel below), GitHub is a queue of
recorded responses behind httpx.MockTransport, and the schema is the same synthetic SDL the 3.2
and 3.3 suites use, built through the real introspection-to-client-schema path.

The test this sub-step exists for is the first one: a question whose first query is broken is
repaired within the cap, and the repair count is in the state. The rest are the failures that
must NOT turn into a loop -- an error no rewrite can fix, a model that gives up, and the cap
itself.
"""

import json

import httpx
import pytest
from graphql import build_schema, introspection_from_schema
from langchain_core.messages import AIMessage, ToolMessage

from copilot_agent.github_agent import (
    EVIDENCE_BYTE_CAP,
    QUERY_TOOL,
    Attempt,
    add_usage,
    build_github_agent,
    evidence_of,
    query_arguments,
)
from copilot_agent.github_query import (
    PREFLIGHT_ALIAS,
    SPEND_ALIAS,
    GitHubQueries,
    RunOutcome,
)
from copilot_agent.github_schema import SchemaCache
from copilot_agent.planner import NO_USAGE, TokenUsage

pytestmark = pytest.mark.anyio

SDL = """
type Query {
  repository(owner: String!, name: String!): Repository
  rateLimit(dryRun: Boolean = false): RateLimit
}
type Repository {
  name: String!
  releases(first: Int, last: Int): ReleaseConnection!
}
type ReleaseConnection {
  pageInfo: PageInfo!
  nodes: [Release]
}
type PageInfo { hasNextPage: Boolean!, endCursor: String }
type Release { tagName: String! }
type RateLimit { cost: Int!, nodeCount: Int!, remaining: Int! }
"""

# Locally valid against this schema -- `first` is declared optional -- and refused by GitHub.
# That gap is the whole point of the repair loop, and 3.3 measured it: the pre-flight prices such
# a query happily, so P1's failure mode costs a paid request before anyone learns about it.
NO_FIRST = """
query Releases($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) { releases { nodes { tagName } } }
}
"""
WITH_FIRST = """
query Releases($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) { releases(first: 1) { nodes { tagName } } }
}
"""
VARIABLES = {"owner": "vercel", "name": "ai"}
QUESTION = "What was the most recent release of vercel/ai?"


@pytest.fixture(scope="module")
def introspection() -> dict:
    return introspection_from_schema(build_schema(SDL), descriptions=True)


class Recorder:
    """A MockTransport handler that answers from a queue and keeps every request it saw."""

    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.queries: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.queries.append(json.loads(request.content)["query"])
        if not self.responses:
            raise AssertionError(f"unexpected request number {len(self.queries)}")
        return httpx.Response(200, json=self.responses.pop(0))


class Failing:
    """A transport that cannot reach GitHub at all."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")


class ScriptedModel:
    """The two methods the subgraph uses, and a record of how it was bound.

    Not a BaseChatModel subclass on purpose. The subgraph calls bind_tools once and ainvoke per
    step; a real fake chat model would add a `_generate` contract to satisfy and nothing to test.
    """

    def __init__(self, *replies: AIMessage) -> None:
        self.replies = list(replies)
        self.prompts: list[list] = []
        self.tool_names: list[str] = []
        self.bind_kwargs: dict = {}

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [t.name for t in tools]
        self.bind_kwargs = kwargs
        return self

    async def ainvoke(self, messages):
        self.prompts.append(list(messages))
        if not self.replies:
            raise AssertionError(f"the model was called {len(self.prompts)} times, one too many")
        return self.replies.pop(0)


def says(name: str, arguments: dict, *, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id}],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


def asks_query(query: str, *, call_id: str = "c1") -> AIMessage:
    return says(QUERY_TOOL, {"query": query, "variables": VARIABLES}, call_id=call_id)


def priced(cost: int = 1, nodes: int = 5) -> dict:
    # dryRun makes the siblings ABSENT, not null (measured 2026-09-15). The fixture says so too:
    # a fixture that invents a friendlier shape tests a service that does not exist.
    return {"data": {PREFLIGHT_ALIAS: {"cost": cost, "nodeCount": nodes}}}


def answered(tag: str = "ai@7.0.101", spent: int = 1) -> dict:
    return {
        "data": {
            "repository": {"releases": {"nodes": [{"tagName": tag}]}},
            SPEND_ALIAS: {"cost": spent, "nodeCount": 5},
        }
    }


def refused(error_type: str, message: str = "no `first`") -> dict:
    """A GitHub 200 whose errors array carries a path AND a type. Both halves matter.

    `path` says where it happened; `type` is the only thing that says whether a rewrite could
    fix it. Reading repairability off `path` is the mistake 3.3 measured, and it pointed the
    wrong way on the single most common failure there is.
    """
    return {
        "data": {"repository": {"releases": None}, SPEND_ALIAS: {"cost": 1, "nodeCount": 1}},
        "errors": [{"type": error_type, "path": ["repository", "releases"], "message": message}],
    }


def agent(introspection: dict, model: ScriptedModel, recorder=None, **kwargs):
    async def fetch() -> dict:
        return introspection

    cache = SchemaCache(fetch=fetch)
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder or Recorder()))
    runner = GitHubQueries(client=client, token="t", schema=cache)
    return build_github_agent(model=model, cache=cache, runner=runner, **kwargs)


# --- the done-when ----------------------------------------------------------------------


async def test_a_broken_first_query_is_repaired_within_the_cap(introspection) -> None:
    model = ScriptedModel(
        says("github_type", {"name": "Repository"}),
        asks_query(NO_FIRST),
        asks_query(WITH_FIRST, call_id="c2"),
    )
    recorder = Recorder(priced(), refused("MISSING_PAGINATION_BOUNDARIES"), priced(), answered())

    state = await agent(introspection, model, recorder).ainvoke({"question": QUESTION})
    result = state["result"]

    assert result.ok
    assert result.attempts == 2
    assert result.repairs == 1
    assert not result.first_try_valid, "the first query was accepted, so nothing was repaired"
    assert result.stages == ["field-error", "ok"]
    assert result.lookups == 1
    # Both attempts were paid for: the free pre-flight prices a pagination-less query happily.
    assert result.points_spent == 2
    assert "ai@7.0.101" in result.evidence


async def test_every_attempt_is_its_own_step(introspection) -> None:
    # Spec decision 5: a node with a for loop inside would hide all of this. The claim is only
    # worth making if the steps are actually observable, so this asserts on the stream rather
    # than on the final state.
    model = ScriptedModel(
        asks_query(NO_FIRST),
        asks_query(WITH_FIRST, call_id="c2"),
    )
    recorder = Recorder(priced(), refused("MISSING_PAGINATION_BOUNDARIES"), priced(), answered())

    compiled = agent(introspection, model, recorder)
    steps: list[str] = []
    async for chunk in compiled.astream({"question": QUESTION}, stream_mode="updates"):
        steps.extend(chunk)

    assert steps == ["explore", "run_query", "explore", "run_query", "summarise"]


async def test_the_repair_count_stops_at_the_cap(introspection) -> None:
    model = ScriptedModel(*(asks_query(NO_FIRST, call_id=f"c{n}") for n in range(3)))
    recorder = Recorder(
        *(r for _ in range(3) for r in (priced(), refused("MISSING_PAGINATION_BOUNDARIES")))
    )

    state = await agent(introspection, model, recorder).ainvoke({"question": QUESTION})
    result = state["result"]

    # Three attempts in all: the first, plus MAX_REPAIRS of 2.
    assert result.attempts == 3
    assert result.repairs == 2
    assert not result.ok
    assert model.replies == [], "the model should have been asked exactly three times"


# --- the failures that must not become a loop -------------------------------------------


async def test_an_error_no_rewrite_can_fix_is_not_repaired(introspection) -> None:
    model = ScriptedModel(asks_query(WITH_FIRST))
    recorder = Recorder(priced(), refused("NOT_FOUND", "Could not resolve to a Repository"))

    state = await agent(introspection, model, recorder).ainvoke({"question": QUESTION})
    result = state["result"]

    assert result.attempts == 1
    assert result.repairs == 0
    assert not result.ok
    # The loop exists to fix queries, not to argue with the world about a repository that is
    # not there. One attempt, and the model is never asked again.
    assert model.replies == []


async def test_a_failure_cannot_be_read_as_data(introspection) -> None:
    model = ScriptedModel(asks_query(WITH_FIRST))
    recorder = Recorder(priced(), refused("NOT_FOUND", "Could not resolve to a Repository"))

    state = await agent(introspection, model, recorder).ainvoke({"question": QUESTION})

    # The through-line of phase 3: an empty `result:` block under a query reads exactly like a
    # repository with no releases. A failure does not get that shape at all.
    evidence = state["result"].evidence
    assert not state["result"].ok
    assert "returned nothing" in evidence
    assert "Could not resolve" in evidence
    # Nothing in it can be read as a result: no query block, no JSON.
    assert "result:" not in evidence
    # And nothing in it can be read as a TOKEN. It opened `NO GITHUB DATA:` until 2026-09-15,
    # when a turn whose lookup had SUCCEEDED answered the user with that phrase -- copied from
    # the generation prompt, which explained what the marker meant. `ok` above is the machine
    # signal, and always was.
    assert "NO GITHUB DATA" not in evidence


async def test_a_model_that_gives_up_reports_no_query(introspection) -> None:
    model = ScriptedModel(AIMessage(content="I could not work out how to ask this."))

    state = await agent(introspection, model).ainvoke({"question": QUESTION})
    result = state["result"]

    assert not result.ok
    assert result.attempts == 0
    assert result.query is None
    assert "no query" in result.evidence
    assert "NO GITHUB DATA" not in result.evidence


async def test_a_transport_failure_is_its_own_stage_and_not_repairable(introspection) -> None:
    model = ScriptedModel(asks_query(WITH_FIRST))

    state = await agent(introspection, model, Failing()).ainvoke({"question": QUESTION})
    result = state["result"]

    # Rewriting a perfectly good query does not bring GitHub back. Recorded as its own stage so
    # 3.6 can tell a subagent that failed from a network that did.
    assert result.stages == ["transport"]
    assert result.attempts == 1
    assert model.replies == []


async def test_exploring_forever_is_stopped_by_the_lookup_cap(introspection) -> None:
    looks = (says("github_type", {"name": "Repository"}, call_id=f"c{n}") for n in range(3))
    model = ScriptedModel(*looks)

    state = await agent(introspection, model, max_lookups=2).ainvoke({"question": QUESTION})
    result = state["result"]

    assert result.lookups == 2
    assert result.attempts == 0
    assert not result.ok


# --- the wiring -------------------------------------------------------------------------


async def test_the_model_is_offered_three_tools_one_at_a_time(introspection) -> None:
    model = ScriptedModel(AIMessage(content="giving up"))

    await agent(introspection, model).ainvoke({"question": QUESTION})

    assert model.tool_names == ["github_schema", "github_type", QUERY_TOOL]
    # One reply, one destination. A reply holding a lookup and a query at once has two.
    assert model.bind_kwargs == {"parallel_tool_calls": False}


def test_the_subgraph_is_compiled_without_a_checkpointer(introspection) -> None:
    model = ScriptedModel()

    compiled = agent(introspection, model)

    # False, not None: None would INHERIT the parent's saver once this graph is nested, and
    # persist every schema outline the loop read. 2.5 measured what that costs -- 213 of 216 KiB
    # per turn -- on a list of retrieved text of exactly this shape.
    assert compiled.checkpointer is False


async def test_the_query_that_failed_travels_back_for_the_repair(introspection) -> None:
    model = ScriptedModel(
        asks_query(NO_FIRST),
        asks_query(WITH_FIRST, call_id="c2"),
    )
    recorder = Recorder(priced(), refused("MISSING_PAGINATION_BOUNDARIES"), priced(), answered())

    await agent(introspection, model, recorder).ainvoke({"question": QUESTION})

    # The second prompt is the first conversation plus GitHub's refusal, as an ordinary tool
    # reply. That is the whole of the repair: no second prompt, no repair node.
    second = model.prompts[1]
    refusal = next(m for m in second if isinstance(m, ToolMessage) and m.name == QUERY_TOOL)
    assert "Refused at the field-error stage." in refusal.content
    assert "no `first`" in refusal.content
    assert "Attempts left after this one: 2." in refusal.content


# --- the pure parts ---------------------------------------------------------------------


def test_the_evidence_carries_the_query_that_produced_it() -> None:
    text = evidence_of("q?", WITH_FIRST, {"repository": {"name": "ai"}})

    # A release date with the query that fetched it is a fact with provenance. On its own it is
    # a number from somewhere, and the faithfulness judge measures grounding, not truth.
    assert "releases(first: 1)" in text
    assert '"name": "ai"' in text


def test_a_result_over_the_cap_says_it_was_cut() -> None:
    big = {"repository": {"releases": {"nodes": [{"tagName": f"ai@7.0.{n}"} for n in range(600)]}}}

    text = evidence_of("q?", WITH_FIRST, big)

    assert len(text.encode()) <= EVIDENCE_BYTE_CAP
    assert "result lines shown" in text


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param({"query": "q"}, {}, id="absent"),
        pytest.param({"query": "q", "variables": {"a": 1}}, {"a": 1}, id="object"),
        pytest.param({"query": "q", "variables": '{"a": 1}'}, {"a": 1}, id="json-string"),
        pytest.param({"query": "q", "variables": "not json"}, {}, id="unparseable"),
        pytest.param({"query": "q", "variables": None}, {}, id="null"),
    ],
)
def test_variables_arrive_in_more_than_one_shape(arguments: dict, expected: dict) -> None:
    # A model told the parameter is an object still sometimes sends the object as text. An
    # unparseable one falls through to gate 5, which names the variables with no value -- a
    # better message than a type error from our own code.
    assert query_arguments(arguments)[1] == expected


def test_tokens_sum_across_calls_and_an_unknown_one_is_not_a_total_of_none() -> None:
    known = AIMessage(
        content="", usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    )
    unknown = AIMessage(content="")

    assert add_usage(None, known) == TokenUsage(3, 4)
    assert add_usage(TokenUsage(3, 4), known) == TokenUsage(6, 8)
    # A turn where one call of three reported nothing has still spent the other two.
    assert add_usage(TokenUsage(3, 4), unknown) == TokenUsage(3, 4)
    assert add_usage(None, unknown) == NO_USAGE


def test_an_attempt_keeps_the_numbers_and_drops_the_data() -> None:
    outcome = RunOutcome(stage="ok", repairable=False, data={"x": 1}, cost=1, node_count=5, spent=1)

    attempt = Attempt.of("query { x }", outcome)

    assert attempt.ok and attempt.spent == 1
    # The data is not a field of Attempt: attempts append, and an appending list of retrieved
    # text is what 2.5 caught costing 213 of 216 KiB a turn.
    assert not hasattr(attempt, "data")
