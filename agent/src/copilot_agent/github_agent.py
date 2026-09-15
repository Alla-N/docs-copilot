"""The GitHub subagent: one turn's worth of writing a query, running it, and fixing it.

A LangGraph subgraph (spec decision 5), not a node with a loop inside it, so that every attempt
is its own step: it appears in the `updates` stream and gets its own Langfuse span, and the
repair count is state rather than a local variable. 2.3 is the reason to care -- inside a graph
a call silently changed shape, and nothing outside the node could see that it had.

The loop is the tool loop, and a repair is the tool loop going round again:

    explore --lookup call--> lookup (github_schema, github_type) --> explore
       |
       +--github_query call--> run_query --repairable, under the cap--> explore
       |                           |
       |                           | ok, unrepairable, or out of repairs
       v                           v
    summarise <--------------------+

`github_query` is declared to the model as a tool and executed by a NODE, never by the tool
node. That is what makes an attempt a step. Its reply goes back as an ordinary ToolMessage, so a
repair needs no node and no second prompt of its own: the model reads GitHub's refusal in exactly
the place it reads a schema lookup, and writes the next query. The spec's diagram drew a `repair`
box, but it holds no work -- and a node that holds no work is a span that says nothing.

**What the spec's diagram got wrong, and why it is not being followed.** It draws write, validate,
preflight and run as four nodes. Three of them need the parsed document, so four nodes means
either a `DocumentNode` in state -- which the strict serializer (invariant 13) will not rebuild --
or parsing the same text three times, at which point the node boundary buys spans and nothing
else. `GitHubQueries.run()` already holds all seven gates behind one call and `RunOutcome.stage`
says which one refused, so one `run_query` node per attempt keeps decision 5's claim exactly true
and the finer split was a diagram drawn before the serializer was remembered.

**Not checkpointed.** Compiled with `checkpointer=False`, so nesting it under the parent's saver
does not persist this loop. Nothing here needs to survive a restart -- invariant 13 records a
turn only when it completes, and a half-explored schema is not a turn. The cost of the
alternative is measured and known: in 2.5, 213 of the 216 KiB written per turn were retrieval
candidate texts that the next turn read back only to reset them. A `messages` list carrying two
or three type outlines is the same mistake at the same order of magnitude.

**What is deliberately NOT in the system prompt: GitHub's `first`/`last` rule.** P1 predicts a
missing pagination bound is the most common first-try failure, and one sentence here would make
that prediction untestable. The rule is already in `github_type`'s detail output, which is where
a model that looks before it writes will find it -- so first-try validity measures whether it
looked, which is the question worth asking. If the number is bad, the prompt is the first knob,
and turning it will then be a measured change instead of a starting assumption.
"""

import json
import logging
import operator
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from copilot_agent.github_query import GitHubQueries, RunOutcome
from copilot_agent.github_schema import (
    DEFAULT_BYTE_CAP,
    SchemaCache,
    capped_block,
    describe_type,
    github_fetch,
    schema_summary,
)
from copilot_agent.planner import NO_USAGE, TokenUsage
from copilot_agent.settings import Settings

logger = logging.getLogger(__name__)

# Spec decision 9: three attempts in all. A guess, to be measured in 3.6. What the cap protects
# against is not cost -- a refused query usually spends nothing -- but a loop that talks itself
# into the same query twice.
MAX_REPAIRS = 2

# Schema lookups before the subagent is made to stop exploring. Not a cost guard either: a
# lookup is free of GitHub points and cheap in tokens. It is here because a model that has asked
# for six type descriptions and still not written a query is not converging, and the recursion
# limit is a worse way to find that out than a number with a name.
MAX_LOOKUPS = 6

# What the answer model is allowed to read from one GitHub result. Well under the 8000 byte
# schema cap: this competes with five reranked documentation chunks for the same prompt.
EVIDENCE_BYTE_CAP = 4000

# The tool the model calls to run a query. Declared to the model, executed by run_query.
QUERY_TOOL = "github_query"

SYSTEM_PROMPT = """\
You answer questions about the vercel/ai repository on GitHub by writing GraphQL queries against
the GitHub API, and you answer no other way. You cannot read documentation and you cannot rely on
what you remember about this repository; the only facts you may report are the ones a query
returns.

You have three tools:
- github_schema() lists the entry points of the schema.
- github_type(name, fields) describes one type. With no fields it lists every field and its type,
  which is how you find out what exists. With fields it gives the full signature of those fields,
  including which arguments are required.
- github_query(query, variables) runs one read-only query and returns its data, or the reason it
  was refused.

Work in this order: look at the types you need, then write the query. Call one tool at a time.
Use GraphQL variables for the owner and the repository name rather than writing them into the
query text. Ask only for the fields that answer the question.

When the question names one particular thing -- a tag name, an issue or pull request number, a
login -- use the field that takes it as an argument instead of listing the collection it belongs
to. Look that field up if you are not sure it exists. A connection answers questions about sets.

If github_query refuses a query, read the reason, decide which part of the query it is about, and
send a corrected one. Do not resend the same query. You get very few attempts, so spend one on
looking a type up rather than on guessing a field name.

When you have the data, stop and reply with nothing. Do not write the answer yourself: something
else turns the data into the answer."""


@tool(QUERY_TOOL)
async def github_query_tool(query: str, variables: dict[str, Any] | None = None) -> str:
    """Run one read-only GraphQL query against the GitHub API and return its data.

    Args:
        query: the GraphQL document, containing exactly one query operation.
        variables: values for every variable the operation declares.
    """
    # Declared so the model can call it; run by the run_query node, which is what makes each
    # attempt a step of the graph. It is never given to the tool node, so reaching here means
    # the routing is wrong, and a tripwire is better than a quiet second execution path.
    raise RuntimeError("github_query is run by the run_query node, not by the tool node")


def read_tools(cache: SchemaCache, *, byte_cap: int = DEFAULT_BYTE_CAP) -> list[BaseTool]:
    """The two tools that only read the cached schema. Closures over one cache per process."""

    @tool("github_schema")
    async def github_schema() -> str:
        """List the entry points of the GitHub GraphQL schema: the fields of the Query root."""
        return schema_summary(await cache.get(), byte_cap=byte_cap)

    @tool("github_type")
    async def github_type(name: str, fields: list[str] | None = None) -> str:
        """Describe one type of the GitHub GraphQL schema.

        Args:
            name: the type name, spelled as the schema spells it.
            fields: the fields to describe in full, with arguments and descriptions. Omit it to
                list every field of the type with its result type, and nothing else.
        """
        return describe_type(await cache.get(), name, fields=fields, byte_cap=byte_cap)

    return [github_schema, github_type]


@dataclass(frozen=True)
class Attempt:
    """One generated query and what became of it.

    The data is not in here. It is written to its own state key, because a record kept per
    attempt is a record the repair loop appends to, and 2.5 is the measurement of what happens
    when retrieved text rides along in an appending list.
    """

    query: str
    stage: str
    ok: bool
    repairable: bool
    messages: list[str]
    cost: int | None
    node_count: int | None
    spent: int | None

    @classmethod
    def of(cls, query: str, outcome: RunOutcome) -> "Attempt":
        return cls(
            query=query,
            stage=outcome.stage,
            ok=outcome.ok,
            repairable=outcome.repairable,
            messages=list(outcome.messages),
            cost=outcome.cost,
            node_count=outcome.node_count,
            spent=outcome.spent,
        )


@dataclass(frozen=True)
class GitHubEvidence:
    """What the subagent hands back: the facts, and what getting them cost.

    Spec decision 12 -- the metrics travel with the turn rather than in a side channel, because
    the eval harness cannot see inside the service. Everything 3.6 has to report is a field here:
    first-try validity, validity after repairs (`ok` with `repairs` > 0), points per question,
    and how much of the turn went on looking rather than writing.
    """

    question: str
    ok: bool
    # What generation reads. On failure it says there is nothing, in those words.
    evidence: str
    query: str | None
    attempts: int
    repairs: int
    first_try_valid: bool
    # One per attempt, in order: which gate refused it, or "ok".
    stages: list[str]
    lookups: int
    points_spent: int
    node_count: int | None
    usage: TokenUsage


class GitHubAgentInput(TypedDict):
    question: str


class GitHubAgentState(GitHubAgentInput, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    attempts: Annotated[list[Attempt], operator.add]
    # The data of the last attempt that returned any. Replaced, never appended.
    data: dict | None
    usage: TokenUsage
    result: GitHubEvidence


GitHubAgent = CompiledStateGraph[GitHubAgentState, None, GitHubAgentInput, GitHubAgentState]


def lookups_so_far(messages: list[AnyMessage]) -> int:
    """Schema lookups answered so far: tool replies that were not query attempts."""
    return sum(
        1 for m in messages if isinstance(m, ToolMessage) and m.name not in (QUERY_TOOL, None)
    )


def add_usage(total: TokenUsage | None, reply: AIMessage) -> TokenUsage:
    """Accumulate one model call's tokens. None and a number add to a number, not to None.

    The provider reports None when it does not know, and a turn where one call of three reported
    nothing has still spent the other two. Summing to None would throw those away; treating a
    missing number as zero would report a total as if it were complete. This reports the sum of
    what was known, which is what `NO_USAGE` already means everywhere else.
    """
    current = total or NO_USAGE
    metadata = reply.usage_metadata or {}

    def plus(have: int | None, more: object) -> int | None:
        if not isinstance(more, int):
            return have
        return more if have is None else have + more

    return TokenUsage(
        input_tokens=plus(current.input_tokens, metadata.get("input_tokens")),
        output_tokens=plus(current.output_tokens, metadata.get("output_tokens")),
    )


def query_arguments(arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The query text and variables out of one tool call's arguments.

    `variables` arrives as a JSON string often enough to be worth handling: a model that has been
    told the parameter is an object still sometimes sends the object as text. A string that does
    not parse is passed on as no variables, and gate 5 then says which variables have no value --
    a better message than a type error from our own code.
    """
    query = str(arguments.get("query") or "")
    variables = arguments.get("variables")
    if isinstance(variables, str):
        try:
            variables = json.loads(variables)
        except ValueError:
            variables = None
    return query, variables if isinstance(variables, dict) else {}


def attempt_reply(outcome: RunOutcome, *, repairs_left: int) -> str:
    """What the model is told about its query. Never dressed up, never dressed down.

    A refusal says what was refused and whether another attempt is coming. Saying "try again" to
    a model that has no attempts left is how a loop learns to argue with a NOT_FOUND; saying
    nothing when an attempt IS left wastes it.

    There is no third line for "repairable, but out of attempts", because the routing sends that
    case straight to summarise and the model never reads the reply. A branch nobody can reach is
    a claim nobody can check -- phase 3 finding 9, written down rather than repeated.
    """
    if outcome.ok:
        return json.dumps(outcome.data, ensure_ascii=False)
    lines = [f"Refused at the {outcome.stage} stage.", *outcome.messages]
    if not outcome.repairable:
        lines.append("This cannot be fixed by rewriting the query. Do not send another.")
    elif repairs_left > 0:
        lines.append(f"Send a corrected query. Attempts left after this one: {repairs_left}.")
    return "\n".join(lines)


def evidence_of(question: str, query: str, data: dict | None) -> str:
    """The successful result as generation will read it: the question, the query, the JSON.

    The query is included on purpose. A release date with the query that fetched it is a fact
    with provenance; on its own it is a number from somewhere, and phase 2's faithfulness judge
    measures grounding, not truth.
    """
    header = [f"GitHub GraphQL result for: {question}", "query:", query.strip(), "result:"]
    body = json.dumps(data, ensure_ascii=False, indent=2).splitlines()
    return capped_block(header, body, EVIDENCE_BYTE_CAP, "result lines")


def no_evidence_of(attempts: list[Attempt]) -> str:
    """The failed result, in words that cannot be mistaken for an answer OR quoted as a token.

    The through-line of this whole phase is that a correctly shaped output carrying no information
    reads exactly like a real one. This block is the place that would happen: an empty `result:`
    under a query looks like a repository with no releases. So a failure does not use that shape
    at all.

    **It also does not use a marker.** The first version opened `NO GITHUB DATA:`, and on
    2026-09-15 a turn whose lookup had SUCCEEDED answered the user with "NO GITHUB DATA. I could
    not find any open issues...". The phrase was never in that turn's evidence -- the evidence was
    the success shape -- it came from the generation prompt, which explained what the marker meant
    and so taught the model to write it. A token that is both a machine signal and prompt text
    gets quoted eventually; `NO RELEVANT DOCUMENTATION FOUND` and the retrieval prompt's internal
    marker are the same bug already open in the backlog.

    The deeper point is that the marker was redundant from the start. `GitHubEvidence.ok` is the
    machine signal for a failed lookup, and it always was; this string only ever had to be read
    by a model. So the token is deleted rather than reworded, and what is left is prose that
    stays true if the model does repeat it.
    """
    if not attempts:
        return "The GitHub lookup produced no query for this question, so it returned nothing."
    last = attempts[-1]
    reason = "; ".join(last.messages) or "no reason given"
    return (
        f"The GitHub lookup returned nothing for this question. It made {len(attempts)} "
        f"attempt(s); the last was refused at the {last.stage} stage ({reason}). Nothing about "
        f"the repository can be answered from this turn."
    )


def build_github_agent(
    *,
    model: BaseChatModel,
    cache: SchemaCache,
    runner: GitHubQueries,
    max_repairs: int = MAX_REPAIRS,
    max_lookups: int = MAX_LOOKUPS,
    byte_cap: int = DEFAULT_BYTE_CAP,
) -> GitHubAgent:
    """Wire the subagent around one schema cache and one query runner, and compile it.

    Dependencies come in rather than being built here, exactly as build_graph takes its planner
    and its search: the tests drive a scripted model and a mock transport, and nothing in this
    module knows what a network is.

    parallel_tool_calls=False, explicitly. The loop reads one reply, routes on it, and answers
    it; a reply holding a lookup and a query at once has two destinations. The routing below
    still handles a mixed batch rather than trusting the flag, because a flag that only one
    provider honours is not a guarantee, but with it set the case should never arise.
    """
    # Built once, so the tool the model is offered and the tool the node runs are the same
    # object. Two calls to read_tools would give two closures that differ only in the day one of
    # them is edited.
    lookup_tools = read_tools(cache, byte_cap=byte_cap)
    bound = model.bind_tools([*lookup_tools, github_query_tool], parallel_tool_calls=False)

    async def explore(state: GitHubAgentState) -> dict[str, object]:
        opening: list[AnyMessage] = []
        if not state.get("messages"):
            opening = [SystemMessage(SYSTEM_PROMPT), HumanMessage(state["question"])]
        conversation = [*state.get("messages", []), *opening]
        reply = await bound.ainvoke(conversation)
        return {
            "messages": [*opening, reply],
            "usage": add_usage(state.get("usage"), reply),
        }

    async def run_query(state: GitHubAgentState) -> dict[str, object]:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        repairs_left = max_repairs - len(state.get("attempts", []))
        replies: list[AnyMessage] = []
        attempts: list[Attempt] = []
        data = state.get("data")

        for call in calls:
            if call["name"] != QUERY_TOOL:
                # Only reachable from a mixed batch, which parallel_tool_calls=False should
                # prevent. Answered rather than dropped: an unanswered tool call is a 400 from
                # the provider on the next turn of the loop, and a deadlock is a worse bug than
                # a wasted call.
                replies.append(
                    ToolMessage(
                        content="One tool call at a time. This one was not run.",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
                continue
            query, variables = query_arguments(call.get("args") or {})
            try:
                outcome = await runner.run(query, variables)
            except httpx.HTTPError as error:
                # Not a query problem, so not repairable: rewriting a perfectly good query does
                # not bring GitHub back. Recorded as its own stage so 3.6 can tell a subagent
                # that failed from a network that did.
                logger.warning("github query transport failure", exc_info=True)
                outcome = RunOutcome(
                    stage="transport",
                    repairable=False,
                    messages=[f"The request to GitHub failed: {error}"],
                )
            attempts.append(Attempt.of(query, outcome))
            if outcome.data is not None:
                data = outcome.data
            replies.append(
                ToolMessage(
                    content=attempt_reply(outcome, repairs_left=repairs_left),
                    tool_call_id=call["id"],
                    name=QUERY_TOOL,
                )
            )

        return {"messages": replies, "attempts": attempts, "data": data}

    async def summarise(state: GitHubAgentState) -> dict[str, object]:
        attempts = state.get("attempts", [])
        messages = state.get("messages", [])
        succeeded = attempts[-1] if attempts and attempts[-1].ok else None
        return {
            "result": GitHubEvidence(
                question=state["question"],
                ok=succeeded is not None,
                evidence=(
                    evidence_of(state["question"], succeeded.query, state.get("data"))
                    if succeeded is not None
                    else no_evidence_of(attempts)
                ),
                query=attempts[-1].query if attempts else None,
                attempts=len(attempts),
                repairs=max(len(attempts) - 1, 0),
                first_try_valid=bool(attempts) and attempts[0].ok,
                stages=[a.stage for a in attempts],
                lookups=lookups_so_far(messages),
                points_spent=sum(a.spent or 0 for a in attempts),
                node_count=succeeded.node_count if succeeded is not None else None,
                usage=state.get("usage") or NO_USAGE,
            )
        }

    def after_explore(state: GitHubAgentState) -> Literal["lookup", "run_query", "summarise"]:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            # The model replied with prose. It was told not to answer, so this is giving up, and
            # the honest handling is to say so rather than to prompt it again.
            return "summarise"
        if any(call["name"] == QUERY_TOOL for call in calls):
            return "run_query"
        if lookups_so_far(state["messages"]) >= max_lookups:
            return "summarise"
        return "lookup"

    def after_run_query(state: GitHubAgentState) -> Literal["explore", "summarise"]:
        attempts = state.get("attempts", [])
        if not attempts:
            return "summarise"
        last = attempts[-1]
        if last.ok or not last.repairable or len(attempts) > max_repairs:
            return "summarise"
        return "explore"

    builder = StateGraph(GitHubAgentState, input_schema=GitHubAgentInput)
    builder.add_node("explore", explore)
    builder.add_node("lookup", ToolNode(lookup_tools))
    builder.add_node("run_query", run_query)
    builder.add_node("summarise", summarise)
    builder.add_edge(START, "explore")
    builder.add_conditional_edges("explore", after_explore, ["lookup", "run_query", "summarise"])
    builder.add_edge("lookup", "explore")
    builder.add_conditional_edges("run_query", after_run_query, ["explore", "summarise"])
    builder.add_edge("summarise", END)
    # checkpointer=False, not None: None would INHERIT the parent's saver once this graph is
    # nested, and persist every schema outline the loop read. See the module docstring.
    #
    # Measured in 3.5 (experiments/subgraph_stream.py, 2026-09-15), because up to then it was a
    # reading of the documentation: the same run nested under a parent saver wrote 5 checkpoints
    # with False and 9 with None, and the None run put the subgraph's `messages`, `result` and
    # `steps` into the parent's channel values. It holds for a wrapper node calling ainvoke as
    # well as for a directly attached subgraph, which was the half nobody had checked.
    return builder.compile(checkpointer=False)


# What the subagent is allowed to emit in one call. It writes GraphQL documents, which are the
# longest thing any model in this service produces; the planner's 512 would truncate a query
# with two or three nested connections and the truncation would arrive as a parse error, which
# reads exactly like a model that cannot write GraphQL. Not a budget: the repair cap and the
# lookup cap are what bound a turn.
GITHUB_MAX_OUTPUT_TOKENS = 1500

GITHUB_MAX_RETRIES = 2


def openai_github_model(settings: Settings, **client_options: Any) -> ChatOpenAI:
    """The subagent's chat model: the planner's model and the planner's explicit switches.

    Same model as the planner, the router and the answer (spec decision 8), so phase 6's Bedrock
    A/B moves one variable. streaming=False for the reason planner.openai_planner_model gives:
    LangChain streams any call made while a streaming callback handler is attached, and
    LangGraph's "messages" mode attaches one to every model call inside a graph.
    """
    return ChatOpenAI(
        model=settings.planner_model,
        temperature=0,
        max_tokens=GITHUB_MAX_OUTPUT_TOKENS,
        max_retries=GITHUB_MAX_RETRIES,
        use_responses_api=True,
        streaming=False,
        api_key=settings.openai_api_key,
        **client_options,
    )


@asynccontextmanager
async def open_github_agent(settings: Settings) -> AsyncIterator[GitHubAgent | None]:
    """The subagent for the service's lifespan, or None when there is no token.

    Decision 11, as one branch in one place: without GITHUB_TOKEN this yields None, api.py passes
    None to the graph, and graph.build_graph then compiles the phase 2 graph with no router node
    and no GitHub node in it. Not a flag read at request time -- a capability behind a runtime
    check somebody can get wrong is worse than a capability that is absent.

    Nothing is fetched here. The schema is fetched on the first GitHub turn (decision 6): adding
    GitHub to startup readiness would make this service's availability depend on GitHub's, for a
    capability most turns never use.

    One client and one cache for the process, shared by the two tools and the runner, so the
    schema is introspected once and the lock in SchemaCache has something to protect.
    """
    if settings.github_token is None:
        yield None
        return
    token = settings.github_token.get_secret_value()
    async with httpx.AsyncClient() as client:
        # path=None deliberately: the on-disk cache is a development convenience (github_schema
        # says so), and a container has no durable disk and should not run on a schema older
        # than its image.
        cache = SchemaCache(fetch=github_fetch(client, token))
        yield build_github_agent(
            model=openai_github_model(settings),
            cache=cache,
            runner=GitHubQueries(client=client, token=token, schema=cache),
        )
