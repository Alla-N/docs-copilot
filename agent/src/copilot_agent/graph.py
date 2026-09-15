"""The chat pipeline as a LangGraph graph: plan, route, retrieve every sub-query, merge, generate.

The Python counterpart of the middle of app/api/chat/route.ts (plannedRetrieve in lib/plan.ts,
then the canned reply or the grounded streamText call), with the same behaviour:

    START -> plan --route--> canned -> END                      greeting, off-topic
                        \\--> retrieve x N -> merge -> generate -> END

  - a greeting or an off-topic message costs one planner call and nothing else;
  - every sub-query is retrieved in parallel (TypeScript: Promise.all; here: one Send per
    sub-query, so each retrieval is its own node run, and its own span once Langfuse traces it);
  - the answer is generated for the planner's resolved sub-queries (invariant 4).

Step 3.5 adds a second shape, taken ONLY when a router and a GitHub subagent are handed in
(spec decision 11: no GITHUB_TOKEN, no GitHub path, and then this file compiles exactly the
graph above, so CI, Docker and a laptop keep running and measuring the phase 2 pipeline):

    START -> plan --route--> canned -> END
                        \\--> router --> retrieve x N --\\
                                    \\--> github --------+--> merge -> generate -> END

  - the router is its own node, not a field on the planner (spec decision 1): extending the
    planner's structured output would change its prompt, fork it from the TypeScript baseline,
    and cost the 23x5 planner eval and the phase 2 retrieval comparison their meaning;
  - its three routes fan out through the same mechanism as the sub-queries, one Send each, so a
    `both` turn runs retrieval and the subagent in one superstep and they meet at merge;
  - the subagent is reached through a WRAPPER node, not attached as a node itself. Its state
    schema is not this one, and a compiled subgraph attached directly writes back only the keys
    the parent declares -- silently. Measured (experiments/subgraph_stream.py, 2026-09-15): the
    direct attachment ran to completion, wrote five checkpoints and produced an answer with the
    subagent's evidence missing from the state and no error anywhere. The wrapper IS the
    translation between the two schemas, and it is also the only place a subagent failure can be
    given a shape this graph understands.

Conversation state (step 2.5): compiled with a checkpointer, the graph keeps each conversation's
turns under its thread id, so a request brings only the new question and the thread id comes in
the run's config ({"configurable": {"thread_id": ...}}). A turn is recorded only when it
completes: the last node (canned or generate) appends the question and the answer together, so a
turn that fails or is cancelled (Stop, a closed tab) leaves the thread as it was. Everything else
in ChatState is per turn, and the plan node resets what would otherwise carry over (see
turn_reset()).

The graph knows nothing about HTTP or the AI SDK. It emits LangGraph stream events: "updates"
when a node finishes (the plan, each retrieval, the merged sources, the canned or generated
answer) and "messages" for the answer's tokens. ui_stream.py turns those into the AI SDK's UI
message stream for POST /chat (api.py).

Dependencies come in through build_graph() and are captured by the node functions (a closure):
the planner, the search function and the chat model are built once at startup and shared by
every run. Per-request values do not: the thread id travels in the run's config, which is where
LangGraph's checkpointer looks for it (trace ids, in 2.7, likewise per run).
"""

import operator
import time
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite, Send

from copilot_agent.generation import (
    FinishReason,
    finish_reason,
    generation_messages,
    openai_generation_model,
)
from copilot_agent.github_agent import GitHubAgent, GitHubEvidence
from copilot_agent.history import capped_history, keep_recent
from copilot_agent.planner import (
    GREETING_MESSAGE,
    NO_USAGE,
    HistoryTurn,
    Plan,
    Planner,
    TokenUsage,
    build_planner,
    openai_planner_model,
    plan_query,
)
from copilot_agent.refusal import REFUSAL_MESSAGE
from copilot_agent.retrieval import (
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
    SearchDocs,
    union_relevant,
)
from copilot_agent.router import (
    Route,
    Router,
    build_router,
    openai_router_model,
    route_question,
)
from copilot_agent.settings import Settings


@dataclass(frozen=True)
class SubQueryRetrieval:
    """One sub-query's retrieval, as the rest of the turn needs it.

    Not the whole RetrievalResult: its 100 cosine candidates, texts included, are what the
    reranker read, and nothing after retrieval reads them (the merge needs only whether a rerank
    call went out). Kept, they were 213 of the 216 KiB every turn saved to the checkpoint tables
    (experiments/checkpoint_overhead.py, 2026-09-11), and the next turn read them all back
    before its first token, only to reset them.

    The reducer appends them in PLAN order, not in the order the searches finished: LangGraph
    applies the writes of one step's parallel tasks sorted by task path (apply_writes in
    langgraph/pregel/_algo.py), and each Send's path follows its position in the list. The union
    depends on that order when scores tie; test_graph.py pins it with searches that finish
    backwards.
    """

    query: str
    relevant: list[RetrievedChunk]
    mode: RetrievalMode
    # A rerank call reached Cohere: reranked mode, and at least one candidate to rerank.
    reranked: bool
    timings_ms: dict[str, float]

    @classmethod
    def of(cls, query: str, result: RetrievalResult) -> "SubQueryRetrieval":
        return cls(
            query=query,
            relevant=result.relevant,
            mode=result.mode,
            reranked=result.mode == "reranked" and bool(result.candidates),
            timings_ms=result.timings_ms,
        )


@dataclass(frozen=True)
class GenerationMetrics:
    """What the answer cost and how long it took, for the query log (step 2.6).

    ttft_ms is measured from the call to the first token with text, generation_ms to the end of
    the stream. TypeScript takes both from the AI SDK's step.performance; this is the same idea
    measured here, not the same timer. finish_reason goes into the stream's `finish` chunk.
    """

    usage: TokenUsage
    ttft_ms: float | None
    generation_ms: float
    finish_reason: FinishReason


class ChatInput(TypedDict):
    # The new message. The turns before it come from the thread (`turns`), never from the caller.
    question: str


class ChatState(ChatInput, total=False):
    # Kept across turns by the checkpointer: the conversation so far, oldest first. Appended one
    # question + answer pair per completed turn, and bounded to what a request can ever read
    # (history.keep_recent). What the planner and the model see is history.capped_history().
    turns: Annotated[list[HistoryTurn], keep_recent]

    # Per turn. A checkpointed thread starts every turn from the state the last one left, so a
    # key that a turn does not write keeps the previous turn's value: turn_reset() clears the ones
    # that would otherwise leak into the new turn.
    plan: Plan
    # The reducer: each retrieve run appends its result instead of replacing the list. With a
    # checkpointer it would also append to the PREVIOUS turn's results, so turn_reset() empties
    # it first.
    retrievals: Annotated[list[SubQueryRetrieval], operator.add]
    relevant: list[RetrievedChunk]
    mode: RetrievalMode
    rerank_calls: int
    # Step 3.5. None means NO ROUTER RAN, which is not the same as "docs" and must never be read
    # as it: a graph built without a GitHub agent has no router node at all, and a turn that was
    # never routed is a different event from one routed to the documentation. db/009 keeps the
    # same distinction in the column.
    route: Route | None
    router_usage: TokenUsage
    # What the GitHub subagent found, or what it failed to find. None on a docs-only turn.
    github: GitHubEvidence | None
    answer: str
    # None on a canned turn, which calls no model.
    generation: GenerationMetrics | None


def turn_reset() -> dict[str, object]:
    """What the first node of a turn writes so that nothing of the previous turn leaks into it.

    Only the keys a turn might read before writing, or might not write at all: `retrievals` (an
    append reducer: the previous turn's chunks would be merged into this answer, and shown as its
    sources) and `generation` (a canned turn writes none, and would report the last answer's
    tokens). Overwrite bypasses the reducer. Every other per-turn key is always written before
    it is read.

    Step 3.5 adds three, and `github` is the one that matters: a docs-only turn does not run the
    subagent, so without this the previous turn's GitHub facts would be in the prompt of a
    question that never asked for them, and cited as this turn's evidence. That is 2.5's bug
    exactly -- the append reducer carrying turn 1's chunks into turn 2 -- in a key that does not
    even need a reducer to do it. `route` and `router_usage` go with it so that a turn that did
    not route says None rather than repeating the last decision and re-reporting its tokens.
    """
    return {
        "retrievals": Overwrite([]),
        "generation": None,
        "github": None,
        "route": None,
        "router_usage": NO_USAGE,
    }


def history_of(state: ChatState) -> list[HistoryTurn]:
    return capped_history(state.get("turns", []), state["question"])


def completed_turn(state: ChatState, answer: str) -> list[HistoryTurn]:
    return [HistoryTurn("user", state["question"]), HistoryTurn("assistant", answer)]


# The compiled graph: state ChatState, no runtime context yet, input ChatInput.
ChatGraph = CompiledStateGraph[ChatState, None, ChatInput, ChatState]


class RetrieveTask(TypedDict):
    """What one Send carries to a retrieve run: not the chat state, just its sub-query."""

    query: str
    embed_text: str


class GitHubTask(TypedDict):
    """What one Send carries to the GitHub wrapper node: the question, and nothing else.

    Deliberately not the chat state. The subagent is pointed at one repository and answers from
    queries alone (spec: not a general text-to-GraphQL system), and handing it retrieval results
    or the planner's sub-queries would give it documentation text to paraphrase -- which is the
    one way it could produce a GitHub-shaped answer that GitHub never said.
    """

    question: str


def retrieve_sends(plan: Plan) -> list[Send]:
    """One retrieve run per sub-query.

    HyDE: embed the hypothetical, or the query when there is none (q.hypothetical || q.query in
    lib/plan.ts); rerank always with the query.
    """
    return [
        Send("retrieve", RetrieveTask(query=q.query, embed_text=q.hypothetical or q.query))
        for q in plan.queries
    ]


def route_after_plan(state: ChatState) -> Literal["canned"] | list[Send]:
    """The planner's gate, as an edge, in the graph with no GitHub path: greeting and off-topic
    go to the canned reply, a search fans out to one retrieve run per sub-query."""
    plan = state["plan"]
    if plan.intent != "search":
        return "canned"
    return retrieve_sends(plan)


def route_after_plan_routed(state: ChatState) -> Literal["canned", "router"]:
    """The same gate in the graph that has a router: a search goes to the router first.

    The planner still decides greeting and off-topic, so the router never sees a message that is
    not a question -- which is why its prompt has no rules about either.
    """
    return "canned" if state["plan"].intent != "search" else "router"


def route_after_router(state: ChatState) -> list[Send]:
    """The router's decision, as a fan-out: retrieval, the subagent, or both in one superstep.

    Never empty, and not defended against being empty: `plan.intent` is "search" to have got
    here, plan_query guarantees a search carries at least one sub-query, and `route` is one of
    three values of which two include the documentation. An unreachable branch here would be a
    claim no test could check (phase 3 finding 9).
    """
    route = state["route"]
    sends: list[Send] = []
    if route != "github":
        sends.extend(retrieve_sends(state["plan"]))
    if route != "docs":
        sends.append(Send("github", GitHubTask(question=state["question"])))
    return sends


def merge_retrievals(retrievals: list[SubQueryRetrieval]) -> dict[str, object]:
    """plannedRetrieve's union, mode and rerank count, over the per-sub-query results."""
    if not retrievals:
        # A github-only turn: the router sent no retrieve run, so there is nothing to union and
        # no Cohere call was made. "skipped" is already what this column and the UI's retrieval
        # pill mean by "retrieval did not happen" (the canned path writes it), and inventing a
        # fourth mode would change the stream's data-retrieval part, which the TypeScript client
        # parses with a strict schema.
        return {"relevant": [], "mode": "skipped", "rerank_calls": 0}
    return {
        "relevant": union_relevant([r.relevant for r in retrievals]),
        # One fallback makes the whole answer a fallback answer: the UI says so.
        "mode": (
            "cosine-fallback"
            if any(r.mode == "cosine-fallback" for r in retrievals)
            else "reranked"
        ),
        # Rerank calls that reached Cohere: an empty candidate list makes none.
        "rerank_calls": sum(r.reranked for r in retrievals),
    }


def build_graph(
    *,
    planner: Planner,
    search: SearchDocs,
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    router: Router | None = None,
    github_agent: GitHubAgent | None = None,
) -> ChatGraph:
    """Wire the nodes around the given services and compile the graph.

    With a checkpointer, every run needs a thread id in its config and continues that thread's
    conversation. Without one, every run is a conversation of its own (the tests of single turns).

    `router` and `github_agent` are both given or neither is (spec decision 11). One without the
    other is refused here rather than degraded into the other shape: a router with nowhere to
    route pays for a decision that cannot be acted on, and a subagent with no router is a
    capability nothing can reach. Both are failures a deployment should hear about at startup,
    which is the one moment anybody is looking.

    The planner and the router are tagged "nostream". LangGraph's "messages" mode reports every
    chat model called inside a node, and would otherwise put their JSON in the answer stream (as
    one message at the end of the call). The tag only hides the output: keeping the requests
    non-streaming is each model's own streaming=False (planner.openai_planner_model,
    router.openai_router_model).

    The subagent's OWN model calls need no tag. Measured, not assumed
    (experiments/subgraph_stream.py, 2026-09-15): with subgraphs left off -- which is how api.py
    streams -- a wrapper node's nested run does not reach the parent's "messages" stream at all.
    Turning subgraphs on would put them there, named by their own node, and the answer fence in
    ui_stream.py would become the only thing between the subagent's query-writing and the user's
    answer bubble. It is off, so there is nothing to fence.
    """
    if (router is None) != (github_agent is None):
        raise ValueError("build_graph needs a router and a github agent together, or neither")
    quiet_planner = planner.with_config(tags=[TAG_NOSTREAM])

    async def plan(state: ChatState) -> dict[str, object]:
        plan = await plan_query(state["question"], history_of(state), planner=quiet_planner)
        return {"plan": plan, **turn_reset()}

    async def canned(state: ChatState) -> dict[str, object]:
        # No retrieval and no model call: the reply is known, and the user's text cannot argue
        # with it. Logged as mode "skipped", like TypeScript. Recorded in the thread like any
        # answer: TypeScript signs canned replies too, so they come back as history there.
        text = GREETING_MESSAGE if state["plan"].intent == "greeting" else REFUSAL_MESSAGE
        return {
            "answer": text,
            "relevant": [],
            "mode": "skipped",
            "rerank_calls": 0,
            "turns": completed_turn(state, text),
        }

    async def retrieve(task: RetrieveTask) -> dict[str, object]:
        result = await search(task["query"], task["embed_text"])
        return {"retrievals": [SubQueryRetrieval.of(task["query"], result)]}

    async def merge(state: ChatState) -> dict[str, object]:
        return merge_retrievals(state["retrievals"])

    async def generate(state: ChatState) -> dict[str, object]:
        messages = generation_messages(
            state["relevant"],
            history_of(state),
            state["question"],
            [q.query for q in state["plan"].queries],
            github=state.get("github"),
        )
        started = time.perf_counter()
        first_token: float | None = None
        answer: AIMessageChunk | None = None
        # astream, not ainvoke, only to time the first token. The tokens reach the client either
        # way: LangGraph's "messages" mode listens to the model's callbacks.
        async for chunk in model.astream(messages):
            if first_token is None and chunk.text:
                first_token = time.perf_counter()
            answer = chunk if answer is None else answer + chunk
        finished = time.perf_counter()

        usage = answer.usage_metadata if answer is not None else None
        # str(): .text is LangChain's TextAccessor, a str subclass; the thread stores plain text.
        text = str(answer.text) if answer is not None else ""
        return {
            "answer": text,
            # Written only here, after the stream ended: a model that fails or is cancelled
            # mid-answer never gets this far, and the thread stays as it was.
            "turns": completed_turn(state, text),
            "generation": GenerationMetrics(
                usage=(
                    TokenUsage(usage["input_tokens"], usage["output_tokens"]) if usage else NO_USAGE
                ),
                ttft_ms=None if first_token is None else (first_token - started) * 1000,
                generation_ms=(finished - started) * 1000,
                finish_reason=finish_reason(answer.response_metadata if answer else {}),
            ),
        }

    builder = StateGraph(ChatState, input_schema=ChatInput)
    builder.add_node("plan", plan)
    builder.add_node("canned", canned)
    builder.add_node("retrieve", retrieve)
    builder.add_node("merge", merge)
    builder.add_node("generate", generate)
    builder.add_edge(START, "plan")

    if router is not None and github_agent is not None:
        # Bound here rather than read from the enclosing scope inside the nodes: these two exist
        # only on this branch, and a closure that has to re-check that at run time is a branch
        # no test can reach.
        quiet_router = router.with_config(tags=[TAG_NOSTREAM])
        agent = github_agent

        async def route(state: ChatState) -> dict[str, object]:
            routing = await route_question(
                state["question"], history_of(state), router=quiet_router
            )
            return {"route": routing.route, "router_usage": routing.usage}

        async def github(task: GitHubTask) -> dict[str, object]:
            """Run the subagent and translate its state into this one's.

            That translation is the whole node. Attaching the compiled subgraph in its place
            would drop the write, silently, because its output key is not one this schema
            declares -- see the module docstring for the run that showed it.

            `.get`, not `[...]`: a subagent that ended without writing a result leaves the turn
            with no GitHub evidence, and generation then answers as if GitHub had nothing, rather
            than raising a KeyError inside an answer stream that is already a 200.
            """
            result = await agent.ainvoke({"question": task["question"]})
            return {"github": result.get("result")}

        builder.add_node("router", route)
        builder.add_node("github", github)
        builder.add_conditional_edges("plan", route_after_plan_routed, ["canned", "router"])
        builder.add_conditional_edges("router", route_after_router, ["retrieve", "github"])
        builder.add_edge("github", "merge")
    else:
        builder.add_conditional_edges("plan", route_after_plan, ["canned", "retrieve"])

    builder.add_edge("retrieve", "merge")
    builder.add_edge("merge", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("canned", END)
    return builder.compile(checkpointer=checkpointer)


def openai_chat_graph(
    settings: Settings,
    search: SearchDocs,
    checkpointer: BaseCheckpointSaver | None = None,
    github_agent: GitHubAgent | None = None,
) -> ChatGraph:
    """The production graph: the OpenAI planner and answer model around the given search.

    Built once at startup (the API's lifespan, chat_cli) and shared by every run.

    The router is built here, and only when there is a subagent for it to route to: the caller
    decides whether GitHub exists at all (api.py, from GITHUB_TOKEN), and this function does not
    read settings to second-guess it. One switch, in one place, is the whole of decision 11.
    """
    return build_graph(
        planner=build_planner(openai_planner_model(settings)),
        search=search,
        model=openai_generation_model(settings),
        checkpointer=checkpointer,
        router=None if github_agent is None else build_router(openai_router_model(settings)),
        github_agent=github_agent,
    )
