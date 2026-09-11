"""The chat pipeline as a LangGraph graph: plan, retrieve every sub-query, merge, generate.

The Python counterpart of the middle of app/api/chat/route.ts (plannedRetrieve in lib/plan.ts,
then the canned reply or the grounded streamText call), with the same behaviour:

    START -> plan --route--> canned -> END                      greeting, off-topic
                        \\--> retrieve x N -> merge -> generate -> END

  - a greeting or an off-topic message costs one planner call and nothing else;
  - every sub-query is retrieved in parallel (TypeScript: Promise.all; here: one Send per
    sub-query, so each retrieval is its own node run, and its own span once Langfuse traces it);
  - the answer is generated for the planner's resolved sub-queries (invariant 4).

The graph knows nothing about HTTP or the AI SDK. It emits LangGraph stream events: "updates"
when a node finishes (the plan, each retrieval, the merged sources, the canned or generated
answer) and "messages" for the answer's tokens. ui_stream.py turns those into the AI SDK's UI
message stream for POST /chat (api.py).

Dependencies come in through build_graph() and are captured by the node functions (a closure):
the planner, the search function and the chat model are built once at startup and shared by
every run. Per-request values (a thread id in 2.5, trace ids in 2.7) belong in LangGraph's
runtime context instead.
"""

import operator
import time
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from copilot_agent.generation import (
    REFUSAL_MESSAGE,
    FinishReason,
    finish_reason,
    generation_messages,
    openai_generation_model,
)
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
from copilot_agent.retrieval import (
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
    SearchDocs,
    union_relevant,
)
from copilot_agent.settings import Settings


@dataclass(frozen=True)
class SubQueryRetrieval:
    """One sub-query's retrieval.

    The reducer appends them in PLAN order, not in the order the searches finished: LangGraph
    applies the writes of one step's parallel tasks sorted by task path (apply_writes in
    langgraph/pregel/_algo.py), and each Send's path follows its position in the list. The union
    depends on that order when scores tie; test_graph.py pins it with searches that finish
    backwards.
    """

    query: str
    result: RetrievalResult


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
    question: str
    # The turns before the question, as in TypeScript: text only, user and assistant only.
    history: list[HistoryTurn]


class ChatState(ChatInput, total=False):
    plan: Plan
    # The reducer: each retrieve run appends its result instead of replacing the list.
    retrievals: Annotated[list[SubQueryRetrieval], operator.add]
    relevant: list[RetrievedChunk]
    mode: RetrievalMode
    rerank_calls: int
    answer: str
    generation: GenerationMetrics


# The compiled graph: state ChatState, no runtime context yet, input ChatInput.
ChatGraph = CompiledStateGraph[ChatState, None, ChatInput, ChatState]


class RetrieveTask(TypedDict):
    """What one Send carries to a retrieve run: not the chat state, just its sub-query."""

    query: str
    embed_text: str


def route_after_plan(state: ChatState) -> Literal["canned"] | list[Send]:
    """The planner's gate, as an edge: greeting and off-topic go to the canned reply, a search
    fans out to one retrieve run per sub-query."""
    plan = state["plan"]
    if plan.intent != "search":
        return "canned"
    return [
        # HyDE: embed the hypothetical, or the query when there is none (q.hypothetical ||
        # q.query in lib/plan.ts); rerank always with the query.
        Send("retrieve", RetrieveTask(query=q.query, embed_text=q.hypothetical or q.query))
        for q in plan.queries
    ]


def merge_retrievals(retrievals: list[SubQueryRetrieval]) -> dict[str, object]:
    """plannedRetrieve's union, mode and rerank count, over the per-sub-query results."""
    results = [r.result for r in retrievals]
    return {
        "relevant": union_relevant([r.relevant for r in results]),
        # One fallback makes the whole answer a fallback answer: the UI says so.
        "mode": (
            "cosine-fallback" if any(r.mode == "cosine-fallback" for r in results) else "reranked"
        ),
        # Rerank calls that reached Cohere: an empty candidate list makes none.
        "rerank_calls": sum(r.mode == "reranked" and bool(r.candidates) for r in results),
    }


def build_graph(*, planner: Planner, search: SearchDocs, model: BaseChatModel) -> ChatGraph:
    """Wire the nodes around the given services and compile the graph.

    The planner is tagged "nostream". LangGraph's "messages" mode reports every chat model
    called inside a node, and would otherwise put the planner's JSON in the answer stream (as
    one message at the end of the call). The tag only hides the output: keeping the planner's
    request non-streaming is the model's own streaming=False (planner.openai_planner_model).
    """
    quiet_planner = planner.with_config(tags=[TAG_NOSTREAM])

    async def plan(state: ChatState) -> dict[str, object]:
        return {
            "plan": await plan_query(state["question"], state["history"], planner=quiet_planner)
        }

    async def canned(state: ChatState) -> dict[str, object]:
        # No retrieval and no model call: the reply is known, and the user's text cannot argue
        # with it. Logged as mode "skipped", like TypeScript.
        text = GREETING_MESSAGE if state["plan"].intent == "greeting" else REFUSAL_MESSAGE
        return {"answer": text, "relevant": [], "mode": "skipped", "rerank_calls": 0}

    async def retrieve(task: RetrieveTask) -> dict[str, object]:
        result = await search(task["query"], task["embed_text"])
        return {"retrievals": [SubQueryRetrieval(task["query"], result)]}

    async def merge(state: ChatState) -> dict[str, object]:
        return merge_retrievals(state["retrievals"])

    async def generate(state: ChatState) -> dict[str, object]:
        messages = generation_messages(
            state["relevant"],
            state["history"],
            state["question"],
            [q.query for q in state["plan"].queries],
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
        return {
            "answer": answer.text if answer is not None else "",
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
    builder.add_conditional_edges("plan", route_after_plan, ["canned", "retrieve"])
    builder.add_edge("retrieve", "merge")
    builder.add_edge("merge", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("canned", END)
    return builder.compile()


def openai_chat_graph(settings: Settings, search: SearchDocs) -> ChatGraph:
    """The production graph: the OpenAI planner and answer model around the given search.

    Built once at startup (the API's lifespan, chat_cli) and shared by every run.
    """
    return build_graph(
        planner=build_planner(openai_planner_model(settings)),
        search=search,
        model=openai_generation_model(settings),
    )
