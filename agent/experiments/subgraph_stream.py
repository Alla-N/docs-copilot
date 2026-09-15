"""What does nesting a subgraph do to the parent's stream, and to the parent's checkpoint?

    cd agent && uv run python experiments/subgraph_stream.py

Free, instant fakes, no network. Written before 3.5 is built, because 3.4 finding 8 is that
three API facts went out in a build unmeasured and were right by luck. These are the facts 3.5
would otherwise guess:

  1. How a subgraph's steps appear in the PARENT's stream_mode=["updates", "messages"] stream,
     with subgraphs=False (the default, what api.py does today) and with subgraphs=True.
  2. What exactly changes shape when subgraphs=True is set. ui_stream.py reads part["type"] and
     part["data"], and query_log.observe reads the same two keys, on EVERY path including the
     docs-only one. If the chunk becomes a (namespace, chunk) pair, both have to change.
  3. Whether tokens streamed by a model called INSIDE the subgraph reach the parent's "messages"
     stream, and what langgraph_node the metadata reports for them. ui_stream fences the answer
     on langgraph_node == "generate"; if a nested model's chunks arrive naming their own node,
     the fence holds, and if they arrive naming the subgraph's node in the parent, it does not.
  4. Whether a compiled subgraph can be attached as a node DIRECTLY when its state schema is not
     the parent's, or whether it needs a wrapper function that calls ainvoke.
  5. Whether checkpointer=False keeps the subgraph's own channels (messages, steps) out of the
     parent's checkpoint, which is what spec decision 15 claims.

Nothing here is asserted in prose. Every line of the output is printed from the run.
"""

import asyncio
import itertools
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

CHILD_REPLY = "child tokens here"
PARENT_REPLY = "parent tokens here"


# ---- the child: a tiny stand-in for github_agent.py -------------------------------------------


class ChildInput(TypedDict):
    question: str


class ChildState(ChildInput, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: Annotated[list[str], operator.add]
    result: str


def build_child(*, checkpointer: Any) -> Any:
    model = GenericFakeChatModel(messages=itertools.cycle([CHILD_REPLY]))

    async def explore(state: ChildState) -> dict[str, object]:
        reply = await model.ainvoke(state["question"])
        return {"messages": [reply], "steps": ["explore"]}

    async def summarise(state: ChildState) -> dict[str, object]:
        return {"steps": ["summarise"], "result": f"evidence for {state['question']}"}

    builder = StateGraph(ChildState, input_schema=ChildInput)
    builder.add_node("explore", explore)
    builder.add_node("summarise", summarise)
    builder.add_edge(START, "explore")
    builder.add_edge("explore", "summarise")
    builder.add_edge("summarise", END)
    return builder.compile(checkpointer=checkpointer)


# ---- the parent: a tiny stand-in for graph.py -------------------------------------------------


class ParentInput(TypedDict):
    question: str


class ParentState(ParentInput, total=False):
    plan: str
    evidence: str
    answer: str


def build_parent(*, attach: str, checkpointer: Any, child_checkpointer: Any) -> Any:
    """attach: "wrapper" calls child.ainvoke inside a node; "direct" adds the compiled child."""
    child = build_child(checkpointer=child_checkpointer)
    model = GenericFakeChatModel(messages=itertools.cycle([PARENT_REPLY]))

    async def plan(state: ParentState) -> dict[str, object]:
        return {"plan": "search"}

    async def github_wrapper(state: ParentState) -> dict[str, object]:
        out = await child.ainvoke({"question": state["question"]})
        return {"evidence": out["result"]}

    async def generate(state: ParentState) -> dict[str, object]:
        reply = await model.ainvoke(state.get("evidence", ""))
        return {"answer": str(reply.text)}

    builder = StateGraph(ParentState, input_schema=ParentInput)
    builder.add_node("plan", plan)
    builder.add_node("github", github_wrapper if attach == "wrapper" else child)
    builder.add_node("generate", generate)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "github")
    builder.add_edge("github", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=checkpointer)


# ---- reporting --------------------------------------------------------------------------------


def describe(chunk: Any) -> str:
    """One stream chunk, described by what it IS rather than by what it was expected to be."""
    kind = type(chunk).__name__
    if isinstance(chunk, tuple):
        parts = ", ".join(describe(item) for item in chunk)
        return f"{kind}(len {len(chunk)}) [{parts}]"
    if isinstance(chunk, dict):
        keys = sorted(chunk)
        head = f"{kind} keys={keys}"
        mode = chunk.get("type")
        if mode == "updates":
            nodes = sorted(chunk.get("data", {}))
            return f"{head} type=updates nodes={nodes}"
        if mode == "messages":
            data = chunk.get("data")
            node = None
            text = None
            if isinstance(data, tuple) and len(data) == 2:
                message, metadata = data
                node = (metadata or {}).get("langgraph_node")
                ns = (metadata or {}).get("langgraph_checkpoint_ns")
                text = getattr(message, "text", None)
                text = str(text) if text else ""
                return f"{head} type=messages langgraph_node={node!r} ns={ns!r} text={text!r}"
            return f"{head} type=messages data={type(data).__name__}"
        return head
    return f"{kind} {chunk!r}"


async def stream_once(graph: Any, *, subgraphs: bool, config: dict | None) -> list[str]:
    seen: list[str] = []
    stream = graph.astream(
        {"question": "who merged PR 1"},
        config,
        stream_mode=["updates", "messages"],
        version="v2",
        subgraphs=subgraphs,
    )
    async for chunk in stream:
        seen.append(describe(chunk))
    return seen


async def variant(name: str, *, attach: str, subgraphs: bool) -> None:
    print(f"\n=== {name} (attach={attach}, subgraphs={subgraphs}) ===")
    try:
        graph = build_parent(attach=attach, checkpointer=None, child_checkpointer=False)
        for line in await stream_once(graph, subgraphs=subgraphs, config=None):
            print("  " + line)
    except Exception as error:  # the failure IS the result here
        print(f"  RAISED {type(error).__name__}: {error}")


async def checkpoint_probe(attach: str, child_checkpointer: Any) -> None:
    """What the parent's checkpoint holds, and whether the parent got the child's result.

    child_checkpointer is run BOTH ways on purpose. Spec decision 15 says False keeps the child's
    channels out of the parent's checkpoint where None would inherit the parent's saver and put
    them in. Running only False would confirm the first half and leave the second half an
    assumption -- and a decision that turns out to change nothing should be said to change
    nothing, which is the same rule the spec applies to its own points budget under P3.
    """
    name = "False" if child_checkpointer is False else repr(child_checkpointer)
    print(f"\n=== parent checkpoint (attach={attach}, child checkpointer={name}) ===")
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t1"}}
    try:
        graph = build_parent(
            attach=attach, checkpointer=saver, child_checkpointer=child_checkpointer
        )
        await graph.ainvoke({"question": "who merged PR 1"}, config)
        snapshot = await graph.aget_state(config)
        # Printed as its own line rather than left to be spotted in a key list: whether the
        # parent RECEIVED the child's output is the question, and a missing key in a sorted
        # list is exactly the kind of nothing that reads like something.
        print(f"  parent evidence: {snapshot.values.get('evidence', 'MISSING')!r}")
        print(f"  parent answer:   {snapshot.values.get('answer', 'MISSING')!r}")
        print(f"  parent state keys: {sorted(snapshot.values)}")
        tuples = [t async for t in saver.alist(config)]
        print(f"  checkpoints written: {len(tuples)}")
        written: set[str] = set()
        for item in tuples:
            written |= set(item.checkpoint.get("channel_values", {}))
        print(f"  channel_values seen across all checkpoints: {sorted(written)}")
        leaked = sorted(written & {"messages", "steps", "result"})
        print(f"  child-only channels in the parent checkpoint: {leaked or 'none'}")
    except Exception as error:
        print(f"  RAISED {type(error).__name__}: {error}")


async def main() -> None:
    from importlib.metadata import version

    print(f"langgraph {version('langgraph')}, langchain-core {version('langchain-core')}")
    await variant("A", attach="wrapper", subgraphs=False)
    await variant("B", attach="wrapper", subgraphs=True)
    await variant("C", attach="direct", subgraphs=False)
    await variant("D", attach="direct", subgraphs=True)
    for attach in ("wrapper", "direct"):
        for child_checkpointer in (False, None):
            await checkpoint_probe(attach, child_checkpointer)


if __name__ == "__main__":
    asyncio.run(main())
