"""Run the whole Python pipeline on one question, streaming, against the real services.

    uv run python -m copilot_agent.chat_cli "how do I stream text"
    uv run python -m copilot_agent.chat_cli "and how do I configure it?" --thread <id it printed>

Each run is one turn of a conversation kept in Postgres by the checkpointer (checkpoint.py; the
tables must be set up). Without --thread it starts a new conversation and prints its id; pass
that id to continue it, the way the chat UI's follow-ups do.

Prints each graph step as it finishes (the plan, every retrieval, the merged sources), streams
the answer's tokens as they arrive, then the timings: time to the first token as the CALLER sees
it (planner + retrieval + model), the model's own time to first token, and the total.

Costs what one chat request costs: a planner call, one embedding and one rerank per sub-query,
and the answer (about a cent). Local only, like search_cli: nothing here is a server.

With the Langfuse keys set (step 2.7) each run is a trace, in the same session as the rest of the
thread and tagged `cli`, and the run prints its trace id. This is the shortest way to look at a
run: no server, no route, one command.
"""

import argparse
import asyncio
import logging
import secrets
import time
from typing import Any

from copilot_agent.checkpoint import open_checkpointer
from copilot_agent.graph import openai_chat_graph
from copilot_agent.retrieval import open_search
from copilot_agent.settings import get_settings
from copilot_agent.tracing import open_tracing


def show_update(node: str, update: dict[str, Any], started: float) -> None:
    at = f"[{(time.perf_counter() - started) * 1000:6.0f} ms]"
    if node == "plan":
        plan = update["plan"]
        print(f"{at} plan: {plan.intent}  {[q.query for q in plan.queries]}")
    elif node == "retrieve":
        for r in update["retrievals"]:
            stages = "  ".join(f"{k} {v:.0f}" for k, v in r.timings_ms.items())
            print(f"{at} retrieve: {r.mode}, {len(r.relevant)} kept  ({stages})  {r.query}")
    elif node == "merge":
        print(f"{at} merge: {update['mode']}, {update['rerank_calls']} rerank calls")
        for chunk in update["relevant"]:
            print(f"           {chunk.score:.3f}  {chunk.title}")
    elif node == "canned":
        print(f"{at} canned reply, mode {update['mode']}:\n\n{update['answer']}")


async def run(question: str, thread_id: str) -> None:
    settings = get_settings()
    async with (
        open_tracing(settings) as tracing,
        open_search(settings) as search,
        open_checkpointer(settings) as saver,
    ):
        graph = openai_chat_graph(settings, search, saver)
        trace_id = tracing.new_trace_id()
        if trace_id:
            print(f"trace {trace_id}")
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            **tracing.run_config(trace_id=trace_id, session_id=thread_id, tags=["cli"]),
        }
        before = (await graph.aget_state(config)).values.get("turns", [])
        print(f"thread {thread_id}: {len(before)} earlier messages")
        started = time.perf_counter()
        first_token: float | None = None
        async for part in graph.astream(
            {"question": question},
            config,
            stream_mode=["updates", "messages"],
            version="v2",
            durability=settings.checkpoint_durability,
        ):
            if part["type"] == "messages":
                message, metadata = part["data"]
                if metadata["langgraph_node"] == "generate" and message.text:
                    if first_token is None:
                        first_token = time.perf_counter()
                        print()
                    print(message.text, end="", flush=True)
            else:
                for node, update in part["data"].items():
                    if node == "generate":
                        metrics = update["generation"]
                        total = (time.perf_counter() - started) * 1000
                        caller_ttft = (first_token - started) * 1000 if first_token else None
                        print("\n")
                        print(f"first token (caller)  {caller_ttft:.0f} ms" if caller_ttft else "")
                        print(f"first token (model)   {metrics.ttft_ms:.0f} ms")
                        print(f"answer                {metrics.generation_ms:.0f} ms")
                        print(f"total                 {total:.0f} ms")
                        print(f"answer tokens         {metrics.usage}")
                    else:
                        # The same grounding span POST /chat writes (tracing.record_context):
                        # without it a CLI trace shows the tree but not what the model was shown.
                        if node == "merge":
                            tracing.record_context(trace_id, update["relevant"])
                        show_update(node, update, started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument(
        "--thread", help="continue this conversation (the id an earlier run printed)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run(args.question, args.thread or "cli-" + secrets.token_hex(8)))


if __name__ == "__main__":
    main()
