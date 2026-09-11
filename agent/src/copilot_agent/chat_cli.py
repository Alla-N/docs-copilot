"""Run the whole Python pipeline on one question, streaming, against the real services.

    uv run python -m copilot_agent.chat_cli "how do I stream text"
    uv run python -m copilot_agent.chat_cli "and how do I configure it?" \
        --history user "how do I stream text" --history assistant "Use streamText (Source 1)."

Prints each graph step as it finishes (the plan, every retrieval, the merged sources), streams
the answer's tokens as they arrive, then the timings: time to the first token as the CALLER sees
it (planner + retrieval + model), the model's own time to first token, and the total.

Costs what one chat request costs: a planner call, one embedding and one rerank per sub-query,
and the answer (about a cent). Local only, like search_cli: nothing here is a server.
"""

import argparse
import asyncio
import logging
import time
from typing import Any

from copilot_agent.graph import openai_chat_graph
from copilot_agent.planner import HistoryTurn
from copilot_agent.retrieval import open_search
from copilot_agent.settings import get_settings


def show_update(node: str, update: dict[str, Any], started: float) -> None:
    at = f"[{(time.perf_counter() - started) * 1000:6.0f} ms]"
    if node == "plan":
        plan = update["plan"]
        print(f"{at} plan: {plan.intent}  {[q.query for q in plan.queries]}")
    elif node == "retrieve":
        for r in update["retrievals"]:
            stages = "  ".join(f"{k} {v:.0f}" for k, v in r.result.timings_ms.items())
            kept = len(r.result.relevant)
            print(f"{at} retrieve: {r.result.mode}, {kept} kept  ({stages})  {r.query}")
    elif node == "merge":
        print(f"{at} merge: {update['mode']}, {update['rerank_calls']} rerank calls")
        for chunk in update["relevant"]:
            print(f"           {chunk.score:.3f}  {chunk.title}")
    elif node == "canned":
        print(f"{at} canned reply, mode {update['mode']}:\n\n{update['answer']}")


async def run(question: str, history: list[HistoryTurn]) -> None:
    settings = get_settings()
    async with open_search(settings) as search:
        graph = openai_chat_graph(settings, search)
        started = time.perf_counter()
        first_token: float | None = None
        async for part in graph.astream(
            {"question": question, "history": history},
            stream_mode=["updates", "messages"],
            version="v2",
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
                        show_update(node, update, started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument(
        "--history",
        nargs=2,
        action="append",
        default=[],
        metavar=("ROLE", "TEXT"),
        help="an earlier turn (user or assistant); repeat for more",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    history = [HistoryTurn(role, text) for role, text in args.history]
    asyncio.run(run(args.question, history))


if __name__ == "__main__":
    main()
