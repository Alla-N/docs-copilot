"""Run search_docs by hand against the real services.

    uv run python -m copilot_agent.search_cli "how do I stream text"
    uv run python -m copilot_agent.search_cli "what is tool calling" --embed-text "Tools let..."

Prints the mode, per-stage timings, the top cosine candidates (what the reranker was
handed) and the chunks that survived rerank + threshold (what the model would see).
"""

import argparse
import asyncio
import logging

from copilot_agent.retrieval import RetrievalResult, open_search
from copilot_agent.settings import get_settings


async def run(query: str, embed_text: str | None) -> RetrievalResult:
    async with open_search(get_settings()) as search:
        return await search(query, embed_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--embed-text", help="embed this instead of the query (HyDE)")
    parser.add_argument("--candidates", type=int, default=5, help="cosine candidates to show")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    result = asyncio.run(run(args.query, args.embed_text))

    total = sum(result.timings_ms.values())
    stages = "  ".join(f"{k} {v:.0f}ms" for k, v in result.timings_ms.items())
    print(f"mode {result.mode}   total {total:.0f}ms   ({stages})")
    print(f"\ncosine top {args.candidates} of {len(result.candidates)}:")
    for c in result.candidates[: args.candidates]:
        print(f"  {c.similarity:.3f}  {c.title}")
    print(f"\nrelevant ({len(result.relevant)}):")
    for chunk in result.relevant:
        print(f"  {chunk.score:.3f}  {chunk.title}  {chunk.source_url}")


if __name__ == "__main__":
    main()
