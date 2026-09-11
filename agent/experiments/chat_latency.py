"""How fast is POST /chat once the service is warm? Sequential requests to a running service.

    cd agent
    uv run uvicorn --factory copilot_agent.api:create_app            # in one terminal
    uv run python experiments/chat_latency.py "how do I stream text"  # in another

Costs what N chat requests cost (about a cent each; 5 by default). Reads AGENT_API_KEY from
the same settings as the service. For each request it prints the moments the client sees:
  headers   the 200 and its headers (immediate: /chat streams from the start)
  start     the `start` chunk, sent with the first data part: planner + retrieval done
  ttft      the first text-delta: what a reader waits for
  total     [DONE]
Request 1 pays for cold connections (OpenAI, Cohere, the pooler); the medians are over the
rest, and are what the warm service in production looks like. chat_cli.py measures the same
pipeline cold, in a fresh process each time, which is the number to compare with.
"""

import argparse
import json
import statistics
import time

import httpx

from copilot_agent.settings import get_settings


def one_request(client: httpx.Client, url: str, question: str) -> dict[str, float]:
    marks: dict[str, float] = {}
    started = time.perf_counter()

    def mark(name: str) -> None:
        marks.setdefault(name, (time.perf_counter() - started) * 1000)

    with client.stream("POST", f"{url}/chat", json={"question": question}) as response:
        response.raise_for_status()
        mark("headers")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                mark("total")
                break
            chunk = json.loads(payload)
            if chunk["type"] == "start":
                mark("start")
            elif chunk["type"] == "text-delta":
                mark("ttft")
            elif chunk["type"] == "error":
                raise RuntimeError(f"the stream failed: {chunk}")
    return marks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    key = get_settings().agent_api_key
    if key is None:
        raise SystemExit("AGENT_API_KEY is not set (.env.local)")
    headers = {"authorization": f"Bearer {key.get_secret_value()}"}
    columns = ["headers", "start", "ttft", "total"]
    runs = []
    with httpx.Client(headers=headers, timeout=60, trust_env=False) as client:
        print("run  " + "  ".join(f"{c:>8}" for c in columns) + "   (ms)")
        for i in range(1, args.runs + 1):
            marks = one_request(client, args.url, args.question)
            runs.append(marks)
            print(f"{i:>3}  " + "  ".join(f"{marks.get(c, float('nan')):8.0f}" for c in columns))
    if len(runs) > 1:
        warm = runs[1:]
        medians = [statistics.median(r[c] for r in warm if c in r) for c in columns]
        print("med  " + "  ".join(f"{m:8.0f}" for m in medians) + f"   warm, n={len(warm)}")


if __name__ == "__main__":
    main()
