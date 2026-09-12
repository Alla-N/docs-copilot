"""Print what actually reached Langfuse for one trace (step 2.7).

    uv run python experiments/trace_check.py <trace id>

Look, do not guess. A dashboard screenshot proves a trace exists; this prints the tree the service
produced, so the claims the code makes can be checked one by one: a span per graph node, the
generation with its model and tokens and time to first token, the session and tags on the trace,
the `context` span with the chunks the prompt was built from, and no secret anywhere in it.

Spans leave on a batch exporter and Langfuse ingests them asynchronously, so this waits a little
for them rather than reporting an empty trace that is merely late.

Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL through Settings, the same
three the service reads. Free: the API is not billed and nothing here calls a model.
"""

import argparse
import json
import sys
import time
from typing import Any

from langfuse import Langfuse

from copilot_agent.settings import get_settings

WAIT_S = 30.0
POLL_S = 3.0

# Attributes worth showing per observation type, in the order a reader wants them.
FIELDS = "core,basic,time,io,model,usage"

# NOT parse_io_as_json: the installed client still offers it, the v2 endpoint answers a 400 and
# says input and output are always raw strings now. The generated client is older than the API,
# so a parameter existing in the SDK proves nothing about the service accepting it."


def client() -> Langfuse:
    settings = get_settings()
    if not settings.langfuse_public_key or settings.langfuse_secret_key is None:
        sys.exit("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set (see .env.example)")
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
    )


def observations(langfuse: Langfuse, trace_id: str) -> list[Any]:
    deadline = time.monotonic() + WAIT_S
    while True:
        page = langfuse.api.observations.get_many(trace_id=trace_id, fields=FIELDS, limit=100)
        if page.data or time.monotonic() > deadline:
            return list(page.data)
        print("waiting for the trace to be ingested ...")
        time.sleep(POLL_S)


def ms(observation: Any) -> str:
    start, end = observation.start_time, getattr(observation, "end_time", None)
    if start is None or end is None:
        return "    -   "
    return f"{(end - start).total_seconds() * 1000:7.0f}"


def short(value: Any, width: int = 90) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def as_json(value: Any) -> Any:
    """Observation input and output come back as raw strings; JSON is ours to parse."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace_id", help="the 32 hex characters the service printed or logged")
    args = parser.parse_args()

    found = observations(client(), args.trace_id)
    if not found:
        sys.exit(f"no observations for trace {args.trace_id} after {WAIT_S:.0f}s")

    found.sort(key=lambda o: o.start_time)
    print(f"trace {args.trace_id}: {len(found)} observations\n")
    print(f"{'ms':>7}  {'type':<10} {'name':<24} detail")
    for observation in found:
        detail = []
        usage = getattr(observation, "usage_details", None)
        if observation.model:
            detail.append(observation.model)
        if usage:
            detail.append(json.dumps(usage, ensure_ascii=False))
        if getattr(observation, "completion_start_time", None) and observation.start_time:
            first = (observation.completion_start_time - observation.start_time).total_seconds()
            detail.append(f"first token {first * 1000:.0f} ms")
        if observation.level and observation.level != "DEFAULT":
            detail.append(f"{observation.level} {observation.status_message or ''}".strip())
        print(
            f"{ms(observation)}  {str(observation.type).lower():<10} "
            f"{(observation.name or ''):<24} {'  '.join(detail)}"
        )

    # The two things the service promises and a dashboard glance does not prove.
    root = min(found, key=lambda o: o.start_time)
    print(f"\nsession {getattr(root, 'session_id', None)}   input {short(root.input)}")
    context = [o for o in found if o.name == "context"]
    if context:
        chunks = as_json(context[0].output) or []
        print(f"context {len(chunks)} chunks:")
        for chunk in chunks:
            score = chunk.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "  -  "
            print(f"   {score_text}  {chunk.get('title')}: {short(chunk.get('text'), 70)}")
    else:
        print("context: no span (a canned turn retrieves nothing, so it writes none)")


if __name__ == "__main__":
    main()
