"""What GitHub's schema costs to hold: bytes on the wire, bytes in the process, and per lookup.

Run on purpose, against the real API. Free: introspection is a single query and, by the
rate-limit arithmetic, one point.

    cd agent && uv run python experiments/github_schema_size.py

**Why this is an experiment and not a note.** Phase 2b decision 8 set the task at
`--cpu 256 --memory 512`, from a measured working set of 145 MiB with about 3.5 times headroom
over the part that can kill a task. A cached GraphQL schema is the first thing phase 3 adds to
that number, and nobody knows what it weighs. If the built schema is 150 MiB of Python objects
then decision 8 is wrong and 2b has to be redeployed at a bigger size, which is a finding worth
having now rather than from an OOM kill on AWS.

Three numbers, because they answer three different questions:

  - **wire bytes** - what the fetch costs and what the disk cache holds;
  - **tracemalloc** - what the Python objects of the parsed schema weigh, attributable to this
    code and to nothing else;
  - **RSS delta** - what the container actually sees, which is the only one an OOM killer reads.
    It is the loosest of the three (the allocator keeps what it frees) and it is the one that
    decides whether 512 MiB still holds.

The per-type outline sizes are the second question: DEFAULT_BYTE_CAP is 6000 and the whole point
of the two reading modes is that an outline of the widest type fits under it. If `Repository`
does not fit, the cap is wrong, not the type.
"""

import asyncio
import json
import resource
import sys
import tracemalloc

import httpx
from graphql import build_client_schema

from copilot_agent.github_schema import (
    DEFAULT_BYTE_CAP,
    describe_type,
    fetch_introspection,
    schema_summary,
)
from copilot_agent.settings import get_settings

# The types this project's questions actually touch. Not a sample: the outline of each of these
# is what a real turn pays for.
TYPES = [
    "Repository",
    "Release",
    "Issue",
    "PullRequest",
    "SearchResultItem",
    "GitObject",
    "RateLimit",
]


def rss_bytes() -> int:
    """Maximum resident set size so far, in bytes.

    ru_maxrss is kilobytes on Linux and bytes on macOS, which is exactly the kind of difference
    that turns a measurement into a wrong number quietly. Both are handled, and the platform is
    printed with the result so the reader can check the conversion rather than trust it.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


async def main() -> None:
    settings = get_settings()
    if settings.github_token is None:
        raise SystemExit("GITHUB_TOKEN is not set; this experiment needs the real schema.")

    async with httpx.AsyncClient() as client:
        data = await fetch_introspection(client, settings.github_token.get_secret_value())

    wire = len(json.dumps(data).encode())

    before_rss = rss_bytes()
    tracemalloc.start()
    schema = build_client_schema(data)
    traced, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after_rss = rss_bytes()

    print(f"platform            {sys.platform}")
    print(f"types in schema     {len(schema.type_map)}")
    print(f"introspection JSON  {wire / 1024 / 1024:.1f} MiB")
    print(f"tracemalloc current {traced / 1024 / 1024:.1f} MiB")
    print(f"tracemalloc peak    {traced_peak / 1024 / 1024:.1f} MiB")
    print(f"RSS before build    {before_rss / 1024 / 1024:.1f} MiB")
    print(f"RSS after build     {after_rss / 1024 / 1024:.1f} MiB")
    print(f"RSS delta           {(after_rss - before_rss) / 1024 / 1024:.1f} MiB")
    print()

    # UNCAPPED as well as capped. A capped size says the outline is over the cap and not by how
    # much, which is the shape of 2b finding 1 all over again: a number that cannot tell you
    # what it does not contain. The cap is set from the uncapped column, not from this one.
    print(f"{'type':<22} {'uncapped':>9} {'capped':>8} {'fields':>7}")
    for name in ["Query", *TYPES]:
        whole = describe_type(schema, name, byte_cap=10**9)
        capped = describe_type(schema, name)
        cut = "  CUT" if "byte cap" in capped else ""
        fields = sum(1 for line in whole.splitlines() if line.startswith("  "))
        print(f"{name:<22} {len(whole.encode()):>9} {len(capped.encode()):>8} {fields:>7}{cut}")

    summary = schema_summary(schema)
    print()
    print(f"schema_summary is {len(summary.encode())} bytes at the {DEFAULT_BYTE_CAP} byte cap")


if __name__ == "__main__":
    asyncio.run(main())
