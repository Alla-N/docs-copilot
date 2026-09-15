"""What `rateLimit(dryRun: true)` actually does. Three requests, at most three points, no money.

    cd agent && uv run python experiments/dry_run_semantics.py

**Why this exists.** Spec decision 10 builds the budget gate on GitHub computing a query's `cost`
and `nodeCount` before anything is spent. The entire published description of that mechanism is
one sentence in the Meta reference: dryRun "calculate[s] the cost for the query without
evaluating it." Two readings fit that sentence and they lead to different runners:

  - **It suppresses evaluation.** The pre-flight is its own request that returns no data, and the
    runner makes two round trips per query: cost first, then the real call. That is what the spec
    assumed and priced in.
  - **It only avoids the charge.** Then a spliced `rateLimit(dryRun: true)` reports the cost of a
    query that already ran, which is an invoice and not a gate, and decision 10 has to be
    hand-rolled by counting nodes ourselves -- our approximation of GitHub's arithmetic, which is
    exactly what decision 2 chose dryRun to avoid.

Reading documentation harder will not settle it; this project has been wrong four times in one
phase by treating a sentence as a measurement (`--cpu-architecture` absent from the reference and
present in the installed CLI, and the whole of 2b finding 6). So: ask the API.

Three questions, three requests:

  1. does a query carrying `rateLimit(dryRun: true)` still return its sibling data?
  2. what cost and nodeCount does it report?
  3. did `remaining` move afterwards?

Question 1 is the one that decides the design. Questions 2 and 3 are what turn the answer into a
number that can go in the README.
"""

import asyncio
import json

import httpx

from copilot_agent.github_schema import GITHUB_GRAPHQL_URL, USER_AGENT
from copilot_agent.settings import get_settings

REMAINING = """
query Remaining {
  rateLimit { remaining used resetAt }
}
"""

# A real, cheap, single-repo query with a connection in it, so the cost arithmetic has something
# to chew on. The same shape the subagent will emit.
DRY_RUN = """
query DryRun {
  rateLimit(dryRun: true) { cost nodeCount remaining }
  repository(owner: "vercel", name: "ai") {
    name
    releases(first: 5) { nodes { tagName } }
  }
}
"""

WET_RUN = """
query WetRun {
  rateLimit { cost nodeCount remaining }
  repository(owner: "vercel", name: "ai") {
    name
    releases(first: 5) { nodes { tagName } }
  }
}
"""


async def run(client: httpx.AsyncClient, token: str, query: str) -> dict:
    response = await client.post(
        GITHUB_GRAPHQL_URL,
        json={"query": query},
        headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def describe(label: str, payload: dict) -> None:
    print(f"--- {label} ---")
    print(json.dumps(payload, indent=2)[:1200])
    print()


async def main() -> None:
    settings = get_settings()
    if settings.github_token is None:
        raise SystemExit("GITHUB_TOKEN is not set.")
    token = settings.github_token.get_secret_value()

    async with httpx.AsyncClient() as client:
        before = await run(client, token, REMAINING)
        dry = await run(client, token, DRY_RUN)
        after = await run(client, token, REMAINING)
        wet = await run(client, token, WET_RUN)

    describe("remaining, before", before)
    describe("dryRun: true", dry)
    describe("remaining, after the dry run", after)
    describe("dryRun absent, the same query for comparison", wet)

    # The verdict, stated rather than left for the reader to infer from four JSON blobs. A
    # printout that makes the reader do the comparison is how a run gets read as whatever the
    # reader expected. Every line below is computed from the responses; the first version of
    # this block ended with the parenthetical "a plain rateLimit query costs 1 itself", which
    # was not measured and is not true. An asserted aside inside a measurement is still an
    # assertion.
    def rate(payload: dict) -> dict:
        return (payload.get("data") or {}).get("rateLimit") or {}

    dry_data = dry.get("data") or {}
    print("=== verdict ===")
    print(f"sibling data returned under dryRun: {dry_data.get('repository') is not None}")
    print(f"keys in the dryRun response data:   {sorted(dry_data)}")
    print()
    print(f"{'':<28} {'cost':>6} {'nodeCount':>10} {'remaining':>10} {'used':>6}")
    for label, payload in (
        ("rateLimit only, before", before),
        ("dryRun: true", dry),
        ("rateLimit only, after", after),
        ("the same query, for real", wet),
    ):
        r = rate(payload)
        print(
            f"{label:<28} {r.get('cost')!s:>6} {r.get('nodeCount')!s:>10} "
            f"{r.get('remaining')!s:>10} {r.get('used')!s:>6}"
        )
    print()
    dry_rate, wet_rate = rate(dry), rate(wet)
    if dry_rate.get("cost") is not None and wet_rate.get("cost") is not None:
        exact = (dry_rate["cost"], dry_rate["nodeCount"]) == (
            wet_rate["cost"],
            wet_rate["nodeCount"],
        )
        print(f"dry run predicted the real cost exactly: {exact}")
    r_before, r_after = rate(before).get("remaining"), rate(after).get("remaining")
    if r_before is not None and r_after is not None:
        print(f"points spent by the dry run and the two probes around it: {r_before - r_after}")
    if dry.get("errors"):
        print(f"errors on the dry run: {dry['errors']}")


if __name__ == "__main__":
    asyncio.run(main())
