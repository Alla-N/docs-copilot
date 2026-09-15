"""Two shapes the 3.6 baseline produced that nobody here has actually looked at.

Run on purpose, against the real API. Free: two queries, one point each by the arithmetic of 3.3.

    cd agent && uv run python experiments/github_error_shapes.py 2>&1 \
        | tee "../Claude outputs/github-error-shapes.txt"

**Question 1: what type does GitHub put on an over-large `first`?** In the baseline the subagent
asked for `pullRequests(first: 500)` and GitHub refused with "Requesting 500 records on the
pullRequests connection exceeds the first limit of 100 records." The turn then stopped after ONE
attempt, so `is_repairable` said no, so the type is not in REPAIRABLE_ERROR_TYPES. That is a
repairable error by any reading -- the fix is a smaller number, and the model is holding the pen
-- and the repair cap of 2 sat unused. The allow-list carries provenance per entry precisely so
that nothing is added to it from a guess, so this prints the raw error object.

**Question 2: is the ai@5.0.0 tag annotated or lightweight?** evals/github-cases.ts currently
says lightweight, and that was an inference, not a measurement: the freeze run asked
`object(expression: "ai@5.0.0")` and got `__typename: Commit`, and I read a peeled result as a
direct one. The baseline then contradicted it -- the subagent asked
`ref(qualifiedName: "refs/tags/ai@5.0.0") { target { ... on Commit { oid } } }` and got back
`"target": {}`, an inline fragment matching nothing, which happens when the target is a Tag. Both
observations fit exactly one story: the tag is ANNOTATED, `ref.target` is the Tag object and
`object(expression:)` peels it to the commit. This settles it by asking for `__typename` at both
ends instead of by reasoning about it.

That empty `{}` is worth its own line in the write-up whichever way this comes out: a query that
parsed, validated, passed the pre-flight, cost its point and returned an object with nothing in
it, and every counter on the turn reported success.
"""

import asyncio
import json

import httpx

from copilot_agent.github_schema import GITHUB_GRAPHQL_URL, github_headers
from copilot_agent.settings import get_settings

OVER_LARGE_FIRST = """
query OverLargeFirst($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 500) { nodes { number } }
  }
}
"""

TAG_SHAPE = """
query TagShape($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    byRef: ref(qualifiedName: "refs/tags/ai@5.0.0") {
      name
      target {
        __typename
        oid
        ... on Tag { message target { __typename oid } }
      }
    }
    byExpression: object(expression: "ai@5.0.0") { __typename oid }
    byRelease: release(tagName: "ai@5.0.0") { tagName publishedAt tagCommit { oid } }
  }
}
"""


async def ask(client: httpx.AsyncClient, token: str, query: str) -> dict:
    response = await client.post(
        GITHUB_GRAPHQL_URL,
        headers=github_headers(token),
        json={"query": query, "variables": {"owner": "vercel", "name": "ai"}},
    )
    print(f"HTTP {response.status_code}")
    return response.json()


async def main() -> None:
    token = get_settings().github_token
    if token is None:
        raise SystemExit("GITHUB_TOKEN is not set; this experiment only runs against the real API")
    secret = token.get_secret_value()

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("---- 1. pullRequests(first: 500): the error object, in full ----")
        body = await ask(client, secret, OVER_LARGE_FIRST)
        for error in body.get("errors") or []:
            print(json.dumps(error, indent=2, ensure_ascii=False))
        types = [e.get("type") for e in body.get("errors") or []]
        paths = ["path" in e for e in body.get("errors") or []]
        print(f"types: {types}   has a path: {paths}")
        print(f"data was returned: {body.get('data') is not None}")

        print()
        print("---- 2. the tag, asked three ways ----")
        body = await ask(client, secret, TAG_SHAPE)
        for error in body.get("errors") or []:
            print(f"error: {error.get('type')} at {error.get('path')}: {error.get('message')}")
        print(json.dumps((body.get("data") or {}).get("repository"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
