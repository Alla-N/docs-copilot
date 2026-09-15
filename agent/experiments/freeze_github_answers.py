"""Freeze the answers for the 3.6 labelled set, from the repository itself.

Run on purpose, against the real API. Free: one query, and by the rate-limit arithmetic of 3.3
one point.

    cd agent && uv run python experiments/freeze_github_answers.py 2>&1 \
        | tee "../Claude outputs/freeze-3.6.txt"

**Why this exists.** 3.6 measures answer accuracy against about twelve frozen literals. A literal
that came from memory rather than from the repository would make the metric agree with whatever
the model already believes, which is the one failure mode this phase keeps finding. So every
literal in `GITHUB_CASES` is copied out of this run's output, and this file is the provenance:
the question set is a query, the answers are its result, and the date they were frozen is at the
top of the output.

**Why one query and not twelve.** The anchors are all fields of one repository, so aliases put
them in a single document: one request, one cost, and a single point in time for every answer.
It also exercises the thing 3.5b's finding is about from the other side. A field-level failure
(an issue or a pull request number that does not exist) comes back as a null in `data` next to an
entry in `errors`, and the run keeps going -- which is exactly the split `github_query.py` sorts
by `path`. Anchors that come back null are not in the labelled set.

**What makes an anchor usable.** Two properties, and both are load-bearing:

  - *Frozen.* The answer cannot change. A publication date, a merge date, the author of an
    issue, the commit a tag points at: all settled. Star counts, open-issue counts and "the
    latest release" are not, and none of them is here.
  - *Unguessable, and not a side effect of a listing.* 3.5b watched the subagent answer "when was
    ai 5.0.0 released" by listing the ten most recently created releases: valid, well formed, and
    with no answer in it. A frozen literal a generic listing could accidentally contain would
    score that run as correct. Dates from years ago, a commit oid and a login cannot appear in
    such a listing by accident.

The oldest-releases block is the deliberate opposite: it can only be answered by ordering
ascending, so a model that reaches for the default listing gets the wrong end of the repository.
"""

import asyncio
import json
from datetime import UTC, datetime

import httpx

from copilot_agent.github_schema import GITHUB_GRAPHQL_URL, github_headers
from copilot_agent.settings import get_settings

OWNER = "vercel"
NAME = "ai"

# Aliases are the case ids in waiting: whatever this returns is what the labelled set can ask
# about. More anchors than the set needs, because some of them will come back null.
FREEZE_QUERY = """
query Freeze($owner: String!, $name: String!) {
  rateLimit { cost nodeCount remaining resetAt }
  repository(owner: $owner, name: $name) {
    createdAt
    licenseInfo { spdxId }
    defaultBranchRef { name }
    ai5: release(tagName: "ai@5.0.0") { tagName name publishedAt createdAt isPrerelease }
    ai4: release(tagName: "ai@4.0.0") { tagName name publishedAt }
    ai3: release(tagName: "ai@3.0.0") { tagName name publishedAt }
    ai2: release(tagName: "ai@2.0.0") { tagName name publishedAt }
    ai1: release(tagName: "ai@1.0.0") { tagName name publishedAt }
    react1: release(tagName: "@ai-sdk/react@1.0.0") { tagName publishedAt }
    oldestReleases: releases(first: 5, orderBy: {field: CREATED_AT, direction: ASC}) {
      nodes { tagName publishedAt }
    }
    newestReleases: releases(first: 5, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes { tagName publishedAt }
    }
    ai5Tag: object(expression: "ai@5.0.0") {
      __typename
      oid
      ... on Tag { target { oid } }
    }
    issue1: issue(number: 1) { number title state createdAt closedAt author { login } }
    issue2: issue(number: 2) { number title state createdAt closedAt author { login } }
    issue50: issue(number: 50) { number title state createdAt closedAt author { login } }
    pr100: pullRequest(number: 100) { number title state merged mergedAt author { login } }
    pr500: pullRequest(number: 500) { number title state merged mergedAt author { login } }
    pr1000: pullRequest(number: 1000) { number title state merged mergedAt author { login } }
  }
}
"""


def show(label: str, value: object) -> None:
    print(f"{label:<18} {json.dumps(value, ensure_ascii=False)}")


async def main() -> None:
    settings = get_settings()
    token = settings.github_token
    if token is None:
        raise SystemExit("GITHUB_TOKEN is not set; this experiment only runs against the real API")

    print(f"frozen at {datetime.now(UTC).isoformat(timespec='seconds')}  repo {OWNER}/{NAME}")
    print(f"endpoint  {GITHUB_GRAPHQL_URL}")
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GITHUB_GRAPHQL_URL,
            headers=github_headers(token.get_secret_value()),
            json={"query": FREEZE_QUERY, "variables": {"owner": OWNER, "name": NAME}},
        )
    print(f"HTTP {response.status_code}")
    body = response.json()

    # Field-level failures are expected and are not a reason to stop: a null anchor is an anchor
    # the labelled set cannot use, which is a result.
    for error in body.get("errors") or []:
        print(f"error at {error.get('path')}: {error.get('message')}")
    print()

    data = body.get("data") or {}
    limit = data.get("rateLimit") or {}
    print(
        f"cost {limit.get('cost')}  nodeCount {limit.get('nodeCount')}  "
        f"remaining {limit.get('remaining')}"
    )
    print()

    repo = data.get("repository") or {}
    if not repo:
        print(json.dumps(body, indent=2, ensure_ascii=False))
        raise SystemExit("no repository in the response; nothing was frozen")

    print("---- anchors, one per line: alias, then what it returned ----")
    for alias, value in repo.items():
        show(alias, value)

    print()
    print("---- raw response, which is the provenance ----")
    print(json.dumps(body, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
