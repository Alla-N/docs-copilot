"""The runner against the REAL API, because the offline suite tests my beliefs about it.

Marked integration, so the pre-commit hook skips it. Needs GITHUB_TOKEN; costs a handful of
rate-limit points and no money.

    uv run pytest -m integration tests/test_github_query_live.py

Every fixture in tests/test_github_query.py is a shape I asserted GitHub produces: that a dry run
omits its siblings rather than nulling them, that a connection without `first` comes back as a
request-level error with no `path`, that a missing repository comes back as a field-level error
with one. Those fixtures cannot fail when the belief behind them is wrong -- they fail only when
the code stops matching the belief. This file is the other half: it checks the belief.
"""

import httpx
import pytest

from copilot_agent.github_query import GitHubQueries
from copilot_agent.github_schema import SchemaCache, github_fetch
from copilot_agent.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

RELEASES = """
query Releases($owner: String!, $name: String!, $n: Int!) {
  repository(owner: $owner, name: $name) {
    name
    releases(first: $n, orderBy: {field: CREATED_AT, direction: DESC}) { nodes { tagName } }
  }
}
"""

# The GitHub rule that is not in the schema: `first` is declared optional and GitHub rejects a
# connection without it anyway. Prediction P1 says this is the most common first-try failure.
NO_FIRST = """
query NoFirst {
  repository(owner: "vercel", name: "ai") { releases { nodes { tagName } } }
}
"""

MISSING_REPO = """
query Missing {
  repository(owner: "vercel", name: "this-repository-does-not-exist-xyzzy") { name }
}
"""

# Two nested connections: 100 issues each asking for 100 comments. About 10,100 nodes by
# GitHub's own count, which is well inside its 500,000 ceiling -- the test lowers OUR cap
# instead of building a monstrous query, because what is under test is the gate reading
# GitHub's number, not how large a number can be made.
NESTED = """
query Nested {
  repository(owner: "vercel", name: "ai") {
    issues(first: 100) { nodes { comments(first: 100) { nodes { id } } } }
  }
}
"""


@pytest.fixture(scope="module")
async def github_factory():
    """One client and one introspection fetch for the whole module.

    A factory rather than a runner, so a test can lower a cap by constructing its own instead
    of reaching into a private attribute of a shared one.

    Module scope, the widest an async fixture here can be: conftest pins anyio_backend to
    module scope and a fixture cannot outlive it.
    """
    settings = get_settings()
    assert settings.github_token is not None, "GITHUB_TOKEN is not set"
    token = settings.github_token.get_secret_value()
    async with httpx.AsyncClient() as client:
        cache = SchemaCache(fetch=github_fetch(client, token))

        def make(**kwargs) -> GitHubQueries:
            return GitHubQueries(client=client, token=token, schema=cache, **kwargs)

        yield make


async def test_a_real_query_is_priced_then_run(github_factory) -> None:
    outcome = await github_factory().run(RELEASES, {"owner": "vercel", "name": "ai", "n": 5})

    assert outcome.ok, outcome.messages
    assert outcome.data["repository"]["name"] == "ai"
    assert len(outcome.data["repository"]["releases"]["nodes"]) == 5
    # The splice is ours and must not reach the caller.
    assert set(outcome.data) == {"repository"}
    # Measured 2026-09-15: the dry run predicts the real charge exactly, it does not estimate it.
    assert outcome.cost == outcome.spent
    assert outcome.cost == 1, "P3 says a single-repo query costs one point"


async def test_a_connection_without_first_is_a_repairable_request_error(github_factory) -> None:
    outcome = await github_factory().run(NO_FIRST, {})

    # Local validation passed it: the schema says the argument is optional. GitHub checks
    # pagination bounds in its RESOLVERS, so the error arrives with a path and a type --
    # field-level by position, repairable by type. Getting that wrong meant the repair loop
    # never fired on the most common failure there is.
    # One stage, not a set: a test that accepts either answer cannot tell you which one you
    # have. Measured 2026-09-15, and it is the expensive answer. The dry run PRICES this query
    # happily and only the paid call refuses it, so the most common failure a model makes
    # cannot be caught for free. GitHub computes what a query costs and whether a query is
    # runnable in different places, and dryRun only runs the first.
    assert outcome.stage == "field-error"
    assert outcome.repairable
    assert outcome.field_errors[0].type == "MISSING_PAGINATION_BOUNDARIES"
    assert outcome.field_errors[0].path == ["repository", "releases"]
    assert any("first" in message or "last" in message for message in outcome.messages)
    # The paid request is what found it. This is the cost of P1, stated rather than assumed.
    assert outcome.spent is not None


async def test_a_missing_repository_is_a_field_error_and_not_repairable(github_factory) -> None:
    outcome = await github_factory().run(MISSING_REPO, {})

    assert outcome.stage == "field-error"
    assert not outcome.repairable
    assert outcome.field_errors[0].type == "NOT_FOUND"
    # It has a path, which is the whole basis of the split. If this ever comes back without
    # one, split_errors is classifying on something GitHub no longer sends.
    assert outcome.field_errors[0].path == ["repository"]


async def test_the_budget_gate_refuses_on_githubs_own_node_count(github_factory) -> None:
    # Our cap is lowered rather than the query made monstrous: what is under test is the gate
    # reading GitHub's number, not how large a number can be made.
    github = github_factory(max_nodes=1000)

    outcome = await github.run(NESTED, {})

    assert outcome.stage == "budget"
    assert outcome.repairable
    assert outcome.node_count is not None and outcome.node_count > 1000
    # Nested connections multiply: 100 issues times 100 comments, plus the issues themselves.
    assert outcome.node_count >= 10_000
    assert outcome.spent is None, "the expensive query must never have been sent"
