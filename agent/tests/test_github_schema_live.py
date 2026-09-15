"""Introspection against the REAL GitHub GraphQL API.

Marked integration, so the pre-commit hook skips it. Needs GITHUB_TOKEN and network; costs one
rate-limit point and no money. Run it on purpose:

    uv run pytest -m integration tests/test_github_schema_live.py

What it is here to catch, which the offline tests cannot: that the token type works at all. A
fine-grained personal access token with public-repository read-only access is what the spec
chose; the classic token with public_repo is the fallback. This test is where that is settled,
and the assertions below are about GitHub's real schema, not about a schema we wrote.
"""

import httpx
import pytest

from copilot_agent.github_schema import (
    SchemaCache,
    describe_type,
    fetch_introspection,
    github_fetch,
)
from copilot_agent.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
async def github_schema():
    settings = get_settings()
    assert settings.github_token is not None, "GITHUB_TOKEN is not set"
    async with httpx.AsyncClient() as client:
        cache = SchemaCache(fetch=github_fetch(client, settings.github_token.get_secret_value()))
        yield await cache.get()


async def test_the_token_can_introspect(github_schema) -> None:
    # About 1,100 types at the time of writing. The assertion is deliberately loose: the number
    # moves every week and the thing being tested is that introspection came back at all.
    assert len(github_schema.type_map) > 500
    assert github_schema.query_type is not None


async def test_the_shapes_the_subagent_relies_on_are_still_there(github_schema) -> None:
    repository = describe_type(github_schema, "Repository")
    assert "[connection of Issue]" in repository
    assert "[connection of Release]" in repository

    # Both of these need an inline fragment, and both are in the labelled set for 3.6.
    assert "inline fragment" in describe_type(github_schema, "SearchResultItem")
    assert "GitObject" in describe_type(github_schema, "GitObject")

    # dryRun is what the budget gate of decision 10 is built on. If it disappears, the gate has
    # to be hand-rolled and the spec is wrong.
    query_root = describe_type(github_schema, github_schema.query_type.name, fields=["rateLimit"])
    assert "dryRun" in query_root


async def test_a_bad_token_fails_as_a_schema_error() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(Exception) as raised:
            await fetch_introspection(client, "not-a-token")
    # Either shape is fine; what must not happen is a silent empty schema.
    assert "failed" in str(raised.value) or "errors" in str(raised.value)
