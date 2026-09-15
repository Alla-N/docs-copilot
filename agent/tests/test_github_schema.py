"""The schema cache and the two readers, against a schema built here rather than fetched.

No network and no recorded megabytes. A small SDL is built with build_schema, turned into an
introspection result with introspection_from_schema, and fed back through build_client_schema --
which is exactly the path production takes, so these tests exercise the real conversion and not
a shortcut around it. The SDL carries the four GitHub shapes that generated queries trip over:
a connection, a connection whose type name does NOT end in Connection, a union, and a non-null
argument that has a default.

The live introspection against GitHub is test_github_schema_live.py, behind the integration
marker.
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from graphql import build_client_schema, build_schema, introspection_from_schema

from copilot_agent.github_schema import (
    DEFAULT_BYTE_CAP,
    GitHubSchemaError,
    SchemaCache,
    connection_node_name,
    describe_type,
    fetch_introspection,
    is_connection,
    schema_summary,
)

pytestmark = pytest.mark.anyio

SDL = '''
"""The entry points."""
type Query {
  """Look up one repository."""
  repository(owner: String!, name: String!): Repository
  search(query: String!, type: SearchType!, first: Int): SearchResults!
  rateLimit(dryRun: Boolean = false): RateLimit
}

"""A repository contains the content for a project."""
type Repository {
  name: String!
  createdAt: DateTime!
  issues(first: Int, last: Int, states: [IssueState!]): IssueConnection!
  releases(first: Int, last: Int): ReleaseConnection!
  stargazers(first: Int): StargazerConnection!
  owner: Actor!
}

type IssueConnection {
  totalCount: Int!
  pageInfo: PageInfo!
  nodes: [Issue]
}

"""Named without the Connection suffix on purpose: the shape is what counts."""
type ReleaseConnection {
  pageInfo: PageInfo!
  nodes: [Release]
}

type SearchResults {
  pageInfo: PageInfo!
  nodes: [SearchResultItem]
}

"""Edges and no nodes shortcut: GitHub has connections of both shapes."""
type StargazerConnection {
  pageInfo: PageInfo!
  edges: [StargazerEdge]
}

type StargazerEdge {
  cursor: String!
  node: Actor!
}

type PageInfo {
  hasNextPage: Boolean!
  endCursor: String
}

interface Node {
  id: ID!
}

type Issue implements Node {
  id: ID!
  number: Int!
  title: String!
  state: IssueState!
}

type Release {
  tagName: String!
  publishedAt: DateTime
}

interface Actor {
  login: String!
}

union SearchResultItem = Issue | Release | Repository

enum IssueState { OPEN CLOSED }
enum SearchType { ISSUE REPOSITORY }
scalar DateTime

type RateLimit {
  cost: Int!
  nodeCount: Int!
  remaining: Int!
}
'''


@pytest.fixture(scope="module")
def introspection() -> dict:
    """The introspection result for SDL, the same dict shape GitHub returns."""
    return introspection_from_schema(build_schema(SDL), descriptions=True)


@pytest.fixture(scope="module")
def schema(introspection: dict):
    return build_client_schema(introspection)


# --- the readers -------------------------------------------------------------------------


def test_outline_lists_every_field_with_its_sdl_type(schema) -> None:
    outline = describe_type(schema, "Repository")

    assert "type Repository" in outline
    assert "A repository contains the content for a project." in outline
    # SDL notation, wrappers included: a model writing a variable declaration needs the bang.
    assert "name: String!" in outline
    assert "issues: IssueConnection!" in outline
    # Outline mode carries no arguments and no per-field descriptions.
    assert "first: Int" not in outline
    assert "required arguments" not in outline


def test_a_connection_is_recognised_by_its_page_info_and_not_by_its_name(schema) -> None:
    repository = schema.get_type("Repository")

    assert is_connection(repository.fields["issues"])
    # ReleaseConnection is a connection and IssueConnection is too; the point is the reverse
    # case: a field whose type has no pageInfo is not one, whatever it is called.
    assert not is_connection(repository.fields["name"])
    assert not is_connection(repository.fields["owner"])

    outline = describe_type(schema, "Repository")
    # The NODE type, not the connection type. The first version of this said
    # [connection of IssueConnection]: true, well formed, and the one fact the model already
    # had from the field type.
    assert "[connection of Issue]" in outline
    assert "[connection of Release]" in outline
    assert "[connection of IssueConnection]" not in outline


def test_a_connection_with_edges_and_no_nodes_still_names_its_node_type(schema) -> None:
    repository = schema.get_type("Repository")

    assert connection_node_name(repository.fields["stargazers"]) == "Actor"
    assert connection_node_name(repository.fields["name"]) is None
    assert "[connection of Actor]" in describe_type(schema, "Repository")


def test_detail_names_the_required_arguments_and_the_connection_rule(schema) -> None:
    detail = describe_type(schema, "Query", fields=["repository", "search", "rateLimit"])

    assert "repository(owner: String!, name: String!): Repository" in detail
    assert "required arguments: owner, name" in detail
    # search returns a connection, so the GitHub rule that is not in the schema is stated.
    assert "GitHub requires first or last, 1 to 100" in detail
    # dryRun is non-null-free and has a default, so it is NOT required. A model told otherwise
    # would put it in every query.
    assert "required arguments: dryRun" not in detail
    assert "Look up one repository." in detail


def test_a_union_says_that_a_selection_needs_an_inline_fragment(schema) -> None:
    described = describe_type(schema, "SearchResultItem")

    assert "union SearchResultItem" in described
    assert "inline fragment" in described
    assert "Issue" in described and "Release" in described and "Repository" in described


def test_an_interface_lists_its_fields_like_a_type(schema) -> None:
    described = describe_type(schema, "Actor")

    assert described.startswith("interface Actor")
    assert "login: String!" in described


def test_the_implements_line_appears_only_when_there_is_one(schema) -> None:
    assert "implements: Node" in describe_type(schema, "Issue")
    # No empty header line on a type that implements nothing: a field of the output that is
    # always present teaches the model to expect it and to wonder what an empty one means.
    assert "implements:" not in describe_type(schema, "Repository")


def test_a_guessed_name_gets_an_answer_and_not_an_exception(schema) -> None:
    assert describe_type(schema, "Repositry") == "No type named Repositry in this schema."
    detail = describe_type(schema, "Repository", fields=["titel"])
    assert "titel: no such field on Repository" in detail


def test_a_scalar_or_enum_is_printed_whole(schema) -> None:
    assert "DateTime" in describe_type(schema, "DateTime")
    assert "IssueState" in describe_type(schema, "IssueState")


def test_schema_summary_is_the_query_root(schema) -> None:
    summary = schema_summary(schema)

    assert "type Query" in summary
    assert "repository: Repository" in summary


# --- the byte cap ------------------------------------------------------------------------


def test_the_byte_cap_cuts_at_a_field_boundary_and_says_how_many_it_dropped(schema) -> None:
    whole = describe_type(schema, "Repository")
    capped = describe_type(schema, "Repository", byte_cap=200)

    assert len(capped.encode()) <= 200
    assert "of 6 fields shown; cut at the 200 byte cap" in capped
    # Cut BETWEEN fields, never inside one: a half signature is worse than a missing one,
    # because the model will use it. Every field line that survived is a whole line of the
    # uncapped description, character for character.
    kept = [line for line in capped.splitlines() if line.startswith("  ")]
    assert kept
    assert all(line in whole.splitlines() for line in kept)


def test_an_outline_that_exactly_fits_is_not_cut_by_the_footer_reserve(schema) -> None:
    whole = describe_type(schema, "Repository", byte_cap=10**9)

    exact = describe_type(schema, "Repository", byte_cap=len(whole.encode()))

    # Whether the whole thing fits is asked first, before any entry is weighed against a
    # footer that may never be written. Without that question the walk holds back 80 bytes at
    # every step and a description that fits its cap exactly still loses entries -- which is
    # how the real Repository, 5971 bytes, came back cut at a cap of 6000.
    assert exact == whole
    assert "byte cap" not in exact


def test_an_uncapped_description_says_nothing_about_a_cap(schema) -> None:
    whole = describe_type(schema, "Repository", byte_cap=DEFAULT_BYTE_CAP)

    assert "byte cap" not in whole
    assert len(whole.encode()) < DEFAULT_BYTE_CAP


# --- the cache ---------------------------------------------------------------------------


async def test_two_concurrent_first_turns_fetch_once(introspection: dict) -> None:
    calls = 0

    async def fetch() -> dict:
        nonlocal calls
        calls += 1
        # Yield, so the second caller reaches the lock while the first is still fetching.
        # Without the lock this is where the second fetch would start.
        await asyncio.sleep(0)
        return introspection

    cache = SchemaCache(fetch=fetch)
    first, second = await asyncio.gather(cache.get(), cache.get())

    assert calls == 1
    assert first is second


async def test_the_cache_file_is_read_instead_of_fetched(
    introspection: dict, tmp_path: Path
) -> None:
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(introspection))

    async def fetch() -> dict:
        raise AssertionError("fetched despite a cache file on disk")

    schema = await SchemaCache(fetch=fetch, path=path).get()

    assert schema.get_type("Repository") is not None


async def test_a_fetch_writes_the_cache_file_and_leaves_no_partial(
    introspection: dict, tmp_path: Path
) -> None:
    path = tmp_path / "nested" / "schema.json"

    async def fetch() -> dict:
        return introspection

    await SchemaCache(fetch=fetch, path=path).get()

    assert json.loads(path.read_text())["__schema"]["types"]
    assert list(path.parent.glob("*.partial")) == []


# --- the fetch ---------------------------------------------------------------------------


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_errors_in_a_200_response_are_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # A GraphQL endpoint answers 200 and puts the failure in the body. raise_for_status
        # alone would call this a successful introspection.
        return httpx.Response(200, json={"errors": [{"message": "Bad credentials"}]})

    async with _client(handler) as client:
        with pytest.raises(GitHubSchemaError, match="Bad credentials"):
            await fetch_introspection(client, "token")


async def test_a_200_response_with_no_schema_is_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {}})

    async with _client(handler) as client:
        with pytest.raises(GitHubSchemaError, match="no __schema"):
            await fetch_introspection(client, "token")


async def test_an_http_failure_is_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Requires authentication"})

    async with _client(handler) as client:
        with pytest.raises(GitHubSchemaError, match="introspection request failed"):
            await fetch_introspection(client, "token")


async def test_the_request_carries_the_token_and_a_user_agent(introspection: dict) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"data": introspection})

    async with _client(handler) as client:
        data = await fetch_introspection(client, "a-token")

    assert seen["authorization"] == "Bearer a-token"
    # GitHub rejects a request without one, and the rejection does not say so clearly.
    assert seen["user-agent"] == "docs-copilot-agent"
    assert "__schema" in data
