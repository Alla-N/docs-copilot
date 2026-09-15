"""The seven gates, the splice, and the two classes of error -- all offline.

Every GitHub response here is a recorded shape fed through httpx.MockTransport, so the suite
costs nothing and cannot be made green or red by what vercel/ai happens to look like today. The
schema is the same synthetic SDL the schema tests use, built through the real
introspection-to-client-schema path.

The three tests that matter most, in the order they would bite in production:

  - a mutation never leaves the process;
  - a NOT_FOUND is not handed to the repair loop;
  - the pre-flight refuses an expensive query BEFORE the real request goes out.
"""

import json

import httpx
import pytest
from graphql import build_schema, introspection_from_schema, parse

from copilot_agent.github_query import (
    PREFLIGHT_ALIAS,
    SPEND_ALIAS,
    FieldError,
    GitHubQueries,
    is_repairable,
    missing_variables,
    sole_operation,
    split_errors,
    with_rate_limit,
)
from copilot_agent.github_schema import SchemaCache

pytestmark = pytest.mark.anyio

SDL = """
type Query {
  repository(owner: String!, name: String!): Repository
  rateLimit(dryRun: Boolean = false): RateLimit
}
type Mutation {
  addStar(starrableId: ID!): Repository
}
type Repository {
  name: String!
  releases(first: Int, last: Int): ReleaseConnection!
}
type ReleaseConnection {
  pageInfo: PageInfo!
  nodes: [Release]
}
type PageInfo { hasNextPage: Boolean!, endCursor: String }
type Release { tagName: String! }
type RateLimit { cost: Int!, nodeCount: Int!, remaining: Int! }
"""

GOOD = """
query Releases($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) { name releases(first: 5) { nodes { tagName } } }
}
"""
VARIABLES = {"owner": "vercel", "name": "ai"}


@pytest.fixture(scope="module")
def schema_cache_factory():
    introspection = introspection_from_schema(build_schema(SDL), descriptions=True)

    def make() -> SchemaCache:
        async def fetch() -> dict:
            return introspection

        return SchemaCache(fetch=fetch)

    return make


class Recorder:
    """A MockTransport handler that answers from a queue and keeps every request it saw."""

    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.queries: list[str] = []
        self.variables: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.queries.append(body["query"])
        self.variables.append(body.get("variables") or {})
        if not self.responses:
            raise AssertionError(f"unexpected request number {len(self.queries)}")
        return httpx.Response(200, json=self.responses.pop(0))


def priced(cost: int = 1, nodes: int = 5) -> dict:
    # dryRun makes the siblings ABSENT, not null. Measured 2026-09-15; the fixture says so too,
    # because a fixture that invents a friendlier shape tests a service that does not exist.
    return {"data": {PREFLIGHT_ALIAS: {"cost": cost, "nodeCount": nodes}}}


def answered(spent: int = 1) -> dict:
    return {
        "data": {
            "repository": {"name": "ai", "releases": {"nodes": [{"tagName": "ai@7.0.101"}]}},
            SPEND_ALIAS: {"cost": spent, "nodeCount": 5},
        }
    }


def runner(cache_factory, recorder: Recorder, **kwargs) -> GitHubQueries:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return GitHubQueries(client=client, token="t", schema=cache_factory(), **kwargs)


# --- gates that need no network ----------------------------------------------------------


async def test_a_mutation_never_leaves_the_process(schema_cache_factory) -> None:
    recorder = Recorder()  # no responses queued: any request at all is a failure
    outcome = await runner(schema_cache_factory, recorder).run(
        "mutation Star($id: ID!) { addStar(starrableId: $id) { name } }", {"id": "x"}
    )

    assert outcome.stage == "read-only"
    # Not repairable on purpose: this is not a mistake to correct, it is a thing the subagent
    # is not allowed to do, and a repair prompt would be an invitation to try again.
    assert not outcome.repairable
    assert recorder.queries == []


async def test_syntax_and_validation_are_different_stages(schema_cache_factory) -> None:
    recorder = Recorder()
    github = runner(schema_cache_factory, recorder)

    syntax = await github.run("query { repository(owner: }", {})
    invalid = await github.run("query { repository(owner: 1, name: 2) { titel } }", {})

    assert syntax.stage == "syntax"
    assert invalid.stage == "validation"
    assert syntax.repairable and invalid.repairable
    # The model gets the schema's own words, which name the fix.
    assert any("titel" in message for message in invalid.messages)
    assert recorder.queries == []


async def test_local_validation_cannot_catch_a_missing_first(schema_cache_factory) -> None:
    # The GitHub rule the SCHEMA does not carry: first is declared optional, and GitHub rejects
    # a connection without it. Here the schema is our synthetic one, so this documents what
    # local validation can and cannot do -- the argument is optional, so validation passes it.
    recorder = Recorder(priced(), answered())
    outcome = await runner(schema_cache_factory, recorder).run(
        'query { repository(owner: "vercel", name: "ai") { releases { nodes { tagName } } } }',
        {},
    )

    assert outcome.stage == "ok", (
        "local validation cannot catch a missing first: the schema says the argument is "
        "optional. Only GitHub rejects it, which is why the request-error path exists."
    )


async def test_two_operations_are_refused_before_the_network(schema_cache_factory) -> None:
    recorder = Recorder()
    outcome = await runner(schema_cache_factory, recorder).run(
        "query A { rateLimit { cost } } query B { rateLimit { cost } }", {}
    )

    assert outcome.stage == "one-operation"
    assert recorder.queries == []


async def test_a_declared_variable_with_no_value_is_caught_offline(schema_cache_factory) -> None:
    recorder = Recorder()
    outcome = await runner(schema_cache_factory, recorder).run(GOOD, {"owner": "vercel"})

    assert outcome.stage == "variables"
    assert "name" in outcome.messages[0]
    assert recorder.queries == []


def test_a_variable_with_a_default_is_not_missing() -> None:
    operation = sole_operation(parse("query Q($n: Int = 10, $o: String!) { rateLimit { cost } }"))

    # GraphQL validation never looks at the values at all, so this gap is ours; the default is
    # the half of it that is easy to get wrong in the strict direction.
    assert missing_variables(operation, {}) == ["o"]
    assert missing_variables(operation, {"o": "x"}) == []


# --- the splice --------------------------------------------------------------------------


def test_the_splice_adds_an_aliased_rate_limit_and_keeps_the_operation() -> None:
    document = parse(GOOD)
    operation = sole_operation(document)

    spliced = with_rate_limit(
        document, operation, parse("{ rateLimit(dryRun: true) { cost nodeCount } }"), "pre"
    )

    assert "pre: rateLimit(dryRun: true)" in spliced
    assert "query Releases($owner: String!, $name: String!)" in spliced
    assert "releases(first: 5)" in spliced


def test_the_splice_does_not_collide_with_a_rate_limit_the_model_selected() -> None:
    # Two fields with the same response name and different arguments are a validation error.
    # Two with different response names are not, which is the whole reason for the alias.
    document = parse('query Q { rateLimit { cost } repository(owner: "v", name: "a") { name } }')
    operation = sole_operation(document)

    spliced = with_rate_limit(
        document, operation, parse("{ rateLimit(dryRun: true) { cost nodeCount } }"), "pre"
    )

    assert "rateLimit {" in spliced
    assert "pre: rateLimit(dryRun: true)" in spliced


# --- the two classes of error -------------------------------------------------------------


def test_path_says_where_and_type_says_whether_it_is_worth_repairing() -> None:
    # Observed 2026-09-15 against the live API. GitHub checks pagination bounds in its
    # resolvers, so a connection missing `first` comes back WITH a path -- and it is the single
    # most common thing a model gets wrong. Classifying repairability by path alone meant the
    # repair loop would never fire on the failure it exists for.
    pagination = FieldError(
        message="You must provide a `first` or `last` value to properly paginate the "
        "`releases` connection.",
        path=["repository", "releases"],
        type="MISSING_PAGINATION_BOUNDARIES",
    )
    missing = FieldError(message="no such repo", path=["repository"], type="NOT_FOUND")

    assert is_repairable(pagination)
    assert not is_repairable(missing)
    # Unknown types default to no: an omitted repairable type costs one failed question, an
    # included unrepairable one costs a loop.
    assert not is_repairable(FieldError(message="?", path=["x"], type="SOMETHING_NEW"))
    assert not is_repairable(FieldError(message="?", path=["x"], type=None))


async def test_a_pagination_error_from_the_preflight_short_circuits(schema_cache_factory) -> None:
    # One request, not two. The first version discarded the pre-flight's field errors and paid
    # for the real call to learn what the free one already knew.
    recorder = Recorder(
        {
            "data": {PREFLIGHT_ALIAS: None},
            "errors": [
                {
                    "type": "MISSING_PAGINATION_BOUNDARIES",
                    "path": ["repository", "releases"],
                    "message": "You must provide a `first` or `last` value",
                }
            ],
        }
    )

    outcome = await runner(schema_cache_factory, recorder).run(GOOD, VARIABLES)

    assert outcome.stage == "preflight"
    assert outcome.repairable
    assert len(recorder.queries) == 1
    assert outcome.spent is None


def test_errors_are_split_by_path_and_not_by_whether_data_is_present() -> None:
    # A non-null field failing deep in a selection propagates null to data. Reading
    # `data is None` would file this ordinary NOT_FOUND as a request error and send the model
    # off to rewrite a query that was correct.
    payload = {
        "data": None,
        "errors": [
            {
                "type": "NOT_FOUND",
                "path": ["repository"],
                "message": "Could not resolve to a Repository",
            }
        ],
    }

    request, fields = split_errors(payload)

    assert request == []
    assert fields == [
        FieldError(
            message="Could not resolve to a Repository", path=["repository"], type="NOT_FOUND"
        )
    ]


def test_a_validation_error_from_github_has_no_path() -> None:
    payload = {"errors": [{"message": "Field 'titel' doesn't exist on type 'Issue'"}]}

    request, fields = split_errors(payload)

    assert len(request) == 1 and fields == []


async def test_a_not_found_is_not_handed_to_the_repair_loop(schema_cache_factory) -> None:
    recorder = Recorder(
        priced(),
        {
            "data": {"repository": None, SPEND_ALIAS: {"cost": 1, "nodeCount": 5}},
            "errors": [{"type": "NOT_FOUND", "path": ["repository"], "message": "no such repo"}],
        },
    )

    outcome = await runner(schema_cache_factory, recorder).run(GOOD, VARIABLES)

    assert outcome.stage == "field-error"
    # Rewriting the query does not conjure a missing repository. A loop that retries this three
    # times is a bug that looks like a feature. NOT_FOUND is not in the allow-list.
    assert not outcome.repairable
    assert outcome.field_errors[0].type == "NOT_FOUND"
    assert outcome.spent == 1


async def test_a_request_error_from_the_real_call_is_repairable(schema_cache_factory) -> None:
    recorder = Recorder(
        priced(),
        {"errors": [{"message": "You must provide a first or last value"}]},
    )

    outcome = await runner(schema_cache_factory, recorder).run(GOOD, VARIABLES)

    assert outcome.stage == "request-error"
    assert outcome.repairable


# --- the budget gate ----------------------------------------------------------------------


async def test_an_expensive_query_is_refused_before_the_real_request(schema_cache_factory) -> None:
    recorder = Recorder(priced(cost=3, nodes=200_000))

    outcome = await runner(schema_cache_factory, recorder).run(GOOD, VARIABLES)

    assert outcome.stage == "budget"
    assert outcome.repairable
    assert outcome.node_count == 200_000
    # One request, not two: the point of a free pre-flight is that the expensive call never
    # happens.
    assert len(recorder.queries) == 1
    assert PREFLIGHT_ALIAS in recorder.queries[0]
    # The refusal carries the numbers, so the repair prompt can say what was too big.
    assert "200000" in outcome.messages[0]


async def test_the_points_cap_also_refuses(schema_cache_factory) -> None:
    recorder = Recorder(priced(cost=50, nodes=10))

    outcome = await runner(schema_cache_factory, recorder, max_points=10).run(GOOD, VARIABLES)

    assert outcome.stage == "budget"
    assert outcome.cost == 50


# --- the happy path -----------------------------------------------------------------------


async def test_a_good_query_is_priced_then_run_and_reports_what_it_spent(
    schema_cache_factory,
) -> None:
    recorder = Recorder(priced(cost=1, nodes=5), answered(spent=1))

    outcome = await runner(schema_cache_factory, recorder).run(GOOD, VARIABLES)

    assert outcome.ok
    assert outcome.cost == 1 and outcome.node_count == 5 and outcome.spent == 1
    assert outcome.data == {
        "repository": {"name": "ai", "releases": {"nodes": [{"tagName": "ai@7.0.101"}]}}
    }
    # The splice is ours, not the caller's: it must not reach the model.
    assert SPEND_ALIAS not in (outcome.data or {})
    assert len(recorder.queries) == 2
    assert PREFLIGHT_ALIAS in recorder.queries[0]
    assert SPEND_ALIAS in recorder.queries[1]
    # Both requests carry the same variables, or the pre-flight would price a different query.
    assert recorder.variables == [VARIABLES, VARIABLES]
