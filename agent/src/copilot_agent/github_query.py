"""Running a query the model wrote: refuse it, price it, or send it -- and say which happened.

Seven gates, in this order, because each one is cheaper than the next and each one produces a
different repair message:

  1. **parse** -- is it GraphQL at all? A syntax error is not a schema problem and the model
     should not be handed the schema to fix it.
  2. **one operation** -- a document with two operations needs an `operationName` we do not send,
     and GitHub would pick the argument fight rather than the query.
  3. **read only** -- the operation must be a `query`. This is half of spec decision 4; the token
     scope is the other half. This half is the one with a test, because it needs no secret and no
     network: a mutation never leaves the process.
  4. **validate** -- against the cached schema, offline and free. Wrong field names, missing
     required arguments, a selection on a union without an inline fragment: all caught here, and
     all repairable from the message.
  5. **variables supplied** -- every variable the operation declares has a value. GraphQL
     validation checks declarations against *usage*, never against the values you are about to
     send, so this gap is ours to close. It is the difference between "the query is wrong" and
     "the arguments are wrong", which are different repairs.
  6. **pre-flight** -- `rateLimit(dryRun: true)` spliced into the operation. Measured 2026-09-15
     (experiments/dry_run_semantics.py): dryRun makes the sibling fields **absent**, not null, so
     this is necessarily its own request; it costs **zero** points, so every query gets one
     rather than only the suspicious ones; and the cost and nodeCount it reports are **exactly**
     what the real call then charges, not an estimate.

     What it does **not** do is validate. A connection missing its `first` is priced happily by
     the dry run and refused by the paid call
     (tests/test_github_query_live.py). GitHub computes what a query costs and whether a query
     is runnable in different places, and dryRun only runs the first. So the pre-flight is a
     budget gate and nothing more, and P1s failure mode -- the most common one there is --
     necessarily costs a paid request.
  7. **budget** -- refuse before spending if the pre-flight's numbers exceed the caps.

Only then does the query go out, with a `rateLimit` of its own spliced in so the turn can report
what it actually spent (spec decision 12).

**Where an error happened and whether it is worth repairing are two questions, and one signal
does not answer both.** A GraphQL endpoint answers 200 and puts failures in an `errors` array.
`path` answers the first question: an error with one happened during execution, an error without
one killed the query before it ran. `data is None` looks like the same signal and is not -- a
non-null field failing deep in a selection propagates null all the way up -- so `path` is what
the split uses.

But `path` does NOT answer the second question, and assuming it did was wrong in the most
expensive possible place. Measured 2026-09-15 against the live API: a connection missing its
`first` or `last` comes back as a **field-level** error, with `path: ["repository", "releases"]`
and `type: MISSING_PAGINATION_BOUNDARIES`. GitHub checks pagination bounds in its resolvers, not
in schema validation, so the error legitimately has a path -- and P1 says this is the single most
common thing a model gets wrong. Classifying it as unrepairable meant the repair loop would never
fire on the failure it exists for.

So repairability is decided by `type`, against an allow-list (REPAIRABLE_ERROR_TYPES), and the
default is no. An omitted repairable type costs one failed question; an included unrepairable one
costs a loop that retries `NOT_FOUND` three times and looks like a feature. That asymmetry sets
the direction of the guess.

Three discriminators were tried before that landed, each one a single structural signal:
`data is None`, then `path`, then `path` for where and `type` for whether. The mistake each time
was looking for one answer to two questions.
"""

from dataclasses import dataclass, field
from typing import Literal

import httpx
from graphql import (
    DocumentNode,
    FieldNode,
    GraphQLSchema,
    GraphQLSyntaxError,
    NameNode,
    OperationDefinitionNode,
    OperationType,
    SelectionSetNode,
    parse,
    print_ast,
    validate,
)

from copilot_agent.github_schema import (
    GITHUB_GRAPHQL_URL,
    SchemaCache,
    github_headers,
)

# Spec decision 10. MAX_POINTS is almost certainly inert: a single-repo query costs 1 point and
# the arithmetic floors at 1, so reaching 10 takes a query nobody here will write. It is kept
# because it costs nothing to keep and because the day it fires it will have found something
# real. MAX_NODES is the half doing the work -- nested connections multiply, and GitHub's own
# ceiling is 500,000, ten times this one.
MAX_POINTS = 10
MAX_NODES = 50_000

QUERY_TIMEOUT_S = 30.0

# GitHub error types that describe the QUERY and can therefore be fixed by rewriting it.
# Everything else -- NOT_FOUND, FORBIDDEN, RATE_LIMITED, SERVICE_UNAVAILABLE, INTERNAL --
# describes the world rather than the query, and no rewrite conjures a missing repository.
#
# An allow-list, deliberately, and short. Provenance per entry, because this file has already
# been wrong once by asserting a shape nobody had seen:
#   MISSING_PAGINATION_BOUNDARIES  observed 2026-09-15, tests/test_github_query_live.py
#   MAX_NODE_LIMIT_EXCEEDED        from GitHub documentation; not observed here, because the
#                                  budget gate is meant to refuse such a query one round trip
#                                  earlier. If this ever fires, the gate did not.
REPAIRABLE_ERROR_TYPES = frozenset(
    {
        "MISSING_PAGINATION_BOUNDARIES",
        "MAX_NODE_LIMIT_EXCEEDED",
    }
)

# Aliased, so the splice can never collide with a `rateLimit` the model selected itself. Two
# fields with the same response name and different arguments are a validation error; two with
# different response names are not. An alias may not begin with two underscores -- those are
# reserved for introspection -- so these are spelled out.
PREFLIGHT_ALIAS = "preflightRateLimit"
SPEND_ALIAS = "spentRateLimit"

_PREFLIGHT_TEMPLATE = parse("{ rateLimit(dryRun: true) { cost nodeCount } }")
_SPEND_TEMPLATE = parse("{ rateLimit { cost nodeCount } }")

Stage = Literal[
    "syntax",
    "one-operation",
    "read-only",
    "validation",
    "variables",
    "preflight",
    "budget",
    "request-error",
    "field-error",
    "ok",
]


@dataclass(frozen=True)
class FieldError:
    """One execution error: where it happened, and GitHub's own name for it.

    Plain types only. These end up in the graph state and therefore in a checkpoint, and 2.5
    settled that the serializer has no tuple: a list stays a list on the way back.
    """

    message: str
    path: list[str]
    type: str | None


@dataclass(frozen=True)
class RunOutcome:
    """What happened to one generated query, in a shape the repair loop and the log both read.

    `repairable` is the only field the loop branches on, and it is set by the gate that failed
    rather than inferred later. A read-only violation and a NOT_FOUND are both failures and
    neither is worth another attempt; a missing argument is worth exactly one.
    """

    stage: Stage
    repairable: bool
    # What to hand the model on a repair. Empty on success.
    messages: list[str] = field(default_factory=list)
    data: dict | None = None
    # From the pre-flight when it got that far, so a refusal can say what it refused.
    cost: int | None = None
    node_count: int | None = None
    # What the real call charged, which is the number the eval harness reports per question.
    spent: int | None = None
    field_errors: list[FieldError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.stage == "ok"


def sole_operation(document: DocumentNode) -> OperationDefinitionNode | None:
    operations = [d for d in document.definitions if isinstance(d, OperationDefinitionNode)]
    return operations[0] if len(operations) == 1 else None


def with_rate_limit(
    document: DocumentNode,
    operation: OperationDefinitionNode,
    template: DocumentNode,
    alias: str,
) -> str:
    """The same document with an aliased `rateLimit` added to the operation's top level.

    Built by lifting the field out of a parsed template rather than by assembling FieldNode,
    ArgumentNode and BooleanValueNode by hand. The hand-built version is four times the code and
    its correctness is only checkable by printing it, which is the test either way -- so the
    version that cannot get an argument node wrong wins.

    Fragment definitions and any other definitions travel unchanged; only the operation is
    rebuilt, because AST nodes are not mutated in place.
    """
    lifted = template.definitions[0].selection_set.selections[0]
    extra = FieldNode(
        alias=NameNode(value=alias),
        name=lifted.name,
        arguments=lifted.arguments,
        selection_set=lifted.selection_set,
    )
    rebuilt = OperationDefinitionNode(
        operation=operation.operation,
        name=operation.name,
        variable_definitions=operation.variable_definitions,
        directives=operation.directives,
        selection_set=SelectionSetNode(
            selections=(*operation.selection_set.selections, extra),
        ),
    )
    others = [d for d in document.definitions if d is not operation]
    return print_ast(DocumentNode(definitions=(rebuilt, *others)))


def is_repairable(error: FieldError) -> bool:
    """Whether rewriting the query could fix this execution error. Unknown types: no."""
    return error.type in REPAIRABLE_ERROR_TYPES


def any_repairable(errors: list[FieldError]) -> bool:
    """Repairable if any single error is.

    A response can carry a NOT_FOUND and a pagination error together. One rewrite fixes one of
    them, which is a better outcome than refusing to try because the other is hopeless.
    """
    return any(is_repairable(error) for error in errors)


def split_errors(payload: dict) -> tuple[list[str], list[FieldError]]:
    """Request-level messages and field-level errors, told apart by `path` -- where, not whether.

    See the module docstring. `path` says an error happened during execution; it says nothing
    about whether a rewrite can fix it, and MISSING_PAGINATION_BOUNDARIES is the proof.
    """
    request: list[str] = []
    fields: list[FieldError] = []
    for error in payload.get("errors") or []:
        message = str(error.get("message", ""))
        if "path" in error:
            fields.append(
                FieldError(
                    message=message,
                    path=[str(part) for part in error.get("path") or []],
                    type=error.get("type"),
                )
            )
        else:
            request.append(message)
    return request, fields


def missing_variables(operation: OperationDefinitionNode, variables: dict) -> list[str]:
    """Declared variables with no value supplied.

    Only the ones with no default: `$n: Int = 10` is declared and optional. GraphQL's own
    validation never looks at the values, so without this the first sign of a missing variable
    is GitHub rejecting the whole request.
    """
    return [
        definition.variable.name.value
        for definition in operation.variable_definitions
        if definition.default_value is None and definition.variable.name.value not in variables
    ]


class GitHubQueries:
    """The runner. One per process, sharing the client, the token and the schema cache."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        token: str,
        schema: SchemaCache,
        max_points: int = MAX_POINTS,
        max_nodes: int = MAX_NODES,
    ) -> None:
        self._client = client
        self._token = token
        self._schema = schema
        self._max_points = max_points
        self._max_nodes = max_nodes

    async def _post(self, query: str, variables: dict) -> dict:
        response = await self._client.post(
            GITHUB_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=github_headers(self._token),
            timeout=QUERY_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json()

    async def check(
        self, query: str, variables: dict
    ) -> RunOutcome | tuple[DocumentNode, OperationDefinitionNode]:
        """Gates 1 to 5: everything decidable without the network.

        A RunOutcome means refused. Otherwise the parsed document and operation come back, so
        run() does not parse the same text twice.
        """
        try:
            document = parse(query)
        except GraphQLSyntaxError as error:
            return RunOutcome(stage="syntax", repairable=True, messages=[str(error)])

        operation = sole_operation(document)
        if operation is None:
            return RunOutcome(
                stage="one-operation",
                repairable=True,
                messages=["Send exactly one operation per query, with no other operations."],
            )

        if operation.operation is not OperationType.QUERY:
            # Not repairable, and deliberately so: this is not a mistake to be corrected but a
            # thing the subagent is not allowed to do. A repair prompt here would be an
            # invitation to try again.
            return RunOutcome(
                stage="read-only",
                repairable=False,
                messages=[f"This agent runs queries only; {operation.operation.value} refused."],
            )

        schema: GraphQLSchema = await self._schema.get()
        errors = validate(schema, document)
        if errors:
            return RunOutcome(
                stage="validation",
                repairable=True,
                messages=[str(error) for error in errors],
            )

        absent = missing_variables(operation, variables)
        if absent:
            return RunOutcome(
                stage="variables",
                repairable=True,
                messages=[f"No value supplied for the declared variable(s): {', '.join(absent)}."],
            )

        return document, operation

    async def run(self, query: str, variables: dict | None = None) -> RunOutcome:
        """Check it, price it, refuse it or send it."""
        variables = variables or {}
        checked = await self.check(query, variables)
        if isinstance(checked, RunOutcome):
            return checked
        document, operation = checked

        priced = await self._post(
            with_rate_limit(document, operation, _PREFLIGHT_TEMPLATE, PREFLIGHT_ALIAS),
            variables,
        )
        request_errors, preflight_errors = split_errors(priced)
        if request_errors:
            # The schema said yes and GitHub said no. Worth surfacing as itself rather than as a
            # validation error, because it means the cached schema and the live one disagree --
            # a fact about the cache, not about the query.
            return RunOutcome(stage="preflight", repairable=True, messages=request_errors)
        if preflight_errors:
            # The first version threw these away. Keeping them is right in principle and, so
            # far, never observed to fire: the one field-level error we can reliably provoke
            # (MISSING_PAGINATION_BOUNDARIES) does NOT appear here, because the dry run prices
            # without validating. Tested offline, not observed live -- said plainly rather than
            # listed as a feature, because an untested branch that claims to save a round trip
            # is worse than no branch at all.
            return RunOutcome(
                stage="preflight",
                repairable=any_repairable(preflight_errors),
                messages=[error.message for error in preflight_errors],
                field_errors=preflight_errors,
            )

        rate = (priced.get("data") or {}).get(PREFLIGHT_ALIAS) or {}
        cost, node_count = rate.get("cost"), rate.get("nodeCount")
        over_budget = (
            cost is not None
            and node_count is not None
            and (cost > self._max_points or node_count > self._max_nodes)
        )
        if over_budget:
            return RunOutcome(
                stage="budget",
                repairable=True,
                cost=cost,
                node_count=node_count,
                messages=[
                    f"Too expensive before running: {cost} points and {node_count} nodes, "
                    f"against limits of {self._max_points} and {self._max_nodes}. "
                    "Ask for fewer items per connection, or fewer nested connections."
                ],
            )

        payload = await self._post(
            with_rate_limit(document, operation, _SPEND_TEMPLATE, SPEND_ALIAS),
            variables,
        )
        request_errors, field_errors = split_errors(payload)
        data = payload.get("data")
        spent = ((data or {}).get(SPEND_ALIAS) or {}).get("cost")
        if isinstance(data, dict):
            # The splice is ours, not the caller's: it never reaches the model.
            data = {key: value for key, value in data.items() if key != SPEND_ALIAS}

        if request_errors:
            return RunOutcome(
                stage="request-error",
                repairable=True,
                messages=request_errors,
                cost=cost,
                node_count=node_count,
                spent=spent,
            )
        if field_errors:
            return RunOutcome(
                stage="field-error",
                repairable=any_repairable(field_errors),
                messages=[error.message for error in field_errors],
                data=data,
                cost=cost,
                node_count=node_count,
                spent=spent,
                field_errors=field_errors,
            )
        return RunOutcome(
            stage="ok",
            repairable=False,
            data=data,
            cost=cost,
            node_count=node_count,
            spent=spent,
        )
