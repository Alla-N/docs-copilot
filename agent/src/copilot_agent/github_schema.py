"""The GitHub GraphQL schema: fetched once, cached, and served to the model a type at a time.

Measured 2026-09-14 (experiments/github_schema_size.py): GitHub publishes **1,826 types**, and
the introspection result with descriptions is **3.2 MiB of JSON** -- roughly 800,000 tokens, so
the schema cannot go in a prompt and there is no version of this that works by making the prompt
bigger. What the model actually needs, for any one question, is five or six types. So this module
is two things:

  - a cache (fetch introspection once, build a GraphQLSchema from it, keep it for the life of
    the process);
  - a pair of readers that turn one type into a few hundred tokens.

**Lazily, never at startup** (spec decision 6). The service already exits 3 when a dependency is
not ready; putting GitHub in that list would make this service's availability depend on GitHub's,
for a capability most turns never use. The first GitHub turn pays for the fetch, under a lock so
two concurrent first turns do not both pay.

**Two reading modes** (an evolution of spec decision 7, which assumed one). A single budgeted
description cannot serve both jobs: `Repository` has over a hundred fields, and one byte cap over
a full description would cut the list somewhere in the alphabet, so `releases` would exist or not
depending on where the knife fell. Instead:

  - **outline** (`fields=None`): every field name, its type in SDL notation, and a marker for
    connections. No arguments, no descriptions. Cheap and complete, which is what a model needs
    to decide where to look.
  - **detail** (`fields=[...]`): the full signature of the named fields, with arguments, which
    of them are required, and one line of description each.

The byte cap still applies to both, and a truncated result says how many fields it dropped. A
result that is silently short is the same bug as `greatest(reltuples, 0)`: nothing has to look
like nothing, but it must not look like an answer.

Nothing here writes. The read-only guarantee lives with the runner (3.3), which refuses any
operation that is not a query before it reaches the network.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
from graphql import (
    GraphQLArgument,
    GraphQLField,
    GraphQLInterfaceType,
    GraphQLNamedType,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLUnionType,
    Undefined,
    build_client_schema,
    get_introspection_query,
    get_named_type,
    is_non_null_type,
)

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

# GitHub rejects a request with no User-Agent. It asks for the application name.
USER_AGENT = "docs-copilot-agent"

# Introspection is one large request; the rest of the service uses much shorter timeouts.
INTROSPECTION_TIMEOUT_S = 60.0

# Big enough for the whole outline of the widest type this project touches, and small enough
# that a model asking for the wrong type has not spent the turn on it. Measured 2026-09-14
# (experiments/github_schema_size.py), uncapped outline bytes: Repository 5971 over 145 fields,
# PullRequest 4290, Issue 3357, the Query root 1362, everything else under 800. 8000 is the
# widest of those with about a third in hand, because GitHub adds fields to Repository steadily
# and a cap with no headroom starts clipping on a schema change nobody here will notice.
#
# The first value was 6000, chosen before any of those numbers existed. It cut Repository by 35
# bytes -- one field line -- and the reason was not only the cap (see FOOTER_RESERVE).
DEFAULT_BYTE_CAP = 8000

# Room kept for the "(N of M ... shown)" footer, so a truncated result can always say it was
# truncated. Reserved only once truncation is known to be happening; see _capped.
FOOTER_RESERVE = 80

# Descriptions are the bulk of the introspection payload and most of a field line. One line each.
DESCRIPTION_CHARS = 120


class GitHubSchemaError(RuntimeError):
    """Introspection did not come back usable: HTTP failed, or the payload carried errors.

    Separate from the query runner's errors (3.3) on purpose. This one means the tool cannot
    work at all; those mean one generated query was wrong, which is a normal event here.
    """


def introspection_query() -> str:
    """The introspection query, with descriptions.

    Descriptions are most of the bytes and they are also the part a model reads to choose
    between two similarly named fields. Kept, and truncated at the point of use instead
    (DESCRIPTION_CHARS), so the cost is paid once in the cache and never in a prompt.
    """
    return get_introspection_query(descriptions=True)


def github_headers(token: str) -> dict[str, str]:
    """Every request to the API carries these two.

    The User-Agent is not optional: GitHub rejects a request without one, and the rejection does
    not say that is why. Shared with the query runner so there is one place it can be wrong.
    """
    return {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}


async def fetch_introspection(client: httpx.AsyncClient, token: str) -> dict:
    """POST the introspection query and return its `data`, or raise GitHubSchemaError.

    A GraphQL endpoint answers 200 with an `errors` array; `raise_for_status` alone would let a
    failed introspection through as a successful fetch. Both checks, in that order.
    """
    try:
        response = await client.post(
            GITHUB_GRAPHQL_URL,
            json={"query": introspection_query()},
            headers=github_headers(token),
            timeout=INTROSPECTION_TIMEOUT_S,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise GitHubSchemaError(f"introspection request failed: {error}") from error

    payload = response.json()
    if payload.get("errors"):
        messages = "; ".join(str(error.get("message")) for error in payload["errors"])
        raise GitHubSchemaError(f"introspection returned errors: {messages}")
    data = payload.get("data")
    if not isinstance(data, dict) or "__schema" not in data:
        raise GitHubSchemaError("introspection returned no __schema")
    return data


Introspect = Callable[[], Awaitable[dict]]


class SchemaCache:
    """One GraphQLSchema per process, built on first use.

    `fetch` is injected rather than called directly so that the tests and the experiments can
    supply a recorded introspection result and never touch the network (the same shape as
    build_graph taking its planner and search).

    `path`, when given, is a JSON file holding the introspection result. It is read before
    fetching and written after, which makes local development and the live integration test cost
    one fetch between them rather than one each. It is not a production mechanism: a container
    has no durable disk and does not want a schema older than its image.
    """

    def __init__(self, *, fetch: Introspect, path: Path | None = None) -> None:
        self._fetch = fetch
        self._path = path
        self._schema: GraphQLSchema | None = None
        # Two turns can arrive together and both find an empty cache. Without the lock both
        # would fetch tens of megabytes; with it the second waits and then finds the schema.
        self._lock = asyncio.Lock()

    async def get(self) -> GraphQLSchema:
        if self._schema is not None:
            return self._schema
        async with self._lock:
            # Re-checked inside the lock: whoever waited here is the second caller, and the
            # first one has already filled it.
            if self._schema is None:
                self._schema = build_client_schema(await self._introspection())
            return self._schema

    async def _introspection(self) -> dict:
        if self._path is not None and self._path.exists():
            # to_thread: reading tens of megabytes off disk blocks the event loop, and the
            # ASYNC ruff rules exist to catch exactly this.
            return await asyncio.to_thread(lambda: json.loads(self._path.read_text()))
        data = await self._fetch()
        if self._path is not None:
            await asyncio.to_thread(self._write, data)
        return data

    def _write(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Written next to the target and moved, so a killed process cannot leave a half-written
        # cache that the next start reads as a schema.
        temporary = self._path.with_suffix(self._path.suffix + ".partial")
        temporary.write_text(json.dumps(data))
        temporary.replace(self._path)


def github_fetch(client: httpx.AsyncClient, token: str) -> Introspect:
    """The production `fetch` for SchemaCache: introspection against GitHub with this token."""

    async def fetch() -> dict:
        return await fetch_introspection(client, token)

    return fetch


def _one_line(description: str | None) -> str:
    if not description:
        return ""
    line = " ".join(description.split())
    if len(line) > DESCRIPTION_CHARS:
        line = line[: DESCRIPTION_CHARS - 3] + "..."
    return line


def connection_node_name(field: GraphQLField) -> str | None:
    """What a connection field yields, or None when the field is not a connection.

    Detected by shape, not by name. `...Connection` is a naming convention a schema is free to
    break; a `pageInfo` field is what makes cursor pagination work, and it is what forces the
    `first` or `last` argument GitHub rejects a query for omitting.

    The NAME returned is the node type, not the connection type. Found in 3.2 by the test:
    the first version reported `[connection of IssueConnection]`, which is true, well formed,
    and useless -- it hands the model back the wrapper name it already had in the field type,
    instead of the one thing it cannot see from there. A wrong answer that reads like a right
    one, which is this project's most frequent bug by some distance.

    Two shapes, because GitHub uses both: `nodes: [Issue]` is the shortcut, and `edges { node }`
    is the long form that some connections have and others do not. A connection with neither
    falls back to its own name, so the marker is never empty.
    """
    named = get_named_type(field.type)
    if not isinstance(named, GraphQLObjectType) or "pageInfo" not in named.fields:
        return None
    nodes = named.fields.get("nodes")
    if nodes is not None:
        return get_named_type(nodes.type).name
    edges = named.fields.get("edges")
    if edges is not None:
        edge = get_named_type(edges.type)
        if isinstance(edge, GraphQLObjectType) and "node" in edge.fields:
            return get_named_type(edge.fields["node"].type).name
    return named.name


def is_connection(field: GraphQLField) -> bool:
    return connection_node_name(field) is not None


def is_required(argument: GraphQLArgument) -> bool:
    """Non-null and without a default: the caller must supply it.

    Computed rather than taken from a predicate so the rule is visible: a non-null argument WITH
    a default is optional, which is the case that makes `first: Int` on GitHub connections look
    optional when the server still rejects a query that omits it. That rejection is GitHub's own
    rule on top of the schema, not something the schema says; 3.3 is where we catch it.
    """
    return is_non_null_type(argument.type) and argument.default_value is Undefined


def _argument_signature(name: str, argument: GraphQLArgument) -> str:
    return f"{name}: {argument.type}"


def _field_outline(name: str, field: GraphQLField) -> str:
    node = connection_node_name(field)
    marker = f"  [connection of {node}]" if node is not None else ""
    return f"  {name}: {field.type}{marker}"


def _field_detail(name: str, field: GraphQLField) -> str:
    arguments = ", ".join(
        _argument_signature(argument_name, argument)
        for argument_name, argument in field.args.items()
    )
    required = [
        argument_name for argument_name, argument in field.args.items() if is_required(argument)
    ]
    lines = [f"  {name}({arguments}): {field.type}" if arguments else f"  {name}: {field.type}"]
    if required:
        lines.append(f"    required arguments: {', '.join(required)}")
    node = connection_node_name(field)
    if node is not None:
        lines.append(f"    connection of {node}; GitHub requires first or last, 1 to 100")
    description = _one_line(field.description)
    if description:
        lines.append(f"    {description}")
    return "\n".join(lines)


def _capped(header: list[str], entries: list[str], byte_cap: int, noun: str) -> str:
    """Join header and entries under a byte cap, and say how many entries were dropped.

    Asked in the order that matters: does the whole thing fit? Only when it does not is anything
    dropped, and only then is room kept for the footer that says so.

    That question was missing from the first two versions. Both walked the entries holding back
    80 bytes for a footer at every step, against a footer that would never be written if the walk
    simply finished -- so a description that fit its cap exactly still lost entries. The real
    Repository is 5971 bytes and came back cut at a cap of 6000. The second version tried to
    exempt the final entry, which fixed nothing: the reserve is applied at entry three, long
    before the last one is in sight. The bug was never the arithmetic. It was answering a local
    question (does this entry fit, plus a footer?) in place of the only one that decides the
    outcome (does all of it fit?).

    Cut at an entry boundary, never mid-line: half a field definition is worse than a missing
    one, because the model will use it.
    """
    whole = "\n".join(header + entries)
    if len(whole.encode()) <= byte_cap:
        return whole

    body = "\n".join(header)
    kept = 0
    for entry in entries:
        candidate = f"{body}\n{entry}"
        # Something is being dropped, so the footer will exist and has to fit inside the cap
        # too -- a capped answer that overruns the cap while announcing the cap is no answer.
        if len(candidate.encode()) + FOOTER_RESERVE > byte_cap:
            break
        body = candidate
        kept += 1
    return body + f"\n({kept} of {len(entries)} {noun} shown; cut at the {byte_cap} byte cap)"


def describe_type(
    schema: GraphQLSchema,
    name: str,
    *,
    fields: list[str] | None = None,
    byte_cap: int = DEFAULT_BYTE_CAP,
) -> str:
    """One type, budgeted. Outline when `fields` is None, full signatures when it is given.

    Returns a message rather than raising when the type or a field does not exist: the caller is
    a model that guessed a name, and the useful answer to a guess is what the name should have
    been, in the same channel as a success.
    """
    named: GraphQLNamedType | None = schema.get_type(name)
    if named is None:
        return f"No type named {name} in this schema."

    if isinstance(named, GraphQLUnionType):
        members = [f"  {member.name}" for member in named.types]
        header = [
            f"union {named.name}",
            "A selection on a union needs an inline fragment per member: ... on Member { ... }",
        ]
        if _one_line(named.description):
            header.append(_one_line(named.description))
        return _capped(header, members, byte_cap, "members")

    if not isinstance(named, GraphQLObjectType | GraphQLInterfaceType):
        # Scalars, enums and input objects: small enough to print whole.
        return f"{named.name} ({type(named).__name__}): {_one_line(named.description)}".strip()

    kind = "interface" if isinstance(named, GraphQLInterfaceType) else "type"
    header = [f"{kind} {named.name}"]
    if _one_line(named.description):
        header.append(_one_line(named.description))
    if named.interfaces:
        header.append(f"implements: {', '.join(i.name for i in named.interfaces)}")

    if fields is None:
        entries = [_field_outline(n, f) for n, f in named.fields.items()]
        return _capped(header, entries, byte_cap, "fields")

    entries = []
    for wanted in fields:
        field = named.fields.get(wanted)
        if field is None:
            entries.append(f"  {wanted}: no such field on {named.name}")
        else:
            entries.append(_field_detail(wanted, field))
    return _capped(header, entries, byte_cap, "fields")


def schema_summary(schema: GraphQLSchema, *, byte_cap: int = DEFAULT_BYTE_CAP) -> str:
    """The entry points: the fields of the Query root, in outline.

    This is where a model starts. It is deliberately the same shape as describe_type on any
    other type, so the model learns one format instead of two.
    """
    query_type = schema.query_type
    if query_type is None:
        return "This schema has no Query root."
    return describe_type(schema, query_type.name, byte_cap=byte_cap)
