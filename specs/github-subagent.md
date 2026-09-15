# Spec — Phase 3: the GitHub GraphQL data subagent

**Status:** IN PROGRESS, opened 2026-09-14. Phase 3 of `claude/windward-plan.md`. Phases 1, 2,
the before-2b block and 2b are complete and measured. Results and findings will go in the phase 3
section of `agent/README.md`; this file is the design record and is kept with its wrong
predictions in it, like `specs/aws-deploy.md`.

**Why now:** phase 2b proved the service can be deployed and measured. What the role brief
actually asks for is an agent that **composes its own queries against a real API** and repairs
them when the API says no. That is the one capability docs-copilot does not have, and it is the
headline of the CV case. Retrieval becomes one subagent of two.

## What is being claimed — and what is not

**Claimed:** an agent that introspects a large third-party GraphQL schema, writes its own queries
from a natural-language question, validates them before spending anything, repairs them from the
server's validation errors, and stays inside a points budget — measured on a labelled set with
answer accuracy, first-try validity, validity after repairs, points per question and routing
accuracy.

**Not claimed:** a general text-to-GraphQL system. The subagent is pointed at one repository
(`vercel/ai`) and one schema. Generalising it is a different piece of work and saying so is
cheaper than being caught.

**Also not claimed:** that the router is the right architecture. It is variant A. Phase 4 puts a
DeepAgents orchestrator next to it and the A/B is the point.

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | **A separate router node**, not a new field on the planner | Extending the planner's structured output would change its prompt, which forks it from the TypeScript baseline and invalidates `tests/test_planner_request_parity.py` and the 23x5 planner eval suite. Phase 2's retrieval numbers would stop being a clean comparison for the sake of saving one small call. A node of its own also makes routing accuracy a number that can be measured alone, which is what the phase-3 done-when asks for. Cost: one extra model call per turn, on every turn including docs-only ones. That is a regression to the existing 27-case latency marks and prediction P6 says where it will show. |
| 2 | **Local validation first, then `rateLimit(dryRun: true)`** | Two failure classes, two mechanisms. A malformed or misspelled query is caught offline and free against the cached schema (`graphql-core`: `build_client_schema` then `parse` + `validate`), which makes the repair loop cheap and makes first-try validity measurable without the network. A well-formed but expensive query is caught by GitHub's own `cost` and `nodeCount` under `dryRun: true`, which calculates both **without evaluating the query**. Counting braces ourselves would be our approximation of GitHub's arithmetic; this is GitHub's answer. Cost: one extra round trip before every real query. |
| 3 | **Frozen-answer labelled set** | Only questions whose true answer cannot change: the author of a merged PR, which release first carried a tag, the state a closed issue ended in, a release's published date, the content of a file at a pinned commit. The suite stays green a year from now with no maintenance and still exercises every query shape we care about. Live counts are deliberately excluded: a failing case must mean the agent moved, not the repository. |
| 4 | **Read-only enforced twice: token scope and a code check** | The token is a fine-grained PAT with public-repository read-only access and no account permissions. Independently, the runner parses the operation and **refuses anything whose operation type is not `query`** before it reaches the network. The token makes a mutation fail; the code check makes it never leave the process, and unlike the token it is testable offline with no secret. Defence in depth is the reason given in interviews; the real reason is that the code check is the one of the two that has a test. |
| 5 | **The subagent is a LangGraph subgraph, not a node** | Its loop (write, validate, pre-flight, run, repair) is nodes and edges. Consequence that matters: every attempt is its own step, so it appears in the LangGraph `updates` stream and as its own span in Langfuse, and the repair count is state rather than a local variable. A single node with a `for` loop inside would hide all of it, which is exactly finding 2.3 (inside the graph, a call silently changed shape) waiting to happen again. |
| 6 | **The schema is fetched lazily on first use, never at startup** | Startup readiness already exits 3 when a dependency is not ready (the before-2b block). Adding GitHub to that list would make the service's availability depend on GitHub's, for a capability most turns do not use. So: fetched on the first GitHub turn, cached in memory for the process lifetime and persisted to a local path for development. **Tests never touch the network**: they build the schema from a recorded fixture, the same discipline as the golden request tests. |
| 7 | **Tool 2 returns a budgeted type description, not a type dump** | `__type` on `Repository` is thousands of tokens, most of it descriptions. **Revised in 3.2 to two reading modes**, because one budgeted description cannot do both jobs: `Repository` has 145 fields, so a single cap over a full description cuts the list somewhere in the alphabet and `releases` exists or not depending on where the knife fell. **Outline** (no arguments, no descriptions, one line per field with its SDL type and a `[connection of X]` marker) is complete and cheap; **detail** (named fields only) carries full signatures, required arguments and one line of description each. The byte cap applies to both and a truncated result says how many fields it dropped. **How much to return is the interesting knob of this phase**, and it is a variable worth an A/B in phase 5. |
| 8 | **`gpt-4o-mini` for the subagent too** | Same model as the planner and the answer, so phase 6's Bedrock A/B moves one variable. If query-writing turns out to need a stronger model, that is a finding with a number attached, not a starting assumption. |
| 9 | **Repair cap N = 2**, three attempts in total | A guess, to be measured. The thing the cap protects against is not cost (a validation failure costs one round trip and, with decision 2, usually zero points) but a loop that convinces itself. Prediction P2 says most first tries are already valid; if the second attempt rarely helps, the cap comes down to 1. |
| 10 | **Budget gate: reject before running if `cost` > 10 points or `nodeCount` > 50,000** | Both are from the `dryRun` pre-flight, so both are GitHub's numbers. 10 points is generous for a single-repo query that should cost 1 (P3); it is set where it is so that tripping it is a signal and not a nuisance. A rejection is fed back to the model as a repairable error with the number in it, which is a more useful message than a depth limit. |
| 11 | **`GITHUB_TOKEN` is optional in `Settings`, and the GitHub route is only wired when it is set** | Same discipline as `enable_search_endpoint`: a deployment without the token has no GitHub subagent at all, and the router only ever emits `docs`. A capability behind a runtime check somebody can get wrong is worse than a capability that is absent. The AWS secret grows from six JSON keys to seven. |
| 12 | **The subagent's own metrics travel with the turn**, not in a side channel | The harness cannot see inside the service, so the per-turn record (`query_log` and the stream's finish metadata) grows a GitHub block: attempts, first-try valid, final valid, points spent, node count, which errors were request-level and which were field-level. This is how first-try validity and points per question become numbers the eval harness can report next to cost, the same way it already reports tokens. |

## Architecture

```
                 START
                   |
                 plan            (unchanged: intent + sub-queries)
                   |
             route_after_plan
              /          \
          canned      router      <- NEW (decision 1): docs | github | both
                        / | \
                       /  |  \
          retrieve x N    |   github_subagent      <- NEW subgraph (decision 5)
                 \        |        /
                  \       |       /
                   ------ merge --
                          |
                       generate
                          |
                         END
```

The GitHub subagent's own loop:

```
  write_query  ->  validate_local  --invalid--> repair (<= 2)  -> write_query
       |                |
       |              valid
       |                v
       |            preflight (rateLimit dryRun: cost, nodeCount)
       |                |
       |          over budget --> repair
       |                |
       |              within
       |                v
       |             run_query
       |                |
       |     request-level errors --> repair
       |     field-level errors   --> stop, report (NOT repairable)
       |                |
       |              data
       v                v
             summarise -> back to merge as evidence
```

The two evidence sets meet at `merge`. Documentation chunks and GitHub facts are different kinds
of thing and the generation prompt must say which is which, or the answer will cite a release
date to a documentation page. That prompt change is the one place phase 3 touches generation.

## The three tools

| Tool | Signature | Notes |
|---|---|---|
| `github_schema` | `() -> SchemaSummary` | Fetches introspection once (decision 6), caches, returns only the top-level entry points and the handful of type names that matter for this repository. Not the schema. |
| `github_type` | `(name: str, fields: list[str] \| None) -> str` | The budgeted type description of decision 7: outline when `fields` is omitted, full signatures when it is given. This is the tool the model will call most. |
| `github_query` | `(query: str, variables: dict) -> QueryResult` | Read-only check (decision 4), local validation (decision 2), `dryRun` pre-flight and budget gate (decision 10), then the real call. Returns data, or a typed error that says which class it is. |

## Done when

A labelled set of about 12 frozen-answer GitHub questions plus the existing 27-case documentation
suite, run twice (HyDE is still in the docs path), reporting:

1. **answer accuracy** on the GitHub set;
2. **first-try query validity** — the share of questions whose first emitted query was both
   locally valid and accepted by GitHub;
3. **validity after repairs** — the same after at most 2 repairs;
4. **points per question** — median and max, from `rateLimit.cost`;
5. **routing accuracy** — on a combined set with a known label per question of `docs`, `github`
   or `both`.

And: the existing 27-case suite is re-run, so the router's cost to every docs-only turn is a
measured number rather than an assumption (P6).

## Measured in 3.2 (2026-09-14)

The schema cache, from `experiments/github_schema_size.py` on the Mac against the live API:

| | |
|---|---|
| types in GitHub's schema | 1,826 |
| introspection JSON, descriptions included | 3.2 MiB |
| built schema, tracemalloc | 5.3 MiB |
| built schema, RSS delta | **2.5 MiB** |

**Phase 2b decision 8 survives.** The task is 256 CPU units and 512 MiB from a measured 145 MiB
working set, and a cached GitHub schema adds about 1.7 percent of that. The fear written into
the experiment -- that a parsed 1,800-type schema might weigh 150 MiB and force a redeploy at a
bigger size -- was wrong by two orders of magnitude.

**Two numbers in the first draft of this spec and of `github_schema.py` were invented**, and are
corrected above and in the source: "about 1,100 types" (it is 1,826) and "tens of megabytes of
JSON" (it is 3.2 MiB). The argument they supported still holds -- 3.2 MiB is roughly 800,000
tokens, so the schema cannot go in a prompt -- but it was being argued from a number nobody had
measured.

Uncapped outline bytes, which is what `DEFAULT_BYTE_CAP` is now set from: Repository 5,971 over
145 fields, PullRequest 4,290, Issue 3,357, the Query root 1,362, and everything else under 800.

## Measured before 3.3 (2026-09-15): what dryRun does

`experiments/dry_run_semantics.py`, three requests against the live API. The published
description of the mechanism decision 10 is built on is one sentence, and two readings of it
build different runners, so it was asked rather than read.

| | cost | nodeCount | remaining | used |
|---|---|---|---|---|
| `rateLimit` only, before | - | - | 5000 | 0 |
| `rateLimit(dryRun: true)` + a repository query | 1 | 5 | 5000 | 0 |
| `rateLimit` only, after | - | - | 5000 | 0 |
| the same query, for real | 1 | 5 | 4999 | - |

Three results, and all three are better than the reading the spec assumed:

1. **dryRun suppresses evaluation.** `repository` is not null in the dry run response -- it is
   **absent**. The response data holds `rateLimit` and nothing else. So the pre-flight is
   necessarily its own request, and the runner makes two round trips per query, exactly as
   decision 2 priced.
2. **The dry run is free.** `remaining` did not move, and neither did `used`. So is a plain
   `rateLimit` query: three requests were made before the second probe and `used` was still 0.
   The gate therefore costs latency and nothing else, which means **every** generated query can
   be pre-flighted rather than only the suspicious ones.
3. **The prediction is exact, not an estimate.** The dry run reported cost 1 and nodeCount 5;
   the same query run for real reported cost 1 and nodeCount 5 and spent exactly one point.
   The gate reads the number the real call will charge.

**P3 is confirmed early**: a five-release single-repo query costs 1 point. Which means, as the
spec already warned, that the 10-point ceiling of decision 10 is probably inert, and the
`nodeCount` half of the gate is the half doing the work.

## Findings so far

1. **A marker that names the wrapper.** The first `_field_outline` reported
   `[connection of IssueConnection]`, which is true, well formed, and the one fact the model
   already had from the field type. What it cannot see from there is the node type, `Issue`.
   Caught by a test that knew the right answer; nothing about the output looked wrong. Same
   shape as phase 2b finding 1, arriving for the fifth time: **nothing has to look different
   from no answer, and a correctly shaped answer that carries no information looks like both.**
2. **A cap that could not be reached, and a fix that did not fix it.** Room for the truncation
   footer was held back at every step of the walk, against a footer that would never be written
   if the walk simply finished. The real `Repository` outline is 5,971 bytes and was cut at a
   6,000 byte cap -- by one field line -- and then told the truth about having been cut. The
   first fix exempted the final entry and changed nothing: the reserve bites at entry three,
   long before the last one is in sight. **The bug was never the arithmetic. It was answering a
   local question -- does this entry fit, plus a footer? -- in place of the only one that decides
   the outcome: does all of it fit?** The cap now asks that first, and reserves footer room only
   once truncation is known to be happening.
3. **The capped measurement could not measure the cap.** The first version of the experiment
   printed only the capped size, so `Repository` read as "5,936 bytes, CUT" -- over the cap, by
   an unknown amount. A measurement tool that reports a clipped number is the same bug it is
   meant to find. It now prints uncapped, capped and field count side by side.
4. **A fine-grained personal access token works against the GraphQL API.** Settled by the live
   test, not by the changelog. The classic `public_repo` token is not needed.
5. **An asserted aside inside a measurement is still an assertion.** The first version of the
   dryRun experiment printed `points moved by: 0 (a plain rateLimit query costs 1 itself)`. The
   parenthetical was invented, and the same run's own numbers disprove it -- `used` was 0 after
   three requests. A measurement tool is the last place an unmeasured claim should appear,
   because it arrives wearing the authority of everything around it. Every line of the verdict
   block is now computed from the responses.

## Predictions, written before the measurement

- **P1.** The most common first-try failure will be a **missing `first` or `last` on a
  connection**, not a wrong field name. GitHub requires it on every connection and it is a rule
  of GitHub's schema rather than of GraphQL, so a model writing from general GraphQL knowledge
  will omit it.
- **P2.** Local validation will catch **more than 80 percent** of first-try failures, so GitHub's
  own errors will be needed for fewer than one in five.
- **P3.** **Median points per question will be 1.** Every query is single-repo and most
  connections will ask for well under 100 nodes, so the division-by-100-and-round rule floors
  almost everything at the minimum.
- **P4.** **Routing accuracy will be higher than answer accuracy.** Choosing between two sources
  is an easier classification than authoring a correct query, and the errors will cluster on
  `both` — questions that need a release date *and* a documentation page will be routed to one
  of the two.
- **P5.** **Inline fragments on unions will be the failure the model does not learn from in one
  repair.** `search` returns `SearchResultItem` and `object` returns `GitObject`; both need
  `... on X`. Local validation will catch these, and the repair will fix the named case without
  generalising to the other one in the same turn.
- **P6.** The router node will add **200 to 400 ms to every turn**, docs-only turns included, and
  it will show up in the existing suite's **to-sources** mark rather than in first-token or done,
  because it sits before retrieval. Phase 2b's finding 3 is the reason to expect that shape: the
  marks before generation are the tight ones.

If P3 holds, the points budget of decision 10 is doing nothing and should be said to be doing
nothing rather than quietly kept as a feature.

## Measured in 3.3 (2026-09-15): what the pre-flight is, and is not

The runner was built on the belief that a request-level error has no `path` and a field-level
error does, and that the first class is the repairable one. The live suite disproved the second
half of that immediately, and then a tightened assertion disproved a consolation prize.

1. **A connection missing `first` is a FIELD-level error.** `path: ["repository", "releases"]`,
   `type: MISSING_PAGINATION_BOUNDARIES`. GitHub checks pagination bounds in its resolvers, not
   in schema validation, so the error legitimately carries a path -- and P1 says this is the
   single most common thing a model gets wrong. Deciding repairability from `path` classified it
   as hopeless, so **the repair loop would never have fired on the failure it exists for**, with
   37 offline tests agreeing. Repairability is now decided by `type` against a short allow-list
   (default: no), with provenance recorded per entry.
2. **Where an error happened and whether it is worth repairing are two questions.** Three
   discriminators were tried, each a single structural signal: `data is None` (wrong -- null
   propagates upward), `path` (right about where, wrong about whether), and finally `path` for
   where and `type` for whether. The mistake each time was looking for one answer to two
   questions.
3. **The pre-flight prices without validating.** The dry run costs the pagination-less query
   happily and only the paid call refuses it. So the pre-flight is a budget gate and nothing
   more, and **P1's failure mode necessarily costs a paid request**. GitHub computes what a query
   costs and whether a query is runnable in different places, and dryRun only runs the first.
4. **A test that accepts two answers is a measurement that cannot measure.** The first version
   asserted `stage in {"preflight", "field-error"}` and passed without saying which. Tightened to
   one value, it failed and produced finding 3. This is the same defect as `greatest(reltuples,
   0)` and the `[connection of IssueConnection]` marker, committed into a test by the person who
   had just written those two up.
5. **A second bug was hiding behind the first.** `run()` discarded the pre-flight's field errors
   entirely, so a failure the free request already knew about was rediscovered by the paid one.
   The branch that surfaces them is now there, tested offline -- and, given finding 3, **never
   observed to fire**. That is written in the source as a limitation rather than listed as a
   feature.

## Sub-steps

Each ends with a measured result, the Mac gate, a commit and CI, in the project's usual order.

- **3.1 — this spec and the lesson.** No code. Done when the spec is committed.
- **3.2 — the GraphQL client, introspection and the schema cache.** `github_schema` and
  `github_type`. Tests against a recorded introspection fixture, no network. One live integration
  test behind the `integration` marker. Done when a type lookup for `Repository`, `Release`,
  `Issue` and the `SearchResultItem` union returns under the byte cap and the omitted-field count
  is right.
- **3.3 — the query runner.** Read-only check, local validation, `dryRun` pre-flight, the budget
  gate, and the split between request-level and field-level errors. Done when each of the five
  failure shapes has a test built from a recorded GitHub response, and a mutation is refused
  offline.
- **3.4 — the subagent subgraph.** Write, validate, pre-flight, run, repair, summarise. Done when
  a question with a deliberately broken first attempt is repaired within the cap, the repair
  count is in the state, and every attempt appears as its own Langfuse span.
- **3.5 — the router and the graph wiring.** The router node, the three-way route, the merge of
  two evidence sets, and the generation prompt change that keeps them apart. Done when the
  existing 27-case suite still passes with the router in the path, twice, with the latency delta
  recorded against P6.
- **3.6 — the labelled set and the done-when run.** The 12 frozen questions, the routing labels,
  the harness reading the new per-turn GitHub block, and the five measures, run twice.

## Open questions

- **Where the labelled set lives.** The existing harness is TypeScript (`evals/run.ts`,
  `evals/dataset.ts`) and already has a Python target (`evals/agent-target.ts`,
  `EVAL_TARGET=python`). Adding the GitHub set there keeps one command for everything, which is
  worth more than keeping the phase's code in one language. Decision 12 is what makes that
  possible. To confirm in 3.6.
- **Whether the router should see the sub-queries.** It runs after `plan`, so the planner's
  resolved sub-queries are available to it. Giving it more context probably helps routing and
  certainly costs tokens. Start without them and see whether the failures are context failures.
- **Secondary rate limits.** Finding 9 of phase 2b says the harness is one client, so
  concurrency limits should not fire. If they do, that is a finding, not a bug to route around.

## New dependency

`graphql-core` (3.2.12 or newer at the time of writing; re-check with `uv` on the Mac before
pinning). It is the reference implementation, it is what `build_client_schema`, `parse` and
`validate` come from, and it has no dependencies of its own. `httpx` is already in the project
and is the client for the GitHub endpoint; no new HTTP library.
