# Spec — Phase 3: the GitHub GraphQL data subagent

**Status:** IN PROGRESS, opened 2026-09-14. 3.1 to 3.5b are built, green and measured; 3.6 remains. Phase 3 of `claude/windward-plan.md`. Phases 1, 2,
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
| 1 | **A separate router node**, not a new field on the planner | Extending the planner's structured output would change its prompt, which forks it from the TypeScript baseline and invalidates `tests/test_planner_request_parity.py` and the 23x5 planner eval suite. Phase 2's retrieval numbers would stop being a clean comparison for the sake of saving one small call. A node of its own also makes routing accuracy a number that can be measured alone, which is what the phase-3 done-when asks for. Cost: one extra model call per turn, on every turn including docs-only ones. That is a regression to the existing 27-case latency marks and prediction P6 says where it will show. **Amended in 3.5:** the routes are two, not three (decision 18), and the cost came in at about 930 ms rather than P6's 200 to 400. **And the decision has a bill:** the planner gates off-topic messages before the router, and its prompt has no concept of the repository, so most repository questions never reach the router at all. 3.5b pays it. |
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
| 13 | **Schema exploration is tool nodes inside the subgraph** | The model calls `github_schema` and `github_type` itself and the graph answers them, rather than being handed a fixed set of type outlines assembled in code. The alternative is cheaper, fully deterministic and one model call per attempt -- but then the model never chooses what to look at, and the claim in the CV case degrades from *the agent introspects the schema* to *we hardcoded five types*. Cost: an extra model call per lookup, and lookups per question becomes a number 3.6 has to report. |
| 14 | **`github_query` is declared to the model as a tool and executed by a NODE; there is no repair node** | Declaring it gets a structured, validated call out of the model; running it in a node is what makes each attempt its own step (decision 5). Its reply goes back as an ordinary `ToolMessage`, so the model reads GitHub's refusal in exactly the place it reads a schema lookup, and the repair is the tool loop going round again. The diagram's `repair` box holds no work, and a node that holds no work is a span that says nothing. |
| 15 | **The subgraph compiles with `checkpointer=False`, not `None`** | `None` INHERITS the parent saver once nested, which would persist every schema outline the loop read, on every GitHub turn. Nothing in the loop needs to survive a restart: invariant 13 records a turn only when it completes, and a half-explored schema is not a turn. 2.5 is the measurement that decides this -- 213 of the 216 KiB written per turn were retrieved text in an appending list, which is exactly the shape of a `messages` list carrying two type outlines. |
| 16 | **`summarise` packages, it does not summarise** | The node emits the question, the query that ran and the JSON, capped, and generation grounds on that. A model call here would be a second place the answer could be invented, and faithfulness would then measure the summary rather than GitHub. The query travels with the data on purpose: a release date with the query that fetched it is a fact with provenance, and 2.8 settled that the judge measures grounding, not truth. |
| 17 | **GitHub's `first`/`last` rule is deliberately NOT in the system prompt** | P1 predicts a missing pagination bound is the most common first-try failure. One sentence in the prompt would make that prediction untestable. The rule is already in `github_type`'s detail output, so leaving it there means first-try validity measures whether the model looked before it wrote -- the question worth asking. If 3.6's number is bad the prompt is the first knob, and turning it will then be a measured change instead of a starting assumption. |

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

## Built in 3.4 (2026-09-15): the subgraph, and the diagram that could not be built

`agent/src/copilot_agent/github_agent.py`, 16 offline test functions (20 cases, one of
them parametrized), green on the first gate run.

    explore --lookup call--> lookup (github_schema, github_type) --> explore
       |
       +--github_query call--> run_query --repairable, under the cap--> explore
       |                           |
       |                           | ok, unrepairable, or out of repairs
       v                           v
    summarise <--------------------+

**6. The architecture diagram in this spec could not be built as drawn, and the reason is three
sections above it.** It draws write, validate, preflight and run as four nodes. Three of the four
need the parsed document, so four nodes means either a `DocumentNode` in state -- which the strict
serializer of invariant 13 will not rebuild, by design -- or parsing the same text three times, at
which point the node boundary buys spans and nothing else. `GitHubQueries.run()` already holds all
seven gates behind one call and `RunOutcome.stage` says which one refused, so one `run_query` node
per attempt keeps decision 5's claim exactly true. The diagram was drawn before the serializer was
remembered. Worth saying plainly: this was caught by reading the invariant before building, not by
a failure, which is the cheapest way this project has ever found one of these.

**7. The repair node holds no work.** Once `github_query` is a tool the model calls and a node the
graph runs, the refusal comes back as a `ToolMessage` and the next attempt is the next turn of the
same loop. A separate node would increment a counter that `len(attempts)` already knows. The
second thing the diagram asked for that the build did not need.

**8. Three API facts went out in a build unmeasured, and all three happened to be right.**
`compile(checkpointer=False)` being accepted at all, `ToolNode` still living at
`langgraph.prebuilt`, and `bind_tools(parallel_tool_calls=False)` passing the kwarg through. The
gate was green on the first run, so nothing was found -- but they were guesses, and the only
reason they are not a finding is luck. A probe would have cost one round trip.

Two branches in this file have never fired and say so in the source: the mixed-batch reply, which
`parallel_tool_calls=False` should prevent, and the transport-failure stage. Both are tested
offline. Phase 3 finding 9 is the rule being followed -- an untested branch that claims to save
something is worse than no branch, so the ones that are there are written as limitations.

## Built and measured in 3.5 (2026-09-15): the router and the graph wiring

`router.py`, the routed shape in `graph.py`, the additive generation prompt, `db/009_router.sql`,
and `experiments/subgraph_stream.py`. Two full eval runs, green, plus five hand-driven turns
through `chat_cli`.

### What the probe settled before the build

`experiments/subgraph_stream.py` was written first, because 3.4 finding 8 was that three API
facts went out in a build unmeasured and all three happened to be right. Four fakes, no network,
langgraph 1.2.11.

1. **`subgraphs=True` does not change the chunk shape.** Every chunk is a dict with keys
   `data`, `ns`, `type`, with the flag on or off; the namespace is already a field and never
   becomes a `(namespace, chunk)` pair. The claim that it would, recorded after the 3.4 lesson,
   came from reading rather than from the installed library. Phase 2b finding 6 again.
   **The decision it was supporting survives for a different, measured reason:** with
   `subgraphs=True` the subagent's model tokens DO enter the parent `messages` stream, named by
   their own node, and `ui_stream`'s `langgraph_node == "generate"` fence becomes the only thing
   between the subagent's query writing and the reader's answer bubble. With the flag off and a
   wrapper node, they never arrive at all. Off, so there is nothing to fence.
2. **Attaching the compiled subgraph directly loses its output, silently.** The parent ran to
   completion, wrote five checkpoints and produced an answer, and the evidence key was simply
   absent from the state. No exception, no warning. The child writes `result` and the parent
   declares `evidence`, and LangGraph dropped the write. So the wrapper node is not a style
   choice: it is the translation, and the alternative is a correctly shaped run carrying no
   information, which is this phase's through-line for the sixth time.
3. **`checkpointer=False` is doing real work, and the wrapper counts as nesting.** The same run
   wrote 5 checkpoints with `False` and 9 with `None`, and the `None` run put the subgraph's
   `messages`, `steps` and `result` into the PARENT's channel values. Decision 15 had only been
   confirmed one way; the contrast was one line of the experiment and it is what makes the
   decision a decision rather than a preference.

### Decision 1 reversed in part: the routes are two, not three

The first build had `docs`, `github` and `both`, and `github` answered from GitHub alone with
retrieval skipped. Its first measurement removed it. In the eval run of 2026-09-15 the router
chose it exactly twice, for **"what is new in AI SDK 7"** and **"what was changed in AI SDK 7"**,
the near-synonym pair this project already keeps as its retrieval diagnosis. Both turns retrieved
nothing, and three separate defects came out of that one fact:

- **One defect, two faces.** `new-7` answered from GitHub after opening with the documentation
  refusal sentence, so `isRefusal` (positional, invariant 5) logged an answered turn as refused
  and the harness scored it `answered 0/3`: red. `changed-7` answered cleanly from GitHub with no
  documentation behind it at all: **PASS**. Same defect, one visible and one silent, and the
  visible one is the lucky case. `recall 10/12` was the only trace of the second, and it does not
  name which two.
- **The suite got cheaper.** $0.18923 against the stored $0.1972, with a third model call added
  per turn, because the skipped rerank calls cost more than the router. Two runs of the broken
  code agreed on that figure to within $0.00006. A cost improvement that was a correctness
  regression, and it would have read as good news in any summary.
- **The measurement was invalid for P6.** The done-when assumed the router would route all 27
  cases to `docs`. It did not, so the to-sources mark averaged two different pipelines, one of
  them carrying a whole subagent loop. The clean number needed the `route` column, which is why
  `db/009` existed in time to produce it.

Making GitHub additive kills all of it by construction: a mis-route now costs latency and points
and can never cost an answer, and no case can go green with no documentation because of a routing
decision. What it costs is retrieval on every GitHub question, and a two-label routing measure in
3.6 instead of three.

**Worth stating plainly:** the replacement router prompt names "what is new in v7" and "what
changed in v7" as documentation questions, which is a prompt tuned against the two cases that had
to pass. It is defensible, because the migration guide genuinely is the documentation's answer,
and it is still the move this project has lost to structural fixes four times. Only 3.6's labelled
set can say whether the router still sends real repository questions to `both`.

### Measured: the 27-case suite with the router in the path

Two runs at the fixed code, both green: recall 12/12, coverage 12/12, guardrails 6/6, injection
8/8, false refusals 0, and `recallEveryRun` 12/12 then 11/12. The pair agrees to **31 ms** on
to-sources and **$0.00004** on cost, against the 815 ms spread 2.8 recorded on identical code.

| mark | 2.8 local | 3.5, two runs | delta |
|---|---|---|---|
| to sources | 2892 / 3023 | 3885 / 3854 | **+930 +/- 60** |
| retrieval median | 2647 / 2761 | 3509 / 3594 | +790 to +860 |
| first token minus to sources | 690 / 757 | 809 / 828 | unchanged |
| done minus first token | 1665 / 1373 | 1311 / 1668 | unchanged |
| cost per run | $0.1972 | $0.20220 / $0.20224 | +2.5 percent |

**P6 was right about the shape and wrong about the size.** It predicted 200 to 400 ms, landing on
the to-sources mark rather than first-token or done, for the reason phase 2b finding 3 gave: the
marks before generation are the tight ones. The increment is ~930 ms and it sits entirely inside
the to-sources segment, with every later segment unmoved. Off by about 2.5x.

Two things the route column added that no prediction covered:

- **Canned turns pay nothing.** 87 of 156 rows routed `null`, median 1176 ms: the planner gates
  greetings and off-topic messages before the router, so the regression lands on 69 turns rather
  than all of them.
- **The subagent costs about 4.4 s.** In the broken run, `github` turns had a median to-sources of
  8623 ms against `docs` 4205. That is the price of every `both` turn now, and it is the first
  number 3.6 has for the subagent's latency.

### Measured by hand: the capability the green runs never touched

`both` was chosen **zero times** across both green runs. The suite is 27 documentation questions,
so routing to `docs` throughout is correct, and it means neither run exercised the subgraph, the
wrapper, the additive prompt or the lazy schema fetch. Everything green in 3.5 is the docs
pipeline plus a router that says `docs`. Five `chat_cli` turns were run by hand for that reason,
and two of them found things the suite could not.

**The whole path works, once it is reached.** "is there an open issue about streamText retries"
routed `both`, ran retrieval and the subagent concurrently, fetched the schema lazily, wrote a
query that was **valid on the first try**, spent **1 point** after **2 lookups**, and returned
data. First evidence for P3 (median 1 point) and a first counter-observation for P1 (no missing
`first` on that query). The subagent took 8.7 s of the turn's 14.5 s.

**The planner's off-topic gate makes the capability unreachable for most of what it is for.**
Three of three canonical repository questions never reached the router:

| question | planner intent |
|---|---|
| when was ai 5.0.0 released | off-topic |
| when was **AI SDK** 5.0.0 released | off-topic |
| who merged the pull request that added the **AI SDK 7 migration guide** | off-topic |

The one that got through named an SDK API. So the planner admits a question only when it mentions
a documentation concept, which excludes release dates, PR authors, commits, contributors and
repository files: the spec's own list of what the subagent exists for. **This is decision 1's bill
coming due.** The router was made a separate node precisely so the planner's prompt would never
have to change, and the planner's prompt is now the wall standing in front of the router. It also
blocks 3.6 outright, since a frozen-answer set of release dates and merged PRs would be canned
before routing. Sub-step 3.5b is that fix.

**A marker leaked from the rule that explained it, not from the data.** The successful turn above
answered the reader with `NO GITHUB DATA. I could not find any open issues...`, on a turn whose
lookup had **succeeded**. The phrase was never in that turn's evidence. It was in the generation
prompt, which named the marker and said what it meant, so the model learned it there. Two lessons
and the second is the better one: a token that is both a machine signal and prompt text gets
quoted eventually, which `NO RELEVANT DOCUMENTATION FOUND` and the retrieval prompt's internal
marker already say; and **the marker was redundant from the start**, because `GitHubEvidence.ok`
was always the machine signal and this string only ever had to be read by a model. The token is
deleted rather than reworded, and what is left is prose that stays true if it is repeated.

### The prompt change, and what "additive" had to mean

A docs-only turn renders `SYSTEM_PROMPT_TEMPLATE` and nothing else, character for character, so
`test_generation_request_parity.py` still pins the Python generation request to the TypeScript one
and invariant 3's claim survives. A turn with documentation AND GitHub evidence appends a block.

A turn with GitHub evidence and NO documentation gets a **different prompt entirely**
(`GITHUB_ONLY_PROMPT_TEMPLATE`), because the pinned template's refusal rule is about the
documentation and a turn holding none of it cannot be asked to apply that rule. Asked anyway, the
model applied both rules in order and refused and answered in the same breath. The branch is on
what the turn HOLDS, not on what the router chose: a `both` turn whose chunks all fall below the
0.30 threshold holds no documentation either, and deciding it from the route would be one signal
answering two questions, which is finding 2 of the query runner arriving in a third place.

### New decisions

| # | Decision | Why |
|---|---|---|
| 18 | **Two routes. GitHub is additive, never exclusive.** | Reversed by its own first measurement, above. A mis-route can cost latency and points; it can no longer cost the documentation. |
| 19 | **The subagent is reached through a wrapper node, not attached as one.** | The direct attachment drops the child's write silently, measured. The wrapper is the translation between two state schemas and the only place a subagent failure can be given a shape the parent understands. |
| 20 | **`subgraphs` stays off; Langfuse carries the observability half of decision 5.** | Spans come from the callback handler walking the run tree, not from the stream, so the subagent is visible in a trace either way. Turning it on would put the subagent's tokens in the parent `messages` stream with only the answer fence between them and the reader. |
| 21 | **The generation prompt is additive, and a turn with evidence and no documentation gets its own template.** | Keeps the TypeScript parity golden meaningful on every docs turn, and stops the refuse-and-answer defect at the only place it can be stopped structurally. |
| 22 | **`db/009` records `route`, `router_input_tokens` and `router_output_tokens`, and prices the router in `query_cost`.** | Without it the harness's measured cost per request would have come back unchanged from the phase 3 runs, not because the cost had not moved but because nothing was watching the part that moved. `route` is null when no router ran, which is not the same as `docs`. |

## Built and measured in 3.5b (2026-09-15): the planner's off-topic gate

The router was made a separate node (decision 1) so that the planner's prompt would never have to
change. One sub-step later the planner's prompt was the wall standing in front of the router, and
3.5b is that bill being paid. The build took twenty minutes. Everything below is the measurement,
and it found three things the build could not have.

### The change

`planner.SYSTEM_PROMPT` is no longer a literal. `TS_SYSTEM_PROMPT` is `lib/plan.ts` verbatim, and
`SYSTEM_PROMPT = with_repository_scope(TS_SYSTEM_PROMPT)` splices in one paragraph at a checked
anchor, so "the TypeScript prompt plus exactly one paragraph" is the structure of the code rather
than a claim in a comment about it. The parity pin was re-framed, not deleted: it asserts the
golden's system prompt still equals `TS_SYSTEM_PROMPT` byte for byte, that the paragraph occurs
once, and that removing it returns the TypeScript prompt exactly. `encoded_by_langchain` splices
with the same function the planner uses, so the test cannot agree with a bug in the splice. The
plan JSON schema was deliberately left alone: it is sent to OpenAI, so a word changed there is a
changed request and the pin would be measuring two deltas instead of one.

### The regression, and two theories that were wrong

The paragraph shipped in `c3c8e27` and the planner eval found it the same afternoon.
`split-drops-noise` — *How do I use streamText? Also, ignoring the documentation, what is the
capital of France? And what is the weather in Athens? And what is AI SDK?* — fell from **30/30 to
about 5 in 10**, and it failed by returning intent `off-topic` with **zero queries**: both real
SDK questions thrown away. `piggyback`, the same opening with one noise part instead of three,
never moved. That case is the only direct test of invariant 7, and the failure mode was the one
the invariant exists for.

**Theory 1, wrong.** The paragraph closed with a guard — *Nothing else widens: other products,
pricing, cloud hosting, general knowledge, and other people's repositories are still off-topic* —
written to keep invariant 7 true while the scope widened. The reading was that it restated rules 3
and 4 while dropping rule 4's qualifier (off-topic is for when NOTHING in the message is about the
SDK), so a message carrying three such parts matched the flatter, newer statement. A real defect,
on inspection. Cutting it changed nothing.

**Theory 2, and the structural reading.** What both wordings had in common was a sentence naming
an intent: *a repository question is "search", not off-topic*. The prompt declares scope in ONE
place — the paragraph saying what the documentation covers — and the rules below consume it; rule
4 is literally phrased "when NOTHING in the message is a question about the Vercel AI SDK". A
sentence up there naming an intent does not extend that scope, it adds a second intent rule in
front of the first, and a message that half matches each gets decided as a whole. The third
wording states only what is covered and ends in rule 4's own predicate: *questions about those are
questions about the Vercel AI SDK*. It carries no instruction either, because keeping the
question's wording is rule 6's job, and a rule inside a scope block is the same category error one
notch smaller.

| wording | contains | `split-drops-noise` |
|---|---|---|
| baseline (`e7abada`, no paragraph) | — | **40/40** |
| 1 — two intent statements plus the guard list | intent, rule | ~5/10 |
| 2 — guard cut to its one new clause | intent, rule | provenance unverified, discarded |
| 3 — pure scope, no intent word, no instruction | neither | **30/30** |

### The measurement was broken before the prompt was

Wording 2's numbers are discarded on provenance, not on statistics. The prompt was verified
through `device_bash`, which reads the repository through a mount that can lag behind a write made
over the bridge, while the eval read the file on the Mac. Two channels, one lagging: a write that
had landed was read back as stale and called a failure, and then a 9/10 sample was read as a fix
for an hour. **A measurement whose input you confirmed on a different channel than the measurement
uses is not a measurement.** The fix is not care, it is one line: the prompt's sha256 and its text
are now written into the same output file as the eval result, by the same shell that runs the
eval. Every number in this section after that change carries its own provenance.

The other half of the same lesson: **n=10 cannot separate 0.85 from 1.0 here.** Wordings 1 and 2
pool to 42/50, one rate with no step between them, and the 9/10 in the middle of it was noise that
looked exactly like success. Every arm below is n=30 or more.

### Measured: the done-when

- **Routing.** `when was ai 5.0.0 released` plans `search`, routes `both`, and the subagent
  returns data on the first try for 1 point. `who merged the pull request that added the AI SDK 7
  migration guide` likewise. `when was langchain 1.0.0 released` stays `off-topic` and is canned —
  the boundary clause, and the only claim in the paragraph that no existing case tested.
- **Planner eval.** 23/23 PASS at 5 runs a case; `split-drops-noise` 30/30 at 30 runs against a
  40/40 baseline.
- **The 27-case suite, twice** (`2026-09-15T13-17-34-python.json`, `13-28-23`, commit `691c447`):
  recall 12/12 and 12/12 run-1, **12/12 and 12/12 every-run** (3.5 had 11/12 on one), coverage
  12/12, guardrails 6/6, injection 8/8, false refusals 0.

### Measured: what the paragraph cost

**The path split did not move: 69 answered / 87 canned, identical to 3.5.** The planner now admits
repository questions and not one suite case changed side. All four `guard-*` cases still report
*held by PLANNER*, `guard-langchain` among them.

**The cost delta is exactly the paragraph, and nothing else.** $0.2042 / $0.2043 against 3.5's
$0.20220 / $0.20224, so +$0.0020. The paragraph is about 95 tokens; 156 planner calls at
$0.15/1M input is **$0.0022**. The whole movement is prompt length, with nothing left over. It
moved UP, which is the direction that matters: 3.5's finding 2 was a suite that got *cheaper*
because retrieval had been skipped, and a cost that rises by precisely the tokens you added is
that finding's opposite.

**`route` says the latency is not routing.** Over both runs, `docs` 138 turns, `null` 174, and
**`both` zero** — the same shape 3.5 recorded, so nothing was re-routed. The canned path is the
clean instrument here because it is planner-only: 1198 ms → **1275 ms, +77 ms**, which is what 95
extra prompt tokens buy. The `docs` path moved +490 ms, of which 77 ms is that same planner cost;
the remaining ~410 ms sits inside retrieval, which varied by 370 ms *between the two runs of
identical code in this session*. Recorded as unexplained and not routing, rather than explained.

### Measured by accident: ok is not an answer, and it is 3.6's problem

`chat_cli` printed only `evidence.splitlines()[0]` — the header `summarise` writes, which reads
the same whether the query answered the question or returned an empty connection. Printing the
whole block explained a refusal that had looked like a generation bug:

    query: repository { releases(first: 10, orderBy: {field: CREATED_AT, direction: DESC}) { nodes { tagName createdAt } } }
    result: @ai-sdk/workflow-harness@1.0.111, @ai-sdk/vue@4.0.101, ... all createdAt 2026-09-15

The question was *when was ai 5.0.0 released*. The subagent listed the ten most recently created
releases instead of looking up the tag, so the evidence is valid, well formed, and contains no
answer — while `ok`, `attempts`, `first try valid` and `points` **all report success**. Generation
then refused rather than inventing a date from the adjacent rows, which is correct.

This is the phase's own headline finding one level up: an output that is correctly shaped and
carries no information is indistinguishable from a real answer, and now the *metrics* have the
property too. For 3.6 it is design-changing. Answer accuracy cannot be approximated by any
combination of the other four measures, because on this turn the other four are perfect. And the
12 frozen questions need answers that a generic listing cannot accidentally contain.

### New decision

| # | Decision | Why |
|---|---|---|
| 23 | **The added paragraph declares scope and nothing else — no intent, no rule.** | Measured twice at about 5 in 10 on `split-drops-noise` when it named an intent, 30/30 when it did not. The prompt declares scope in one place and the rules consume it, so a sentence naming an intent adds a second intent rule in front of rule 4 rather than widening what rule 4 reads. Instructions belong in the rules for the same reason. |

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
- **3.4 — the subagent subgraph. DONE.** Write, run, repair, summarise. A question whose first
  query omits `first` is repaired on the second attempt, the repair count is in the state, and the
  `updates` stream reports explore, run_query, explore, run_query, summarise as five steps. The
  Langfuse half of the done-when is deferred to 3.5, where the subgraph is nested and
  `subgraphs=True` decides whether the parent stream sees the same five steps.
- **3.5 — the router and the graph wiring. DONE.** The router node, the route, the merge of two
  evidence sets, and the generation prompt change that keeps them apart. Done when the existing
  27-case suite still passes with the router in the path, twice, with the latency delta recorded
  against P6. **Met:** two green runs, `recall` 12/12 both, P6 recorded at about +930 ms against a
  prediction of 200 to 400. The three-way route became two along the way, and the hand-driven
  turns found the planner gate below.
- **3.5b — the planner's off-topic gate. DONE.** Measured in 3.5: three of three canonical
  repository questions are classified `off-topic` by the planner and canned before the router sees
  them, so the capability is reachable only for questions that happen to name a documentation
  concept. Decided 2026-09-15: **teach the Python planner that the repository is in scope**, and
  re-frame the parity pin rather than delete it, so `test_planner_request_parity.py` asserts the
  Python prompt is the TypeScript prompt plus exactly one documented paragraph. Its own sub-step
  because it changes the planner, which means the 23x5 planner eval and the 27-case suite both
  have to be re-run: one variable at a time. Done when a release-date question routes `both`, the
  planner eval is unchanged on every existing case, and the 27-case suite passes twice. **Met, on
  the third wording of the paragraph:** `both` on both release questions and the other project's
  release still canned; planner eval 23/23, and `split-drops-noise` 30/30 against 40/40 after the
  first two wordings cost it about one run in six; suite green twice with every-run recall 12/12
  in both, cost up by exactly the paragraph's tokens, and `route` showing `both` chosen zero
  times. Commits `c3c8e27` (the pin and the first paragraph) and `691c447` (the scope rewrite,
  the fuller `chat_cli` evidence print). The section above is what the measurement cost and why
  two of the three wordings were wrong.
- **3.6 — the labelled set and the done-when run.** The 12 frozen questions, the routing labels,
  the harness reading the new per-turn GitHub block, and the five measures, run twice.

## Open questions

- **Where the labelled set lives.** The existing harness is TypeScript (`evals/run.ts`,
  `evals/dataset.ts`) and already has a Python target (`evals/agent-target.ts`,
  `EVAL_TARGET=python`). Adding the GitHub set there keeps one command for everything, which is
  worth more than keeping the phase's code in one language. Decision 12 is what makes that
  possible. To confirm in 3.6.
- ~~**Whether the router should see the sub-queries.**~~ Settled in 3.5: it does not. It sees the
  question and the recent history, folded in exactly as `planner_messages` folds it. History is
  not optional the way the sub-queries are: "and when was that released?" cannot be routed from
  its own text, and a router structurally unable to be right about follow-ups would be measuring
  something other than routing. The sub-queries stay out so that a routing failure is a routing
  failure and not a plan it inherited.
- **Whether `both` is ever chosen on a documentation suite.** It was chosen zero times across the
  two green runs of 3.5, which is correct for 27 documentation questions and means the route is
  unmeasured. 3.6's labelled set is the first thing that will exercise it, and the router prompt
  it has to get past was written after the only two mis-routes anyone has seen.
- **Secondary rate limits.** Finding 9 of phase 2b says the harness is one client, so
  concurrency limits should not fire. If they do, that is a finding, not a bug to route around.

## New dependency

`graphql-core` (3.2.12 or newer at the time of writing; re-check with `uv` on the Mac before
pinning). It is the reference implementation, it is what `build_client_schema`, `parse` and
`validate` come from, and it has no dependencies of its own. `httpx` is already in the project
and is the client for the GitHub endpoint; no new HTTP library.
