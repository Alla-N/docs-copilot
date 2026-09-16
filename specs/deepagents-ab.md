# Phase 4: a DeepAgents orchestrator beside the hand-built one

Design record, written 2026-09-16 before any code. Decisions first, predictions second, and both
scored at the done-when run the way `specs/eval-subsystem.md` scored phase 5's.

Phase 3 built a router node and two capabilities behind it. Phase 5 built the machinery to tell a
real change from noise. This phase spends both: the same two subagents get a second orchestrator,
and the question is whether a tool-calling agent harness beats a declared edge at a job the
declared edge already does at ceiling.

## What this phase asks

One sentence: **does replacing a classifier node and a deterministic fan-out with an agent that
decides at inference time change accuracy, tokens, latency or cost enough to pay for itself.**

The honest version of that question needs the two arms to differ in exactly one thing. Most of
this spec is about arranging that.

## Decisions

**1. The arm is an environment variable on one service.**
`Settings` gains `orchestrator: Literal["langgraph", "deepagents"] = "langgraph"`. One image, one
commit, one URL. The graph is built once at startup, so the service is restarted between arms --
the existing rule, not a new one.

Rejected: a second route (`POST /chat/deep`), which makes every measurement depend on which URL
the harness used, and a second eval target, which would duplicate `agent-service.ts`'s readers for
two arms that speak the identical HTTP contract.

**2. The arm goes in the run record, not only in the shell.**
A new record field `orchestrator`, typed `Maybe<T>` like every other, so a run stored before this
phase reads as *unknown* and never as `"langgraph"`. **A measurement whose arm you know only from
your own shell history is not a measurement** -- 3.5b's provenance rule, applied to an A/B where
the two arms produce byte-identical file names. `schemaVersion` goes up.

**3. The answer prompt does not change.** `generate()` keeps `generation.py`'s prompt byte for
byte and reads the same `ChatState` keys. Guardrails, injection, false refusals and faithfulness
then measure orchestration alone. This is 3.5b's lesson used before the fact rather than after it:
a metric that moves for two reasons is not a measurement.

**4. The planner does not change.** The `plan` node is untouched, `test_planner_request_parity.py`
keeps its hash pin to `lib/plan.ts`, and the 23x5 planner eval stays closed. A planner change
re-opens that suite and forks the Python planner from the TypeScript baseline that phases 1 and 2
exist to have kept comparable.

**5. Variant B replaces the router node and the retrieval fan-out, and nothing else.**
The deep agent receives the question and the planner's resolved sub-queries, and owns both
subagent calls. So two things that are *declared edges* in A become *model choices* in B:

- whether GitHub is consulted at all;
- whether each sub-query is actually retrieved.

That is the orchestration question stated as a difference in the graph. It also means **a recall
drop in B is an orchestration result, not a retrieval bug**, and the write-up must be able to tell
those apart -- which it can, because the record stores the plan's sub-queries and the retrievals
separately.

**6. The GitHub subagent attaches as a TOOL, not as a `CompiledSubAgent`.**
DeepAgents can take a compiled graph as a subagent, but it must have a `messages` state key and
what comes back to the orchestrator is a message. `GitHubEvidence` has fourteen fields and
**every process metric the 3.6 labelled set reports lives in the ones that are not prose** --
`first_try_valid`, `stages`, `lookups`, `points_spent`, `node_count`, `types_read`. So the tool
closes over a per-turn collector that the wrapper node drains after the agent run, and the
structured evidence reaches `ChatState` intact. See P3.

**7. The deep agent runs inside a wrapper node.**
Same reason as the GitHub subgraph's wrapper in 3.4: its state schema is not `ChatState`, and a
compiled graph attached directly writes back only the keys the parent declares, silently
(`experiments/subgraph_stream.py`, 2026-09-15). The wrapper writes `retrievals`, `github` and
`route`, and it is also the one place a deep-agent failure can be given a shape this graph
understands.

**8. `route` is declared in A and derived in B, and the write-up says so.**
In B the wrapper sets `both` if the GitHub tool was called during the turn and `docs` if it was
not. `None` still means no orchestrator ran at all. `db/009`'s column and the routing evaluator
work unchanged -- but a declared route and an observed one are not the same measurement even
though they land in the same column, and reporting them as one number without that sentence would
be this project's own favourite mistake.

**9. The virtual filesystem, the planning tool (`write_todos`) and skills stay off in 4.3.**
Each is its own sub-step with its own diff if it is ever turned on. A capability added and
measured in the same commit is 3.6's finding 3, and it cost that phase a week of ambiguity.

**10. No checkpointer inside the deep agent.** Conversation state stays the parent graph's job, as
it already is for the GitHub subgraph.

**11. Each arm is diffed against its OWN pair before the arms are diffed against each other.**
Non-negotiable, and the whole reason phase 5 came first. The GitHub set's aggregate is conserved
but its composition is not: three runs of identical code gave 15/26 every time while two cases
traded PASS and VARIED on each. An A/B verdict on that set needs the per-case lines, not the
headline.

**12. Budget: the golden 27 twice and the GitHub 13 twice, per arm.** Eight runs, about $1.10.

## The two shapes

```
A (today)     START -> plan --route--> canned -> END
                                  \--> router --> retrieve x N --\
                                              \--> github -------+--> merge -> generate -> END

B (phase 4)   START -> plan --route--> canned -> END
                                  \--> deep ------------------------> merge -> generate -> END

              deep = wrapper node around create_deep_agent(
                         model, tools=[search_docs_tool, github_tool],
                         system_prompt=..., subagents=None)
                     draining a per-turn collector into retrievals / github / route
```

`plan`, `canned`, `merge`, `generate` are shared, unmodified, and that is decisions 3 and 4 drawn
rather than written.

## What each arm can and cannot measure

| | A | B |
|---|---|---|
| retrievals per sub-query | exactly one, by construction | a model choice, observed |
| route | declared by a classifier | derived from whether the tool was called |
| subagent evidence | structured, through the wrapper | structured, through the collector (decision 6) |
| parallelism | one superstep, Sends meet at merge | whatever the loop emits |
| answer prompt | identical | identical |

## Predictions

Scored at 4.5, wrong ones included, as in the last two specs.

**P1. The deep agent will not call `search_docs` once per sub-query.** A guarantees one retrieval
per sub-query; B makes it a choice. Expect at least one golden case where fewer retrievals go out
than the plan has sub-queries, and recall run 1 below 12/12 at least once across the two runs.

**P2. Routing will tie.** A scored 74/74, 75/75, 75/75, and phase 3's finding 7 was that routing
was never the weak link. B's derived route should land inside that noise floor. If B loses here it
will be by not calling the tool, not by calling the wrong one.

**P3. The subagent-as-a-tool boundary will cost structure unless it is designed around.** Any path
carrying the subagent's result as a message loses every field that is not prose. Decision 6 is the
design around it; P3 says the design was necessary. **If the collector turns out to be
unnecessary, P3 is wrong, and that is the better outcome.**

**P4. B costs more tokens per turn.** An orchestrator system prompt plus two tool schemas plus the
loop's intermediate messages, against A's one short router call. Guess: +25% to +60% on the
router-equivalent tokens, and a suite cost visibly above $0.2042 -- which would be the first time
this suite's cost has moved past the fourth decimal since 3.5b.

**P5. B is slower, and the cause is serialisation.** A fans retrieval and GitHub out in one
superstep; a ReAct loop issues tool calls in sequence unless it emits them in parallel. Expect the
done-median above A's 5848 / 6294 ms.

**P6. The verdict will be "not worth it here, and the reason is the shape of the task".**
DeepAgents is a context-management harness for long, open-ended work. This turn is one question
deep and 12 to 18 seconds long, and the context it would isolate is already isolated by hand.
Writing the verdict down before measuring is what makes it falsifiable.

## Sub-steps

- **4.1 this spec.** No code, no money.
- **4.2 the dependency and a probe.** `uv add deepagents`, then a standalone script that builds a
  two-tool deep agent and prints what it calls on a handful of real questions. Settles the open
  facts below by running them rather than reading about them.
- **4.3 the build.** The settings field, `deep_orchestrator.py`, the wrapper node, the derived
  route, the second shape in `build_graph()`. Gate, commit, CI.
- **4.4 the record and the harness.** `orchestrator` in the run record and in `eval:diff`; tests.
- **4.5 the done-when.** Eight runs, six diffs (A against A, B against B, A against B on each
  set), the per-case read on the GitHub set, the A/B table, the written verdict, P1 to P6 scored.

## Done when

An A/B table over answer accuracy, tokens, latency and cost for both orchestrators on both
datasets; each arm diffed against its own same-commit pair first; the GitHub set read per case;
and a written verdict on when DeepAgents earns its overhead here.

## Open facts -- probe before building, do not write from memory

1. **The installed version.** `claude/windward-plan.md` says deepagents 0.7.13 (PyPI, 2026-09-11);
   the project's GitHub releases page on 2026-09-16 shows **0.6.11** stable with **0.7.0a1** as a
   pre-release. They disagree. `uv` and the lock file settle it, and the answer goes here.
2. **How the orchestrator's own loop is bounded.** 3.6's finding 2 -- a budget that can be spent
   entirely on preparation is a race the turn loses silently -- applies to a ReAct loop exactly as
   it applied to `MAX_LOOKUPS`. Find the recursion limit and set it deliberately.
3. **Whether tool calls are emitted in parallel by default**, and whether that is configurable.
   P5 depends on the answer and 4.2 should measure it, not assume it.
4. **Whether `create_deep_agent` accepts a model instance the way the existing nodes build one**
   (`openai_*_model(settings)`), so the arm difference is not secretly a model difference.
