# Phase 5 — the eval subsystem in the vocabulary of the role

Written 2026-09-16, before any code, at `3548c5f`. This spec covers the **renaming half** of phase 5: datasets, evaluators, baselines, run diffing, aggregate metrics. It is the last item on the minimum-before-applying list in `claude/windward-plan.md`.

The other half of phase 5 — A/B-ing a single subagent (docs with and without HyDE) and mirroring datasets into Langfuse experiments — is **out of scope here** and stays in the plan for later.

---

## 1. Why this phase is a rename and not a rewrite

The survey came first. What is in `evals/` today, measured rather than remembered:

| File | Lines | What it actually is |
|---|---:|---|
| `run.ts` | 1299 | Config, two targets, the case loop, the scoring, every evaluator inline, the GitHub section, the report, the record writer, five exit codes |
| `dataset.ts` | 468 | The golden 27 — a dataset, with its criteria documented case by case |
| `github-cases.ts` | 260 | The 13 frozen-answer GitHub questions — a dataset, with its own denominators, cost line and exit code |
| `planner-cases.ts` | 170 | The 23 planner cases — a dataset |
| `planner.ts` | 71 | Its runner |
| `judge.ts` | 459 | The faithfulness evaluator, verdict computed in code — already the shape every evaluator should have |
| `agent-target.ts` | 324 | The Python target: HTTP, thread ids, and the readers for `query_log` and Langfuse |
| `langfuse-api.ts` | 93 | Trace reading |
| `calibrate-judge.ts` | 172 | The judge's calibration experiment |
| `results/*.json` | 23 files | Stored runs with a `summary` block — baselines in all but name |

**Three datasets, several evaluators, twenty-three baselines and two targets already exist.** None of them are called that, one file holds most of them, and there is no way to compare two stored runs except by eye. So the work is to make the structure say what the code already does, and to add the one thing that is genuinely missing — the diff.

That framing matters for a second reason. A restructure that also changes behaviour cannot be verified, because there is nothing to compare it against. This one can: see prediction P1.

### What the survey turned up on the way

Recorded here because a survey that finds nothing usually means nobody looked.

1. **The injection numerator is computed twice** — once for the console line (`resisted`) and once, as a separately written expression, inside the record's `summary.injection`. Two spellings of one metric is exactly what an `evaluators/` directory exists to prevent.
2. **The console guards an empty latency population and the record does not.** The console prints `no successful retrievals` when every retrieval errored; the record writes `Math.round(Math.max(...sorted))`, which is `-Infinity`, which `JSON.stringify` silently turns into `null`. Only reachable on a run that is already `incomplete`, so it is a latent inconsistency rather than a live bug — **backlog, not this step** (decision 7).
3. **Three record schemas are already in the results directory.** The five TS runs have no `target`, `dirty`, `incomplete` or `errored`; `dirty` arrives with the Python target on 2026-09-11; `errored` and `incomplete` arrive on 2026-09-14 with the ERROR verdict; `github` and `githubCost` arrive on 2026-09-15 with the labelled set. Six of the 23 runs carry the GitHub set. Counted, not remembered — the first draft of this spec said four TS runs. Any diff has to have an opinion about this before it can print one line.
4. **`vitest.config.ts` includes `tests/**/*.test.ts` only.** Unit tests for the extracted evaluators have to live in `tests/`, not beside the evaluators (decision 8).

### Two findings the phase produced that the survey could not have

5. **A stable aggregate can hide an unstable composition (5.5).** Three runs of identical code reported the GitHub set as 15/26 every time. Underneath, two different cases traded PASS and VARIED on every run: baseline to run 1 moved `gh-issue-1-title` and `gh-issue-2-closed`; baseline to run 2 moved `gh-issue-1-title` and `gh-pr-1000-merged`; run 1 to run 2 moved `gh-issue-2-closed` and `gh-pr-1000-merged`. **The total is conserved, not reproducible.** A suite printing only the headline would call that perfectly stable and would then report a real fix as having done nothing, because a different question would have slipped to make room for it. This is finding 1 of phase 3.6 one level up: there, process metrics at ceiling hid missing answers; here, a stable aggregate hides a moving set of them. It is visible only because section 4 specified per-case lines for changed verdicts before any of the diff was written.

6. **A census over an open set is a test that fails when you do your job (5.5).** `tests/record.test.ts` asserted that exactly six stored runs carry the labelled GitHub set. The done-when run added two, and the assertion failed on the one commit of the phase that changed nothing whatever about the reader it tests. The trap is the repair: bump six to eight, green again, breaks at ten — **an assertion whose fix is always to edit the number teaches you to edit the test rather than read it.** It now asserts the contract it always meant, which is that the reader agrees with the file, checked against every stored run. The same file already had both spellings and got them right for the right reasons: the count of stored runs is a floor because runs get added, and the count of TypeScript runs is exact because that target stopped being written to and the set can never grow again. **A census is fine over a closed set and a bug over an open one, and nothing in the assertion says which you are looking at** — only whether the thing being counted can still change.

---

## 2. Decisions

All four of the first ones were put to Alla on 2026-09-16 with a recommendation; all four recommendations were taken.

**1. Full physical restructure.** `evals/datasets/`, `evals/evaluators/`, `evals/targets/`, `evals/baselines.json`, with `run.ts` reduced to an orchestrator. The alternatives were extracting the evaluators while leaving the datasets where they are, and a README-only vocabulary layer. Chosen because the phase exists so that the tree reads in the role's words in ten seconds, and because neither cheaper option delivers the done-when command.

**2. A baseline is a pointer, not a copy.** `evals/baselines.json` maps a name to a file already in `evals/results/`:

```json
{
  "ts-baseline":  { "file": "2026-09-08T14-35-14.json",        "note": "the TypeScript pipeline on main, 29a14c1" },
  "python-2.8":   { "file": "2026-09-11T20-43-10-python.json", "note": "the port at parity, 8146975" },
  "python-3.6c":  { "file": "2026-09-15T18-40-11-python.json", "note": "current head of v2, a0c083c" }
}
```

Copied snapshots were rejected on this project's own evidence: a second copy of a number that can drift is the failure mode phase 3 hit three times in one day. Raw file paths were rejected because then nothing is named and the README cannot cite a baseline.

**3. `eval:diff` does not spend money unless told to.** `npm run eval:diff -- --baseline python-3.6c --variant <name-or-file>` compares two stored runs for free. `--run` executes the variant first and then diffs. The done-when sentence is satisfied by `npm run eval:diff -- --baseline python-3.6c --run`, and the diff's own formatting can be developed against the 23 files already on disk without paying once.

**4. Records carry a `schemaVersion`; older files are normalised on read.** A reader upgrades the three shapes into one. **An absent field reads as `unknown` and prints as `—`, never as `0` or as absent-equals-unchanged.** This is the whole point: a field the old file never had must not render as a regression, and must not render as agreement either.

**5. A dataset is a descriptor, not a bare array.** `{ id, title, cases, evaluators, denominators, exitCode }`. Today the golden set's exit code is 1 and the GitHub set's is 5, and that difference is load-bearing — two different claims about the system, deliberately not collapsed. The descriptor is where that stops being a comment in `main()` and becomes data.

**6. An evaluator is a pure function.** `(case, runs) => score`, with the verdict computed in code. `judge.ts` already works this way — a model produces claims and quotes, and the pass/fail is decided by `isQuoteFound` in TypeScript — and that is the shape the others get. No evaluator calls a model except the faithfulness one, and that stays opt-in.

**7. The renaming half must not move a single number.** No verdict rule, denominator, exit code or default changes in this step. The two defects in section 1 are written into the backlog, not fixed here. Anything that would change a number gets its own sub-step after the restructure is verified green, so that the restructure stays falsifiable.

**8. Evaluator unit tests go in `tests/`,** because that is where `vitest.config.ts` looks. `tests/evaluators.test.ts`, one case per evaluator, seeded from the stored runs rather than from a live call.

**9. The planner eval becomes the third dataset but keeps its own runner and command.** It takes no target and does no retrieval, so folding it into `run.ts` would mean a target-shaped hole in it. It gets a descriptor and its metrics in the vocabulary; `npm run eval:planner` stays.

**10. Langfuse datasets and experiments are out of scope.** The plan already marks them optional. Adding a second home for the same numbers on the same day as the restructure would break P1's verification for a feature nobody has asked for yet.

---

## 3. Target shape

```
evals/
  run.ts                  orchestrator: knobs, the case loop, the report, the exit codes
  datasets/
    index.ts              the registry: id → descriptor
    golden.ts             the 27 (was dataset.ts)
    github.ts             the 13 (was github-cases.ts)
    planner.ts            the 23 (was planner-cases.ts)
    types.ts              EvalCase, GitHubCase, PlannerCase, Dataset
  evaluators/
    index.ts              the registry: id → evaluator
    recall.ts             expected page survived rerank + threshold; run 1 and every-run
    coverage.ts           answerable questions actually answered
    guardrails.ts         out-of-corpus questions refused, and which layer held
    injection.ts          adversarial prompts that got nothing
    refusal.ts            false refusals: page retrieved, answer refused
    faithfulness.ts       the judge (was judge.ts), opt-in
    github-answer.ts      frozen-literal accuracy, normalised for rendering only
    github-process.ts     first-try validity, validity after repairs, points, routing
  targets/
    index.ts              id → target
    in-process.ts         the TS pipeline (was inline in run.ts)
    agent-service.ts      the Python service over HTTP (was agent-target.ts)
    langfuse-api.ts       unchanged, used by the agent-service target
  baselines.json          name → stored run
  record.ts               the record schema, the writer, and the normalising reader
  diff.ts                 npm run eval:diff
  results/                unchanged, 23 files, still the single source of numbers
  experiments/
    calibrate-judge.ts    unchanged in content
```

`package.json` keeps `eval`, `eval:ci`, `eval:planner`, `eval:planner:ci`, `eval:calibrate` at the same names and adds `eval:diff`. **Prediction P4 depends on this: `.github/workflows/eval.yml` should need no edit at all.**

### Where the built tree differs from the sketch above, and why (5.4b)

Three deviations, all deliberate.

1. **`datasets/types.ts` is a barrel, not the home of the case types.** `evals/datasets/planner.ts` is pinned by SHA256 from the Python side — `agent/tests/test_planner_request_parity.py` hashes it by path — so lifting a type out of it costs a golden regeneration to buy nothing but symmetry. A file pinned by hash cannot be refactored for free, and that is a property of the file rather than an oversight. `EvalCase`'s field comments are also the golden set's criteria rationale, which belongs beside the cases. So: one import surface, three homes.
2. **`targets/agent-run.ts` was added.** `evaluators/types.ts` type-imports `TurnFacts` from `targets/agent-service.ts`, because a `Result` records the route the service chose. Putting that target's case loop in the same file would have made the two mutually dependent — erased today, since the import is type-only, and a real runtime cycle the first time either side needs a value. The client and the loop that drives it are separate files instead.
3. **The three `index.ts` files are descriptor tables, not dispatchers.** A registry that the record is BUILT by iterating would be the better end state, and it is also a change to how the summary object is assembled, which is exactly how a key stops being written or a conditional loses its guard. Decision 7 rules that out of this half. What the tables do instead is tie each evaluator to the `METRICS` id it is stored under, so that a number which cannot be diffed is findable — and `tests/evaluators.test.ts` fails if one is not.

---

## 4. The diff

The output is the thing a reviewer reads, so it is specified before it is built.

```
eval diff   baseline python-3.6c  (a0c083c, 2026-09-15T18-40-11)
            variant  <file>       (<commit>, <date>)

golden set (27 cases)
  recall run 1        12/12   →  12/12     =
  recall every run    12/12   →  12/12     =
  coverage            12/12   →  12/12     =
  guardrails            6/6   →    6/6     =
  injection             8/8   →    8/8     =
  false refusals          0   →      0     =
  cost              $0.2044   → $0.2044    =
  to sources          4289ms  →  4301ms    +12ms

github set (13 cases, 52 observations)
  answer accuracy     31/52   →  31/52     =
  first-try valid     43/43   →  43/43     =

per case
  (only cases whose verdict changed)

not comparable
  faithfulness        baseline did not record it
```

Rules:

- **A metric absent from either side prints under `not comparable` with the reason.** It never prints as `=`, and it never prints as a change.
- **Per-case lines are printed only for cases whose verdict changed**, plus any case present in one run and not the other. A 27-line table where 26 lines say `=` is how a reviewer stops reading diffs.
- A case that is `ERROR` on either side is called out separately: an incomplete run is not a baseline, and the diff says so at the top rather than quietly comparing 26 cases against 27.
- Exit code 0 always. The diff **reports**; it is not a gate. Making it a gate is a later decision with its own evidence, and a gate that nobody asked for would fire on HyDE variance on its first week.

---

## 5. Sub-steps

- **5.1 — this spec.** Committed before any code moves.
- **5.2 — `record.ts`:** the schema, `schemaVersion: 1`, the writer lifted out of `main()` unchanged, and the normalising reader with its unit tests against all 23 stored files. Built first because it is the only part with a testable contract that costs nothing to exercise, and because the diff depends on it.
- **5.3 — `diff.ts` and `baselines.json`,** developed against stored runs only. Free. At the end of 5.3 the done-when command exists in its compare-only form.
- **5.4 — the move:** datasets, evaluators, targets, and `run.ts` reduced to an orchestrator. The largest and least interesting diff, done last so that 5.2 and 5.3 are already green when it lands. **Split in two once it started:**
  - **5.4a — the renames.** Eight files moved into `datasets/`, `evaluators/` and `targets/`; five string-shaped references fixed; the frozen planner golden regenerated, and its three-line diff is what makes "the move changed only paths" a measurement rather than a claim.
  - **5.4b — the evaluators.** Everything `main()` was deciding for itself, pulled out into pure functions: the population rules, the per-case verdict, recall, coverage, guardrails with the layer attribution, injection, false refusals, and the labelled set's answer and process measures. Both targets' case loops moved out with them. The two duplicated numerators are gone, and the registries (`datasets/index.ts`, `evaluators/index.ts`, `targets/index.ts`) say what the suite measures and over what.
  - **The free check that belongs to neither:** `eval:diff`, built in 5.3, had never actually been RUN. It was exercised against five pairs of stored runs before 5.4b touched anything, so that a non-empty diff at 5.5 could only mean the extraction. It worked on the first attempt and overturned P1 — see section 6.
- **5.5 — the done-when run:** DONE, `1bf2a2f`. Two full runs at `b73a554`, three diffs, the README section in the new vocabulary, both result files committed so every figure in it traces to a stored run. Golden set `=` on fifteen of sixteen metrics; the sixteenth is `followup` and the stored record proves it HyDE rather than the extraction (see P1). **Phase 5's renaming half is complete.**

Each sub-step ends the usual way: her Mac gate, a commit message in `Claude outputs/`, CI. Only 5.5 spends money.

---

## 6. Predictions

Written before the build, checked afterwards, wrong ones kept — same as phases 2b and 3.

**P1. The restructure moves zero numbers.** A full run at the end of 5.4 diffs empty against `python-3.6c` on every metric except latency and the fourth decimal of cost. This is the verification the phase gets for free, and it is why decision 7 exists.

> **WRONG, and corrected before it could mislead 5.5 (2026-09-16, 5.4b).** Before touching the
> evaluators, `eval:diff` was pointed at `python-3.6c` and its own PAIR — `18-40-11` against
> `18-21-55`, the same commit `a0c083c`, two runs twenty minutes apart, no restructure anywhere
> near them. The golden set came back `=` on every ratio, with only latency and the fourth
> decimal of cost moving, exactly as P1 says. **The GitHub set did not:**
>
> ```
>   ▲ answer accuracy            15/26  →  16/26   +1
>   ! first-try query valid      21/21  →  22/22   denominator 21 → 22
>   ! valid after repairs        21/21  →  22/22   denominator 21 → 22
>   ! routing accuracy           37/37  →  38/38   denominator 37 → 38
>   ▲ canned by the planner          5  →      4   −1
>   github per case  (1 changed)
>     gh-issue-1-title         VARIED 1/2  →  PASS 2/2
> ```
>
> So P1 is true of the golden set and false of the labelled one, and 5.5 may only claim the
> first. An empty GitHub diff there would be luck; a non-empty one is not evidence of anything.
>
> **And the part worth keeping: the denominator is a random variable.** `21/21 → 22/22` on
> identical code is the same rate over a different number of attempts. How many turns reach the
> subagent depends on how many the planner cans — 5 here, 4 there — and the planner is model
> output. Finding 3 in section 1 read the 20/20-to-21/21 move between 3.6 and 3.6c as a
> consequence of the change between those two states. It is not. It is what this metric does
> when nothing changes at all, and the diff's `!` flag is the only reason either reading is
> visible. **A rate whose population is itself nondeterministic cannot be compared run to run
> without saying so**, which is the strongest argument the phase has produced for the diff
> existing.
>
> The check cost nothing: two stored files, no model calls. Running the diff against a run of
> the SAME commit, before trusting it against a different one, is now the rule.
>
> **SCORED AT 5.5, and the correction above was itself half wrong (2026-09-16, `1bf2a2f`).**
> Two full runs at `b73a554`, diffed against `python-3.6c`. The golden set came back `=` on
> **fifteen of sixteen** metrics — recall run 1, coverage, guardrails, injection, false
> refusals, every per-case verdict, cost at the fourth decimal. P1's golden-set claim holds.
>
> The sixteenth moved: `recall every run` read 11/12 in both runs against 12/12 in the
> baseline, on the case `followup`. **Not a number the restructure moved, and the stored file
> proves it rather than suggesting it:** the record says `foundRuns: 2/3` and
> `retrievedEvery: varied`, which are consistent, and `varied` is the *correct* verdict for 2
> of 3 — the evaluator did the right thing with the data it was handed. A broken aggregation
> returns a constant, not a fraction landing in the middle. `recall run 1` reads the same
> `foundPerRun` array through the same `expectedFound` and is 12/12, so a defect would have to
> hit the every-run filter while sparing run 1, and that filter is one line. The only remaining
> route is retrieval itself, which the extraction does not touch. `followup` has varied in
> **twelve of the twenty stored Python runs, across seven commits**.
>
> **And the GitHub set did not move at all.** Identical aggregates in all three files: 15/26,
> validity 21/21, routing 37/37, five turns canned. The correction above said this set was too
> noisy to read. The headline was perfectly steady — so that correction was right that the
> noise exists and wrong about where it lives. See finding 8 below, which is what was actually
> under it.

**P2. At least three metrics in the 23 stored files cannot be compared without normalisation.** Named in advance: `target` (absent in the five TS runs), `errored` (absent before 2026-09-14), `github` and `githubCost` (absent before 2026-09-15). If the reader finds a fourth, the prediction was too conservative and that is worth saying.

**P3. Extracting the evaluators finds at least one more duplicated or divergent metric** beyond the two in section 1. Stated as a bet: one file of 1299 lines that grew across six sub-steps does not have exactly two.

> **RIGHT (5.4b).** Faithfulness had the same defect as injection, and had had it longer: the
> console printed `judged.filter(yes).length / judged.length` while the record wrote
> `results.filter(=== "yes").length / results.filter(!== "—").length` — two separately written
> expressions over the same array, agreeing by arithmetic rather than by construction. Both now
> read `aggregateFaithfulness`. A second, smaller one turned up beside it: `GitHubRun.intent`
> was set on every observation of the labelled set and read by nothing at all.

**P4. `eval.yml` needs no edit,** because every script name survives. If it does need one, the restructure reached further than a rename should.

> **RIGHT, for `eval.yml`, and it nearly cost something anyway (5.4a, re-checked 5.4b).** No
> workflow script name moved. But `agent.yml` and `.pre-commit-config.yaml` both name
> `evals/datasets/planner.ts` as a PATH STRING, and 5.4a had to fix both — see finding 5. The
> prediction was about the wrong file. At 5.4b the unfiltered grep came back clean: the new
> modules are reached by extension (`\.(ts|tsx|mts)$`), so tsc, eslint and vitest picked them up
> with no config change at all, which is the difference between adding a file and moving one.

**P5. `run.ts` ends under 500 lines.** A number to be wrong about out loud. The orchestrator keeps the case loop, the console report and the exit codes, and those are most of what makes it long today.

> **WRONG (5.4b): 733 lines, down from 1303.** And the prediction was wrong for the reason it
> named in its own second sentence, which is the annoying part. What came out was every
> evaluator, both targets' case loops and the per-case types — 570 lines. What stayed is the
> console report and the record writer, and those two are nearly 400 lines between them because
> almost every line of them is a sentence explaining a number rather than computing one. Getting
> under 500 would have meant a `report.ts`, and splitting a report away from the exit codes it
> justifies is a change to how the run is assembled, which decision 7 puts outside this half.

---

## 7. What this phase is not

- Not a change to any verdict rule, denominator or exit code (decision 7).
- Not the A/B half of phase 5 (docs subagent with and without HyDE).
- Not Langfuse datasets or experiments (decision 10).
- Not a fix for the two defects the survey found — they are backlog items in `claude/docs-copilot-state.md`.
- Not a new gate. The diff reports and exits 0 (section 4).
