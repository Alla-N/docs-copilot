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
3. **Three record schemas are already in the results directory.** The four TS runs have no `target`, `dirty`, `incomplete` or `errored`; `errored` arrives on 2026-09-14; `github` and `githubCost` arrive on 2026-09-15. Any diff has to have an opinion about this before it can print one line.
4. **`vitest.config.ts` includes `tests/**/*.test.ts` only.** Unit tests for the extracted evaluators have to live in `tests/`, not beside the evaluators (decision 8).

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
- **5.4 — the move:** datasets, evaluators, targets, and `run.ts` reduced to an orchestrator. The largest and least interesting diff, done last so that 5.2 and 5.3 are already green when it lands.
- **5.5 — the done-when run:** the full suite against the Python service at the new structure, diffed against `python-3.6c`, plus the README section in the new vocabulary.

Each sub-step ends the usual way: her Mac gate, a commit message in `Claude outputs/`, CI. Only 5.5 spends money.

---

## 6. Predictions

Written before the build, checked afterwards, wrong ones kept — same as phases 2b and 3.

**P1. The restructure moves zero numbers.** A full run at the end of 5.4 diffs empty against `python-3.6c` on every metric except latency and the fourth decimal of cost. This is the verification the phase gets for free, and it is why decision 7 exists.

**P2. At least three metrics in the 23 stored files cannot be compared without normalisation.** Named in advance: `target` (absent in the four TS runs), `errored` (absent before 2026-09-14), `github` and `githubCost` (absent before 2026-09-15). If the reader finds a fourth, the prediction was too conservative and that is worth saying.

**P3. Extracting the evaluators finds at least one more duplicated or divergent metric** beyond the two in section 1. Stated as a bet: one file of 1299 lines that grew across six sub-steps does not have exactly two.

**P4. `eval.yml` needs no edit,** because every script name survives. If it does need one, the restructure reached further than a rename should.

**P5. `run.ts` ends under 500 lines.** A number to be wrong about out loud. The orchestrator keeps the case loop, the console report and the exit codes, and those are most of what makes it long today.

---

## 7. What this phase is not

- Not a change to any verdict rule, denominator or exit code (decision 7).
- Not the A/B half of phase 5 (docs subagent with and without HyDE).
- Not Langfuse datasets or experiments (decision 10).
- Not a fix for the two defects the survey found — they are backlog items in `claude/docs-copilot-state.md`.
- Not a new gate. The diff reports and exits 0 (section 4).
