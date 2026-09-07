@AGENTS.md

# docs-copilot — project instructions for coding agents

A RAG assistant over the Vercel AI SDK docs (Next.js 16, AI SDK 7, OpenAI `gpt-4o-mini`,
`text-embedding-3-small`, Cohere `rerank-v3.5`, Supabase pgvector, Upstash). Live at
https://docs-copilot-w89t.vercel.app — one Vercel project, do not create a second.

The project's thesis is **every design decision is measured**. An agent working here is
expected to hold to that, not just to keep the tests green.

## Map

| Path | Owns |
|---|---|
| `app/api/chat/route.ts` | The **only** route: rate-limit → parse → plan → retrieve → generate |
| `lib/plan.ts` | Query planner (plan-and-execute) + HyDE hypotheticals; intents search / greeting / off-topic; `DEBUG_PLAN=1` |
| `lib/corpus.ts` | The 37-page list, shared by ingestion and experiments |
| `lib/retrieve.ts` | embed → pgvector (top 40) → rerank (top 5) → threshold 0.30; `buildSystemPrompt` |
| `lib/generation.ts` | Generation settings + the resolved-query message swap, shared by route, harness and judge calibration |
| `lib/env.ts` | `requireEnv` — the only way to read a required variable |
| `lib/refusal.ts` | `REFUSAL_MESSAGE` + `isRefusal` — dependency-free, shared by route/log/UI/evals |
| `lib/chat-request.ts` | zod parse-then-construct of the request body |
| `lib/rate-limit.ts` | Upstash sliding windows; fails OPEN when unconfigured (local dev) |
| `lib/visitor.ts` · `lib/landing.ts` | Visitor attribution: server-side sanitised headers → `query_log`; client captures referrer/UTM once per session |
| `scripts/ingest.ts` | Terminal-only ingestion; dry run by default, `--write` opt-in |
| `tests/` | Vitest unit tests for the pure functions; `npm test`, first CI step |
| `evals/dataset.ts` · `run.ts` | 27 hand-labelled cases; the harness that gates CI |
| `evals/planner.ts` | Planner-only eval: intent + sub-query assertions, no retrieval |
| `scripts/experiments/` | Runnable sources for every README number (threshold sweep, chunking) |
| `evals/judge.ts` · `calibrate-judge.ts` | Faithfulness judge (opt-in) and its calibration |
| `specs/` | Specs written before builds — read the relevant one before touching a subsystem |
| `db/000..004_*.sql` | schema · content hash · query log · visitor attribution + views · retrieval health view |

## Invariants — do not break these

1. **The content hash is byte-identical in TypeScript and SQL.** `scripts/ingest.ts`
   computes `sha256(source_url + "\n" + content)` exactly as `db/001_content_hash.sql`
   does. Change one and idempotent ingestion silently re-embeds the whole corpus.
2. **`app/api/` holds exactly one route.** Ingestion is a terminal script. Never add a
   mutating public endpoint — the old `GET /api/ingest` was an unauthenticated way to
   spend the owner's API credits.
   Analytics follows the same rule: page views beacon to Vercel, and question attribution
   rides on the existing rate-limited chat write — there is no `/api/track`.
3. **The eval harness and production share one code path.** Both call `plannedRetrieve`
   from `lib/plan.ts` and `buildSystemPrompt` from `lib/retrieve.ts`. Never re-implement
   retrieval inside `evals/` — a copy drifts, and drifts toward passing.
4. **Generation answers the planner's *resolved* query, not the raw message** — in the
   route, the harness AND the judge calibration, all through `lib/generation.ts`
   (`generationSettings` + `generationMessages`). Never inline a generation call with its
   own settings; the calibration script drifted that way once (Day 15). This is what makes
   "What is SDK?" behave like "What is the AI SDK?".
5. **Refusal detection is positional.** `isRefusal` matches the refusal's core sentence at
   the *start* (or after a "The documentation doesn't cover…" prefix). Do not loosen it to
   `includes()` — that scored partial answers as refusals and poisoned the production
   query log (Day 12). Do not tighten it to the full message — the model drops the polite
   tail and a must-refuse case went flaky (Day 13).
6. **Thresholds are calibrated, not guessed.** Rerank 0.30, cosine-fallback 0.45, 40
   candidates, top 5. Move one only with an eval run showing coverage *and* guardrails.
7. **Planner rewrites nothing off-topic into something retrievable.** Off-topic and
   "ignore the docs" parts are *dropped*; a wholly off-topic message gets intent
   `off-topic` (retrieval skipped, canned refusal). Never expand "deploy to AWS" into
   "deploy the Vercel AI SDK to AWS" — the planner did exactly that until Day 14 and
   guard-aws held only because the prompt refused. `evals/planner.ts` asserts this
   directly; the main suite can only see its consequences. Greeting intent applies only
   when the whole message is a greeting — a greeting attached to a question is a search.
8. **Requests are parsed-then-constructed.** Only `role` + text parts are read; `system`
   is never an accepted role; caps 20 msgs / 4k chars / 24k total — the total cap trims the
   OLDEST turns (it once dropped the newest and 400'd). Malformed JSON is a 400. Never
   `String(err)` to the client. Rate limit runs before any paid work. `IP_HASH_SALT` is
   mandatory whenever Upstash is configured — the module throws otherwise.
9. **Logging runs in `after()`**, never a bare `void promise` — serverless freezes the
   function after the response and the insert is lost. Greetings and off-topic refusals are
   logged too (mode `skipped`).

## How to change things here

- **Baseline first, one variable at a time, full suite after.** `npm run eval` — the
  *whole* thing, not `EVAL_ONLY`. Three fixes in this project passed the case under
  scrutiny and broke others; each was caught only by re-running everything. A change
  that turns two cases green and three red is a regression.
- **Prompt-level fixes have lost to structural fixes four times** (false refusal, judge,
  injection clauses, terse-query refusal). Reach for structure: a stage, a criterion, a
  detector — not a sentence in the prompt. Tightening refusal wording tips borderline
  cases to refuse; loosening it weakens guardrails. They are one dial.
- **Criteria are as likely to be the bug as the system.** Before adding `mustContain`,
  grep the corpus for the phrase — `"set of tools"` was a model paraphrase that never
  existed in the docs. Use `shouldAnswer: "either"` + `mustNotContain` when
  answer-vs-refuse is the wrong axis. The canned greeting is *not* a refusal, so
  `shouldAnswer: true` alone passes a short-circuit.
- **Every fixed bug in a pure function gets a unit test with the bug as the case** (`tests/`).
  The Day-12 `includes()` refusal, the newest-turn cap drop, the bag-of-words quote matcher —
  all sit in `tests/` now. `npm test` runs in seconds and before anything paid in CI.
- **Every eval case is added because it was observed**, with a note saying how.
  `evals/manual-qa.md` is where the hand tests live; a bug found there becomes a case.
- **Look, don't guess.** `EVAL_RUNS=0` for retrieval-only; `DEBUG_PLAN=1` prints each
  sub-query's hypothetical and pre-rerank candidates; a non-PASS prints the run that broke
  expectation. Use them before theorising about retrieval.
- **Spec first for anything non-trivial**: write `specs/<thing>.md` with the eval contract
  (which cases flip, which must not move), then build to it.
- Retrieval is nondeterministic even at temp 0 (the HyDE hypothetical is an LLM output).
  Don't reach for `seed`: the default Responses API silently ignores it, and switching the
  planner to `openai.chat` to make it count changed its outputs and regressed two cases
  (Day 14, reverted). A case near a threshold moves ±0.05 between runs; `FLAKY`
  is a real verdict, not a rounding error. Don't declare a case stable from one run. When a
  question has several genuinely correct pages, list them all in `expectedSource` (any-of);
  never pad it with neighbours to make recall pass.
- Refusal detection is compositional (lib/refusal.ts): negative opener about the docs +
  only canonical refusal sentences after it. Don't add one-off regexes per new shape.
- **A guardrail is held by one of three layers** — planner (off-topic), threshold, or
  prompt — and the harness prints which. Post-HyDE the threshold holds almost nothing on
  its own (see the sweep in the README); a "4/4 held" that moved from PROMPT to PLANNER or
  back is a change worth understanding, not a pass.
- **Every number in the README has a runnable source** under `scripts/experiments/`. A
  number whose script was deleted is a rumour; re-run the script, don't re-type the number.

## Commands

```
npm run dev                          # local app; limiter fails open without Upstash keys
npm test                             # Vitest unit tests (Mac, not the bridge VM)
npm run eval                         # full suite: 27 cases × 3 gens (8 for injection); exits non-zero on fail
EVAL_RUNS=0 npm run eval             # retrieval-only (still pays planner+embed+rerank); fails on a recall miss twice
npm run eval:planner                 # planner-only: intent, sub-query count, must/must-not strings; cheap
EVAL_ONLY=id1,id2 npm run eval       # subset — for diagnosis only, never as the pass signal
DEBUG_PLAN=1 EVAL_RUNS=0 npm run eval # hypotheticals + vector candidates per sub-query
EVAL_JUDGE=1 npm run eval            # + faithfulness judge per answered case
npm run eval:calibrate               # validate the judge against known-labelled answers first (last: 0/12 FA, 0/23 missed, n=35)
npm run exp:sweep                    # threshold / rerank / HyDE gap numbers for the README (~5 min)
npm run exp:chunking                 # chunker comparison on the real corpus vs the eval queries
npm run ingest / -- --write          # dry run prints the diff; --write applies it
npx tsc --noEmit                     # typecheck (CI runs this before eval)
```

## Environment gotchas

- Node **24** (`.nvmrc`). `npm i <one package>` can move others; re-check `tsc` after.
- Secrets live in `.env.local` (never committed). `.env.example` *is* tracked — keep it
  in sync when a new env var is read anywhere (`grep -r "process.env"`).
- Cohere key is a PRODUCTION key since Day 15 (the trial's 1,000 calls/month ran out
  mid-eval; every case silently fell back to cosine and looked like two retrieval failures).
  Rerank costs ~$0.002 per call now, one per sub-query — `RATE_DAILY_GLOBAL` was re-derived
  to 200. The harness refuses to score a cosine-fallback run (exit 3). On a trial key set
  `RERANK_INTERVAL_MS=6500`. `retrieve()` degrades to cosine ordering on failure with one
  retry, not two — don't remove the fallback or raise the retries (12 s per visitor).
- CI (`.github/workflows/eval.yml`): planner eval on every push; retrieval-only gate on
  branches; full suite on pushes to `main`, PRs and manual runs; weekly full + judge on
  Mondays. Needs 4 repo secrets. Workflow files can't be written through the remote bridge.
- Evals must run on the Mac, not in the Cowork bridge VM (`@esbuild/darwin-x64` is what's
  installed). `tsc` works anywhere.

## Out of scope — decided, not forgotten

ReAct / tool-calling loops belong to Artifact 2, not here. Server-side sessions (forged
assistant *text* is still client-supplied) is the real fix for history, deferred.
Content-defined chunk boundaries would remove the ~14× re-ingest write amplification;
deferred until an eval proves retrieval survives it.
