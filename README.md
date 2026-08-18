# docs-copilot

A RAG assistant over the Vercel AI SDK documentation. Ask it a question about the SDK
and it answers from 853 indexed chunks of the real docs, with clickable source pills —
or refuses, when the docs don't cover it.

**Live:** https://docs-copilot-w89t.vercel.app/

The interesting part isn't that it works. It's that every design decision below was
made by measuring the alternative.

---

## Measured results

| Change | Before | After | How it was measured |
|---|---|---|---|
| Structure-aware chunking | 0.546 | **0.643** | Top-1 cosine similarity, same content, same query, only chunk boundaries differ |
| Cohere reranking | — | **2 of 6** | Queries where vector search ranked the right chunk below a worse one, fixed by the cross-encoder |
| Answerable vs unanswerable score gap | ~1.7× | **~8×** | Widening this gap is what makes a threshold able to separate the two |
| Idempotent ingestion | 853 embeds | **154** | Re-ingest after real upstream doc drift — 82% fewer embedding calls |
| Answer coverage (eval harness) | 4/5 | **5/5** | 9 labelled cases × 3 runs; guardrails held 4/4 throughout |

**Retrieval threshold was calibrated, not guessed.** Cosine similarity needed 0.45 to
separate answerable from unanswerable queries; after reranking the useful cut moved to
0.30, because rerank scores are distributed differently. Reusing the old number would
have silently rejected good results.

---

## Architecture

```mermaid
flowchart LR
    Q[User question] --> E[embed<br/>text-embedding-3-small]
    E --> V[(Supabase pgvector<br/>853 chunks · cosine)]
    V -->|top 20| R[rerank<br/>cohere rerank-v3.5]
    R -->|top 5| T{score >= 0.30?}
    T -->|no| REF[Refuse:<br/>not in the docs]
    T -->|yes| S[Grounded system prompt]
    S --> G[streamText · gpt-4o-mini · temp 0]
    G --> UI[Answer + source pills]

    subgraph offline [Ops — runs from a terminal, never deployed]
        D[37 AI SDK doc pages] --> C[structure-aware chunker]
        C --> H[sha256 content hash]
        H -->|diff vs stored| V
    end
```

**Retrieve wide, then narrow.** Vector search is fast but ranks by embedding proximity,
which is not the same as relevance. Pulling 20 candidates and letting a cross-encoder
re-score them recovers cases where the right chunk was buried at rank 8. The threshold
then runs on the *rerank* score, not the cosine score — the reranker is the component
that actually knows what relevance means.

**Two-layer refusal.** A numeric gate drops low-scoring chunks before the model sees
them, and the system prompt instructs refusal when context is empty. Either alone
leaks: the gate can't judge semantics, and the prompt alone will happily answer from
the model's own knowledge.

---

## Idempotent ingestion

Ingestion used to be a public `GET /api/ingest` that blind-inserted every chunk on
each call. Two problems: re-running duplicated the corpus, and once deployed it would
have been an unauthenticated endpoint anyone could use to spend my OpenAI credits — a
mutating GET that link-preview bots trigger on their own.

It's now a terminal script. **The deployed app only ever reads the vector store.**
Writes are opt-in (`--write`), so a mistyped command can't corrupt the table.

Each chunk is keyed by `sha256(source_url + "\n" + content)`, computed identically in
Postgres and TypeScript. A run diffs fresh hashes against stored ones and embeds only
what changed:

```
$ npm run ingest
pages 37  |  chunks 853
unchanged 699  ·  new 154  ·  stale 143
```

That output is from real drift — 7 of 37 AI SDK pages changed upstream between
ingestions. A second run reports `unchanged 853 · new 0 · stale 0` and spends nothing.

**Known limitation, quantified:** those 7 pages gained 11 chunks net, but cost 154
re-embeds — roughly **14× write amplification**. Chunk boundaries are positional, so a
paragraph inserted mid-page shifts every boundary below it and all those chunks hash
differently even where the prose is identical. Content-defined boundaries would fix
it; that change is deferred until an eval harness can prove retrieval quality survives
it.

---

## Other known limitations

- **False refusal on wording.** *"What is new in AI SDK 7"* refuses while *"what was
  changed in AI SDK 7"* answers. Diagnosed as generation-side — retrieval returns the
  right chunks either way, the strict prompt over-refuses on phrasing. Deliberately not
  patched: a narrow prompt tweak is whack-a-mole, and the general fix can't be verified
  without an eval set. Parked as a regression case.
- **Conversation history is client-supplied.** Request validation strips forged *structure*
  — extra roles, oversized payloads, unexpected parts — but not forged *text*. A client can
  still send a fabricated assistant turn. The real fix is server-side sessions.
- **The rate limiter fails open.** If Redis is unreachable, requests are allowed and the
  failure is logged. A demo staying up matters more than a few unmetered minutes; anywhere
  real money is at stake, fail closed and alert.
- **No ANN index on the embedding column** — deliberate. At 853 rows an exact scan has
  perfect recall and is fast; hnsw/ivfflat trades recall for speed and only pays off at
  a far larger corpus.
- **The chunker isn't code-fence-aware,** so reference pages with dense code blocks
  chunk mechanically.

---

## Run it

```bash
git clone https://github.com/Alla-N/docs-copilot && cd docs-copilot
npm install
cp .env.example .env.local     # fill in the four keys
```

```
SUPABASE_URL=          SUPABASE_SERVICE_KEY=
OPENAI_API_KEY=        COHERE_API_KEY=
```

Then, in the Supabase SQL editor, run `db/000_schema.sql` followed by
`db/001_content_hash.sql`. Populate the corpus and start:

```bash
npm run ingest              # dry run — prints the diff, writes nothing
npm run ingest -- --write   # apply it
npm run dev
```

`npm run exp:chunking` re-runs the chunking experiment behind the 0.546 → 0.643 number.

## Security

Public endpoint, personal API keys — so the threat model is cost first, then grounding.

**Cost.** Three sliding windows in Redis (Upstash): burst 10/min, per-visitor 50/day, and a
**global 800/day ceiling ≈ €0.50**, derived from measured per-request cost rather than picked.
The global one is the point: a per-user limit bounds abuse, but a public link means hundreds
of distinct IPs each with their own allowance, so only a global counter bounds spend. Redis
rather than Postgres because a limiter must be atomic — count-then-insert races exactly when
you're being hit hardest. In-memory is worse: serverless instances don't share memory, so the
effective ceiling rises with the load it exists to stop.

Callers are identified by a salted hash of their IP, never the raw address.

**Input.** The request body is parsed and rebuilt rather than trusted — only `role` and text
parts are read, capped at 20 messages / 4,000 chars each / 24,000 total, and `system` is not
an accepted role.

**Injection.** Eight adversarial cases run in the regression suite, 8 attempts each, two of
them multi-turn:

```
injection resisted 8/8   (2 multi-turn, 6 single-turn, 8 attempts each)
```

That number means *those eight attacks don't work* — not that the app is safe. It exists
because a manual attempt extracted the entire system prompt while the suite reported the same
case as passing: every eval case was single-turn, and the attack only reproduced inside a
conversation. Adding history to the dataset reproduced it immediately, and the fix then had a
failing test to prove itself against.

Worth stating plainly: the exposure here is low because the model **has no tools** and the
corpus is public. Posture comes from what capability you expose, not from how the prompt is
worded.

## Evals

```bash
npm run eval                 # 9 labelled cases × 3 generations
EVAL_RUNS=0 npm run eval     # retrieval-only diagnostic — free, no generation calls
```

Reports retrieval recall, answer coverage, guardrails held, and median retrieval latency —
and, per guardrail, **which layer refused it**. That last one matters: two of the four
guardrails retrieve nothing past the threshold, so the model never sees them and they stay
green no matter what the prompt says. A test that can't fail in the direction you're
changing is decoration, and the harness says so out loud rather than quietly counting it
as a pass.

Retrieval runs once per case (deterministic); generation runs N times, because temperature 0
lowers variance without eliminating it. A case passing 2 of 3 is reported `FLAKY`, not
rounded up.

---

## Stack

Next.js 16 · AI SDK 7 (TypeScript) · OpenAI `gpt-4o-mini` + `text-embedding-3-small`
(1536d) · Cohere `rerank-v3.5` · Supabase pgvector · Vercel
