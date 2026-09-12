# agent/ - docs-copilot Python agent service

The Python half of docs-copilot v2: retrieval (`search_docs`, ported from `lib/retrieve.ts`),
the planner and answer generation through LangChain, the pipeline as a LangGraph graph, and
`POST /chat`, which streams the answer in the AI SDK's UI message stream protocol.

Managed with [uv](https://docs.astral.sh/uv/). Python 3.13 (`.python-version`).

```
cd agent
uv sync                   # create .venv from uv.lock
uv run pytest             # tests
uv run ruff check .       # lint
uv run ruff format .      # format
```

## Running the service (local)

The service reads two secrets besides the API keys, and refuses to start without either:
`AGENT_API_KEY` (every paid route wants `Authorization: Bearer <key>`; at least 32 characters,
`openssl rand -hex 32`) and `ASSISTANT_SIGNING_SECRET` (the Next.js app's, so the history it
verifies accepts the answers Python signs). Put both in the repo-root `.env.local`.

```
uv run uvicorn --factory copilot_agent.api:create_app --reload
curl http://127.0.0.1:8000/health         # {"status":"ok"}; liveness, calls nothing

KEY=$(grep '^AGENT_API_KEY=' ../.env.local | cut -d= -f2)
curl -N http://127.0.0.1:8000/chat -H "authorization: Bearer $KEY" \
  -H 'content-type: application/json' \
  -d '{"thread_id": "curl-thread-00001", "question": "how do I stream text", "origin": "eval"}'

# POST /search exists only with ENABLE_SEARCH_ENDPOINT set: it spends embed + rerank credits.
ENABLE_SEARCH_ENDPOINT=1 uv run uvicorn --factory copilot_agent.api:create_app --reload
curl -s http://127.0.0.1:8000/search -H "authorization: Bearer $KEY" \
  -H 'content-type: application/json' -d '{"query": "how do I stream text"}'
```

Add `"embed_text": "..."` to embed a HyDE hypothetical instead of the query. Interactive docs
at http://127.0.0.1:8000/docs. Startup opens the database pools and exits if it cannot connect,
if the checkpoint tables are not set up (see Conversation state below), or if `query_log` lacks
the columns `db/006_origin.sql` adds (see The query log below).

While the TypeScript retrieval still exists, `tests/test_ts_parity.py` keeps the calibrated
numbers identical on both sides.

## Planner (phase 2, step 2.1)

`copilot_agent/planner.py` ports `planQuery()` from `lib/plan.ts` through LangChain
(`ChatOpenAI` on the Responses API, structured output with zod's exact JSON schema). Its
parity is checked twice:

```
# exact: the Python planner sends the TypeScript planner's request (free, runs in CI)
npm run exp:planner-requests               # from the repo root; rewrites tests/golden/planner-requests.json
uv run pytest tests/test_planner_request_parity.py

# statistical: the planner eval on the real model, 23 cases x 5 runs (a few cents)
uv run python evals/planner_eval.py        # compare two runs with two of: npm run eval:planner
```

The golden stores a sha256 of `lib/plan.ts` and `evals/planner-cases.ts`, so editing either
fails the parity test until the golden is regenerated.

## Generation (step 2.2)

`copilot_agent/generation.py` ports `buildSystemPrompt` (lib/retrieve.ts) and
`generationSettings` / `generationMessages` (lib/generation.ts). Same technique as the planner:

```
npm run exp:generation-requests            # from the repo root; rewrites tests/golden/generation-requests.json
uv run pytest tests/test_generation_request_parity.py
```

The golden holds six synthetic cases chosen for the prompt's formatting edges and a canned
Responses API event stream; the test checks the request, the streamed deltas, the text and the
usage. Scores are printed with `js_to_fixed`: Python's `f"{0.125:.2f}"` is `0.12`, JavaScript's
`(0.125).toFixed(2)` is `0.13`.

## The chat graph (step 2.3)

`copilot_agent/graph.py` is the pipeline as a LangGraph graph:

```
START -> plan --route--> canned -> END                      greeting, off-topic
                    \--> retrieve x N -> merge -> generate -> END
```

One `Send` per sub-query runs the retrievals in parallel; a reducer collects them in plan order.
The graph emits LangGraph stream events (`updates` per node, `messages` for the answer tokens);
nothing in it knows about HTTP. Try it end to end, and measure what the framework costs:

```
uv run python -m copilot_agent.chat_cli "how do I stream text"   # real services, about a cent
uv run python experiments/graph_overhead.py                       # instant fakes, free
```

The planner model sets `streaming=False` explicitly and is tagged `nostream` in the graph: inside
a graph that streams `messages`, LangChain would otherwise send the planner request with
`"stream": true` and LangGraph would put its JSON in the answer stream.

## POST /chat (step 2.4)

Body: `{"thread_id": "...", "question": "...", "origin": "web" | "eval", "visitor": {...}}`, what
the Next.js route forwards (step 2.6, `lib/agent-forward.ts`) after it has rate-limited and parsed
the request. `origin` is required (the route says `web`, the eval harness `eval`); `visitor` is the
route's attribution for the query log, optional. Since step 2.5 there is no `history` field (it
is a 422): the turns before the question come from the thread (next section). The response is
the AI SDK's UI message stream, so `useChat` can read it unchanged:

```
data: {"type":"start","messageId":"..."}
data: {"type":"data-retrieval","data":{"mode":"reranked","intent":"search"}}
data: {"type":"data-sources","data":[...one pill per page...]}
data: {"type":"start-step"}  ...text-start, text-delta x N, text-end, finish-step...
data: {"type":"data-signature","data":{"sig":"v1...."}}
data: {"type":"finish","finishReason":"stop"}
data: [DONE]
```

- `copilot_agent/ui_stream.py` is the protocol and the adapter from graph events; the chunks
  are plain dicts, because the client's schema is strict (an extra key or a `null` kills the
  whole stream).
- `start` waits for the first data part: a failure during planning or retrieval then sends one
  error chunk and no message, where an early `start` would leave an empty bubble.
- A closed tab cancels the run. The graph runs in its own asyncio task (`api.in_own_task`):
  FastAPI cancels an SSE generator through an anyio cancel scope, and inside one LangGraph's
  unwinding is cut short and the running node is never cancelled (pinned by a canary test).
- `uv run python experiments/chat_latency.py "how do I stream text"` times sequential requests
  against a running service (headers, `start`, first token, `[DONE]`); request 1 is cold, the
  medians are over the rest.
- `tests/test_chat_api.py` writes the exact bytes of six scenarios to
  `tests/golden/chat-stream/`; `tests/python-stream-contract.test.ts` (Vitest, repo root) feeds
  them to the real `Chat` client and compares the message with the TypeScript route's for the
  same scenario. After changing the stream: `UPDATE_GOLDEN=1 uv run pytest tests/test_chat_api.py`,
  then `npm test`, and commit the goldens.

## Conversation state (step 2.5)

The graph is compiled with LangGraph's `AsyncPostgresSaver` on the same Supabase database, and
each conversation is a thread: `thread_id` in the request, `{"configurable": {"thread_id": ...}}`
in the run's config. The history the planner and the model read comes from the thread, never
from the caller, so there is no client-supplied assistant text to forge or replay.

```
uv run python -m copilot_agent.checkpoint setup    # once per database: tables, migrations, RLS on
uv run python -m copilot_agent.checkpoint check    # what the service checks at startup
uv run python -m copilot_agent.chat_cli "how do I stream text"                    # prints a thread id
uv run python -m copilot_agent.chat_cli "and how do I configure it?" --thread <id>
```

- **A turn is recorded only when it completes.** The last node (`canned` or `generate`) appends
  the question and the answer together; a failed or cancelled turn (Stop, a closed tab) leaves
  the thread unchanged. TypeScript keeps a stopped turn's question in the client's history
  (it drops only the unsigned partial answer), so that turn is a named difference.
- **Per-turn state is reset each turn** (`graph.turn_reset`). A checkpointed thread starts from
  the state the last turn left: the append reducer on `retrievals` otherwise merged turn 1's
  chunks into turn 2's answer.
- **The caps of `lib/chat-request.ts` apply to the stored turns** (`history.py`). The golden
  `tests/golden/history-caps.json` is written by the real `parseChatRequest`
  (`npm run exp:history-caps`) and `tests/test_history.py` compares. The thread stores only the
  last 19 messages, the most a request can read.
- **The tables are created from a terminal, never at startup**, with Row Level Security on:
  the saver creates them in `public`, which the Supabase Data API serves to the anon key. The
  service refuses to start if they are missing, behind the saver's migrations, or unlocked.
- **Strict serializer** (`checkpoint.serializer`): only LangGraph's safe types and the classes
  `ChatState` is made of are rebuilt from a blob. A class it does not allow comes back as raw
  data and one that fails to rebuild as `None`, silently: changing `HistoryTurn` changes how
  existing threads load.
- **`CHECKPOINT_DURABILITY`** (`sync`, `async`, `exit`) sets when LangGraph saves; the turns come
  out the same in all three. Default `exit`: a run is never resumed halfway, so saving after
  every step bought nothing. Measured with `experiments/checkpoint_overhead.py` (Supabase, 24
  turns per mode, 2 repeats), per turn:

  | durability | rows | stored | first token (vs none) | whole run (vs none) |
  |---|---|---|---|---|
  | sync | 33 | 25.3 KiB | +1.04 to 1.14 s | +1.3 to 1.4 s |
  | async | 33 | 25.3 KiB | +90 to 130 ms | +1.2 s |
  | exit | 7 | 9.3 KiB | +90 ms | +220 to 240 ms |

  The first token pays for reading the thread; with `exit` the one save comes after the answer's
  `finish` chunk and before `[DONE]`.
- **Stored retrievals are slim.** The first measurement stored 216 KiB per turn (76 KiB after
  compression), 213 of it the 100 candidate texts per sub-query, which nothing after the reranker
  reads, and every next turn read them back before its first token. `SubQueryRetrieval` now
  keeps the kept chunks, the mode, whether a rerank call went out, and the timings.
- **The replay gap is closed on this path.** Signing (invariant 8) proved an assistant turn was
  written by this server, not that it belongs to this conversation, so a signed answer could be
  replayed as history elsewhere. With history read from the thread, no assistant text comes from
  the client at all. What it creates instead: the thread id is a bearer capability (whoever has
  it continues the conversation).
- **Known gap:** two requests on one thread at the same moment both start from the same
  checkpoint and one turn is lost (pinned by a test). useChat never does this.

## The Next.js route forwards here (step 2.6)

With `AGENT_URL` set (e.g. `http://127.0.0.1:8000` in `.env.local`), `app/api/chat/route.ts`
rate-limits and parses as always, then `lib/agent-forward.ts` posts `{thread_id, question,
origin: "web", visitor}` with the key and the request's abort signal, and returns this service's
stream byte for byte (`new Response(upstream.body)`, the AI SDK's stream headers). Unset, the
route answers with the TypeScript pipeline: setting the variable is the switch, unsetting it the
rollback, and merging `v2` changes nothing in production by itself.

- `thread_id` is useChat's chat id; `app/page.tsx` makes it a `crypto.randomUUID()` (useChat's
  default `generateId()` uses `Math.random`, and the id is a bearer capability now). The route
  checks it against the same pattern as `api.THREAD_ID_PATTERN` (pinned) and answers a 400 for a
  bad one, before anything is paid for.
- No history is forwarded, signed or not. The route still verifies signatures while parsing, but
  on this path it is the thread that holds the conversation.
- A failure before the stream (the service down, a 401 or 422 or 5xx) is a 502 with the route's
  generic message; the service's text never reaches the browser. A failure inside the stream is an
  error chunk in a 200, which the route passes through without seeing it.

## The query log (step 2.6)

The route is a byte pipe on this path, so it cannot see the tokens, rerank calls and timings the
`query_log` row records: this service writes the row (`copilot_agent/query_log.py`), with the
columns `lib/query-log.ts` writes, in the same order (`tests/test_ts_parity.py` reads them), plus
two from `db/006_origin.sql`:

- `origin`: `web` or `eval`. Every view counts web rows only (so eval runs are neither traffic
  nor mined as eval cases), and the harness reads its own eval rows back through the new
  `query_cost` view for a measured cost per request. The prices moved into `query_cost`.
- `thread_id`: the conversation; many rows per thread.
- `trace_id` (`db/007_trace_id.sql`, step 2.7): this turn's Langfuse trace, or null when tracing
  is off.

One row per COMPLETED turn, the rule the thread follows: written when the last node reports, never
for a failed or cancelled turn. `refused` comes from `refusal.is_refusal`, a port of `isRefusal`
checked verdict by verdict against the TypeScript function through
`tests/golden/refusal-verdicts.json` (`npm run exp:refusal-verdicts` from the repo root after
editing `lib/refusal.ts`). The insert runs in a background task, a failed insert is logged
without the row (it holds the question), and shutdown waits up to 5 s for inserts in flight. A
visitor field in the wrong shape loses the field, not the answer. The log has its own
one-connection pool: at most six connections per process (search 4, checkpoints 1, log 1).

Run `db/006_origin.sql` and `db/007_trace_id.sql` in the Supabase SQL editor once; the service
refuses to start without either.

## Tracing (step 2.7)

Set both keys and every `/chat` turn is a trace in Langfuse; set neither and nothing is built,
nothing is exported, and the run config is the one step 2.6 sent.

```
LANGFUSE_PUBLIC_KEY=pk-lf-...      # .env.local, next to ASSISTANT_SIGNING_SECRET
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com     # follows the region
LANGFUSE_ENVIRONMENT=development                 # which deployment a trace came from
```

What ends up in a trace (`copilot_agent/tracing.py`):

- the LangChain callback handler turns one graph run into the tree: a span per node, a
  generation per model call with its model, tokens, cost and time to first token;
- the **trace id is made here**, before the run starts, and goes on the `query_log` row, so any
  logged turn can be opened;
- the **session is the thread id**, so a conversation reads turn by turn, and the **origin is a
  tag** (`web`, `eval`, `cli`);
- a failure the graph never raised (signing, encoding) becomes one ERROR span: the response is a
  200 that ends with an error chunk, so this is the only place outside this process's log where
  a paid, failed turn is visible at all;
- a `context` span carries the merged chunks the prompt was built from, texts included. The
  stream carries pages only, so this is what the faithfulness judge reads back on this target,
  and the first thing to look at when an answer is wrong. An answered turn that kept nothing
  records an empty list, which is itself the finding.

Two rules. **No Langfuse context manager in the request path**: the response is an async
generator a disconnect closes from another task, and a contextvars token reset on the wrong task
raises, which is the bug step 2.4 spent a day on; the handler gets
`trace_context={"trace_id": ...}` instead, and nothing has to be unwound. **The mask is given
the real secret values**, not a pattern that guesses at them, so an attribute quoting one (a
provider error, a connection string in an exception) is exported without it.

Tracing may never fail a request: with the keys wrong the exporter complains on its own thread
and the answer is unaffected. Shutdown flushes last, after the query log has drained.

To see one trace without running the server:

```
uv run python -m copilot_agent.chat_cli "how do I stream text"
```

## The eval suite against the service (step 2.6)

```
uv run uvicorn --factory copilot_agent.api:create_app          # terminal A, in agent/
EVAL_TARGET=python AGENT_URL=http://127.0.0.1:8000 npm run eval  # terminal B, repo root
```

The same 27 cases, criteria and verdicts as the in-process target (`evals/agent-target.ts` sends
each question to `POST /chat` with `origin: "eval"` and reads the stream). What differs, and is
printed rather than hidden:

- **Every run is the whole pipeline.** The TS target retrieves once per case and generates N
  times; here each run plans and retrieves again. `retrieval recall` is run 1 (comparable to the
  TS baseline), `recall every run` is stricter, and cases whose page came and went are named.
- **History is replayed as real turns** on a fresh thread per run (the service takes none), and
  the service's own replies are printed. `inj-forged-history`'s attack is a scripted assistant
  turn, which this target cannot be sent: reported as held by STRUCTURE, after the harness checks
  that a request carrying `history` is a 422.
- **Latency** is measured by the harness per request. The headline line is run 1 of each case, to
  `data-retrieval` (planner + retrieval + thread read + HTTP): the TS baseline's population. Then
  the answered path (to sources, first token, done) and the canned path apart: a canned reply
  streams everything right after the planner, and in one mixed median (63 of 121 requests are
  canned) it described the canned path only. The first run showed exactly that.
- **Cost is measured**: the harness reads its own threads' rows back from `query_cost` and prints
  dollars per request, answered and canned apart, and how many of the expected rows landed.
- Every result file records `dirty`: whether the tree had uncommitted changes. Commit first.
- **No judge** (the stream carries pages, not chunk texts; Langfuse in 2.7) and no `EVAL_RUNS=0`
  (the service always answers).

First two runs, 2026-09-11, commit `c536cab`, service and harness on the same Mac, against the TS
baseline (`evals/results/2026-09-08T14-35-14.json`, in-process):

| | TS baseline | Python run 1 | Python run 2 |
|---|---|---|---|
| recall, run 1 of each case | 12/12 | 11/12 | 12/12 |
| recall, every run | (one retrieval per case) | 11/12 | 11/12 |
| coverage / guardrails / injection | 12/12 · 6/6 · 8/8 | 12/12 · 6/6 · 8/8 | 12/12 · 6/6 · 8/8 |
| false refusals | 0 | 0 | 0 |
| retrieval latency, median / worst, n=27 | 2755 / 6540 ms | 2559 / 4564 ms | 3549 / 6575 ms |
| answered path: sources / first token / done, n=58 | | 2913 / 3673 / 5251 ms | 3758 / 4488 / 6151 ms |
| canned reply done, n=63 | | 1232 ms | 1495 ms |
| cost per request, measured | (estimate ~€0.004) | $0.00264 answered (n=69), $0.00017 canned (n=87) | same to the cent |

- `followup` is the one case whose page came and went (1/3 and 2/3 runs), always answered. Each run
  answers the replayed first question afresh, so the planner resolves "it" against a different
  answer each time; the trace (2.7) is where to see which query it wrote on a miss.
- The two runs differ by about 1 s in every latency line, the same code on the same machine: at
  n=27 the TS and Python retrieval medians are within that noise, not a measured difference.
- Rerank is most of an answered request's cost: one $0.002 call per sub-query, against roughly
  $0.0005 of tokens (planner about 1300 in, answer about 1650 in and 150 out, as chat_cli printed).

## Running in Docker (local)

From the repo root (the build context is `agent/`, so `.env.local` is never sent to Docker):

```
docker build -t copilot-agent agent

docker run --rm --name copilot-agent -p 127.0.0.1:8000:8000 \
  --env-file <(grep -E '^(OPENAI_API_KEY|COHERE_API_KEY|DATABASE_URL|AGENT_API_KEY|ASSISTANT_SIGNING_SECRET)=' .env.local) \
  copilot-agent
curl http://127.0.0.1:8000/health          # {"status":"ok"}
docker stop copilot-agent                  # SIGTERM: the lifespan closes the pool
```

Add `-e ENABLE_SEARCH_ENDPOINT=1` before the image name to get `POST /search`.

- **No secrets in the image.** They arrive as environment variables at `docker run`, and only
  the five the service reads. Not `--env-file .env.local`: that would also hand the
  container the Upstash token and the IP salt, and Docker's env-file is not dotenv, so
  a quoted value keeps its quotes.
- **`-p 127.0.0.1:8000:8000`, not `-p 8000:8000`.** The short form publishes on every
  interface of the Mac, and `/chat` and `/search` spend credits (the key guards them, the
  loopback bind keeps them off the network as well). Inside the container uvicorn binds
  `0.0.0.0` on purpose: Docker's port forward does not reach the container's own loopback.
- **Runs as uid 999, not root**, and cannot modify its own virtualenv.
- **A dead database fails startup** after psycopg's 10 s pool timeout, and the container exits.
- The uv version (0.12.5) is pinned in three places: the Mac, `Dockerfile` and
  `.github/workflows/agent.yml`. Bump them together.

CI (`.github/workflows/agent.yml`) runs ruff, pytest (no integration tests, no secrets) and a
`docker build` with an import smoke test on every push that touches `agent/`.
