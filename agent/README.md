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
  -d '{"thread_id": "curl-thread-00001", "question": "how do I stream text"}'

# POST /search exists only with ENABLE_SEARCH_ENDPOINT set: it spends embed + rerank credits.
ENABLE_SEARCH_ENDPOINT=1 uv run uvicorn --factory copilot_agent.api:create_app --reload
curl -s http://127.0.0.1:8000/search -H "authorization: Bearer $KEY" \
  -H 'content-type: application/json' -d '{"query": "how do I stream text"}'
```

Add `"embed_text": "..."` to embed a HyDE hypothetical instead of the query. Interactive docs
at http://127.0.0.1:8000/docs. Startup opens the database pools and exits if it cannot connect,
or if the checkpoint tables are not set up (see Conversation state below).

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

Body: `{"thread_id": "...", "question": "..."}`, what the Next.js route will forward in step 2.6
after it has rate-limited and parsed the request. Since step 2.5 there is no `history` field (it
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
