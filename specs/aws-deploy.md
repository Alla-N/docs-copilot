# Spec — Phase 2b: the agent service on AWS

**Status:** IN PROGRESS, opened 2026-09-14. Step 2b of `claude/windward-plan.md`.
**Why now:** phase 2 is complete and measured. The role brief asks for AWS, and the honest
version of that claim needs a container the author actually built, pushed, ran behind a load
balancer, measured, and then deleted on purpose.

## What is being claimed — and what is not

**Claimed:** built a Python service into an image, pushed it to ECR, ran it on ECS Fargate
behind an Application Load Balancer with its secrets in Secrets Manager and its logs in
CloudWatch, and ran the project's own 27-case eval suite against the deployed URL.

**Not claimed:** VPC design, multi-account IAM, Kubernetes, Terraform, blue/green, cost
optimisation at scale. ECS Express Mode creates the network and the load balancer from
defaults; the interesting work here is what the service needs to survive being deployed,
not the deployment.

**The service is torn down after the measurement** (decision, 2026-09-14). The claim is
earned by having done it and by the recorded numbers, not by paying about $35 a month to
keep a URL warm. Everything below is scripted so a redeploy for a demo is minutes.

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | **Region `eu-west-1`** | The Supabase session pooler is in eu-west-1. A turn makes several Postgres round trips and only one inbound HTTP hop, so the container belongs next to the database, not next to the author. |
| 2 | **Free account plan at sign-up** | Read from AWS's supported-services list, 2026-09-14: Amazon ECS, ECR, Elastic Load Balancing, Secrets Manager, CloudWatch and Bedrock are all supported. Fargate is in neither list, because it is a launch type of ECS and bills under it rather than a separately listed service, so everything this phase needs is available. The plan also cannot be billed past its credits, which makes a surprise bill structurally impossible, and Free to Paid is one click while Paid to Free is not. **The first reading of that page concluded Fargate was excluded**: a summary saying "not listed" became "not allowed". Absence from a list is not a finding, which is the lesson of e6bbc7d arriving from a different direction. |
| 3 | **An IAM user with an access key, named as a compromise** | AWS IAM Identity Center is on the not-supported list for the Free plan, and so is AWS Organizations, which Identity Center requires. There is therefore no SSO here: the CLI holds a long-lived credential in the local AWS config. That is the older pattern and the wrong default for real work. It is accepted here because the plan caps the account's spend, the account is torn down after the measurement, and it holds nothing but a public documentation corpus. One user only, MFA on it, and the plan closes the account by itself in six months. |
| 4 | **`AGENT_URL` stays out of Vercel Production** | Production keeps answering with the TypeScript pipeline. Invariant 14's rollback story is "unset a variable"; not setting it in Production is the same story one step earlier. The eval harness talks to the AWS URL directly. |
| 5 | **One Secrets Manager secret with six JSON keys**, not six secrets | ECS resolves `valueFrom` as `<secret-arn>:<json-key>::`, so one secret can back six environment variables. Secrets Manager is priced per secret per month, and one blast radius is easier to rotate than six. Cost is the smaller reason; rotation is the real one. |
| 6 | **`maxTaskCount` 2** | Measured, not chosen: the Supabase pool is 15 connections and a busy task holds 6 (`agent/README.md`). Autoscaling that can outgrow the database is a worse outage than autoscaling that cannot. |
| 7 | **`--cpu-architecture X86_64`, passed explicitly** | The Mac is a 2020 Intel i5-1038NG7, settled by the CPU brand string and by `sysctl.proc_translated` not existing, not by `uname -m`, which reports x86_64 under Rosetta too and so cannot answer the question on its own. The image built and measured in 1g is therefore already x86-64, and ARM64 would mean deploying an artifact never built or run here, from an emulated cross build. Graviton is about 20 percent cheaper and would be the right answer on an Apple Silicon machine; it is the wrong one here. Passed explicitly rather than defaulted, so the deployed architecture is a stated choice. **This parameter is not in the published command reference at all** and was found in the synopsis the installed CLI prints. Installed code settles shapes, again. |
| 8 | **`--cpu 256 --memory 512`** | Measured in 2b.2, not chosen. The container's working set (cgroup `anon`) is 144.5 and 146.9 MiB across two caps and two positions: the process is the same size whatever the cap. Worst observed total including page cache is 218.3 MiB. Both 512m and 1024m survived every run with no OOM kill, so 512 MiB wins on about 3.5 times headroom over the part that can actually kill a task. 512 MiB is offered only with 256 CPU units, so one measurement settles both parameters. |

## Step 0 — the account (nothing is deployed here)

Done before any repo change. On the Mac, in this order:

1. Sign up at `aws.amazon.com`. **Choose the free account plan** (decision 2).
2. **Root user: turn on MFA, then stop using root.** Everything after this is the IAM user
   from step 4. Root exists to close the account, to change the plan later, and to fix a
   locked-out IAM user.
3. **Budget before resources.** Billing and Cost Management → Budgets → a monthly cost
   budget of $10 with alerts at 50 / 80 / 100 percent to the sign-up address. A budget set
   after the first deploy is a smoke alarm installed after the fire.
4. **An IAM user for the CLI** (decision 3). Identity Center is not available on this
   plan, so: IAM, Users, create one user with `AdministratorAccess`, turn MFA on for it,
   then create an access key of type CLI. Copy the secret at once; it is shown once.
5. **AWS CLI v2** on the Mac, from the official pkg rather than Homebrew:
   `curl https://awscli.amazonaws.com/AWSCLIV2.pkg -o AWSCLIV2.pkg` then
   `sudo installer -pkg AWSCLIV2.pkg -target /`. Homebrew compiles the aws-c-* libraries
   from source when the Xcode Command Line Tools are out of date, which is what happened
   here; the pkg is a signed universal binary and needs no toolchain at all.
6. `aws configure --profile docs-copilot`, then paste the key and secret, region
   `eu-west-1`, output `json`.

**Two verifications, both of which must pass before step 2b.1 is worth writing:**

```
aws sts get-caller-identity --profile docs-copilot
aws ecs create-express-gateway-service help
```

The second one is not a formality. Express Mode is recent; a CLI older than it has no such
command and every later step would fail at the last one. The project's own rule applies —
installed code settles shapes, a live call settles behaviour — and here the installed CLI
settles whether the plan is even expressible. And if Express Mode does turn out to be
withheld on the Free plan after all, upgrading is one click and nothing here changes but
decision 3.

## What Express Mode creates

Read from AWS's "Resources created by Amazon ECS Express Mode services", 2026-09-14.

| Resource | What we get | What it costs |
|---|---|---|
| Application Load Balancer | internet-facing, HTTPS listener on 443, AWS-provided hostname and certificate | the main standing cost, roughly $17/month |
| Target group | HTTP, target type IP, health check every 30 s | — |
| Security groups | LB group inbound 443; service group outbound only, reachable from the LB on the container port | — |
| VPC and subnets | the default VPC's public subnets, at least two AZs, at least 8 free IPs | — |
| ECS cluster | the `default` cluster with Fargate capacity providers, created if absent | — |
| Fargate task | one per `minTaskCount` | roughly $18/month at 0.5 vCPU / 1 GB |
| CloudWatch log group | `/aws/ecs/<cluster>/<name>-####` | pennies |

Three roles are involved: a **task execution role** (`AmazonECSTaskExecutionRolePolicy`, plus
an inline `secretsmanager:GetSecretValue` for our one secret — the managed policy does not
grant a named secret), an **infrastructure role**
(`AmazonECSInfrastructureRoleforExpressGatewayServices`), and a **task role**, which this
service does not need: the container's application code calls OpenAI, Cohere, Postgres and
Langfuse, and none of them are AWS.

## Parameters that must not be left at their defaults

The interesting part of `create-express-gateway-service` is where its defaults disagree with
this service.

| Parameter | Default | Ours | Why |
|---|---|---|---|
| `--health-check-path` | `/ping` | `/health` | We have `/health`, and it queries nothing on purpose, which is what an ALB health check should hit. `/ping` is a 404, so every task would fail its health check and the deploy would roll back with a healthy service inside it. |
| `containerPort` | 80 | 8000 | The Dockerfile's `EXPOSE`/`CMD`. |
| `--cpu` / `--memory` | 256 / 512 | **256 / 512, the default** | Decision 8. The prediction written here first was 512 / 1024 and it was wrong. That stays on the record: a default being correct is only knowable after measuring it, and a spec that quietly deletes its wrong predictions is not a record of anything. |
| `--scaling-target` | AWS's | `minTaskCount=1,maxTaskCount=2` | Decision 6. |
| `--task-role-arn` | none | none | Stated so that "we did not need one" is a decision on the record rather than an omission. |
| `--cpu-architecture` | `X86_64` | `X86_64` | Decision 7: stated, not defaulted. Absent from the published reference for this command; the installed CLI has it. |

**`stopTimeout` is not a parameter of this command.** The whole shutdown budget in
`agent/README.md` is sized for ECS's 30 s, which is the ECS default — but Express Mode
generates the task definition, and this plan is not allowed to assume what it generated.
2b.5 reads the generated task definition back and checks the number. Same for the target
group's deregistration delay.

### What the container actually costs (measured 2b.2)

`agent/experiments/container_memory.sh`: the real image under a hard cap, with real `/chat`
turns driven through it, three runs covering 512m and 1024m in both orders.

| | position 1 | position 2 |
|---|---|---|
| total (`docker stats`) | 218 to 220 MiB | 148 to 151 MiB |
| working set (cgroup `anon`) | 144.5 MiB | 146.9 MiB |
| page cache (cgroup `file`) | 99.4 MiB | 0.0 MiB |
| boot to `/health` | 8 to 11 s | 5 to 8 s |

**FINDING: the number moved with position, not with the cap, and moved the wrong way** - the
tighter cap appearing to use 70 MiB more, which is backwards for a cgroup. Running both orders
separated the two variables and splitting the cgroup counters named the cause: the first
container to run faults the image layers in from disk and is charged for them, the second finds
them resident and is charged nothing, while `anon` stays constant within 2.4 MiB.

This decides the sizing rather than being a curiosity. **Every Fargate task is a position-1
container**, on a host that has never seen the image, so the larger number is the one the
deployment experiences. Sizing on the 149 MiB a second-position container reports would have
been sizing on a state production never reaches.

**The totals are not `anon` plus `file`, and should not be.** Each column is a maximum taken
independently across the samples, and `docker stats` on cgroup v2 reports `memory.current`
minus `inactive_file`, so it already omits part of what `file` counts.

**A correction from the same runs.** The latency difference was read as an ordering effect too,
and that half was wrong: across three runs the position-1 warm median went 6770, 6439, 5405 ms
and position-2 went 5668, 5722, 7428 ms. Mixed, so latency here is upstream API variance. One
run showed two differences and only one of them was real, which is the whole argument for
running it twice before believing a knob.

## Environment: what goes in plain, what goes in the secret

The service reads exactly what `agent/src/copilot_agent/settings.py` declares. Note what is
absent: `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are the TypeScript app's, not the agent's.
The agent talks to Postgres over psycopg and needs no REST credentials at all.

| Variable | Where | Note |
|---|---|---|
| `OPENAI_API_KEY` | secret | |
| `COHERE_API_KEY` | secret | production key |
| `DATABASE_URL` | secret | session pooler, port 5432; `settings.py` rejects the direct IPv6-only host |
| `AGENT_API_KEY` | secret | same value Vercel sends |
| `ASSISTANT_SIGNING_SECRET` | secret | must be byte-identical to the Next.js app's |
| `LANGFUSE_SECRET_KEY` | secret | |
| `LANGFUSE_PUBLIC_KEY` | plain env | not a secret; it names the project and travels with every export (`settings.py` says so) |
| `LANGFUSE_BASE_URL` | plain env | EU region |
| `LANGFUSE_ENVIRONMENT` | plain env | **`production`** — deploy checklist item 5, so AWS traces do not share a dashboard with the laptop |
| `ENABLE_SEARCH_ENDPOINT` | **never set** | invariant 2: unset means the paid debug route does not exist, rather than existing behind a check |
| `CHECKPOINT_DURABILITY` | not set | `exit` is the default and the measured choice |

**Every value copied out of `.env.local` must be copied unquoted.** Seven of its lines are
written with surrounding quotes, `ASSISTANT_SIGNING_SECRET` and `LANGFUSE_BASE_URL` among
them. pydantic-settings reads that file as dotenv and strips the quotes; nothing else does.
Found by the 2b.2 memory experiment, which passed the same file through Docker's `--env-file`,
which is not a dotenv parser: the container refused to start on `LANGFUSE_BASE_URL`, where a
scheme validator was watching. The two variables that matter for this deploy had no validator
watching. A quoted `ASSISTANT_SIGNING_SECRET` or `AGENT_API_KEY` is simply the wrong secret,
and the symptom is a 401 from the service or history dropped by the route, never an error
naming the variable. So 2b.3 adds a paste check to `settings.py` that refuses a value which
both starts and ends with the same quote character, next to the existing length check on
`AGENT_API_KEY` and for exactly the same reason: it is a typo test, not a strength test.

## The door is the key

The ALB is internet-facing, so `POST /chat` is reachable from anywhere. The only thing in
front of it is the bearer key (constant-time compare, `create_app` refuses to start without
one) — and unlike the Vercel route, there is **no rate limiter behind it**. A leaked key is
therefore unbounded spend until it is rotated, bounded only by `MAX_OUTPUT_TOKENS` per call.

Accepted for a service that exists for a few days, with the rotation path (one Secrets
Manager value, one Vercel variable) written down. Not accepted silently: this is the reason
the service is torn down rather than left running, and it is the honest answer to "what
would you do differently in production" — a WAF rate rule, or an internal load balancer with
the caller inside the VPC.

## Done when

1. `GET /health` on the AWS URL answers 200, and an authenticated `POST /chat` streams a
   grounded answer with sources.
2. The generated task definition and target group are read back and match the shutdown
   budget's assumptions (`stopTimeout` 30, a deregistration delay).
3. The full 27-case suite runs **twice** against the deployed URL, with both results JSONs
   committed next to the 2.8 pair.
4. The runs appear in Langfuse under environment `production`, and `query_log` holds their
   rows with `origin = 'eval'`.
5. The service is deleted, the ALB and log group are confirmed gone, and the actual dollar
   figure is read from Billing rather than estimated.

### The prediction, written before the measurement

Quality numbers should be **identical** to the 2.8 pair: same image, same model, same
corpus. Recall 12/12 run 1, coverage 12/12, guardrails 6/6, injection 8/8, false refusals 0.
Anything that moves here is the deployment, and that would be the finding.

Latency should **improve**, which is the counter-intuitive half. Two changes pull opposite
ways: the harness-to-service hop goes from localhost to Athens→Ireland, adding perhaps
50-70 ms once per request; but every Postgres round trip goes from Athens→Ireland to
in-region, saving a similar amount several times per turn. A turn makes more database round
trips than HTTP hops, so the prediction is that retrieval median drops below the 2647 /
2761 ms of the 2.8 pair.

Cost per run should be unchanged at about $0.197: the same tokens are bought either way.

If latency does **not** improve, the cheap explanation is wrong and the next question is
which hop actually dominates — which is exactly what the per-stage timings in `query_log`
can answer, and the reason they are recorded per stage.

## Teardown

Part of the spec, not an afterthought, because an unused ALB is the classic first AWS bill.
Delete the Express service (which removes the ALB, target group, security groups and
scaling policies it created), then confirm by listing: load balancers, target groups, the
log group, the ECR repository, the secret. The default VPC, the `default` cluster and the
roles are left; they are free and a redeploy reuses them.
