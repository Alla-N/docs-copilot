#!/usr/bin/env bash
#
# Create the ECS Express Mode service. Run from agent/ after infra/roles.sh.
#
#   ./infra/service.sh              # image tag defaults to the current commit
#   ./infra/service.sh f140eb6
#
# Everything this passes is a decision recorded in specs/aws-deploy.md. The defaults it does
# NOT accept are the interesting part:
#
#   --health-check-path   defaults to /ping, and this service has /health. Left alone, every
#                         task would pass its own startup, fail the load balancer health check
#                         and roll the deployment back with a working service inside it.
#   containerPort         defaults to 80. The Dockerfile listens on 8000.
#   --cpu / --memory      measured in 2b.2: the working set is about 145 MiB and both 512m and
#                         1024m survived, so 256 CPU units and 512 MiB. Passed explicitly
#                         because the two AWS pages disagree about what the default is - the CLI
#                         reference says 256 / 512 and the getting-started page says its minimal
#                         command gives 1 vCPU and 2 GB, which is four times the cost.
#   --cpu-architecture    X86_64, matching the Intel Mac the image was built and measured on.
#                         Not in the published CLI reference; found in the installed CLI.
#   --scaling-target      max 2 tasks, which is the measured Supabase pool ceiling (15 connections,
#                         6 per busy task), not a number anyone liked the look of.
#   --monitor-mode        TEXT-ONLY, so the output survives being piped to a file.
#
# stopTimeout is NOT settable here - Express Mode generates the task definition - so 2b.5 reads
# the generated one back rather than trusting that it got the 30 s the shutdown budget assumes.
set -euo pipefail

PROFILE=${PROFILE:-docs-copilot}
REGION=${REGION:-eu-west-1}
SERVICE=${SERVICE:-docs-copilot-agent}
SECRET_NAME=${SECRET_NAME:-docs-copilot-agent}
REPOSITORY=${REPOSITORY:-docs-copilot-agent}
EXEC_ROLE=${EXEC_ROLE:-docsCopilotTaskExecutionRole}
INFRA_ROLE=${INFRA_ROLE:-docsCopilotExpressInfrastructureRole}
LOG_GROUP=${LOG_GROUP:-/ecs/docs-copilot-agent}
TAG=${1:-$(git rev-parse --short HEAD)}

aws_() { aws "$@" --profile "$PROFILE" --region "$REGION"; }

ACCOUNT=$(aws_ sts get-caller-identity --query Account --output text)
SECRET_ARN=$(aws_ secretsmanager describe-secret --secret-id "$SECRET_NAME" \
  --query ARN --output text)
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY:$TAG"

# Fail here rather than in a task that cannot pull.
aws_ ecr describe-images --repository-name "$REPOSITORY" --image-ids imageTag="$TAG" >/dev/null
echo "image: $IMAGE"

# Our own log group, with the same 30 days the checkpoint retention job and the Langfuse free
# tier use, so a trace, the conversation state that produced it and the log line about it all
# expire together. A log group created implicitly never expires, and nobody goes back for it.
# Create-and-tolerate, not check-then-create. The check that was here asked describe-log-groups
# for the group and tested whether the answer was empty - but --output text renders a null
# JMESPath result as the literal string None, which is not empty, so "does not exist" read as
# "exists", nothing was created, and put-retention-policy failed on a group that was never made.
# Same shape as the reltuples bug in e6bbc7d: a placeholder meaning I do not know, read as a value.
LOG_ERR=$(mktemp -t docs-copilot-logs)
if aws_ logs create-log-group --log-group-name "$LOG_GROUP" 2>"$LOG_ERR"; then
  echo "log group $LOG_GROUP created"
elif grep -q ResourceAlreadyExistsException "$LOG_ERR"; then
  echo "log group $LOG_GROUP already exists"
else
  cat "$LOG_ERR" >&2
  rm -f "$LOG_ERR"
  exit 1
fi
rm -f "$LOG_ERR"
aws_ logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 30

# The container payload, built by python so a value can never be mangled by shell quoting, and
# read out of .env.local by the same dotenv parser the service itself uses (see aws_secret.py).
# LANGFUSE_PUBLIC_KEY is plain environment, not a secret: it names the project and travels with
# every export. ENABLE_SEARCH_ENDPOINT is absent on purpose - invariant 2 says the paid debug
# route should not exist in a deployment, rather than exist behind a flag someone can flip.
PAYLOAD=$(SECRET_ARN="$SECRET_ARN" IMAGE="$IMAGE" LOG_GROUP="$LOG_GROUP" uv run python - <<'PY'
import json
import os

from dotenv import dotenv_values

from copilot_agent.settings import REPO_ROOT

env = dotenv_values(REPO_ROOT / ".env.local")
arn = os.environ["SECRET_ARN"]
secret_keys = [
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "DATABASE_URL",
    "AGENT_API_KEY",
    "ASSISTANT_SIGNING_SECRET",
    "LANGFUSE_SECRET_KEY",
]
plain = {
    "LANGFUSE_PUBLIC_KEY": env["LANGFUSE_PUBLIC_KEY"],
    "LANGFUSE_BASE_URL": env.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    # Deploy checklist item 5: not "development", so AWS traces do not share the laptop's view.
    "LANGFUSE_ENVIRONMENT": "production",
}
print(json.dumps({
    "image": os.environ["IMAGE"],
    "containerPort": 8000,
    "awsLogsConfiguration": {"logGroup": os.environ["LOG_GROUP"], "logStreamPrefix": "ecs"},
    "environment": [{"name": k, "value": v} for k, v in plain.items()],
    # <arn>:<json-key>:: is how ECS reads one key out of a JSON secret. The empty version-stage
    # and version-id mean AWSCURRENT, so a new secret version is picked up by the next task.
    "secrets": [{"name": k, "valueFrom": f"{arn}:{k}::"} for k in secret_keys],
}))
PY
)

echo "creating $SERVICE, this provisions a load balancer and takes a few minutes"
aws_ ecs create-express-gateway-service \
  --service-name "$SERVICE" \
  --execution-role-arn "arn:aws:iam::$ACCOUNT:role/$EXEC_ROLE" \
  --infrastructure-role-arn "arn:aws:iam::$ACCOUNT:role/$INFRA_ROLE" \
  --primary-container "$PAYLOAD" \
  --health-check-path /health \
  --cpu 256 \
  --memory 512 \
  --cpu-architecture X86_64 \
  --scaling-target minTaskCount=1,maxTaskCount=2 \
  --tags key=project,value=docs-copilot key=phase,value=2b \
  --monitor-resources DEPLOYMENT \
  --monitor-mode TEXT-ONLY
