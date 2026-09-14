#!/usr/bin/env bash
#
# How much memory does the container actually need?
#
#   cd agent && ./experiments/container_memory.sh            # tries 512m then 1024m
#   cd agent && ./experiments/container_memory.sh 512m       # one limit
#   cd agent && ./experiments/container_memory.sh 1024m 512m # reversed, to separate order
#
# ECS Express Mode defaults --memory to 512 MiB, and this service loads langchain, langgraph
# and the OpenTelemetry SDK before it answers anything. 512 might be ample and it might be an
# OOM kill on the first real turn; the difference between those two is not something to learn
# from a rolling deployment.
#
# So this does not read a gauge and then guess. For each limit it runs the real image under
# that hard cap and drives real /chat turns through it, and reports whether the kernel killed
# it. A limit that survives is a measurement. A limit that does not is a number never deployed.
#
# TWO THINGS MEASURED 2026-09-14, both of which change how the output is read:
#
#  1. POSITION DOMINATES THE LIMIT. Running 512m then 1024m, and then 1024m then 512m, the
#     peak followed the position (about 219 MiB first, about 149 MiB second, within 2.6 MiB
#     across both orders) and did not follow the cap at all. Always run both orders before
#     believing any difference between two limits here.
#  2. WHICH IS WHY anon AND file ARE REPORTED SEPARATELY. docker stats MemUsage is a cgroup
#     total. `anon` is the process working set; `file` is page cache, and the image layers
#     are faulted in by whichever container touches them first, so the second container in a
#     run looks leaner than it is. Size on the FIRST position and on anon: every Fargate task
#     is a first-position container on a host that has never seen the image.
#
# Costs about a cent per /chat turn (3 per limit). Needs Docker running and .env.local with the
# usual keys. Holds 3 to 6 Supabase connections while it runs, like any local service, so do
# not run an eval at the same time.
set -euo pipefail

cd "$(dirname "$0")/.."          # agent/
REPO_ROOT="$(cd .. && pwd)"
ENV_FILE="$REPO_ROOT/.env.local"
NAME=copilot-agent-memtest
QUESTION="how do I stream text"
RUNS=3
LIMITS="${*:-512m 1024m}"

[ -f "$ENV_FILE" ] || { echo "no .env.local at $ENV_FILE"; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker is not running"; exit 2; }

# The same five variables the container needs, plus Langfuse: the OpenTelemetry SDK and its
# exporter thread are part of what is being sized, and the deployment has them.
VARS='^(OPENAI_API_KEY|COHERE_API_KEY|DATABASE_URL|AGENT_API_KEY|ASSISTANT_SIGNING_SECRET|LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY|LANGFUSE_BASE_URL)='

SAMPLER=""
cleanup() {
  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
  docker rm -f "$NAME" >/dev/null 2>&1
  return 0
}
trap cleanup EXIT

to_mib() {  # a stream of "245.3MiB" / "1.02GiB" -> a stream of MiB numbers
  awk '{ n = $0; sub(/[A-Za-z]+$/, "", n);
         if ($0 ~ /GiB$/)      printf "%.1f\n", n * 1024;
         else if ($0 ~ /MiB$/) printf "%.1f\n", n;
         else if ($0 ~ /KiB$/) printf "%.1f\n", n / 1024;
         else                  printf "%.1f\n", n / 1048576 }'
}

# pydantic-settings reads .env.local as dotenv and strips surrounding quotes. Docker's
# --env-file is not dotenv and keeps them, so a quoted value reaches the container WITH its
# quotes. agent/README.md warns about this in prose; this is the line that acts on it.
# LANGFUSE_BASE_URL failed loudly (the scheme validator refused to start). The two that
# would not have failed at all are ASSISTANT_SIGNING_SECRET and AGENT_API_KEY: quoted, they
# are simply the wrong secret, and nothing anywhere would say so.
strip_quotes() {
  sed -E "s/^([A-Za-z_][A-Za-z0-9_]*)=[\"'](.*)[\"']\$/\1=\2/"
}

# anon and file out of the container's own cgroup, in MiB. Prints "- -" if the cgroup
# filesystem is not mounted in the container, rather than inventing a number.
cgroup_mem() {
  docker exec "$NAME" cat /sys/fs/cgroup/memory.stat 2>/dev/null \
    | awk '/^anon /{a=$2} /^file /{f=$2} END{ if (a=="") print "- -"; else printf "%.1f %.1f\n", a/1048576, f/1048576 }'
}

echo "building the image"
docker build -q -t copilot-agent .
IMAGE_MB=$(docker image inspect copilot-agent --format '{{.Size}}' | awk '{printf "%.0f", $1/1048576}')
echo "image: ${IMAGE_MB} MB"
echo

HDR="%-8s  %4s  %7s  %9s  %9s  %9s  %10s  %s\n"
printf "$HDR" limit pos boot_s idle_MiB peak_MiB peak_anon peak_file verdict

POS=0
for LIMIT in $LIMITS; do
  POS=$((POS + 1))
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --memory "$LIMIT" --memory-swap "$LIMIT" \
    -p 127.0.0.1:8000:8000 \
    -e LANGFUSE_ENVIRONMENT=docker-local \
    --env-file <(grep -E "$VARS" "$ENV_FILE" | strip_quotes) \
    copilot-agent >/dev/null

  # Wait for /health. The lifespan opens three pools, so this is not instant.
  BOOT_START=$(date +%s)
  BOOT=""
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
      BOOT=$(( $(date +%s) - BOOT_START ))
      break
    fi
    docker ps -q -f name="$NAME" | grep -q . || break   # it died on the way up
    sleep 1
  done

  if [ -z "$BOOT" ]; then
    OOM=$(docker inspect -f '{{.State.OOMKilled}}' "$NAME" 2>/dev/null || echo "?")
    printf "$HDR" "$LIMIT" "$POS" "-" "-" "-" "-" "-" "never healthy, oom=$OOM"
    docker logs --tail 15 "$NAME" 2>&1 | sed 's/^/    /'
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    continue
  fi

  IDLE=$(docker stats --no-stream --format '{{.MemUsage}}' "$NAME" | awk '{print $1}' | to_mib)

  SAMPLES=$(mktemp -t copilot-memtest)
  ( while true; do
      MU=$(docker stats --no-stream --format '{{.MemUsage}}' "$NAME" 2>/dev/null | awk '{print $1}')
      echo "${MU:--} $(cgroup_mem)"
      sleep 0.5
    done ) > "$SAMPLES" &
  SAMPLER=$!

  # Real turns through the real pipeline, using the experiment that already exists.
  uv run python experiments/chat_latency.py "$QUESTION" --runs "$RUNS" 2>&1 | sed 's/^/    /' || true

  kill "$SAMPLER" 2>/dev/null || true
  wait "$SAMPLER" 2>/dev/null || true
  SAMPLER=""

  PEAK=$(awk '$1 != "-" {print $1}' "$SAMPLES" | to_mib | sort -n | tail -1)
  PEAK_ANON=$(awk '$2 != "-" {print $2}' "$SAMPLES" | sort -n | tail -1)
  PEAK_FILE=$(awk '$3 != "-" {print $3}' "$SAMPLES" | sort -n | tail -1)
  rm -f "$SAMPLES"

  OOM=$(docker inspect -f '{{.State.OOMKilled}}' "$NAME")
  RUNNING=$(docker inspect -f '{{.State.Running}}' "$NAME")
  if [ "$OOM" = "true" ]; then
    VERDICT="KILLED, never deploy at this limit"
  elif [ "$RUNNING" != "true" ]; then
    VERDICT="exited on its own, read the logs"
  else
    VERDICT="survived"
  fi

  printf "$HDR" "$LIMIT" "$POS" "$BOOT" "${IDLE:-?}" "${PEAK:-?}" "${PEAK_ANON:-?}" "${PEAK_FILE:-?}" "$VERDICT"

  docker stop -t 15 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
done

echo
echo "peak columns are sampled about every 2 s (docker stats --no-stream and docker exec are"
echo "not instant), so they are lower bounds: a spike between two samples is never seen."
echo "oom_killed in the verdict is not sampled and has no such weakness, which is why the"
echo "verdict reads from it rather than from any peak. Size on peak_anon in position 1."
