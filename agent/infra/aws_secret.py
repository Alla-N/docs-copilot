"""Put the deployed service's secrets into Secrets Manager, read with the service's own parser.

    cd agent && uv run python infra/aws_secret.py           # dry run: names and lengths only
    cd agent && uv run python infra/aws_secret.py --write

One secret holding six JSON keys, not six secrets: ECS resolves a `valueFrom` of the form
`<secret-arn>:<json-key>::`, and one rotation surface beats six. Secrets Manager is also priced
per secret, but that is the smaller reason.

Nothing here prints a secret value. The dry run prints key names and value lengths, which is
enough to catch a missing or half-pasted one and not enough to leak anything.

WHY IT READS .env.local RATHER THAN TAKING ARGUMENTS. The values go through python-dotenv, the
same parser pydantic-settings uses, so what the deployment gets cannot disagree with what the
service reads locally. Step 2b.2 found the failure this avoids: a quoted value reaching a
container through Docker's env-file, which is not a dotenv parser, arriving with its quotes
attached. Hand-copying into a console has exactly the same failure mode, and for
AGENT_API_KEY and ASSISTANT_SIGNING_SECRET it produces no error at all - just the wrong secret.

And before anything is uploaded, the payload is validated by the service's OWN Settings class.
A value that would stop the container from starting stops here instead, on a laptop, in a second.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values

from copilot_agent.settings import REPO_ROOT, Settings

# Exactly what the deployed service needs that is secret. LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
# and LANGFUSE_ENVIRONMENT are plain environment variables in the task definition: the public key
# names the project and travels with every export, so it is not a secret and pretending otherwise
# would only make it harder to read a task definition.
SECRET_KEYS = (
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "DATABASE_URL",
    "AGENT_API_KEY",
    "ASSISTANT_SIGNING_SECRET",
    "LANGFUSE_SECRET_KEY",
)


def aws(*args: str, profile: str, region: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["aws", *args, "--profile", profile, "--region", region],
        capture_output=True,
        text=True,
        check=False,
    )


def collect() -> dict[str, str]:
    env_file = REPO_ROOT / ".env.local"
    if not env_file.is_file():
        sys.exit(f"no {env_file}")
    values = dotenv_values(env_file)

    payload: dict[str, str] = {}
    missing = []
    for key in SECRET_KEYS:
        value = values.get(key) or os.environ.get(key)
        if not value:
            missing.append(key)
        else:
            payload[key] = value
    if missing:
        sys.exit("missing from .env.local: " + ", ".join(missing))

    # The service's own validation, before anything leaves this machine. Catches a quoted paste,
    # a short AGENT_API_KEY and the IPv6-only direct Supabase host, with the same messages the
    # container would have produced, minus the deployment in between.
    Settings(_env_file=None, **{key.lower(): value for key, value in payload.items()})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="actually upload")
    parser.add_argument("--name", default="docs-copilot-agent")
    parser.add_argument("--profile", default="docs-copilot")
    parser.add_argument("--region", default="eu-west-1")
    args = parser.parse_args()

    payload = collect()
    print(f"{len(payload)} keys, validated by the service's own Settings:")
    for key, value in payload.items():
        print(f"  {key:<26} {len(value):>4} characters")

    if not args.write:
        print("\ndry run. Re-run with --write to upload.")
        return

    # Through a 0600 file rather than argv: process arguments are readable by every process on
    # the machine, and this is six secrets at once.
    handle, path = tempfile.mkstemp(prefix="docs-copilot-secret-")
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w") as out:
            json.dump(payload, out)
        ref = f"file://{path}"
        existing = aws(
            "secretsmanager",
            "describe-secret",
            "--secret-id",
            args.name,
            profile=args.profile,
            region=args.region,
        )
        if existing.returncode == 0:
            result = aws(
                "secretsmanager",
                "put-secret-value",
                "--secret-id",
                args.name,
                "--secret-string",
                ref,
                profile=args.profile,
                region=args.region,
            )
            action = "new version of"
        else:
            result = aws(
                "secretsmanager",
                "create-secret",
                "--name",
                args.name,
                "--description",
                "docs-copilot agent service, phase 2b",
                "--secret-string",
                ref,
                profile=args.profile,
                region=args.region,
            )
            action = "created"
    finally:
        Path(path).unlink(missing_ok=True)

    if result.returncode != 0:
        sys.exit(result.stderr.strip() or "the AWS CLI failed")
    arn = json.loads(result.stdout)["ARN"]
    print(f"\n{action} {args.name}\n{arn}")


if __name__ == "__main__":
    main()
