#!/usr/bin/env bash
#
# The two IAM roles ECS Express Mode needs, plus the one permission its managed policy leaves out.
#
#   ./infra/roles.sh            # after infra/aws_secret.py --write, which this reads the ARN from
#
# Idempotent: a role that already exists is reported and left alone, and attaching a policy twice
# is not an error.
#
# THREE ROLES EXIST IN THIS PICTURE AND ONLY TWO ARE CREATED HERE.
#
#   task execution role   assumed by the ECS agent, NOT by the application. It pulls the image,
#                         writes the log stream and reads the secret. Trusts ecs-tasks.amazonaws.com
#   infrastructure role   assumed by ECS itself to build the load balancer, target group, security
#                         groups and scaling policies on your behalf. Trusts ecs.amazonaws.com
#   task role             assumed by the application code to call AWS. This service calls OpenAI,
#                         Cohere, Postgres and Langfuse, none of which are AWS, so it gets none.
#                         Stated here so that absence is a decision rather than an omission.
#
# THE INLINE POLICY IS NOT DECORATION. AmazonECSTaskExecutionRolePolicy grants
# secretsmanager:GetSecretValue on nothing: the managed policy covers ECR and CloudWatch, and the
# secret has to be named. Without it the task fails to start with a message about the secret, and
# the natural but wrong reaction is to suspect the secret rather than the role.
set -euo pipefail

PROFILE=${PROFILE:-docs-copilot}
REGION=${REGION:-eu-west-1}
SECRET_NAME=${SECRET_NAME:-docs-copilot-agent}
EXEC_ROLE=${EXEC_ROLE:-docsCopilotTaskExecutionRole}
INFRA_ROLE=${INFRA_ROLE:-docsCopilotExpressInfrastructureRole}
MANAGED=arn:aws:iam::aws:policy/service-role
INFRA_POLICY=$MANAGED/AmazonECSInfrastructureRoleforExpressGatewayServices
EXEC_POLICY=$MANAGED/AmazonECSTaskExecutionRolePolicy

aws_() { aws "$@" --profile "$PROFILE" --region "$REGION"; }

SECRET_ARN=$(aws_ secretsmanager describe-secret --secret-id "$SECRET_NAME" \
  --query ARN --output text 2>/dev/null || true)
if [ -z "$SECRET_ARN" ] || [ "$SECRET_ARN" = "None" ]; then
  echo "no secret named $SECRET_NAME. Run infra/aws_secret.py --write first."
  exit 2
fi
echo "secret: $SECRET_ARN"

ensure_role() {  # name, trust json
  if aws_ iam get-role --role-name "$1" >/dev/null 2>&1; then
    echo "role $1 already exists"
  else
    aws_ iam create-role --role-name "$1" --assume-role-policy-document "$2" >/dev/null
    echo "role $1 created"
  fi
}

ensure_role "$EXEC_ROLE" '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

ensure_role "$INFRA_ROLE" '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AllowAccessInfrastructureForECSExpressServices",
    "Effect": "Allow",
    "Principal": {"Service": "ecs.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

aws_ iam attach-role-policy --role-name "$EXEC_ROLE" --policy-arn "$EXEC_POLICY"
aws_ iam attach-role-policy --role-name "$INFRA_ROLE" --policy-arn "$INFRA_POLICY"
echo "managed policies attached"

# Scoped to this one secret, not to secretsmanager:* . The execution role is the most
# over-granted role in most ECS accounts precisely because this statement is easier to write
# with a star in it.
aws_ iam put-role-policy --role-name "$EXEC_ROLE" \
  --policy-name ReadTheAgentSecret \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Action\": \"secretsmanager:GetSecretValue\",
      \"Resource\": \"$SECRET_ARN\"
    }]
  }"
echo "inline policy ReadTheAgentSecret attached to $EXEC_ROLE"

ACCOUNT=$(aws_ sts get-caller-identity --query Account --output text)
echo
echo "execution role   arn:aws:iam::$ACCOUNT:role/$EXEC_ROLE"
echo "infrastructure   arn:aws:iam::$ACCOUNT:role/$INFRA_ROLE"
echo "secret           $SECRET_ARN"
