#!/bin/bash
set -e  # Exit on errors

# Absolute path to project root (directory of this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# If .envrc exists, run direnv allow and source for instant access
if [ -f "$PROJECT_ROOT/.envrc" ]; then
  if command -v direnv >/dev/null 2>&1; then
    echo ".envrc found – running 'direnv allow'"
    direnv allow
    source "$PROJECT_ROOT/.envrc"
  else
    echo "direnv is not installed – skipping 'direnv allow'"
  fi
else
  echo ":information_source:  No .envrc found – using default AWS profile fallback"
fi

# Use AWS_PROFILE from env, or fallback to 'default'
AWS_PROFILE="${AWS_PROFILE:-default}"
AWS_DIR="$PROJECT_ROOT/.aws"

DJANGO_APP_DIR="$PROJECT_ROOT/django_app"
DJANGO_APP_AWS_DIR="$DJANGO_APP_DIR/.aws"

NOTEBOOKS_DIR="$PROJECT_ROOT/notebooks"
NOTEBOOKS_AWS_DIR="$NOTEBOOKS_DIR/.aws"


echo "Using AWS profile: $AWS_PROFILE"

# 1. Ensure main .aws directory and credentials file exist
mkdir -p "$AWS_DIR"
if [ ! -f "$AWS_DIR/credentials" ]; then
  echo "'credentials' file missing. Copying from example..."
  if [ -f "$AWS_DIR/credentials.example" ]; then
    cp "$AWS_DIR/credentials.example" "$AWS_DIR/credentials"
  else
    echo "'credentials.example' not found in $AWS_DIR!"
  fi
fi

# 2. Login with AWS SSO
aws sso login --profile "$AWS_PROFILE"

# 3. Update credentials in main .aws/credentials
aws configure export-credentials --profile "$AWS_PROFILE" --format env-no-export | while IFS= read -r line; do
  key=$(echo "$line" | cut -d'=' -f1)
  value=$(echo "$line" | cut -d'=' -f2-)
  sed -i.bak "s|^${key}=.*|${key}=${value}|" "$AWS_DIR/credentials" || echo "${key}=${value}" >> "$AWS_DIR/credentials"
done

# 4. Clean up backup
rm -f "$AWS_DIR/credentials.bak"

# 5. Ensure django_app/.aws exists and copy over credentials
mkdir -p "$DJANGO_APP_AWS_DIR"
cp "$AWS_DIR/credentials" "$DJANGO_APP_AWS_DIR/credentials"
echo "AWS credentials updated and copied to notebooks/.aws"

# 6. Ensure notebooks/.aws exists and copy over credentials
mkdir -p "$NOTEBOOKS_AWS_DIR"
cp "$AWS_DIR/credentials" "$NOTEBOOKS_AWS_DIR/credentials"
echo "AWS credentials updated and copied to notebooks/.aws"
