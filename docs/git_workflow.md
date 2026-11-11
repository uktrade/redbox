# Git workflow

The workflows for using Git.

## Branch Naming

Consistent branch names help maintain a clean and predictable workflow. CI will fail if your branch does not follow conventions. Use the following prefixes:

- `feature/<name>` — New features or enhancements
- `chore/<name>` — Maintenance tasks that don’t affect functionality
- `bugfix/<name>` — Non-critical bug fixes
- `hotfix/<name>` — Urgent fixes for production issues
- `dependabot/<name>` — Automated dependency updates
- `security/<name>` — Security-related changes
- `revert<name>` — For when git creates a branch to revert a PR

## Pre-commit hooks

- Download and install [pre-commit](https://pre-commit.com) to benefit from pre-commit hooks
  - `pip install pre-commit`
  - `pre-commit install`
