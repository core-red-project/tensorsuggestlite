#!/usr/bin/env bash
# Validate a single commit message against Conventional Commits specification
# Types allowed: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
set -euo pipefail

MESSAGE_FILE="${1:-}"

if [[ -z "$MESSAGE_FILE" || ! -f "$MESSAGE_FILE" ]]; then
    echo "Usage: $0 <path-to-commit-message-file>" >&2
    exit 1
fi

FIRST_LINE=$(head -n 1 "$MESSAGE_FILE")

# Allow automated git commits (Merge, Revert)
if [[ "$FIRST_LINE" =~ ^Merge ]] || [[ "$FIRST_LINE" =~ ^Revert ]]; then
    exit 0
fi

PATTERN="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-zA-Z0-9_.-]+\))?!?: .+$"

if [[ ! "$FIRST_LINE" =~ $PATTERN ]]; then
    echo "ERROR: Commit message does not follow Conventional Commits standard." >&2
    echo "First line was: '$FIRST_LINE'" >&2
    echo "" >&2
    echo "Format: <type>(<optional scope>): <description>" >&2
    echo "Valid types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert" >&2
    echo "Example: feat(cli): add --json output flag" >&2
    exit 1
fi

exit 0

