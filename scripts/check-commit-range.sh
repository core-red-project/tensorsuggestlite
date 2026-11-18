#!/usr/bin/env bash
# Validate all commits in a git revision range against Conventional Commits
set -euo pipefail

BASE_REF="${1:-main}"
HEAD_REF="${2:-HEAD}"

FAILED=0
COMMITS=$(git rev-list "${BASE_REF}..${HEAD_REF}" || true)

if [[ -z "$COMMITS" ]]; then
    echo "No commits found between ${BASE_REF} and ${HEAD_REF}."
    exit 0
fi

PATTERN="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-zA-Z0-9_.-]+\))?!?: .+$"

for COMMIT in $COMMITS; do
    SUBJECT=$(git log -1 --format=%s "$COMMIT")
    # Allow merge and revert
    if [[ "$SUBJECT" =~ ^Merge ]] || [[ "$SUBJECT" =~ ^Revert ]]; then
        continue
    fi
    if [[ ! "$SUBJECT" =~ $PATTERN ]]; then
        echo "ERROR in commit $COMMIT:" >&2
        echo "  '$SUBJECT'" >&2
        FAILED=1
    fi
done

if [[ $FAILED -ne 0 ]]; then
    echo "" >&2
    echo "One or more commits do not follow Conventional Commits standard." >&2
    exit 1
fi

echo "All commits in range follow Conventional Commits."
exit 0

