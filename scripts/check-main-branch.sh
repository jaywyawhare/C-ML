#!/bin/bash
# Check if we're on the main branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "Pre-commit hooks only run on main branch. Current branch: $BRANCH"
    exit 1
fi
exit 0
