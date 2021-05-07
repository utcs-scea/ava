#!/bin/bash

GIT_ROOT=$(git rev-parse --show-toplevel)
chmod +x "$GIT_ROOT"/scripts/pre-commit
ln -s "$GIT_ROOT"/scripts/pre-commit "$GIT_ROOT"/.git/hooks/pre-commit
