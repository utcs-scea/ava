#!/bin/bash

GIT_ROOT=$(git rev-parse --show-toplevel)
PYFMT="${PYFMT:-black}"

set -eu

if [ $# -eq 0 ]; then
  echo "$(basename "$0") [-fix] <paths>" >&2
  exit 1
fi

FIX=
if [ "$1" == "-fix" ]; then
  FIX=1
  shift 1
fi

ROOTS=("$@")
PRUNE_PATHS="llvm third_party"
PRUNE_NAMES=".git build*"

emit_prunes() {
  { for p in ${PRUNE_PATHS}; do echo "-path ${p} -prune -o -path ./${p} -prune -o"; done; \
    for p in ${PRUNE_NAMES}; do echo "-name ${p} -prune -o"; done; } | xargs
}

pushd "$GIT_ROOT" > /dev/null

# shellcheck disable=SC2046,SC2207,SC2038
FILES=($(find "${ROOTS[@]}" $(emit_prunes) -name '*.py' -print | xargs))

if [ -n "${FIX}" ]; then
  ${PYFMT} "${FILES[@]}"
else
  ${PYFMT} --check "${FILES[@]}"
fi

popd > /dev/null
