#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${1:-${REPO_ROOT}/upstream/Mamba-Segmentation}"

mkdir -p "$(dirname "${DEST}")"

if [[ ! -d "${DEST}/.git" ]]; then
  git clone https://github.com/Dineth14/Mamba-Segmentation.git "${DEST}"
fi

cd "${DEST}"
git fetch origin
git checkout 1139642b8dae2ab416dc50995409c1814b9951c4
git apply "${REPO_ROOT}/patches/upstream_compatibility.patch"

mkdir -p reproduction_scripts
cp "${REPO_ROOT}"/scripts/*.py reproduction_scripts/

printf '\nPrepared upstream checkout at %s\n' "${DEST}"
printf 'Activate the environment and follow docs/REPRODUCE.md.\n'
