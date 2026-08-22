#!/usr/bin/env bash
# Reproduces every numerical claim in the paper. See README.md for the map
# from script to location in the text.
set -euo pipefail
cd "$(dirname "$0")"

for s in network neutrality characterization sign_reversal block_reversal; do
  printf '\n===== scripts/%s.py =====\n\n' "$s"
  python3 "scripts/$s.py"
done

printf '\n===== done =====\n'
