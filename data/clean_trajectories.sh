#!/usr/bin/env bash
# Clean generated trajectory outputs.
# Usage: ./clean_trajectories.sh [--yes]
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)/trajectories"

if [ ! -d "$DIR" ]; then
  echo "Nothing to clean ($DIR does not exist)."
  exit 0
fi

echo "Will remove:"
echo "  $DIR/trajectories.jsonl"
echo "  $DIR/artifacts/"

if [[ "${1:-}" != "--yes" ]]; then
  read -r -p "Continue? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

rm -f  "$DIR/trajectories.jsonl"
rm -rf "$DIR/artifacts"
echo "Done."
