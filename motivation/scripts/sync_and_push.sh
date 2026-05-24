#!/usr/bin/env bash
# Sync the latest results + interpretation doc into the EASMO repo and push.
#
# Usage:
#   bash motivation/scripts/sync_and_push.sh "milestone short summary"
#
# Mirrors the canonical guidance docs from /workspace/guidances/ into
# motivation/docs/, stages all changes under motivation/, and pushes
# to origin/main. Designed to be called from a long-running experiment
# wrapper whenever a milestone (M1 done, M3 done, instance_noise done…)
# completes, so the GitHub repo stays a faithful mirror of the current
# state of the analysis.

set -euo pipefail

REPO="/workspace/EASMO"
GUIDANCES_SRC="/workspace/guidances"

MSG="${1:-results update}"

cd "$REPO"

# Mirror the always-canonical docs into the repo. The /workspace/guidances/
# versions are the source of truth; the in-repo copies are a snapshot.
cp -f "$GUIDANCES_SRC/ICLR27_Memory_motivation_experiments.md" \
      motivation/docs/01_experiments_spec.md
cp -f "$GUIDANCES_SRC/ICLR27_Memory_motivation_results_and_interpretation.md" \
      motivation/docs/02_results_and_interpretation.md

# Stage everything under motivation/ + the doc + top-level README.
git add -A motivation/ README.md .gitignore 2>/dev/null || true

if git diff --cached --quiet; then
    echo "[sync_and_push] No changes to commit."
    exit 0
fi

TS="$(date -u +%Y-%m-%dT%H:%MZ)"
git commit -m "results($TS): $MSG"
git push origin main
echo "[sync_and_push] Pushed at $TS — '$MSG'"
