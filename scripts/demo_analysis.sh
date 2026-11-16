#!/usr/bin/env bash
# One-command demonstration of the comparability-analysis product.
#
#   ./scripts/demo_analysis.sh                 # uses http://127.0.0.1:8000
#   BASE=http://host:8000 ./scripts/demo_analysis.sh
#
# For each analysis scenario it resets the service, uploads the simulated
# station waves, runs the pipeline, publishes the analysis definitions,
# generates the comparability report(s), and then runs the independent
# verifier, which recomputes coverage and contributions from ground truth.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
KEY="${WETLAND_API_KEY:-dev-api-key}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

cd "$ROOT"

echo "service: $BASE"
"$PY" - <<PY
import json, urllib.request
with urllib.request.urlopen("$BASE/health", timeout=5) as r:
    print("health:", json.load(r))
PY

# 1. Day/night biased sampling + overlapping stations + an empty stratum.
echo
echo "=== daynight ==="
"$PY" -m simulator.analysis_demo --scenario daynight --base-url "$BASE" --api-key "$KEY" --reset
"$PY" -m verifier.verify_analysis --scenario daynight --base-url "$BASE" --api-key "$KEY"

# 2. Night-time recorder failure reported late => two report versions.
echo
echo "=== nightfailure ==="
"$PY" -m simulator.analysis_demo --scenario nightfailure --base-url "$BASE" --api-key "$KEY" --reset
"$PY" -m verifier.verify_analysis --scenario nightfailure --base-url "$BASE" --api-key "$KEY"

echo
echo "All analysis scenarios verified."
