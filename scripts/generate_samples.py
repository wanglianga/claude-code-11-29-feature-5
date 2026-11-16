"""Regenerate the bundled sample scenarios under ``samples/<scenario>/``.

Each directory gets ``truth.json``: stations, per-boot ground-truth clock models,
calibration-source instants, upload waves (station-side records) and the true
events.  Nothing here talks to the service, so the files are reproducible and
safe to regenerate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator.run_simulator import write_ground_truth  # noqa: E402
from simulator.scenarios import build_scenario, scenario_names  # noqa: E402


def main() -> int:
    out_root = ROOT / "samples"
    summary = {}
    for name in scenario_names():
        scn = build_scenario(name)
        data = write_ground_truth(scn, out_root / name)
        summary[name] = {
            "title": scn.title,
            "stations": len(scn.stations),
            "truth_events": len(data["truth"]["events"]),
            "waves": len(data["truth"]["waves"]),
            "records": sum(len(w["records"]) for w in data["truth"]["waves"]),
        }
        print(f"{name}: {summary[name]}")
    (out_root / "index.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
