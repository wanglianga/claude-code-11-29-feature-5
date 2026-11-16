"""Drive the analysis scenarios end to end against a running service.

For each analysis scenario this driver

1. uploads the station waves (via the standard simulator),
2. runs the processing pipeline,
3. publishes the analysis definitions bundled in the scenario (regions,
   habitat map, reference composition),
4. generates the comparability report — and, for ``nightfailure``, then
   delivers the late gap declarations and issues a second report at a new
   cutoff, so the version difference can be checked.

Usage::

    python -m simulator.analysis_demo --scenario daynight --reset
    python -m simulator.analysis_demo --scenario nightfailure --reset
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import simulator.run_simulator as sim  # noqa: E402
from simulator.scenarios import REFERENCE_EPOCH, local_iso, us  # noqa: E402

EPOCH_US = us(REFERENCE_EPOCH)


def _local_from_ref(boot_truth: dict, ref_s: float) -> str:
    """Station-local reading of a reference instant, from the truth clock."""
    ref_us = EPOCH_US + int(round(ref_s * 1_000_000))
    return local_iso(int(round(boot_truth["slope"] * ref_us + boot_truth["intercept_us"])))


def publish_analysis_definitions(api, analysis: dict) -> None:
    for region in analysis["regions"]:
        api._req("POST", "/analysis/regions", json={
            "id": region["id"],
            "name": region["name"],
            "grid_cell_size_m": analysis["grid_cell_m"],
            "bbox": region["bbox"],
        })
    api._req("POST", "/analysis/habitat-maps", json={
        "id": analysis["habitat_map"]["id"],
        "grid_cell_size_m": analysis["grid_cell_m"],
        "cells": analysis["habitat_map"]["cells"],
    })
    ref = analysis["reference_composition"]
    api._req("POST", "/analysis/reference-compositions", json={
        "id": ref["id"],
        "name": ref["name"],
        "entries": ref["entries"],
    })


def _window_iso(analysis: dict) -> dict:
    fmt = "%Y-%m-%dT%H:%M:%S.000000Z"
    start = REFERENCE_EPOCH + timedelta(seconds=analysis["window"]["start_s"])
    end = REFERENCE_EPOCH + timedelta(seconds=analysis["window"]["end_s"])
    return {"start": start.strftime(fmt), "end": end.strftime(fmt)}


def create_report(api, analysis: dict) -> dict:
    return api._req("POST", "/analysis/reports", json={
        "region_a_id": analysis["region_a_id"],
        "region_b_id": analysis["region_b_id"],
        "habitat_map_id": analysis["habitat_map"]["id"],
        "reference_composition_id": analysis["reference_composition"]["id"],
        "class_label": analysis["class_label"],
        "window": _window_iso(analysis),
        "grid_cell_size_m": analysis["grid_cell_m"],
        "bin_seconds": analysis["bin_seconds"],
        "created_by": "analysis-demo",
    })


def run_daynight(base: str, api_key: str, reset: bool, api=None) -> dict:
    loaded = sim.load_scenario("daynight", base, reset=reset)
    api = api or sim.Api(base, api_key=api_key)
    api._req("POST", "/runs", json={"wait": True})
    analysis = loaded["scn"].analysis
    publish_analysis_definitions(api, analysis)
    report = create_report(api, analysis)
    reproduce = api._req("POST", f"/analysis/reports/{report['id']}/reproduce", json={})
    return {"scenario": "daynight", "report": report, "reproduce": reproduce, "loaded": loaded}


def run_nightfailure(base: str, api_key: str, reset: bool, api=None) -> dict:
    loaded = sim.load_scenario("nightfailure", base, reset=reset)
    api = api or sim.Api(base, api_key=api_key)
    api._req("POST", "/runs", json={"wait": True})
    analysis = loaded["scn"].analysis
    publish_analysis_definitions(api, analysis)
    v1 = create_report(api, analysis)

    # Late evidence: the night-time recorder failures are confirmed only now.
    boots = {st["id"]: st["boots"][0] for st in loaded["truth"]["stations"]}
    for declaration in analysis["late_gap_declarations"]:
        boot = boots[declaration["station_id"]]
        gaps = [
            {
                "from_local_at": _local_from_ref(boot, gap[0]),
                "through_local_at": _local_from_ref(boot, gap[1]),
                "reason": declaration["reason"],
            }
            for gap in declaration["gaps_s"]
        ]
        api._req(
            "POST",
            f"/stations/{declaration['station_id']}/progress",
            json={
                "boot_id": declaration["boot_id"],
                "progress": {
                    "complete_through_local_at": _local_from_ref(boot, 202_200.0),
                    "complete": True,
                    "gaps": gaps,
                },
            },
        )

    v2 = api._req("POST", f"/analysis/reports/{v1['id']}/refresh",
                  json={"created_by": "analysis-demo"})
    compare = api._req(
        "GET",
        f"/analysis/reports/compare?baseline_id={v1['id']}&current_id={v2['id']}",
    )
    reproduce_v1 = api._req("POST", f"/analysis/reports/{v1['id']}/reproduce", json={})
    reproduce_v2 = api._req("POST", f"/analysis/reports/{v2['id']}/reproduce", json={})
    return {
        "scenario": "nightfailure",
        "report_v1": v1,
        "report_v2": v2,
        "compare": compare,
        "reproduce_v1": reproduce_v1,
        "reproduce_v2": reproduce_v2,
        "loaded": loaded,
    }


RUNNERS = {"daynight": run_daynight, "nightfailure": run_nightfailure}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analysis scenario driver")
    parser.add_argument("--scenario", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="dev-api-key")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args(argv)
    result = RUNNERS[args.scenario](args.base_url, args.api_key, args.reset)
    summary = {"scenario": result["scenario"]}
    if "report" in result:
        summary["report_id"] = result["report"]["id"]
        summary["report_hash"] = result["report"]["report_hash"]
        summary["reproducible"] = result["reproduce"]["reproducible"]
    else:
        summary["report_v1"] = {"id": result["report_v1"]["id"], "hash": result["report_v1"]["report_hash"]}
        summary["report_v2"] = {"id": result["report_v2"]["id"], "hash": result["report_v2"]["report_hash"]}
        summary["v1_reproducible"] = result["reproduce_v1"]["reproducible"]
        summary["v2_reproducible"] = result["reproduce_v2"]["reproducible"]
        summary["stratum_changes"] = len(result["compare"]["stratum_changes"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
