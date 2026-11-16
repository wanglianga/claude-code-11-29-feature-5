"""Independent verification of the comparability-analysis reports.

Like ``verifier.verify`` this module shares no code with the service: it
recomputes, straight from the scenario's ``truth.json``,

* the **expected coverage** of every grid cell x period bin from the stations'
  true online windows and hearing ranges,
* the **expected event contributions** per region and stratum from the true
  source positions and emission instants,

and then checks the published report against them: no event may contribute
more than once in total, a stratum without coverage must be *unknown* (never
zero-filled), the common-coverage and standardised differences must reproduce
from the published strata and weights, and — for ``nightfailure`` — the old
report version must stay reproducible at its own cutoff while the new version
marks the failed nights unknown.

Usage::

    python -m verifier.verify_analysis --scenario daynight --base-url http://127.0.0.1:8000

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
AREA_Y_MIN = -1100.0
AREA_X_MIN = 0.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def _iso_to_us(value: str) -> int:
    text = value.replace("Z", "+00:00")
    return int(round(datetime.fromisoformat(text).timestamp() * 1_000_000))


def _us_to_s(value_us: int) -> float:
    return value_us / 1e6


# --------------------------------------------------------------------------- #
# Independent recomputation from truth
# --------------------------------------------------------------------------- #
def grid_cell_of(x: float, y: float, cell_m: float) -> tuple[int, int]:
    return (
        math.floor((x - AREA_X_MIN) / cell_m),
        math.floor((y - AREA_Y_MIN) / cell_m),
    )


def cell_center(ix: int, iy: int, cell_m: float) -> tuple[float, float]:
    return (
        AREA_X_MIN + (ix + 0.5) * cell_m,
        AREA_Y_MIN + (iy + 0.5) * cell_m,
    )


def region_cells(bbox: dict) -> list[tuple[int, int]]:
    return [
        (ix, iy)
        for ix in range(bbox["ix_min"], bbox["ix_max"] + 1)
        for iy in range(bbox["iy_min"], bbox["iy_max"] + 1)
    ]


def expected_bins(truth: dict) -> list[dict]:
    analysis = truth["analysis"]
    start = analysis["window"]["start_s"]
    end = analysis["window"]["end_s"]
    size = analysis["bin_seconds"]
    # Recover the scenario's reference epoch from any event that carries both
    # representations, so bin midpoints can be classified by UTC hour.
    anchor = truth["events"][0]
    epoch_s = _iso_to_us(anchor["emission"]) / 1e6 - anchor["emission_s"]
    bins = []
    k = 0
    cursor = start
    while cursor < end:
        mid = datetime.fromtimestamp(epoch_s + cursor + size / 2, tz=timezone.utc)
        period = "day" if 6 <= mid.hour < 18 else "night"
        bins.append({"index": k, "start_s": cursor, "end_s": cursor + size, "period_class": period})
        cursor += size
        k += 1
    return bins


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def station_windows(truth: dict) -> dict[str, list[tuple[float, float]]]:
    """True online intervals per station (reference seconds)."""
    out = {}
    for st in truth["stations"]:
        windows = st.get("online_windows_s")
        if windows is None:
            out[st["id"]] = [(-1e9, 1e9)]
        else:
            out[st["id"]] = [tuple(w) for w in windows]
    return out


def expected_coverage(
    truth: dict, analysis: dict
) -> dict[tuple[str, str], dict[str, float]]:
    """Expected covered cell-seconds per (region, (habitat, period_class))."""
    cell_m = analysis["grid_cell_m"]
    habitat_of = {
        (c["ix"], c["iy"]): c["habitat"] for c in analysis["habitat_map"]["cells"]
    }
    regions = {r["id"]: region_cells(r["bbox"]) for r in analysis["regions"]}
    bins = expected_bins(truth)
    windows = station_windows(truth)
    stations = {st["id"]: st for st in truth["stations"]}
    effort: dict[tuple[str, str], float] = {}
    full_effort: dict[tuple[str, str], float] = {}
    for region_id, cells in regions.items():
        for cell in cells:
            habitat = habitat_of.get(cell, "unclassified")
            cx, cy = cell_center(*cell, cell_m)
            hearers = [
                sid
                for sid, st in stations.items()
                if math.hypot(st["x"] - cx, st["y"] - cy) <= st["hearing_radius_m"]
            ]
            for b in bins:
                intervals: list[tuple[float, float]] = []
                for sid in hearers:
                    intervals.extend(windows[sid])
                merged = _union(intervals)
                covered = sum(
                    min(e, b["end_s"]) - max(s, b["start_s"])
                    for s, e in merged
                    if min(e, b["end_s"]) > max(s, b["start_s"])
                )
                covered = max(0.0, min(covered, b["end_s"] - b["start_s"]))
                key = (region_id, (habitat, b["period_class"]))
                effort[key] = effort.get(key, 0.0) + covered
                if covered >= (b["end_s"] - b["start_s"]) - 5.0:
                    full_effort[key] = full_effort.get(key, 0.0) + (b["end_s"] - b["start_s"])
    return {
        key: {"effort_cell_hours": effort.get(key, 0.0) / 3600.0,
              "fully_covered_cell_hours": full_effort.get(key, 0.0) / 3600.0}
        for key in set(effort) | set(full_effort)
    }


def expected_contributions(
    truth: dict, analysis: dict
) -> dict[tuple[str, str], float]:
    """Expected event contributions per (region, (habitat, period_class))."""
    cell_m = analysis["grid_cell_m"]
    habitat_of = {
        (c["ix"], c["iy"]): c["habitat"] for c in analysis["habitat_map"]["cells"]
    }
    regions = {r["id"]: set(region_cells(r["bbox"])) for r in analysis["regions"]}
    bins = expected_bins(truth)
    counts: dict[tuple[str, str], float] = {}
    for event in truth["events"]:
        if not event["heard_by"]:
            continue
        cell = grid_cell_of(event["x"], event["y"], cell_m)
        habitat = habitat_of.get(cell, "unclassified")
        region = next((rid for rid, cells in regions.items() if cell in cells), None)
        if region is None:
            continue
        b = next(
            (b for b in bins if b["start_s"] <= event["emission_s"] < b["end_s"]),
            None,
        )
        if b is None:
            continue
        key = (region, (habitat, b["period_class"]))
        counts[key] = counts.get(key, 0.0) + 1.0
    return counts


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
class Api:
    def __init__(self, base: str, api_key: str):
        self.base = base.rstrip("/")
        self.api_key = api_key

    def get(self, path: str, **params):
        r = requests.get(
            self.base + path,
            headers={"X-API-Key": self.api_key},
            params=params or None,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict | None = None):
        r = requests.post(
            self.base + path,
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
            json=body or {},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()


def fetch_reports(api: Api, region_a: str, region_b: str) -> list[dict]:
    listed = api.get("/analysis/reports", limit=200)
    matching = [
        r for r in listed
        if r["config"]["region_a_id"] == region_a and r["config"]["region_b_id"] == region_b
    ]
    reports = [api.get(f"/analysis/reports/{r['id']}") for r in matching]
    reports.sort(key=lambda r: r["as_of"])
    return reports


# --------------------------------------------------------------------------- #
# Shared checks
# --------------------------------------------------------------------------- #
def check_contribution_integrity(report: dict) -> Check:
    """One event contributes exactly one unit in total, never more."""
    bad = []
    total = 0.0
    for item in report["result"]["events"]["contributions"]:
        weight = sum(c["weight"] for c in item["cells"]) + item["outside_regions_weight"]
        total += sum(c["weight"] for c in item["cells"])
        # Weights are published rounded to 1e-6 per cell; the tolerance absorbs
        # that rounding while still catching any real double count or loss.
        if abs(weight - 1.0) > 1e-5:
            bad.append((item["event_id"], round(weight, 6)))
    return Check(
        "every event contributes exactly one unit across its candidate grids",
        not bad,
        f"{report['result']['events']['included']} events, total spatial contribution "
        f"{total:.3f}; violations: {bad[:3]}",
    )


def check_no_population_estimate(report: dict) -> Check:
    text = json.dumps(report["result"]).lower()
    forbidden = [k for k in ("population", "abundance", "density_estimate") if k in text]
    return Check(
        "report contains activity rates only, never a population estimate",
        not forbidden,
        f"forbidden keys found: {forbidden}" if forbidden else "no population-level quantities",
    )


def check_unknown_never_zero(report: dict) -> Check:
    """A stratum without coverage is unknown: rate null, never a filled zero."""
    bad = []
    for s in report["result"]["strata"]:
        for side in ("a", "b"):
            reg = s["regions"][side]
            if reg["status"].startswith("no_coverage"):
                if reg["rate_per_cell_hour"] is not None or reg["rate_fully_covered"] is not None:
                    bad.append((s["habitat"], s["period_class"], side))
    return Check(
        "strata without coverage are unknown, never zero-filled",
        not bad,
        f"violations: {bad[:4]}" if bad else "all no-coverage strata carry null rates",
    )


def check_common_coverage_arithmetic(report: dict) -> Check:
    """The published difference must reproduce from the published strata."""
    cc = report["result"].get("common_coverage")
    if cc is None:
        return Check("common-coverage arithmetic", False, "section missing")
    strata = {(s["habitat"], s["period_class"]): s for s in report["result"]["strata"]}
    total_w = 0.0
    weighted = 0.0
    for entry in cc["strata"]:
        row = strata.get((entry["habitat"], entry["period_class"]))
        if row is None:
            return Check("common-coverage arithmetic", False, "stratum missing from table")
        a, b = row["regions"]["a"], row["regions"]["b"]
        w = min(a["fully_covered_effort_cell_hours"], b["fully_covered_effort_cell_hours"])
        if abs(w - entry["weight"]) > 1e-6:
            return Check(
                "common-coverage arithmetic", False,
                f"weight mismatch on {entry}: published {entry['weight']} vs recomputed {w}",
            )
        total_w += w
        weighted += w * (a["rate_fully_covered"] - b["rate_fully_covered"])
    if total_w == 0:
        ok = cc["difference"] is None
        return Check("common-coverage arithmetic", ok, "no common strata; difference is null")
    recomputed = weighted / total_w
    ok = cc["difference"] is not None and abs(recomputed - cc["difference"]) < 1e-6
    return Check(
        "common-coverage arithmetic",
        ok,
        f"published {cc['difference']} vs recomputed {recomputed:.9f} over "
        f"{len(cc['strata'])} strata",
    )


def check_standardized_arithmetic(report: dict, truth: dict) -> Check:
    std = report["result"].get("standardized")
    if std is None or std.get("status") != "done":
        return Check("standardised arithmetic", False, "section missing or skipped")
    ref_entries = truth["analysis"]["reference_composition"]["entries"]
    weights = {(e["habitat"], e["period_class"]): e["weight"] for e in ref_entries}
    empty_keys = {(e["habitat"], e["period_class"]) for e in std["empty_strata"]}
    expected_excluded = sum(w for k, w in weights.items() if k in empty_keys)
    ok = abs(std["excluded_weight_mass"] - expected_excluded) < 1e-6
    strata = {(s["habitat"], s["period_class"]): s for s in report["result"]["strata"]}
    used = [k for k, w in weights.items() if k not in empty_keys and w > 0]
    num = 0.0
    den = 0.0
    for key in used:
        row = strata.get(key)
        if row is None:
            continue
        a, b = row["regions"]["a"], row["regions"]["b"]
        if a["effort_cell_hours"] > 0 and b["effort_cell_hours"] > 0:
            num += weights[key] * (a["rate_per_cell_hour"] - b["rate_per_cell_hour"])
            den += weights[key]
    recomputed = num / den if den else None
    if recomputed is None:
        ok = ok and std["difference"] is None
    else:
        ok = ok and std["difference"] is not None and abs(recomputed - std["difference"]) < 1e-6
    # Every empty stratum must carry its reference weight and never a zero rate.
    for entry in std["empty_strata"]:
        key = (entry["habitat"], entry["period_class"])
        if abs(entry["reference_weight"] - weights.get(key, -1)) > 1e-9:
            ok = False
    return Check(
        "standardised arithmetic and empty-stratum disclosure",
        ok,
        f"published difference {std['difference']} vs recomputed {recomputed}; "
        f"excluded mass {std['excluded_weight_mass']} (expected {expected_excluded}); "
        f"{len(std['empty_strata'])} empty strata",
    )


def check_efforts_against_truth(report: dict, truth: dict, label: str) -> Check:
    expected = expected_coverage(truth, truth["analysis"])
    regions = {r["id"]: side for side, r in
               (("a", report["result"]["regions"]["a"]), ("b", report["result"]["regions"]["b"]))}
    region_by_side = {"a": report["result"]["regions"]["a"]["id"],
                      "b": report["result"]["regions"]["b"]["id"]}
    worst = 0.0
    worst_key = None
    compared = 0
    for s in report["result"]["strata"]:
        for side in ("a", "b"):
            region_id = region_by_side[side]
            key = (region_id, (s["habitat"], s["period_class"]))
            exp = expected.get(key, {"effort_cell_hours": 0.0})
            got = s["regions"][side]["effort_cell_hours"]
            diff = abs(got - exp["effort_cell_hours"])
            compared += 1
            if diff > worst:
                worst, worst_key = diff, (region_id, s["habitat"], s["period_class"], side)
    ok = worst <= 0.1
    return Check(
        f"coverage efforts match the truth-derived efforts ({label})",
        ok,
        f"{compared} stratum-region efforts compared; worst |diff| = {worst:.4f} "
        f"cell-hours at {worst_key}",
    )


def check_contributions_against_truth(report: dict, truth: dict) -> Check:
    expected = expected_contributions(truth, truth["analysis"])
    region_by_side = {"a": report["result"]["regions"]["a"]["id"],
                      "b": report["result"]["regions"]["b"]["id"]}
    bad = []
    compared = 0
    for s in report["result"]["strata"]:
        for side in ("a", "b"):
            region_id = region_by_side[side]
            key = (region_id, (s["habitat"], s["period_class"]))
            exp = expected.get(key, 0.0)
            got = s["regions"][side]["contributions"]
            compared += 1
            if abs(got - exp) > 0.02:
                bad.append((region_id, s["habitat"], s["period_class"], exp, got))
    return Check(
        "event contributions per stratum match truth (no double counting, none lost)",
        not bad,
        f"{compared} stratum-region counts compared; mismatches: {bad[:4]}",
    )


def check_reproduce(api: Api, report_id: int) -> Check:
    out = api.post(f"/analysis/reports/{report_id}/reproduce")
    return Check(
        f"report {report_id} re-derives identically at its cutoff",
        out["reproducible"] is True,
        f"stored {out['report_hash']} vs recomputed {out['recomputed_hash']}",
    )


# --------------------------------------------------------------------------- #
# Scenario drivers
# --------------------------------------------------------------------------- #
def verify_daynight(api: Api, truth: dict) -> list[Check]:
    analysis = truth["analysis"]
    reports = fetch_reports(api, analysis["region_a_id"], analysis["region_b_id"])
    if not reports:
        return [Check("daynight report exists", False, "no report found for west/east")]
    report = reports[-1]
    result = report["result"]
    expected = analysis["expected"]
    checks = [
        check_contribution_integrity(report),
        check_no_population_estimate(report),
        check_unknown_never_zero(report),
        check_efforts_against_truth(report, truth, "daynight"),
        check_contributions_against_truth(report, truth),
        check_common_coverage_arithmetic(report),
        check_standardized_arithmetic(report, truth),
        check_reproduce(api, report["id"]),
    ]
    naive = result["naive_counts"]
    checks.append(Check(
        "naive counts expose the sampling bias (and carry a warning)",
        naive["difference"] == expected["naive_difference"] and bool(naive.get("warning")),
        f"counts {naive['counts']}, difference {naive['difference']}",
    ))
    common_keys = {(s["habitat"], s["period_class"]) for s in result["common_coverage"]["strata"]}
    expected_common = {tuple(s) for s in expected["common_strata"]}
    diff = result["common_coverage"]["difference"]
    checks.append(Check(
        "common-coverage comparison restricted to the day strata, difference ~ 0",
        common_keys == expected_common
        and diff is not None
        and abs(diff) <= expected["max_abs_common_difference"],
        f"strata {sorted(common_keys)}, difference {diff}",
    ))
    incomparable = {
        (s["habitat"], s["period_class"]) for s in result["common_coverage"]["incomparable_strata"]
    }
    expected_inc = {tuple(s) for s in expected["incomparable_strata"]}
    checks.append(Check(
        "night and mudflat strata are listed as incomparable (unknown on the east side)",
        incomparable == expected_inc
        and all(s["unknown_side"] == "b" for s in result["common_coverage"]["incomparable_strata"]),
        f"incomparable {sorted(incomparable)}",
    ))
    empty = {(s["habitat"], s["period_class"]) for s in result["standardized"]["empty_strata"]}
    expected_empty = {tuple(s) for s in expected["empty_strata_standardized"]}
    checks.append(Check(
        "standardised report discloses every empty stratum with its reference weight",
        empty == expected_empty,
        f"empty {sorted(empty)}",
    ))
    checks.append(Check(
        "standardised report attaches the common-coverage cross-check",
        result["standardized"].get("common_coverage_crosscheck") is not None,
        "crosscheck embedded" if result["standardized"].get("common_coverage_crosscheck")
        else "missing",
    ))
    return checks


def verify_nightfailure(api: Api, truth: dict) -> list[Check]:
    analysis = truth["analysis"]
    reports = fetch_reports(api, analysis["region_a_id"], analysis["region_b_id"])
    if len(reports) < 2:
        return [Check("two report versions exist", False, f"found {len(reports)}")]
    v1, v2 = reports[0], reports[-1]
    expected = analysis["expected"]
    checks = [Check("two report versions exist", True, f"v1 id {v1['id']}, v2 id {v2['id']}")]

    r1 = v1["result"]
    night_v1 = next(
        s for s in r1["strata"] if s["habitat"] == "reed" and s["period_class"] == "night"
    )["regions"]["a"]
    checks.append(Check(
        "v1 scores the failed north nights as covered with a (false) zero rate",
        night_v1["status"] == expected["v1_night_status"]
        and night_v1["rate_per_cell_hour"] == 0.0,
        f"status {night_v1['status']}, rate {night_v1['rate_per_cell_hour']}",
    ))
    d1 = r1["common_coverage"]["difference"]
    checks.append(Check(
        "v1's comparison is visibly biased by the unreported failure",
        d1 is not None and d1 <= expected["v1_common_difference_max_abs"],
        f"v1 difference {d1}",
    ))

    r2 = v2["result"]
    night_v2 = next(
        s for s in r2["strata"] if s["habitat"] == "reed" and s["period_class"] == "night"
    )["regions"]["a"]
    checks.append(Check(
        "v2 marks the failed north nights unknown (no coverage), never zero",
        night_v2["status"] == expected["v2_night_status"]
        and night_v2["rate_per_cell_hour"] is None,
        f"status {night_v2['status']}, rate {night_v2['rate_per_cell_hour']}",
    ))
    d2 = r2["common_coverage"]["difference"]
    checks.append(Check(
        "v2's comparison on the comparable sample is ~ 0",
        d2 is not None and abs(d2) <= expected["v2_common_difference_max_abs"],
        f"v2 difference {d2}",
    ))
    empty_v2 = {(s["habitat"], s["period_class"]) for s in r2["standardized"]["empty_strata"]}
    checks.append(Check(
        "v2 standardised report discloses the night stratum as empty",
        empty_v2 == {tuple(s) for s in expected["v2_empty_strata_standardized"]},
        f"empty {sorted(empty_v2)}",
    ))

    checks.extend(
        [
            check_contribution_integrity(v2),
            check_unknown_never_zero(v2),
            check_efforts_against_truth(v2, truth, "nightfailure v2"),
            check_contributions_against_truth(v2, truth),
            check_common_coverage_arithmetic(v2),
            check_standardized_arithmetic(v2, truth),
            check_no_population_estimate(v2),
        ]
    )

    checks.append(Check(
        "the two versions differ (late evidence changed the report)",
        v1["report_hash"] != v2["report_hash"],
        f"v1 {v1['report_hash']} vs v2 {v2['report_hash']}",
    ))
    checks.append(check_reproduce(api, v1["id"]))
    checks.append(check_reproduce(api, v2["id"]))

    compare = api.get(
        "/analysis/reports/compare", baseline_id=v1["id"], current_id=v2["id"]
    )
    lost = [
        c for c in compare["stratum_changes"]
        if c["change"] == "coverage_lost"
        and c["habitat"] == "reed"
        and c["period_class"] == "night"
        and c["region"] == analysis["region_a_id"]
    ]
    evidence = compare["evidence_arrived_between_cutoffs"]
    checks.append(Check(
        "version diff names the lost night coverage and the late gap declarations",
        bool(lost)
        and evidence["progress_reports_declaring_gaps"] == len(analysis["late_gap_declarations"]),
        f"{len(compare['stratum_changes'])} stratum changes, "
        f"{evidence['progress_reports_declaring_gaps']} gap declarations",
    ))
    return checks


def run_checks(scenario: str, truth_path: Path, base_url: str, api_key: str) -> int:
    truth = json.loads(truth_path.read_text())
    api = Api(base_url, api_key)
    if scenario == "daynight":
        checks = verify_daynight(api, truth)
    elif scenario == "nightfailure":
        checks = verify_nightfailure(api, truth)
    else:  # pragma: no cover
        raise KeyError(f"no analysis checks for {scenario}")
    for check in checks:
        print(check)
    failed = [c for c in checks if not c.ok]
    payload = {
        "scenario": scenario,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "all_passed": not failed,
    }
    out = truth_path.parent / "verification_analysis.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n{payload['passed']}/{len(checks)} checks passed -> {out}")
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent analysis verifier")
    parser.add_argument("--scenario", required=True, choices=["daynight", "nightfailure"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="dev-api-key")
    parser.add_argument(
        "--truth", type=Path, default=None, help="defaults to samples/<scenario>/truth.json"
    )
    args = parser.parse_args(argv)
    truth = args.truth or (
        Path(__file__).resolve().parent.parent / "samples" / args.scenario / "truth.json"
    )
    return run_checks(args.scenario, truth, args.base_url, args.api_key)


if __name__ == "__main__":
    raise SystemExit(main())
