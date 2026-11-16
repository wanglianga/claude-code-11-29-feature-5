"""Clock calibration and geometric primitives.

A station clock is modelled as ``local = a * reference_arrival + b + noise``
inside one boot.  ``reference_arrival`` is the reference-time instant at which
the sound reached the station, i.e. the known emission instant plus the
propagation time derived from distance and configured propagation speed.  That
propagation term is removed *before* the line is fitted, as required.

A reboot can jump the clock by an arbitrary amount, so calibration models are
always keyed by ``(station_id, boot_id)`` and are never shared across boots.
Slow drift is handled by fitting up to ``cal_max_segments`` piecewise linear
segments per boot, with breakpoints chosen from anchor midpoints.

When there are too few anchors, the residual is too large, the anchor span is
too short, or the implied rate is implausible, the model is marked invalid and
every observation of that boot stays uncalibrated.  Upload time is never used
as a substitute for an arrival time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import combinations

from .config import Settings

REF_FLOOR_US = 1_000_000


@dataclass(frozen=True)
class Anchor:
    observation_id: int
    source_id: str
    local_us: int
    reference_arrival_us: float
    propagation_us: float


@dataclass
class Segment:
    local_from_us: int
    local_to_us: int
    slope: float
    intercept_us: float
    anchor_ids: list[int]
    rms_residual_us: float
    max_residual_us: float
    se_slope: float
    se_intercept_us: float

    def to_ref_arrival(self, local_us: int) -> float:
        return (local_us - self.intercept_us) / self.slope


@dataclass
class CalibrationFit:
    station_id: str
    boot_id: str
    valid: bool
    reason: str | None
    segments: list[Segment] = field(default_factory=list)
    anchor_ids: list[str] = field(default_factory=list)
    dropped_anchor_ids: list[str] = field(default_factory=list)
    anchor_count: int = 0
    used_anchor_observation_ids: list[int] = field(default_factory=list)
    rms_residual_us: float | None = None
    max_residual_us: float | None = None
    base_uncertainty_us: float | None = None
    local_min_us: int | None = None
    local_max_us: int | None = None
    local_span_us: int | None = None
    params: dict = field(default_factory=dict)

    def version_hash(self) -> str:
        payload = {
            "station": self.station_id,
            "boot": self.boot_id,
            "anchors": self.anchor_ids,
            "segments": [
                [
                    s.local_from_us,
                    s.local_to_us,
                    round(s.slope, 12),
                    round(s.intercept_us, 3),
                    s.anchor_ids,
                ]
                for s in self.segments
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

    def segment_for(self, local_us: int) -> Segment:
        for seg in self.segments:
            if seg.local_from_us <= local_us <= seg.local_to_us:
                return seg
        # Between two segments: use the nearest one.
        return min(
            self.segments,
            key=lambda s: 0
            if s.local_from_us <= local_us <= s.local_to_us
            else min(abs(local_us - s.local_from_us), abs(local_us - s.local_to_us)),
        )


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """Ordinary least squares.  Returns slope, intercept, se_slope, se_intercept."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two points")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        raise ValueError("degenerate regression: all anchors at the same local instant")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    if n > 2:
        variance = sum(r * r for r in residuals) / (n - 2)
        se_slope = math.sqrt(variance / sxx) if variance > 0 else 0.0
        se_intercept = math.sqrt(variance * (1.0 / n + (mx * mx) / sxx)) if variance > 0 else 0.0
    else:
        se_slope = 0.0
        se_intercept = 0.0
    return slope, intercept, se_slope, se_intercept


def _fit_group(points: list[Anchor]) -> tuple[Segment, float]:
    xs = [p.reference_arrival_us for p in points]
    ys = [float(p.local_us) for p in points]
    slope, intercept, se_slope, se_intercept = _linear_fit(xs, ys)
    seg = Segment(
        local_from_us=min(int(p.local_us) for p in points),
        local_to_us=max(int(p.local_us) for p in points),
        slope=slope,
        intercept_us=intercept,
        anchor_ids=[p.observation_id for p in points],
        rms_residual_us=0.0,
        max_residual_us=0.0,
        se_slope=se_slope,
        se_intercept_us=se_intercept,
    )
    residuals = [p.local_us - (slope * p.reference_arrival_us + intercept) for p in points]
    seg.rms_residual_us = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    seg.max_residual_us = max(abs(r) for r in residuals)
    return seg, sum(r * r for r in residuals)


def _candidate_splits(n: int, parts: int, min_len: int) -> list[tuple[int, ...]]:
    """Indices where new segments start, each segment holding >= min_len anchors."""
    if parts == 1:
        return [()]
    splits = []
    for combo in combinations(range(min_len, n - min_len * (parts - 1) + 1), parts - 1):
        ok = True
        bounds = (0,) + combo + (n,)
        for a, b in zip(bounds, bounds[1:]):
            if b - a < min_len:
                ok = False
                break
        if ok:
            splits.append(combo)
    return splits


def _fit_segmented(
    anchors: list[Anchor], parts: int, min_len: int
) -> tuple[list[Segment], float, list[int]] | None:
    if len(anchors) < parts * min_len:
        return None
    best = None
    for splits in _candidate_splits(len(anchors), parts, min_len):
        bounds = (0,) + splits + (len(anchors),)
        segments: list[Segment] = []
        rss = 0.0
        valid = True
        for a, b in zip(bounds, bounds[1:]):
            group = anchors[a:b]
            try:
                seg, group_rss = _fit_group(group)
            except ValueError:
                valid = False
                break
            segments.append(seg)
            rss += group_rss
        if not valid:
            continue
        if best is None or rss < best[1] - 1e-6:
            best = (segments, rss, list(splits))
    return best


def _slope_ok(segments: list[Segment], cfg: Settings) -> bool:
    return all(cfg.cal_slope_min <= s.slope <= cfg.cal_slope_max for s in segments)


def _apply_uncertainty(segments: list[Segment]) -> None:
    """Segment uncertainty already carries its residuals; nothing to mutate."""


def fit_calibration(
    station_id: str,
    boot_id: str,
    anchors: list[Anchor],
    cfg: Settings,
) -> CalibrationFit:
    """Fit a per-boot piecewise-linear local->reference-arrival map."""
    anchors = sorted(anchors, key=lambda a: a.local_us)
    fit = CalibrationFit(
        station_id=station_id,
        boot_id=boot_id,
        valid=False,
        reason=None,
        anchor_ids=[a.source_id for a in anchors],
        anchor_count=len(anchors),
    )
    if anchors:
        fit.local_min_us = anchors[0].local_us
        fit.local_max_us = anchors[-1].local_us
        fit.local_span_us = anchors[-1].local_us - anchors[0].local_us
    fit.params = {
        "min_anchors": cfg.cal_min_anchors,
        "min_segment_anchors": cfg.cal_min_segment_anchors,
        "max_segments": cfg.cal_max_segments,
        "max_residual_ms": cfg.cal_max_residual_ms,
        "max_rms_residual_ms": cfg.cal_max_rms_residual_ms,
        "min_anchor_span_s": cfg.cal_min_anchor_span_s,
        "coverage_margin_s": cfg.cal_coverage_margin_s,
        "slope_range": [cfg.cal_slope_min, cfg.cal_slope_max],
        "split_rms_ms": cfg.cal_split_rms_ms,
        "split_min_gain": cfg.cal_split_min_gain,
        "propagation_speed_mps": cfg.propagation_speed_mps,
    }

    if len(anchors) < cfg.cal_min_anchors:
        fit.reason = (
            f"too_few_anchors: {len(anchors)} calibration anchors, "
            f"need at least {cfg.cal_min_anchors}"
        )
        return fit
    if fit.local_span_us is None or fit.local_span_us < cfg.cal_min_anchor_span_s * 1e6:
        fit.reason = (
            f"anchor_span_too_short: {((fit.local_span_us or 0) / 1e6):.1f}s, "
            f"need at least {cfg.cal_min_anchor_span_s:.0f}s"
        )
        return fit

    # Robustify: iteratively refit and drop the worst anchor, provided enough
    # anchors remain.  Dropped anchors are retained in the record for audit.
    working = list(anchors)
    dropped: list[Anchor] = []
    for _ in range(max(1, cfg.cal_outlier_rounds)):
        trial = _fit_best_model(working, cfg)
        if trial is None:
            break
        segments, rss, model_parts = trial
        worst_idx = max(
            range(len(working)),
            key=lambda i: abs(
                working[i].local_us
                - _eval_local(segments, working[i].reference_arrival_us)
            ),
        )
        worst_res = abs(
            working[worst_idx].local_us
            - _eval_local(segments, working[worst_idx].reference_arrival_us)
        )
        if (
            worst_res <= cfg.cal_max_residual_ms * 1000.0
            or len(working) - 1 < cfg.cal_min_anchors
        ):
            break
        dropped.append(working.pop(worst_idx))
    if not working:
        fit.reason = "no_anchor_survived_outlier_rejection"
        return fit

    trial = _fit_best_model(working, cfg)
    if trial is None:
        # Distinguish "no line exists at all" from "a line exists but its rate
        # is physically implausible": the latter is the reboot-jump case.
        unconstrained = _fit_unconstrained(working)
        if unconstrained is not None:
            fit.segments = [unconstrained[0]]
            fit.params["implausible_segments"] = [
                {
                    "local_from_us": unconstrained[0].local_from_us,
                    "local_to_us": unconstrained[0].local_to_us,
                    "slope": unconstrained[0].slope,
                    "intercept_us": unconstrained[0].intercept_us,
                }
            ]
            fit.reason = (
                "implausible_clock_rate: fitted slope "
                f"{unconstrained[0].slope:.6f} outside "
                f"[{cfg.cal_slope_min}, {cfg.cal_slope_max}]; a restart jump must be "
                "reported as a new boot, not absorbed as drift"
            )
        else:
            fit.reason = "no_plausible_piecewise_model"
        fit.segments = []
        return fit
    segments, rss, model_parts = trial
    fit.segments = segments
    fit.used_anchor_observation_ids = [a.observation_id for a in working]
    fit.dropped_anchor_ids = [a.source_id for a in dropped]
    fit.rms_residual_us = math.sqrt(rss / max(1, len(working)))
    fit.max_residual_us = max(s.max_residual_us for s in segments)
    fit.params.update(
        {
            "segments": model_parts,
            "used_anchor_ids": [a.source_id for a in working],
            "dropped_anchor_ids": fit.dropped_anchor_ids,
            "rss_us2": rss,
            "residuals_us": [
                working[i].local_us - _eval_local(segments, working[i].reference_arrival_us)
                for i in range(len(working))
            ],
        }
    )

    if not _slope_ok(segments, cfg):
        fit.reason = (
            "implausible_clock_rate: fitted slope "
            + ", ".join(f"{s.slope:.6f}" for s in segments)
            + f" outside [{cfg.cal_slope_min}, {cfg.cal_slope_max}]; "
            "a restart jump must be reported as a new boot, not absorbed as drift"
        )
        return fit
    if fit.rms_residual_us > cfg.cal_max_rms_residual_ms * 1000.0:
        fit.reason = (
            f"residual_too_large: rms {fit.rms_residual_us / 1000.0:.1f}ms exceeds "
            f"{cfg.cal_max_rms_residual_ms:.0f}ms"
        )
        return fit
    if fit.max_residual_us > cfg.cal_max_residual_ms * 1000.0:
        fit.reason = (
            f"residual_too_large: max {fit.max_residual_us / 1000.0:.1f}ms exceeds "
            f"{cfg.cal_max_residual_ms:.0f}ms"
        )
        return fit

    fit.valid = True
    fit.base_uncertainty_us = max(
        fit.rms_residual_us, cfg.cal_uncertainty_floor_ms * 1000.0
    )
    return fit


def _fit_unconstrained(anchors: list[Anchor]) -> tuple[Segment, float] | None:
    try:
        return _fit_group(anchors)
    except ValueError:
        return None


def _fit_best_model(
    anchors: list[Anchor], cfg: Settings
) -> tuple[list[Segment], float, int] | None:
    """Pick the simplest model that is good enough; prefer more segments only
    when the RMS residual improves materially (guards against over-fitting)."""
    candidates: list[tuple[int, list[Segment], float]] = []
    for parts in range(1, cfg.cal_max_segments + 1):
        result = _fit_segmented(anchors, parts, cfg.cal_min_segment_anchors)
        if result is None:
            continue
        segments, rss, _splits = result
        if not _slope_ok(segments, cfg):
            continue
        rms = math.sqrt(rss / len(anchors))
        candidates.append((parts, segments, rms))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    best = candidates[0]
    for parts, segments, rms in candidates[1:]:
        gain = (best[2] - rms) / best[2] if best[2] > 0 else (1.0 if rms == 0 else 0.0)
        if rms < cfg.cal_split_rms_ms * 1000.0 and gain >= cfg.cal_split_min_gain:
            best = (parts, segments, rms)
        else:
            break
    parts, segments, rms = best
    rss = sum(s.rms_residual_us**2 * len(s.anchor_ids) for s in segments)
    return segments, rss, parts


def _eval_local(segments: list[Segment], reference_arrival_us: float) -> float:
    # Evaluate against the segment whose reference span contains the value.
    chosen = segments[0]
    for seg in segments:
        start_ref = (seg.local_from_us - seg.intercept_us) / seg.slope
        end_ref = (seg.local_to_us - seg.intercept_us) / seg.slope
        if start_ref <= reference_arrival_us <= end_ref:
            chosen = seg
            break
    return chosen.slope * reference_arrival_us + chosen.intercept_us


@dataclass
class CalibratedTime:
    status: str  # calibrated | calibrated_extrapolated | uncalibrated
    reference_arrival_us: int | None
    uncertainty_us: int | None
    reason: str | None
    segment_index: int | None


def apply_calibration(fit: CalibrationFit, local_us: int, cfg: Settings) -> CalibratedTime:
    if not fit.valid or not fit.segments:
        return CalibratedTime(
            status="uncalibrated",
            reference_arrival_us=None,
            uncertainty_us=None,
            reason=fit.reason or "no_valid_calibration",
            segment_index=None,
        )
    margin_us = cfg.cal_coverage_margin_s * 1e6
    status = "calibrated"
    extrapolation_us = 0.0
    if fit.local_min_us is not None and local_us < fit.local_min_us:
        extrapolation_us = fit.local_min_us - local_us
    elif fit.local_max_us is not None and local_us > fit.local_max_us:
        extrapolation_us = local_us - fit.local_max_us
    if extrapolation_us > margin_us:
        return CalibratedTime(
            status="uncalibrated",
            reference_arrival_us=None,
            uncertainty_us=None,
            reason=(
                "outside_calibrated_span: local reading is "
                f"{extrapolation_us / 1e6:.0f}s beyond the anchor span "
                f"(margin {cfg.cal_coverage_margin_s:.0f}s)"
            ),
            segment_index=None,
        )
    seg_index = 0
    seg = fit.segments[0]
    for idx, candidate in enumerate(fit.segments):
        if candidate.local_from_us <= local_us <= candidate.local_to_us:
            seg, seg_index = candidate, idx
            break
    else:
        seg_index = min(
            range(len(fit.segments)),
            key=lambda i: min(
                abs(local_us - fit.segments[i].local_from_us),
                abs(local_us - fit.segments[i].local_to_us),
            ),
        )
        seg = fit.segments[seg_index]
    if extrapolation_us > 0:
        status = "calibrated_extrapolated"
    ref = seg.to_ref_arrival(local_us)
    # Uncertainty in *reference* time.  Because local = slope*ref + intercept,
    # a local residual e corresponds to a reference-time error of e/slope, and
    # rate uncertainty compounds the local distance from the segment centre by
    # roughly d * se_slope / slope.  Anchors are drawn on both sides of the
    # query point, so the intercept standard error mostly cancels; only a small
    # extrapolation term is added outside the calibrated span.
    base = max(seg.rms_residual_us, fit.base_uncertainty_us or 0.0)
    centroid = (seg.local_from_us + seg.local_to_us) / 2.0
    rate_component = 2.0 * seg.se_slope * abs(local_us - centroid)
    # The intercept standard error from OLS is dominated by the lever arm of
    # the anchors, so it is a *local-clock* quantity centred on the segment;
    # near the centroid only the residual scatter matters.  Use the slope
    # standard error over the segment half-span instead of the raw
    # se_intercept (which grows with |mean local| and is not a local error).
    intercept_component = 2.0 * seg.se_slope * (seg.local_to_us - seg.local_from_us) / 2.0
    extrap_component = 2e-5 * extrapolation_us
    # Never let a span shorter than the anchors' own scatter imply precision
    # better than the anchor noise floor.
    uncertainty = max(
        cfg.cal_uncertainty_floor_ms * 1000.0,
        base + rate_component + intercept_component + extrap_component,
    )
    return CalibratedTime(
        status=status,
        reference_arrival_us=int(round(ref)),
        uncertainty_us=int(round(uncertainty)),
        reason=None,
        segment_index=seg_index,
    )
