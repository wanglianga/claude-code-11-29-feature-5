"""Runtime configuration for the wetland observation service.

Every threshold that influences an inference is surfaced here (and copied into
each processing-run snapshot) so that a run can be reproduced exactly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Static process configuration (env-driven)."""

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://ccdev:ccdev@127.0.0.1:5432/wetland_dev",
    )
    api_key: str = os.getenv("WETLAND_API_KEY", "dev-api-key")
    auth_disabled: bool = _env_bool("WETLAND_AUTH_DISABLED", False)
    allow_reset: bool = _env_bool("WETLAND_ALLOW_RESET", True)
    samples_dir: Path = Path(os.getenv("WETLAND_SAMPLES_DIR", str(BASE_DIR / "samples")))

    # Physical model
    propagation_speed_mps: float = _env_float("PROPAGATION_SPEED_MPS", 343.0)
    area_x_min: float = _env_float("AREA_X_MIN", 0.0)
    area_x_max: float = _env_float("AREA_X_MAX", 2100.0)
    area_y_min: float = _env_float("AREA_Y_MIN", -1100.0)
    area_y_max: float = _env_float("AREA_Y_MAX", 1100.0)

    # Calibration
    cal_min_anchors: int = _env_int("CAL_MIN_ANCHORS", 4)
    cal_min_segment_anchors: int = _env_int("CAL_MIN_SEGMENT_ANCHORS", 3)
    cal_max_segments: int = _env_int("CAL_MAX_SEGMENTS", 3)
    cal_max_residual_ms: float = _env_float("CAL_MAX_RESIDUAL_MS", 250.0)
    cal_max_rms_residual_ms: float = _env_float("CAL_MAX_RMS_RESIDUAL_MS", 120.0)
    cal_min_anchor_span_s: float = _env_float("CAL_MIN_ANCHOR_SPAN_S", 60.0)
    cal_coverage_margin_s: float = _env_float("CAL_COVERAGE_MARGIN_S", 3600.0)
    cal_slope_min: float = _env_float("CAL_SLOPE_MIN", 0.9)
    cal_slope_max: float = _env_float("CAL_SLOPE_MAX", 1.1)
    cal_split_rms_ms: float = _env_float("CAL_SPLIT_RMS_MS", 40.0)
    cal_split_min_gain: float = _env_float("CAL_SPLIT_MIN_GAIN", 0.15)
    cal_outlier_rounds: int = _env_int("CAL_OUTLIER_ROUNDS", 3)
    cal_uncertainty_floor_ms: float = _env_float("CAL_UNCERTAINTY_FLOOR_MS", 2.0)
    localization_slack_ms: float = _env_float("LOCALIZATION_SLACK_MS", 25.0)
    definite_region_max_extent_m: float = _env_float("DEFINITE_REGION_MAX_EXTENT_M", 200.0)

    # Association
    feature_distance_limit: float = _env_float("FEATURE_DISTANCE_LIMIT", 0.25)
    association_time_slack_ms: float = _env_float("ASSOCIATION_TIME_SLACK_MS", 150.0)
    max_group_size: int = _env_int("MAX_GROUP_SIZE", 8)
    max_candidate_groups: int = _env_int("MAX_CANDIDATE_GROUPS", 3000)
    min_event_stations: int = _env_int("MIN_EVENT_STATIONS", 2)

    # Localization grid
    grid_coarse_m: float = _env_float("GRID_COARSE_M", 100.0)
    grid_refine_factor: int = _env_int("GRID_REFINE_FACTOR", 2)
    grid_refine_levels: int = _env_int("GRID_REFINE_LEVELS", 3)
    grid_max_leaf_cells: int = _env_int("GRID_MAX_LEAF_CELLS", 6000)
    max_returned_cells_per_region: int = _env_int("MAX_RETURNED_CELLS_PER_REGION", 2000)
    geometry_collinear_tolerance: float = _env_float("GEOMETRY_COLLINEAR_TOLERANCE", 1e-6)

    # Solver
    max_solver_nodes: int = _env_int("MAX_SOLVER_NODES", 2000000)
    max_solution_options: int = _env_int("MAX_SOLUTION_OPTIONS", 5)
    residual_tie_epsilon_us: int = _env_int("RESIDUAL_TIE_EPSILON_US", 1)

    # Finalization
    hearing_radius_m: float = _env_float("HEARING_RADIUS_M", 1600.0)
    progress_margin_s: float = _env_float("PROGRESS_MARGIN_S", 120.0)

    # Processing
    max_run_observations: int = _env_int("MAX_RUN_OBSERVATIONS", 20000)

    # Comparability analysis (regional activity-rate reports)
    analysis_grid_cell_m: float = _env_float("ANALYSIS_GRID_CELL_M", 250.0)
    analysis_bin_seconds: int = _env_int("ANALYSIS_BIN_SECONDS", 43200)
    analysis_day_start_hour: int = _env_int("ANALYSIS_DAY_START_HOUR", 6)
    analysis_night_start_hour: int = _env_int("ANALYSIS_NIGHT_START_HOUR", 18)
    # A cell-bin is treated as uncovered when its covered time is below this
    # tolerance, and as fully covered when its uncovered time is below it.
    # The tolerance absorbs the calibration uncertainty of gap endpoints; it
    # is far below any real shutdown.
    analysis_coverage_tolerance_s: float = _env_float(
        "ANALYSIS_COVERAGE_TOLERANCE_S", 5.0
    )

    def as_dict(self) -> dict:
        data = asdict(self)
        data["samples_dir"] = str(self.samples_dir)
        return data


settings = Settings()


@dataclass(frozen=True)
class Area:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("area must have positive width and height")

    def as_dict(self) -> dict:
        return asdict(self)


def default_area() -> Area:
    return Area(settings.area_x_min, settings.area_x_max, settings.area_y_min, settings.area_y_max)
