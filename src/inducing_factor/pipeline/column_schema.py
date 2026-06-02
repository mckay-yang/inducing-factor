"""Explicit column classification schema for data leakage prevention.

Defines which columns are allowed as predictor features, which are labels,
which are metadata identifiers, and which are strictly forbidden (CCDC-derived
or target-dependent) at prediction time.

This module serves as the canonical reference for feature/label separation
throughout the pipeline. No other file should independently decide which
columns are predictors vs. labels.
"""

from typing import Iterable, List, Set

# ═══════════════════════════════════════════════════════════════════════════════
# Labels — NEVER enter predictors at training or prediction time.
# These are the targets the model aims to predict.
# ═══════════════════════════════════════════════════════════════════════════════
LABEL_COLUMNS: Set[str] = {
    "disturbance_flag",   # binary: 1 = disturbance occurred this year
    "disturbance_class",  # categorical: 1=fire, 2=logging, 3=geo_hazard, 4=road_construction
}

# ═══════════════════════════════════════════════════════════════════════════════
# Metadata — identifiers and spatial/temporal bookkeeping.
# These track where/when each row belongs but carry no causal signal.
# ═══════════════════════════════════════════════════════════════════════════════
METADATA_COLUMNS: Set[str] = {
    "point_id",
    "raster_row",
    "raster_col",
    "x",
    "y",
    "year",
    "month",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Static predictors — time-invariant per-point, always available.
# ═══════════════════════════════════════════════════════════════════════════════
STATIC_PREDICTORS: Set[str] = {
    "slope",
    "aspect",
    "dist_to_river",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Historical / dynamic predictors — available at prediction time for a target
# year (computed from prior years' data via shift/lag).
# ═══════════════════════════════════════════════════════════════════════════════
HISTORICAL_PREDICTORS: Set[str] = {
    # rainfall — direct window features (computed from reference_month)
    "rainfall_prev_1m",
    "rainfall_prev_2m",
    "rainfall_prev_3m",
    "rainfall_prev_total_3m",
    # dynamic features with {last, mean, max} rolling aggregations
    "temperature_prev_last",
    "temperature_prev_mean",
    "temperature_prev_max",
    "resident_prev_last",
    "resident_prev_mean",
    "resident_prev_max",
    "dist_to_road_prev_last",
    "dist_to_road_prev_mean",
    "dist_to_road_prev_max",
    "dist_to_railway_prev_last",
    "dist_to_railway_prev_mean",
    "dist_to_railway_prev_max",
    "dist_to_waterway_prev_last",
    "dist_to_waterway_prev_mean",
    "dist_to_waterway_prev_max",
    # categorical temporal — only "last" aggregation
    "landuse_prev_last",
}

# ═══════════════════════════════════════════════════════════════════════════════
# ALL allowed predictors (union of static + historical).
# ═══════════════════════════════════════════════════════════════════════════════
ALLOWED_PREDICTORS: Set[str] = STATIC_PREDICTORS | HISTORICAL_PREDICTORS

# ═══════════════════════════════════════════════════════════════════════════════
# Forbidden — CCDC-derived or target-dependent columns.
# These MUST NEVER appear in:
#   - feature_columns
#   - discretizer input (training or prediction)
#   - predict_dataframe input
#
# Some of these are labels; others are intermediate values used to derive
# features but which encode information from the disturbance event itself
# (e.g. reference_month → rainfall_prev_* calculations).
# ═══════════════════════════════════════════════════════════════════════════════
LEAKAGE_FORBIDDEN: Set[str] = {
    "disturbance_time_value",  # decimal-year timestamp of disturbance (CCDC output)
    "disturbance_type",        # raw CCDC type code
    "disturbance_month",       # month derived from disturbance_time_value
    "disturbance_class",       # categorical disturbance class label
    "disturbance_flag",        # binary disturbance label
    "reference_month",         # assigned month used for rainfall window computation
}


# ═══════════════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def validate_predictor_columns(columns: Iterable[str]) -> List[str]:
    """Check that all columns are in the allowed predictor set.

    Returns a list of column names that are NOT allowed predictors.
    An empty list means all columns are valid.

    Args:
        columns: Iterable of column name strings.

    Returns:
        List of violating column names (empty = valid).
    """
    col_set = set(columns)
    violations = col_set - ALLOWED_PREDICTORS
    return sorted(violations)


def validate_no_leakage(df_columns: Iterable[str]) -> List[str]:
    """Detect CCDC-derived or target-dependent columns present in a DataFrame.

    Returns any columns from LEAKAGE_FORBIDDEN that appear in *df_columns*.
    These columns represent potential data leakage at prediction time.

    Args:
        df_columns: Iterable of column names present in a DataFrame.

    Returns:
        List of forbidden column names found (empty = no leakage detected).
    """
    col_set = set(df_columns)
    violations = col_set & LEAKAGE_FORBIDDEN
    return sorted(violations)


def get_predictor_columns_only(columns: Iterable[str]) -> List[str]:
    """Filter a column list to only allowed predictors.

    Drops labels, metadata, and leakage-forbidden columns.

    Args:
        columns: Iterable of column name strings.

    Returns:
        Sorted list of columns that are in ALLOWED_PREDICTORS.
    """
    col_set = set(columns)
    return sorted(col_set & ALLOWED_PREDICTORS)
