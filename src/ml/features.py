"""
features.py  (v2 – domain-adaptive)
─────────────────────────────────────────────────────────────────
Extracts a comprehensive, adaptive feature vector from any tabular
DataFrame.  Automatically routes to time-series / sensor-IoT feature
paths when the data contains timestamps or strongly autocorrelated
sequential signals, and falls back to standard tabular stats otherwise.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import re
import math
from collections import Counter
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# Time-Series & Sensor features (graceful import)
# ──────────────────────────────────────────────────────────────────
try:
    from ml.timeseries_detector import detect_dataset_type, extract_timeseries_features
    _TS_AVAILABLE = True
except ImportError:
    try:
        from src.ml.timeseries_detector import detect_dataset_type, extract_timeseries_features
        _TS_AVAILABLE = True
    except ImportError:
        _TS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────

def _safe(v, default=0.0):
    """Return `default` if v is NaN / Inf / None."""
    try:
        if v is None or math.isnan(float(v)) or math.isinf(float(v)):
            return default
        return float(v)
    except Exception:
        return default


def _shannon_entropy(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True)
    return float(-np.sum(counts * np.log2(counts + 1e-9)))


def _benford_mae(series: pd.Series) -> float:
    """Mean Absolute Error vs Benford's predicted first-digit distribution."""
    str_vals = series.dropna().astype(str)
    first_digits = str_vals.str.extract(r'([1-9])')[0].dropna().astype(int)
    if len(first_digits) < 20:
        return float('nan')
    counts = first_digits.value_counts(normalize=True).sort_index()
    actual = np.zeros(9)
    for idx in counts.index:
        actual[idx - 1] = counts[idx]
    expected = np.log10(1 + 1 / np.arange(1, 10))
    return float(np.mean(np.abs(actual - expected)))


def _near_duplicate_ratio(df: pd.DataFrame, sample: int = 2000) -> float:
    """
    Hash-based near-duplicate detection: round numeric columns to 2 sig-figs,
    then count duplicates in the simplified hash space.
    """
    try:
        n = min(len(df), sample)
        sub = df.sample(n=n, random_state=42) if len(df) > sample else df.copy()
        num_cols = sub.select_dtypes(include=[np.number]).columns
        simplified = sub.copy()
        for c in num_cols:
            col_data = simplified[c].dropna()
            if len(col_data) > 0 and col_data.std() > 1e-10:
                # Round to 2 decimal places of normalized values
                mn, mx = col_data.min(), col_data.max()
                simplified[c] = ((simplified[c] - mn) / (mx - mn + 1e-10) * 50).round(0)
        return float(simplified.duplicated().mean())
    except Exception:
        return 0.0


def _permutation_entropy_fast(series: pd.Series, order: int = 3) -> float:
    """Fast permutation entropy (lower → more regular / synthetic)."""
    try:
        x = series.dropna().values[:2000]  # cap for speed
        n = len(x)
        if n < order * 2:
            return 0.5
        patterns = []
        for i in range(n - order + 1):
            patterns.append(tuple(np.argsort(x[i:i + order])))
        c = Counter(patterns)
        total = sum(c.values())
        probs = np.array([v / total for v in c.values()])
        pe = -np.sum(probs * np.log2(probs + 1e-12))
        max_pe = math.log2(math.factorial(order))
        return float(pe / max_pe) if max_pe > 0 else 0.5
    except Exception:
        return 0.5


# ──────────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTOR
# ──────────────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> dict:
    """
    Returns a flat dict of scalar features + a 'context_flags' sub-dict
    (the sub-dict is used in the UI and stripped before model input).
    """
    if df is None or df.empty:
        return {}

    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(exclude=[np.number])
    n_rows, n_cols = df.shape

    if n_rows == 0 or n_cols == 0:
        return {}

    features: dict = {}
    context_flags: dict = {}

    # ── 1. Domain / type detection ─────────────────────────────────
    type_info = {"is_timeseries": False, "dataset_type": "tabular", "ts_column": None}
    if _TS_AVAILABLE:
        try:
            type_info = detect_dataset_type(df)
        except Exception:
            pass

    context_flags["dataset_type"]  = type_info.get("dataset_type", "tabular")
    context_flags["is_timeseries"] = type_info.get("is_timeseries", False)
    context_flags["ts_column"]     = type_info.get("ts_column", None)

    # ── 2. Exact duplicates ─────────────────────────────────────────
    raw_dup_pct = float(df.duplicated().mean())

    # ── 3. Near-duplicates ─────────────────────────────────────────
    features["near_duplicate_ratio"] = _near_duplicate_ratio(df)

    # ── 4. Missing ─────────────────────────────────────────────────
    features["missing_pct"]       = _safe(df.isna().mean().mean())
    row_miss = df.isna().sum(axis=1) / n_cols
    features["missing_variance"]  = _safe(row_miss.var() if n_rows > 1 else 0.0)

    # ── 5. Column sanity ────────────────────────────────────────────
    suspicious_cols = sum(
        1 for c in df.columns
        if re.search(r'(unnamed|col_\d+|var_\d+)', str(c).lower())
    )
    features["col_sanity_score"] = suspicious_cols / n_cols if n_cols else 0.0

    # ── 6. Cardinality ─────────────────────────────────────────────
    card_pcts = [df[c].nunique() / n_rows for c in df.columns]
    features["mean_cardinality"] = _safe(np.mean(card_pcts))

    # ── 7. Rounded numbers ratio ────────────────────────────────────
    total_nums = 0
    rounded_nums = 0
    narrow_range_cols = 0

    if not num_df.empty:
        for c in num_df.columns:
            vals = num_df[c].dropna().astype(float)
            if len(vals) == 0:
                continue
            if vals.max() - vals.min() < 50:
                narrow_range_cols += 1
            total_nums += len(vals)
            rounded_nums += int(np.sum((vals % 1 == 0)))

        raw_rounded = rounded_nums / total_nums if total_nums else 0.0

        if not num_df.empty and narrow_range_cols / len(num_df.columns) > 0.5:
            features["rounded_num_ratio"] = raw_rounded * 0.5
            context_flags["narrow_numeric_range"] = True
        else:
            features["rounded_num_ratio"] = raw_rounded
            context_flags["narrow_numeric_range"] = False
    else:
        features["rounded_num_ratio"] = 0.0
        context_flags["narrow_numeric_range"] = False

    # ── 8. Integer column ratio ─────────────────────────────────────
    if not num_df.empty:
        is_int = [1 if pd.api.types.is_integer_dtype(num_df[c]) else 0
                  for c in num_df.columns]
        features["integer_col_ratio"] = float(np.mean(is_int))
    else:
        features["integer_col_ratio"] = 0.0

    # ── 9. Distribution shape: skewness, kurtosis ──────────────────
    skew_vals, kurt_vals, ks_vals = [], [], []
    if not num_df.empty:
        for c in num_df.columns:
            vals = num_df[c].dropna()
            if len(vals) > 3:
                skew_vals.append(vals.skew())
                kurt_vals.append(vals.kurtosis())
                scaled = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
                d, _ = stats.kstest(scaled, "uniform")
                ks_vals.append(d)

    features["mean_skewness"]     = _safe(np.nanmean(skew_vals) if skew_vals else 0.0)
    features["mean_kurtosis"]     = _safe(np.nanmean(kurt_vals) if kurt_vals else 0.0)
    features["uniform_ks_stat"]   = _safe(np.nanmean(ks_vals)  if ks_vals  else 0.0)

    # ── 10. Shapiro-Wilk normality (sample) ────────────────────────
    sw_pvals = []
    if not num_df.empty:
        for c in num_df.columns[:6]:
            s = num_df[c].dropna()
            samp = s.sample(min(len(s), 300), random_state=42) if len(s) > 300 else s
            if len(samp) >= 8:
                try:
                    _, pval = stats.shapiro(samp)
                    sw_pvals.append(pval)
                except Exception:
                    pass
    features["mean_shapiro_pval"] = _safe(np.nanmean(sw_pvals) if sw_pvals else 0.5)

    # ── 11. Correlation structure ─────────────────────────────────
    if num_df.shape[1] > 1:
        corr = num_df.corr().abs().values
        upper = corr[np.triu_indices_from(corr, k=1)]
        features["mean_abs_correlation"] = _safe(np.nanmean(upper) if len(upper) else 0.0)
    else:
        features["mean_abs_correlation"] = 0.0

    # ── 12. Benford's Law (smart – skip for sensors/bounded cols) ───
    is_sensor = type_info.get("dataset_type") == "sensor_iot"
    benford_maes = []

    if not num_df.empty and not is_sensor:
        for c in num_df.columns:
            col_data = num_df[c].dropna()
            col_range = col_data.max() - col_data.min() if len(col_data) > 0 else 0
            if col_data.max() > 100 and col_range > 100 and col_data.var() > 10:
                mae = _benford_mae(col_data)
                if not math.isnan(mae):
                    benford_maes.append(mae)

    features["benfords_law_mae"]         = _safe(np.mean(benford_maes) if benford_maes else 0.0)
    context_flags["benford_bypassed"]    = len(benford_maes) == 0
    features["benford_applicable_cols"]  = len(benford_maes)

    # ── 13. Shannon Entropy ─────────────────────────────────────────
    entropies = [_shannon_entropy(df[c]) for c in df.columns]
    features["mean_entropy"] = _safe(np.mean(entropies) if entropies else 0.0)

    # ── 14. Permutation Entropy (regularity test) ──────────────────
    pe_vals = []
    if not num_df.empty:
        for c in list(num_df.columns[:6]):
            s = num_df[c].dropna()
            if len(s) >= 8:
                pe_vals.append(_permutation_entropy_fast(s))
    features["mean_permutation_entropy"] = _safe(np.mean(pe_vals) if pe_vals else 0.5)

    # ── 15. Exact duplicates (context-aware) ─────────────────────
    if raw_dup_pct > 0.10 and features["mean_entropy"] > 3.5:
        features["duplicate_pct"]              = raw_dup_pct * 0.3
        context_flags["clustered_observations"] = True
    else:
        features["duplicate_pct"]              = raw_dup_pct
        context_flags["clustered_observations"] = False

    # ── 16. Outlier fraction (Isolation Forest) ───────────────────
    if not num_df.empty and len(num_df) > 10:
        num_filled   = num_df.fillna(num_df.mean())
        sample_size  = min(len(num_filled), 1000)
        sample       = num_filled.sample(n=sample_size, random_state=42)
        iso          = IsolationForest(contamination=0.05, random_state=42)
        preds        = iso.fit_predict(sample)
        features["outlier_fraction"] = _safe(float(np.mean(preds == -1)))
    else:
        features["outlier_fraction"] = 0.0

    # ── 17. Text / string variance ────────────────────────────────
    str_vars = []
    if not cat_df.empty:
        for c in cat_df.columns:
            lens = cat_df[c].dropna().astype(str).str.len()
            if len(lens) > 0:
                str_vars.append(float(lens.var()))
    features["mean_string_len_variance"] = _safe(np.nanmean(str_vars) if str_vars else 0.0)

    # ── 18. Near-duplicate ratio (already set above in step 3) ────
    # (already in features dict)

    # ── 19. Value range diversity ─────────────────────────────────
    if not num_df.empty:
        range_ratios = []
        for c in num_df.columns:
            s = num_df[c].dropna()
            if len(s) > 1:
                iqr = float(s.quantile(0.75) - s.quantile(0.25))
                rng = float(s.max() - s.min()) + 1e-10
                range_ratios.append(iqr / rng)
        features["mean_iqr_range_ratio"] = _safe(np.mean(range_ratios) if range_ratios else 0.5)
    else:
        features["mean_iqr_range_ratio"] = 0.5

    # ── 20. Time-series / sensor specific features ─────────────────
    if _TS_AVAILABLE and type_info.get("is_timeseries", False):
        try:
            ts_feats = extract_timeseries_features(df, type_info)
            features.update(ts_feats)
        except Exception as e:
            # Fall back to neutral values
            for key in [
                "is_timeseries", "is_sensor_iot", "adf_p_value", "kpss_p_value",
                "mean_autocorr_lag1", "mean_autocorr_lag5", "seasonality_strength",
                "trend_slope", "residual_variance", "sequence_smoothness",
                "spike_fraction", "noise_level_estimate", "value_jump_freq",
                "higuchi_fd",
            ]:
                features.setdefault(key, 0.5)
    else:
        # Non-time-series: fill TS features with neutral defaults
        features.setdefault("is_timeseries",      0.0)
        features.setdefault("is_sensor_iot",      0.0)
        features.setdefault("adf_p_value",        0.5)
        features.setdefault("kpss_p_value",       0.05)
        features.setdefault("mean_autocorr_lag1", 0.0)
        features.setdefault("mean_autocorr_lag5", 0.0)
        features.setdefault("seasonality_strength", 0.0)
        features.setdefault("trend_slope",        0.0)
        features.setdefault("residual_variance",  0.0)
        features.setdefault("sequence_smoothness", 0.5)
        features.setdefault("spike_fraction",     0.0)
        features.setdefault("noise_level_estimate", 0.0)
        features.setdefault("value_jump_freq",    0.0)
        features.setdefault("higuchi_fd",         1.5)

    # ── Global NaN / Inf sanitisation ──────────────────────────────
    for k in list(features.keys()):
        v = features[k]
        if isinstance(v, (int, float)):
            features[k] = _safe(v)

    features["context_flags"] = context_flags
    return features


# ──────────────────────────────────────────────────────────────────
# UI HELPER – Benford's visualisation
# ──────────────────────────────────────────────────────────────────

def evaluate_benfords_law(series: pd.Series):
    """Returns (digits, actual, expected) distribution for UI charts."""
    str_vals   = series.dropna().astype(str)
    first_digs = str_vals.str.extract(r'([1-9])')[0].dropna().astype(int)
    actual     = np.zeros(9)
    if len(first_digs) > 0:
        counts = first_digs.value_counts(normalize=True).sort_index()
        for idx in counts.index:
            actual[idx - 1] = counts[idx]
    expected = np.log10(1 + 1 / np.arange(1, 10))
    return list(range(1, 10)), actual, expected
