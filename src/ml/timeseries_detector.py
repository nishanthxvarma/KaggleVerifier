"""
timeseries_detector.py
─────────────────────
Detects whether a DataFrame is time-series / sensor / IoT style data
and extracts a rich set of temporal features that are invisible to
standard statistical tests.

Returns a dict of features that is merged into the main feature vector
before model prediction.
"""
import warnings
import math
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from collections import Counter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 1. TIMESTAMP COLUMN DETECTION
# ─────────────────────────────────────────────────────────────────
_TS_KEYWORDS = {
    "time", "timestamp", "date", "datetime", "ts", "recorded_at",
    "created_at", "updated_at", "event_time", "log_time",
    "datetime_utc", "epoch", "measured_at", "sample_time"
}

def _find_timestamp_column(df: pd.DataFrame):
    """Return the first timestamp-like column name, or None."""
    # Check by dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # Check by name
    for col in df.columns:
        if str(col).lower().strip() in _TS_KEYWORDS:
            # Try to parse it
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                return col
            except Exception:
                pass
        # Partial keyword match
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in _TS_KEYWORDS):
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                return col
            except Exception:
                pass
    return None


def _is_monotone_index(series: pd.Series) -> bool:
    """True if series is monotonically increasing (sorted timestamps)."""
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True)
        return bool(parsed.is_monotonic_increasing)
    except Exception:
        return False


def detect_dataset_type(df: pd.DataFrame) -> dict:
    """
    Returns:
        {
          'is_timeseries': bool,
          'dataset_type': 'sensor_iot' | 'timeseries' | 'tabular',
          'ts_column': str | None,
        }
    """
    ts_col = _find_timestamp_column(df)
    num_df = df.select_dtypes(include=[np.number])

    if ts_col is not None:
        # Check monotone → ordered sensor/log
        is_mono = _is_monotone_index(df[ts_col])
        # High lag-1 autocorrelation on numeric columns → sensor readings
        if not num_df.empty:
            autocorrs = []
            for col in num_df.columns[:8]:  # sample up to 8 cols
                s = num_df[col].dropna()
                if len(s) > 20:
                    try:
                        autocorrs.append(abs(s.autocorr(lag=1)))
                    except Exception:
                        pass
            mean_ac = np.nanmean(autocorrs) if autocorrs else 0.0
        else:
            mean_ac = 0.0

        if is_mono and mean_ac > 0.3:
            dtype = "sensor_iot"
        else:
            dtype = "timeseries"
        return {"is_timeseries": True, "dataset_type": dtype, "ts_column": ts_col}

    # No timestamp: check if numeric index-like column exists with high autocorrelation
    if not num_df.empty:
        autocorrs = []
        for col in num_df.columns[:8]:
            s = num_df[col].dropna()
            if len(s) > 20:
                try:
                    autocorrs.append(abs(s.autocorr(lag=1)))
                except Exception:
                    pass
        mean_ac = np.nanmean(autocorrs) if autocorrs else 0.0
        if mean_ac > 0.6:  # very strong autocorrelation even without timestamp
            return {"is_timeseries": True, "dataset_type": "timeseries", "ts_column": None}

    return {"is_timeseries": False, "dataset_type": "tabular", "ts_column": None}


# ─────────────────────────────────────────────────────────────────
# 2. TIME-SERIES FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────

def _safe_adf(series: pd.Series):
    """ADF stationarity test. Returns p-value (lower → more stationary)."""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna(), autolag="AIC", maxlag=20)
        return float(result[1])  # p-value
    except Exception:
        return 0.5  # neutral


def _safe_kpss(series: pd.Series):
    """KPSS stationarity test. Returns p-value (higher → more stationary)."""
    try:
        from statsmodels.tsa.stattools import kpss
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(series.dropna(), regression="c", nlags="auto")
        return float(result[1])  # p-value
    except Exception:
        return 0.05  # neutral


def _autocorr_at_lag(series: pd.Series, lag: int) -> float:
    try:
        return float(series.autocorr(lag=lag))
    except Exception:
        return 0.0


def _stl_seasonality_strength(series: pd.Series) -> tuple:
    """
    Estimates seasonality strength and residual variance using STL.
    Returns (seasonality_strength, residual_variance, trend_slope).
    """
    try:
        from statsmodels.tsa.seasonal import STL
        s = series.dropna()
        if len(s) < 48:
            period = max(2, len(s) // 4)
        else:
            period = 24  # assume hourly → daily cycle

        stl = STL(s, period=period, robust=True)
        res = stl.fit()

        seasonal_var = float(np.var(res.seasonal))
        resid_var = float(np.var(res.resid))
        trend_vals = res.trend
        # Linear trend slope (normalized)
        x = np.arange(len(trend_vals))
        slope, *_ = np.polyfit(x, trend_vals, 1)
        total_var = float(np.var(s)) + 1e-10
        seasonality_strength = seasonal_var / total_var

        return seasonality_strength, resid_var, float(slope)
    except Exception:
        return 0.0, 0.0, 0.0


def _permutation_entropy(series: pd.Series, order: int = 3, delay: int = 1) -> float:
    """
    Permutation entropy – low for overly regular (synthetic) sequences,
    high (closer to 1) for natural chaotic/noisy real signals.
    """
    try:
        x = series.dropna().values
        n = len(x)
        if n < order * 2:
            return 0.5
        patterns = []
        for i in range(n - (order - 1) * delay):
        # Extract embedding
            idx = [i + j * delay for j in range(order)]
            embedded = x[idx]
            patterns.append(tuple(np.argsort(embedded)))
        c = Counter(patterns)
        total = sum(c.values())
        probs = np.array([v / total for v in c.values()])
        pe = -np.sum(probs * np.log2(probs + 1e-12))
        max_pe = math.log2(math.factorial(order))
        return float(pe / max_pe) if max_pe > 0 else 0.5
    except Exception:
        return 0.5


def _higuchi_fd(series: pd.Series, k_max: int = 10) -> float:
    """
    Higuchi Fractal Dimension – real physiological/sensor signals
    typically have FD in [1.5, 1.9]; perfect synthetic signals cluster near 1.0.
    """
    try:
        x = np.array(series.dropna().values, dtype=float)
        n = len(x)
        if n < k_max * 2:
            return 1.5  # neutral
        lk = []
        for k in range(1, k_max + 1):
            lengths = []
            for m in range(1, k + 1):
                indices = np.arange(m - 1, n - k, k)
                if len(indices) < 2:
                    continue
                L_mk = np.mean(np.abs(np.diff(x[indices]))) * ((n - 1) / (len(indices) * k))
                lengths.append(L_mk)
            if lengths:
                lk.append((np.log(k), np.log(np.mean(lengths) + 1e-12)))
        if len(lk) < 3:
            return 1.5
        x_vals, y_vals = zip(*lk)
        slope, _ = np.polyfit(x_vals, y_vals, 1)
        return float(abs(slope))
    except Exception:
        return 1.5


def _sequence_smoothness(series: pd.Series) -> float:
    """
    Mean absolute first-difference normalized by std.
    Real sensor data is moderately smooth; synthetic data can be too jagged or too flat.
    """
    try:
        s = series.dropna().values.astype(float)
        if len(s) < 3:
            return 0.5
        diffs = np.abs(np.diff(s))
        std = np.std(s) + 1e-10
        return float(np.mean(diffs) / std)
    except Exception:
        return 0.5


def _spike_fraction(series: pd.Series, threshold: float = 3.0) -> float:
    """Fraction of values more than `threshold` σ away from rolling median."""
    try:
        s = series.dropna()
        if len(s) < 5:
            return 0.0
        rolling_med = s.rolling(window=5, center=True, min_periods=1).median()
        residuals = (s - rolling_med).abs()
        mad = residuals.median() + 1e-10
        spikes = (residuals / mad) > (threshold * 1.4826)
        return float(spikes.mean())
    except Exception:
        return 0.0


def _noise_level_estimate(series: pd.Series) -> float:
    """
    Noise level as MAD / (IQR + ε).
    Real sensor noise is consistent; synthetic data often has 0 or extreme noise.
    """
    try:
        s = series.dropna().values.astype(float)
        if len(s) < 4:
            return 0.0
        med = np.median(s)
        mad = np.median(np.abs(s - med))
        q75, q25 = np.percentile(s, [75, 25])
        iqr = q75 - q25 + 1e-10
        return float(mad / iqr)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────
# 3. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def extract_timeseries_features(df: pd.DataFrame, type_info: dict) -> dict:
    """
    Given a DataFrame and its type_info from detect_dataset_type(),
    extract all temporal + statistical features.

    Returns a dict of scalar features (no nested objects).
    """
    feats: dict = {}
    num_df = df.select_dtypes(include=[np.number])

    # Flag features
    feats["is_timeseries"] = 1.0 if type_info.get("is_timeseries", False) else 0.0
    feats["is_sensor_iot"] = 1.0 if type_info.get("dataset_type") == "sensor_iot" else 0.0

    if num_df.empty or len(num_df) < 5:
        # Pad with neutral values
        for key in [
            "adf_p_value", "kpss_p_value", "mean_autocorr_lag1",
            "seasonality_strength", "trend_slope", "residual_variance",
            "permutation_entropy", "higuchi_fd", "sequence_smoothness",
            "spike_fraction", "noise_level_estimate", "value_jump_freq",
            "mean_autocorr_lag5",
        ]:
            feats[key] = 0.5
        return feats

    # Sample up to 8 columns for expensive computations
    cols = list(num_df.columns[:8])

    adf_vals, kpss_vals = [], []
    autocorr1_vals, autocorr5_vals = [], []
    smooth_vals, spike_vals, noise_vals = [], [], []
    perm_ent_vals = []
    fd_vals = []
    jump_vals = []

    for col in cols:
        s = num_df[col].dropna()
        if len(s) < 8:
            continue

        adf_vals.append(_safe_adf(s))
        kpss_vals.append(_safe_kpss(s))
        autocorr1_vals.append(_autocorr_at_lag(s, 1))
        autocorr5_vals.append(_autocorr_at_lag(s, min(5, len(s) // 3)))
        smooth_vals.append(_sequence_smoothness(s))
        spike_vals.append(_spike_fraction(s))
        noise_vals.append(_noise_level_estimate(s))
        perm_ent_vals.append(_permutation_entropy(s))
        fd_vals.append(_higuchi_fd(s))

        # Value jump frequency: fraction of rows where value jumps > 3σ from previous
        diffs = s.diff().abs()
        sigma = s.std() + 1e-10
        jump_vals.append(float((diffs > 3 * sigma).mean()))

    def safe_mean(lst, default=0.5):
        return float(np.nanmean(lst)) if lst else default

    feats["adf_p_value"] = safe_mean(adf_vals)
    feats["kpss_p_value"] = safe_mean(kpss_vals)
    feats["mean_autocorr_lag1"] = safe_mean(autocorr1_vals, 0.0)
    feats["mean_autocorr_lag5"] = safe_mean(autocorr5_vals, 0.0)
    feats["sequence_smoothness"] = safe_mean(smooth_vals)
    feats["spike_fraction"] = safe_mean(spike_vals, 0.0)
    feats["noise_level_estimate"] = safe_mean(noise_vals, 0.0)
    feats["permutation_entropy"] = safe_mean(perm_ent_vals)
    feats["higuchi_fd"] = safe_mean(fd_vals, 1.5)
    feats["value_jump_freq"] = safe_mean(jump_vals, 0.0)

    # STL on first usable column (most expensive – only one column)
    stl_done = False
    for col in cols:
        s = num_df[col].dropna()
        if len(s) >= 20:
            seas_str, resid_var, trend_slope = _stl_seasonality_strength(s)
            feats["seasonality_strength"] = float(np.clip(seas_str, 0.0, 1.0))
            feats["residual_variance"] = float(np.clip(resid_var, 0, 1e6))
            feats["trend_slope"] = float(np.clip(trend_slope, -1e6, 1e6))
            stl_done = True
            break

    if not stl_done:
        feats["seasonality_strength"] = 0.0
        feats["residual_variance"] = 0.0
        feats["trend_slope"] = 0.0

    # Sanitize all NaNs
    for k, v in feats.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            feats[k] = 0.5

    return feats
