"""
test_features.py  (v2 – extended)
Tests for the enhanced domain-adaptive feature extractor.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml.features import extract_features


# ── Basic sanity ────────────────────────────────────────────────

def test_extract_features_empty():
    df = pd.DataFrame()
    assert not extract_features(df)


def test_extract_features_basic():
    df = pd.DataFrame({
        "A": np.random.normal(0, 1, 200),
        "B": np.random.uniform(0, 5, 200),
        "Cat": np.random.choice(["Yes", "No"], 200),
    })
    feats = extract_features(df)
    assert isinstance(feats, dict)
    for key in ["duplicate_pct", "missing_pct", "mean_skewness", "mean_entropy",
                "outlier_fraction", "mean_permutation_entropy", "is_timeseries"]:
        assert key in feats, f"Missing key: {key}"


def test_extract_features_duplicates():
    df = pd.DataFrame({"A": [1, 1, 1, 2, 2], "B": [1, 1, 1, 3, 3]})
    feats = extract_features(df)
    # 3 of 5 rows are duplicates
    assert feats["duplicate_pct"] == pytest.approx(0.6, abs=0.01)


def test_missing_values():
    df = pd.DataFrame({"A": [1, np.nan, 3, np.nan], "B": [1, 2, np.nan, 4]})
    feats = extract_features(df)
    assert feats["missing_pct"] == pytest.approx(3 / 8, abs=0.01)
    assert feats["missing_variance"] > 0


def test_string_sanity():
    df = pd.DataFrame({"Unnamed: 0": [1, 2], "col_55": [3, 4], "Valid": ["a", "b"]})
    feats = extract_features(df)
    assert feats["col_sanity_score"] == pytest.approx(2 / 3, abs=0.01)


# ── New features ────────────────────────────────────────────────

def test_permutation_entropy_present():
    df = pd.DataFrame({"x": np.random.normal(0, 1, 200)})
    feats = extract_features(df)
    assert "mean_permutation_entropy" in feats
    # Should be between 0 and 1
    assert 0.0 <= feats["mean_permutation_entropy"] <= 1.0


def test_near_duplicate_ratio():
    # Perfectly identical rows → near_duplicate_ratio should be high
    df = pd.DataFrame({"A": [1.0] * 100, "B": [2.0] * 100})
    feats = extract_features(df)
    assert feats["near_duplicate_ratio"] > 0.5


def test_timeseries_detection_regular_tabular():
    df = pd.DataFrame({
        "age":    np.random.randint(18, 80, 300),
        "income": np.random.normal(50000, 15000, 300),
        "score":  np.random.uniform(0, 100, 300),
    })
    feats = extract_features(df)
    # Should NOT detect as time-series
    assert feats.get("is_timeseries", 0) == 0.0


def test_timeseries_detection_with_timestamp():
    import pandas as pd
    ts = pd.date_range("2023-01-01", periods=500, freq="1H")
    # AR(1) strongly autocorrelated signal
    x = np.cumsum(np.random.normal(0, 1, 500)) + 20.0
    df = pd.DataFrame({"timestamp": ts, "sensor": x})
    feats = extract_features(df)
    # Should detect as time-series
    assert feats.get("is_timeseries", 0) == 1.0
    assert feats.get("mean_autocorr_lag1", 0) > 0.3


def test_benford_bypassed_for_sensor():
    """Bounded sensor columns should bypass Benford's."""
    ts = pd.date_range("2023-01-01", periods=300, freq="5min")
    temp = np.random.normal(25, 2, 300)   # bounded ~23–27 °C
    df   = pd.DataFrame({"timestamp": ts, "temperature": temp})
    feats = extract_features(df)
    ctx   = feats.get("context_flags", {})
    # Should not apply Benford (sensor_iot detected → bypassed)
    assert ctx.get("benford_bypassed", False) == True


def test_all_features_are_floats():
    """No feature in the model input should be NaN or Inf."""
    df = pd.DataFrame({
        "A": np.random.normal(0, 1, 150),
        "B": np.random.exponential(2, 150),
    })
    feats = extract_features(df)
    for k, v in feats.items():
        if k == "context_flags":
            continue
        assert isinstance(v, (int, float)), f"{k} is not numeric"
        import math
        assert not math.isnan(float(v)), f"{k} is NaN"
        assert not math.isinf(float(v)), f"{k} is Inf"
