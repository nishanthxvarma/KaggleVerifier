"""
test_calibration.py
Tests for the pipeline's domain-adaptive calibration rules.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.pipeline import _calibrate


def _base_feats(**overrides):
    """Build a minimal feature dict suitable for _calibrate()."""
    defaults = {
        "context_flags": {
            "dataset_type":             "tabular",
            "is_timeseries":            False,
            "clustered_observations":   False,
            "narrow_numeric_range":     False,
        },
        "noise_level_estimate":     0.0,
        "adf_p_value":              0.5,
        "kpss_p_value":             0.05,
        "mean_autocorr_lag1":       0.0,
        "seasonality_strength":     0.0,
        "residual_variance":        0.0,
        "spike_fraction":           0.0,
        "missing_pct":              0.0,
        "missing_variance":         0.0,
        "mean_permutation_entropy": 0.75,
        "duplicate_pct":            0.05,
        "mean_entropy":             4.0,
        "uniform_ks_stat":          0.3,
        "near_duplicate_ratio":     0.1,
    }
    defaults.update(overrides)
    return defaults


# ── Boost tests ─────────────────────────────────────────────────

def test_sensor_noise_boost():
    """Sensor data with natural noise should be boosted."""
    feats = _base_feats(
        noise_level_estimate=0.05,
        mean_autocorr_lag1=0.5,
    )
    feats["context_flags"].update({"dataset_type": "sensor_iot", "is_timeseries": True})
    raw = 0.55
    cal, reasons = _calibrate(raw, feats)
    assert cal > raw
    assert any("noise" in r.lower() for r in reasons)


def test_stationary_signal_boost():
    """Stationary ADF (p < 0.05) + time-series → boost."""
    feats = _base_feats(adf_p_value=0.01, mean_autocorr_lag1=0.6)
    feats["context_flags"].update({"is_timeseries": True, "dataset_type": "sensor_iot"})
    raw = 0.55
    cal, reasons = _calibrate(raw, feats)
    assert cal > raw
    assert any("stationary" in r.lower() or "adf" in r.lower() for r in reasons)


def test_autocorr_boost():
    """Strong autocorrelation on time-series data → boost."""
    feats = _base_feats(mean_autocorr_lag1=0.8)
    feats["context_flags"].update({"is_timeseries": True, "dataset_type": "sensor_iot"})
    raw = 0.50
    cal, reasons = _calibrate(raw, feats)
    assert cal > raw


def test_natural_missingness_boost():
    """Sparse natural NaN (1–15%) with variance → small boost."""
    feats = _base_feats(missing_pct=0.07, missing_variance=0.02)
    raw = 0.60
    cal, reasons = _calibrate(raw, feats)
    assert cal > raw
    assert any("missing" in r.lower() for r in reasons)


# ── Penalty tests ────────────────────────────────────────────────

def test_low_permutation_entropy_penalty():
    """Very low permutation entropy → synthetic regularity penalty."""
    feats = _base_feats(mean_permutation_entropy=0.2)
    raw = 0.70
    cal, reasons = _calibrate(raw, feats)
    assert cal < raw
    assert any("permutation entropy" in r.lower() for r in reasons)


def test_near_uniform_distribution_penalty():
    """Near-uniform KS stat on tabular data → penalty."""
    feats = _base_feats(uniform_ks_stat=0.02)
    raw = 0.65
    cal, reasons = _calibrate(raw, feats)
    assert cal < raw


def test_high_near_duplicate_penalty():
    """High near-duplicate ratio → generator artifact penalty."""
    feats = _base_feats(near_duplicate_ratio=0.75)
    raw = 0.65
    cal, reasons = _calibrate(raw, feats)
    assert cal < raw
    assert any("near-duplicate" in r.lower() or "near_duplicate" in r.lower() or "near" in r.lower() for r in reasons)


# ── Floor test ────────────────────────────────────────────────────

def test_sensor_floor_applied():
    """Real autocorrelated sensor data should have a floor."""
    feats = _base_feats(mean_autocorr_lag1=0.85)
    feats["context_flags"].update({"is_timeseries": True, "dataset_type": "sensor_iot"})
    raw = 0.20  # Artificially low
    cal, reasons = _calibrate(raw, feats)
    assert cal >= 0.40, f"Expected floor applied but got {cal}"


# ── Output bounds ─────────────────────────────────────────────────

def test_output_always_between_0_and_1():
    """Calibrated probability must always be in [0, 1]."""
    import random
    random.seed(123)
    for _ in range(50):
        raw = random.uniform(0.0, 1.0)
        feats = _base_feats(
            mean_permutation_entropy=random.uniform(0, 1),
            near_duplicate_ratio=random.uniform(0, 1),
            uniform_ks_stat=random.uniform(0, 0.5),
            missing_pct=random.uniform(0, 0.2),
        )
        cal, _ = _calibrate(raw, feats)
        assert 0.0 <= cal <= 1.0, f"Out of bounds: {cal}"
