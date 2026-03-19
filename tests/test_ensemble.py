"""
test_ensemble.py
Tests for the stacked KaggleEnsemble model.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml.ensemble import KaggleEnsemble
from src.ml.features import extract_features


TEST_MODEL = "models/_test_ensemble_tmp.pkl"


@pytest.fixture(scope="module")
def trained_ensemble():
    """Train a minimal ensemble for testing; cleaned up after."""
    rng = np.random.default_rng(42)

    # Minimal feature vector compatible with extract_features() output keys
    real_samples, fake_samples = [], []

    for _ in range(80):
        ts = pd.date_range("2022-01-01", periods=200, freq="1H")
        phi = float(rng.uniform(0.5, 0.95))
        x = np.zeros(200); x[0] = 25.0
        for t in range(1, 200):
            x[t] = phi * x[t-1] + rng.normal(0, 1)
        df = pd.DataFrame({"timestamp": ts, "sensor": x})
        f  = extract_features(df)
        if f:
            real_samples.append(f)

    for _ in range(80):
        df_fake = pd.DataFrame({
            "A": rng.uniform(0, 10, 200),
            "B": rng.uniform(0, 10, 200),
            "C": rng.uniform(0, 10, 200),
        })
        f = extract_features(df_fake)
        if f:
            fake_samples.append(f)

    X_df = pd.DataFrame(real_samples + fake_samples).fillna(0)
    for col in ["context_flags", "dataset_type"]:
        if col in X_df.columns:
            X_df.drop(columns=[col], inplace=True)
    y = pd.Series([1] * len(real_samples) + [0] * len(fake_samples))

    ens = KaggleEnsemble(TEST_MODEL)
    ens.train(X_df, y, report_path="models/_test_report.json")

    yield ens

    # Cleanup
    for f_path in [TEST_MODEL, "models/_test_report.json"]:
        if os.path.exists(f_path):
            os.remove(f_path)


def test_ensemble_is_trained(trained_ensemble):
    assert trained_ensemble.is_trained is True


def test_ensemble_predict_range(trained_ensemble):
    rng = np.random.default_rng(99)
    ts  = pd.date_range("2023-01-01", periods=300, freq="5min")
    phi = 0.85
    x   = np.zeros(300); x[0] = 20.0
    for t in range(1, 300):
        x[t] = phi * x[t-1] + rng.normal(0, 0.8)
    df   = pd.DataFrame({"timestamp": ts, "temperature": x})
    f    = extract_features(df)
    feats = {k: v for k, v in f.items() if k != "context_flags" and isinstance(v, (int, float))}
    prob = trained_ensemble.predict(feats)
    assert 0.0 <= prob <= 1.0


def test_real_sensor_scores_higher(trained_ensemble):
    """A real AR(1) sensor dataset should score higher than uniform random."""
    rng = np.random.default_rng(42)

    # Real sensor
    ts  = pd.date_range("2023-01-01", periods=400, freq="1H")
    phi = 0.9
    x   = np.zeros(400); x[0] = 50.0
    for t in range(1, 400):
        x[t] = phi * x[t-1] + rng.normal(0, 1.5)
    df_real = pd.DataFrame({"timestamp": ts, "sensor": x})
    f_real  = extract_features(df_real)
    feats_r = {k: v for k, v in f_real.items() if k != "context_flags" and isinstance(v, (int, float))}
    p_real  = trained_ensemble.predict(feats_r)

    # Fake uniform data
    df_fake = pd.DataFrame({"A": rng.uniform(0, 100, 400), "B": rng.uniform(0, 100, 400)})
    f_fake  = extract_features(df_fake)
    feats_f = {k: v for k, v in f_fake.items() if k != "context_flags" and isinstance(v, (int, float))}
    p_fake  = trained_ensemble.predict(feats_f)

    assert p_real > p_fake, f"Real={p_real:.3f} should be > Fake={p_fake:.3f}"


def test_confidence_interval_valid(trained_ensemble):
    rng  = np.random.default_rng(7)
    df   = pd.DataFrame({"A": rng.normal(0, 1, 100), "B": rng.uniform(0, 5, 100)})
    f    = extract_features(df)
    feats = {k: v for k, v in f.items() if k != "context_flags" and isinstance(v, (int, float))}
    trained_ensemble.predict(feats)
    lo, hi = trained_ensemble.confidence_interval()
    assert lo is not None and hi is not None
    assert 0.0 <= lo <= hi <= 1.0
