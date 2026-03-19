"""
train_real.py  (v2 – domain-adaptive training)
──────────────────────────────────────────────────────────────────
Trains the KaggleVerifier ensemble using:
  Positives (real=1):
    - sklearn real datasets (diabetes, breast_cancer, iris, wine, california)
    - 50 make_classification variants
    - 50 synthetic sensor/IoT-style AR(1) + random-walk DataFrames
    - 30 multi-sensor correlated streams with natural noise + NaN

  Negatives (fake=0):
    - 7 corruption methods on every real dataset
    - CTGAN-style marginal sampling
    - Perfect grid / periodic fakes
    - Block-missing injection
    - Column-shuffle (destroys all correlations)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml.features import extract_features
    from ml.ensemble import KaggleEnsemble
except ImportError:
    from src.ml.features import extract_features
    from src.ml.ensemble import KaggleEnsemble

from sklearn.datasets import (
    load_diabetes, load_breast_cancer, load_iris,
    load_wine, fetch_california_housing, make_classification,
    load_digits, load_linnerud
)


# ──────────────────────────────────────────────────────────────────
# 1. REAL DATA GENERATORS
# ──────────────────────────────────────────────────────────────────

def _get_sklearn_real():
    """Return list of pandas DataFrames from sklearn bundled datasets."""
    dfs = []
    loaders = [
        lambda: load_diabetes(as_frame=True).frame,
        lambda: load_breast_cancer(as_frame=True).frame,
        lambda: load_iris(as_frame=True).frame,
        lambda: load_wine(as_frame=True).frame,
        lambda: load_digits(as_frame=True).frame.sample(500, random_state=42),
        lambda: load_linnerud(as_frame=True).frame,
        lambda: fetch_california_housing(as_frame=True).frame.sample(2000, random_state=42),
    ]
    for loader in loaders:
        try:
            dfs.append(loader())
        except Exception as e:
            print(f"  [warn] sklearn dataset failed: {e}")
    return dfs


def _gen_classification_variants(n=60):
    """60 make_classification variants. Half are messy (noise+NaN), half are clean (UCI-style)."""
    dfs = []
    for i in range(n):
        rng = np.random.default_rng(i * 17 + 3)
        n_samples  = rng.integers(300, 3000)
        n_features = rng.integers(6, 45)
        n_info     = int(rng.integers(3, max(4, n_features - 2)))
        X, y = make_classification(
            n_samples=int(n_samples), n_features=int(n_features),
            n_informative=n_info, random_state=i,
        )
        df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
        df["target"] = y
        
        # Every 2nd variant is "Perfectly Clean" (No noise, No NaN) to simulate UCI
        if i % 2 == 0:
            pass 
        else:
            # Add natural noise
            for c in df.select_dtypes(include=[np.number]).columns:
                df[c] += rng.normal(0, df[c].std() * 0.02, size=len(df))
            # Add 1-5 % NaN naturally
            mask = rng.random(df.shape) < rng.uniform(0.01, 0.05)
            df[mask] = np.nan
        dfs.append(df)
    return dfs


def _gen_sensor_ar1(n=50):
    """
    50 AR(1) sensor streams with timestamps.
    Simulates temperature / pressure / current readings.
    """
    dfs = []
    for i in range(n):
        rng   = np.random.default_rng(i * 31 + 7)
        phi   = float(rng.uniform(0.5, 0.97))     # AR coefficient
        sigma = float(rng.uniform(0.5, 5.0))       # noise std
        n_pts = int(rng.integers(300, 3000))
        bias  = float(rng.uniform(10, 100))
        trend = float(rng.uniform(-0.003, 0.003))

        x = np.zeros(n_pts)
        x[0] = bias
        for t in range(1, n_pts):
            x[t] = phi * x[t - 1] + rng.normal(0, sigma) + trend * t

        # Add seasonal component
        period = int(rng.choice([24, 48, 96, 144, 288]))
        x += bias * 0.05 * np.sin(2 * np.pi * np.arange(n_pts) / period)

        # Natural spikes (1–3 %)
        spike_mask = rng.random(n_pts) < rng.uniform(0.01, 0.03)
        x[spike_mask] += rng.choice([-1, 1], spike_mask.sum()) * sigma * float(rng.uniform(3, 8))

        # Build DataFrame with timestamp
        start = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=int(rng.integers(0, 8760)))
        freq  = rng.choice(["1min", "5min", "15min", "1H", "1D"])
        ts    = pd.date_range(start=start, periods=n_pts, freq=freq)
        df    = pd.DataFrame({"timestamp": ts, "sensor_value": x})

        # Possibly add 2nd correlated channel
        if rng.random() > 0.4:
            noise2 = rng.normal(0, sigma * 0.5, n_pts)
            df["sensor2"] = x * float(rng.uniform(0.7, 1.3)) + noise2

        # Natural missing values: randomly scattered (NOT block-shaped)
        miss_rate = float(rng.uniform(0.01, 0.08))
        for col in ["sensor_value", "sensor2"]:
            if col in df.columns:
                miss_mask = rng.random(n_pts) < miss_rate
                df.loc[miss_mask, col] = np.nan

        dfs.append(df)
    return dfs


def _gen_multi_sensor(n=30):
    """
    30 multi-column correlated sensor streams (e.g. HVAC, factory floor).
    3–8 sensors, each correlated to a hidden latent baseline.
    """
    dfs = []
    for i in range(n):
        rng    = np.random.default_rng(i * 53 + 11)
        n_cols = int(rng.integers(3, 9))
        n_pts  = int(rng.integers(500, 4000))

        # Latent AR(1) process
        phi   = float(rng.uniform(0.6, 0.95))
        sigma = float(rng.uniform(1.0, 10.0))
        latent = np.zeros(n_pts)
        latent[0] = float(rng.uniform(20, 100))
        for t in range(1, n_pts):
            latent[t] = phi * latent[t - 1] + rng.normal(0, sigma)

        cols = {}
        ts   = pd.date_range("2023-06-01", periods=n_pts, freq="5min")
        cols["timestamp"] = ts

        for col_i in range(n_cols):
            coeff = float(rng.uniform(0.5, 2.0))
            noise = float(rng.uniform(0.1, 2.0))
            signal = latent * coeff + rng.normal(0, noise, n_pts)
            miss   = rng.random(n_pts) < float(rng.uniform(0.01, 0.06))
            signal[miss] = np.nan
            cols[f"sensor_{col_i}"] = signal

        dfs.append(pd.DataFrame(cols))
    return dfs


def get_real_datasets():
    print("Loading real datasets...")
    dfs = []
    dfs += _get_sklearn_real()
    dfs += _gen_classification_variants(50)
    dfs += _gen_sensor_ar1(50)
    dfs += _gen_multi_sensor(30)

    # Load any existing CSVs from data/real/
    real_dir = "data/real"
    if os.path.isdir(real_dir):
        for fn in os.listdir(real_dir):
            if fn.endswith(".csv"):
                try:
                    extra = pd.read_csv(os.path.join(real_dir, fn))
                    if not extra.empty:
                        dfs.append(extra)
                except Exception:
                    pass

    print(f"  → {len(dfs)} real datasets loaded.")
    return dfs


# ──────────────────────────────────────────────────────────────────
# 2. FAKE DATA GENERATORS
# ──────────────────────────────────────────────────────────────────

def corrupt_dataset(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Apply a corruption strategy to simulate synthetic / tampered data."""
    df_f    = df.copy()
    num_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
    rng     = np.random.default_rng(42)

    if method == "uniform":
        for c in num_cols:
            col = df[c].dropna()
            if len(col):
                df_f[c] = rng.uniform(col.min(), col.max(), len(df_f))

    elif method == "normal":
        for c in num_cols:
            col = df[c].dropna()
            if len(col):
                noise = rng.normal(float(col.mean()), float(col.std()) + 1e-6, len(df_f))
                if rng.random() > 0.5:
                    noise = noise ** 2 / (np.abs(noise).max() + 1e-6)
                df_f[c] = noise

    elif method == "linear_combo":
        if len(num_cols) > 2:
            for i, tc in enumerate(num_cols):
                others = [c for c in num_cols if c != tc]
                c1, c2 = rng.choice(others, 2, replace=False)
                df_f[tc] = df[c1] * 0.5 + df[c2] * 0.5 + rng.normal(0, 0.05, len(df_f))

    elif method == "rounded":
        for c in num_cols:
            col = df[c].dropna()
            if len(col):
                scale = 10 ** np.floor(np.log10(np.abs(float(col.mean())) + 1e-5))
                df_f[c] = np.round(df[c] / scale) * scale

    elif method == "shuffled":
        for c in df_f.columns:
            df_f[c] = pd.Series(rng.permutation(df_f[c].values), index=df_f.index)

    elif method == "duplicated":
        drops = df_f.sample(frac=0.4, random_state=42).index
        reps  = df_f.sample(frac=0.4, replace=True, random_state=42)
        df_f  = pd.concat([df_f.drop(drops), reps], ignore_index=True)
        for c in num_cols:
            mask = rng.random(len(df_f)) < 0.15
            df_f.loc[mask, c] = np.nan

    elif method == "outliers":
        for c in num_cols:
            mask = rng.random(len(df_f)) < 0.05
            df_f.loc[mask, c] = (
                float(df_f[c].mean()) +
                10 * float(df_f[c].std()) * float(rng.choice([-1, 1]))
            )

    elif method == "marginal":
        # CTGAN-style: sample each column independently from its empirical distribution
        for c in num_cols:
            col = df[c].dropna().values
            if len(col):
                df_f[c] = rng.choice(col, size=len(df_f), replace=True)
        cat_cols = df_f.select_dtypes(exclude=[np.number]).columns
        for c in cat_cols:
            vals = df[c].dropna().unique()
            if len(vals):
                df_f[c] = rng.choice(vals, size=len(df_f), replace=True)

    elif method == "perfect_grid":
        # Regular grid sampling – common synthetic artifact
        for c in num_cols:
            col = df[c].dropna()
            if len(col) > 1:
                grid = np.linspace(float(col.min()), float(col.max()), len(df_f))
                df_f[c] = grid

    elif method == "block_missing":
        # Block-shaped NaN injection (scraper artifact)
        n = len(df_f)
        if n > 10:
            block_start = int(rng.integers(0, n - 10))
            block_len   = int(rng.integers(max(1, n // 10), max(2, n // 4)))
            block_end   = min(n, block_start + block_len)
            col         = rng.choice(df_f.columns.tolist())
            df_f.loc[df_f.index[block_start:block_end], col] = np.nan

    return df_f


def _augment_real(df: pd.DataFrame, seeds=(42, 73, 101, 2024)) -> list:
    """Create augmented real samples (sub-sampling + tiny noise injection)."""
    out = []
    for seed in seeds:
        sub = df.sample(frac=0.75, random_state=seed)
        num_cols = sub.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            sub[c] = sub[c] + np.random.normal(0, sub[c].std() * 0.01 + 1e-8, len(sub))
        out.append(sub)
    return out


CORRUPTION_METHODS = [
    "uniform", "normal", "rounded", "shuffled", "duplicated",
    "linear_combo", "outliers", "marginal", "perfect_grid", "block_missing",
]


# ──────────────────────────────────────────────────────────────────
# 3. MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────────

def train_robust_model():
    real_dfs = get_real_datasets()
    if not real_dfs:
        print("❌  No real datasets available. Aborting.")
        return

    X_features = []
    y_labels   = []

    # 1. Real Positives
    print(f"  [1/2] Processing {len(real_dfs)} real datasets + augmentations...")
    for df in tqdm(real_dfs, desc="Real"):
        f = extract_features(df)
        if f:
            X_features.append(f)
            y_labels.append(1)

        # Augment real datasets (3 variants each)
        for aug in _augment_real(df, seeds=[42, 73, 101]):
            af = extract_features(aug)
            if af:
                X_features.append(af)
                y_labels.append(1)

    # 2. Fake Negatives (Balance to ~1.5x of real)
    print(f"  [2/2] Generating balanced fake negatives...")
    n_real = len(X_features)
    Target_fakes = int(n_real * 1.5)
    
    # We take the real_dfs and apply a subset of corruptions to keep it balanced
    fakes_count = 0
    for df in tqdm(real_dfs, desc="Fake"):
        # Select 5 random corruption methods per real df
        rng = np.random.default_rng(len(X_features) + fakes_count)
        methods = rng.choice(CORRUPTION_METHODS, size=min(5, len(CORRUPTION_METHODS)), replace=False)
        
        for method in methods:
            try:
                fk = corrupt_dataset(df, method)
                ff = extract_features(fk)
                if ff:
                    X_features.append(ff)
                    y_labels.append(0)
                    fakes_count += 1
                if fakes_count >= Target_fakes:
                    break
            except Exception as e:
                print(f"  [warn] corruption '{method}' failed: {e}")
        if fakes_count >= Target_fakes:
            break

    if not X_features:
        print("❌  Feature extraction returned nothing. Aborting.")
        return

    X_df = pd.DataFrame(X_features).fillna(0)
    # Drop non-numeric meta columns
    for drop_col in ["context_flags", "dataset_type"]:
        if drop_col in X_df.columns:
            X_df = X_df.drop(columns=[drop_col])

    y_ser = pd.Series(y_labels)
    print(f"\nTraining set: {X_df.shape}  |  Real={int((y_ser==1).sum())}  Fake={int((y_ser==0).sum())}")

    ensemble = KaggleEnsemble("models/ensemble_v2.pkl")
    metrics  = ensemble.train(X_df, y_ser)

    # ── Store IsoForest score range (needed for predict) ───────────
    X_real = X_df[y_ser == 1].values
    ensemble._calibrate_iso_range(X_real)
    ensemble.save()  # re-save with iso range

    print("\n🎉  Ensemble v2 trained and saved to models/ensemble_v2.pkl")
    print(f"    OOF AUC: {metrics['oof_auc']}  |  OOF Acc: {metrics['oof_acc']}")
    return metrics


if __name__ == "__main__":
    train_robust_model()
