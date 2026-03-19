"""
simulate_scores.py
# ------------------------------------------------------------------
End-to-end accuracy simulation. Generates 6 test datasets and
asserts each scores within expected thresholds.

Run from project root:
    python tests/simulate_scores.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.pipeline import DetectionPipeline
from sklearn.datasets import load_diabetes

PASS = "[PASS]"
FAIL = "[FAIL]"


def _header(title):
    print(f"\n{'-'*55}")
    print(f"  {title}")
    print('-'*55)


def run_simulation():
    print("\n[Simulation] KaggleVerifier v2 - Score Simulation")
    print("="*55)

    pipeline = DetectionPipeline()
    rng      = np.random.default_rng(2024)
    results  = []

    # -- 1. Noisy AR(1) sensor + timestamp -> expect > 0.65 --------
    _header("Test 1: Noisy AR(1) Sensor with Timestamps")
    ts    = pd.date_range("2023-01-01", periods=1000, freq="5min")
    phi   = 0.88
    x     = np.zeros(1000); x[0] = 25.0
    for t in range(1, 1000):
        x[t] = phi * x[t-1] + rng.normal(0, 1.2)
    x += 3.0 * np.sin(2 * np.pi * np.arange(1000) / 288)   # daily cycle
    spike_mask = rng.random(1000) < 0.02
    x[spike_mask] += rng.choice([-1, 1], spike_mask.sum()) * 5
    miss_mask = rng.random(1000) < 0.04
    x[miss_mask] = np.nan
    df1   = pd.DataFrame({"timestamp": ts, "sensor": x, "sensor2": x * 0.9 + rng.normal(0, 0.5, 1000)})
    prob1, feats1, _ = pipeline.process_file(
        __import__("io").StringIO(df1.to_csv(index=False))
    )
    ok1 = prob1 > 0.65
    print(f"  Score: {prob1*100:.1f}%  (threshold: > 65%)  {PASS if ok1 else FAIL}")
    print(f"  Context: {feats1['context_flags']}")
    results.append(ok1)

    # -- 2. Multi-sensor IoT dataset -> expect > 0.60 -------------
    _header("Test 2: Multi-Sensor IoT (3 correlated channels)")
    latent = np.zeros(800); latent[0] = 50.0
    for t in range(1, 800):
        latent[t] = 0.85 * latent[t-1] + rng.normal(0, 2)
    df2 = pd.DataFrame({
        "timestamp": pd.date_range("2023-06-01", periods=800, freq="1H"),
        "temp":      latent + rng.normal(0, 0.5, 800),
        "humidity":  latent * 0.7 + rng.normal(0, 1.0, 800),
        "pressure":  latent * 1.2 + rng.normal(0, 0.8, 800),
    })
    for col in ["temp","humidity","pressure"]:
        mask = rng.random(800) < 0.03
        df2.loc[mask, col] = np.nan
    prob2, feats2, _ = pipeline.process_file(
        __import__("io").StringIO(df2.to_csv(index=False))
    )
    ok2 = prob2 > 0.60
    print(f"  Score: {prob2*100:.1f}%  (threshold: > 60%)  {PASS if ok2 else FAIL}")
    print(f"  Context: {feats2['context_flags']}")
    results.append(ok2)

    # -- 3. Uniformly sampled fake -> expect < 0.40 ---------------
    _header("Test 3: Pure CTGAN-style Uniform Fake")
    df3 = pd.DataFrame({
        "A": rng.uniform(0, 100, 500),
        "B": rng.uniform(0, 100, 500),
        "C": rng.uniform(0, 100, 500),
        "D": rng.uniform(0, 100, 500),
    })
    prob3, _, _ = pipeline.process_file(
        __import__("io").StringIO(df3.to_csv(index=False))
    )
    ok3 = prob3 < 0.40
    print(f"  Score: {prob3*100:.1f}%  (threshold: < 40%)  {PASS if ok3 else FAIL}")
    results.append(ok3)

    # -- 4. Column-shuffled fake (destroys correlations) -> < 0.45
    _header("Test 4: Column-Shuffled Tabular Fake")
    diabetes  = load_diabetes(as_frame=True).frame
    df4       = diabetes.copy()
    for col in df4.select_dtypes(include=[np.number]).columns:
        df4[col] = rng.permutation(df4[col].values)
    prob4, _, _ = pipeline.process_file(
        __import__("io").StringIO(df4.to_csv(index=False))
    )
    ok4 = prob4 < 0.45
    print(f"  Score: {prob4*100:.1f}%  (threshold: < 45%)  {PASS if ok4 else FAIL}")
    results.append(ok4)

    # -- 5. sklearn Diabetes (real tabular) -> > 0.55 -------------
    _header("Test 5: Real sklearn Diabetes Dataset")
    prob5, feats5, _ = pipeline.process_file(
        __import__("io").StringIO(diabetes.to_csv(index=False))
    )
    ok5 = prob5 > 0.55
    print(f"  Score: {prob5*100:.1f}%  (threshold: > 55%)  {PASS if ok5 else FAIL}")
    print(f"  Context: {feats5['context_flags']}")
    results.append(ok5)

    # -- 6. Perfect grid (synthetic artifact) -> < 0.40 -----------
    _header("Test 6: Perfect Grid Data (Synthetic Artifact)")
    df6 = pd.DataFrame({c: np.linspace(0, 100, 600) for c in ["X","Y","Z","W"]})
    prob6, _, _ = pipeline.process_file(
        __import__("io").StringIO(df6.to_csv(index=False))
    )
    ok6 = prob6 < 0.40
    print(f"  Score: {prob6*100:.1f}%  (threshold: < 40%)  {PASS if ok6 else FAIL}")
    results.append(ok6)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    passed = sum(results)
    total  = len(results)
    print(f"  RESULTS: {passed}/{total} scenarios passed")
    print(f"  Scores: sensor={prob1*100:.0f}%, IoT={prob2*100:.0f}%, "
          f"uniform_fake={prob3*100:.0f}%, shuffled={prob4*100:.0f}%, "
          f"diabetes={prob5*100:.0f}%, grid={prob6*100:.0f}%")

    if passed == total:
        print("  [SUCCESS] ALL SCENARIOS PASSED!")
    else:
        print(f"  [WARN] {total - passed} scenario(s) did not meet thresholds.")

    return passed == total


if __name__ == "__main__":
    success = run_simulation()
    sys.exit(0 if success else 1)
