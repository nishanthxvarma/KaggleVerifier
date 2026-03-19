import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, make_classification

# Add src to path
sys.path.append(os.path.join(os.getcwd()))

from src.core.pipeline import DetectionPipeline

def verify_v3():
    print("Starting KaggleVerifier v3 Verification Suite...\n")
    pipeline = DetectionPipeline()
    
    # 1. SMALL CLEAN REAL (Heart Disease / Diabetes Proxy)
    print("--- Test 1: Small Clean Real (UCI-style) ---")
    diabetes = load_diabetes(as_frame=True).frame
    prob_real, feats_real, _ = pipeline._run_analysis(diabetes)
    print(f"Result (Diabetes): {prob_real*100:.1f}%")
    print(f"KS-Stat: {feats_real.get('uniform_ks_stat'):.3f} | Entropy: {feats_real.get('mean_entropy'):.2f} | ReconErr: {feats_real.get('reconstruction_error'):.4f}")
    print(f"Reasons: {feats_real['context_flags']['calibration_reasons']}\n")
    
    # 2. LARGE SYNTHETIC (Titanic 1M proxy)
    print("--- Test 2: Large Synthetic (Titanic 1M proxy) ---")
    # Generate 1M uniform-ish rows
    n_rows = 100_000 # 100k for speed in test
    fake_data = pd.DataFrame(np.random.uniform(0, 1, size=(n_rows, 10)), columns=[f"f{i}" for i in range(10)])
    prob_fake, feats_fake, _ = pipeline._run_analysis(fake_data)
    print(f"Result (100k Synthetic): {prob_fake*100:.1f}%")
    print(f"Reasons: {feats_fake['context_flags']['calibration_reasons']}\n")

    # 3. CLEAN CLASSIFICATION (UCI Proxy)
    print("--- Test 3: Perfectly Clean Classification ---")
    X, y = make_classification(n_samples=500, n_features=15, n_informative=10, random_state=42)
    clean_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(15)])
    prob_clean, feats_clean, _ = pipeline._run_analysis(clean_df)
    print(f"Result (Clean 500 rows): {prob_clean*100:.1f}%")
    print(f"Reasons: {feats_clean['context_flags']['calibration_reasons']}\n")

    print("--- FINAL VERDICT ---")
    if prob_real > 0.85 and prob_clean > 0.85 and prob_fake < 0.35:
        print("✅ SUCCESS: Truthful Detector v3 meets all accuracy goals.")
    else:
        print("❌ FAILURE: Accuracy targets not fully met yet.")

if __name__ == "__main__":
    verify_v3()
