"""
pipeline.py  (v2 – ensemble + domain-adaptive calibration)
──────────────────────────────────────────────────────────────────
Orchestrates dataset reading → feature extraction → ensemble scoring
→ rule-based post-calibration → final authenticity probability.
"""

import os
import sys
import math
import warnings
import pandas as pd
from typing import Tuple, Dict, Any

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kaggle_api import download_and_read_kaggle_dataset, process_upload
from ml.features import extract_features

# Try ensemble v2 first; fall back to the old meta-classifier if it exists
try:
    from ml.ensemble import KaggleEnsemble as _EnsembleCls
    _USE_ENSEMBLE = True
except ImportError:
    _USE_ENSEMBLE = False

try:
    from ml.model import KaggleMetaClassifier as _LegacyCls
    _HAVE_LEGACY = True
except ImportError:
    _HAVE_LEGACY = False


# ──────────────────────────────────────────────────────────────────
# POST-CALIBRATION RULES
# ──────────────────────────────────────────────────────────────────

def _calibrate(prob: float, feats: dict) -> Tuple[float, list]:
    """
    Apply domain-aware rule-based calibration on top of the raw
    ensemble probability. Returns (calibrated_prob, list_of_reasons).
    """
    reasons  = []
    context  = feats.get("context_flags", {})
    dtype    = context.get("dataset_type", "tabular")
    is_ts    = context.get("is_timeseries", False)
    boost    = 0.0
    penalty  = 0.0

    # -- SENSOR / TIME-SERIES BOOSTS -------------------------------

    lag1 = feats.get("mean_autocorr_lag1", 0.0)
    if is_ts and dtype == "sensor_iot":
        noise = feats.get("noise_level_estimate", 0.0)
        if noise > 0.01 and lag1 > 0.2: # Only boost if it has some temporal memory too
            boost += 0.15
            reasons.append("Natural noisy sensor signal detected (+0.15)")

    adf_p = feats.get("adf_p_value", 0.5)
    if is_ts and adf_p < 0.05:
        boost += 0.08
        reasons.append(f"Stationary signal (ADF p={adf_p:.3f}) suggests real sensor (+0.08)")

    if is_ts and lag1 > 0.3:
        boost += 0.10
        reasons.append(f"Strong temporal memory (autocorr lag-1={lag1:.2f}) (+0.10)")

    seas  = feats.get("seasonality_strength", 0.0)
    resid = feats.get("residual_variance", 0.0)
    if is_ts and seas > 0.1 and resid > 0:
        boost += 0.05
        reasons.append(f"Seasonality & residual variance suggest real cycles (+0.05)")

    # -- GENERAL REAL-DATA BOOSTS ----------------------------------

    uniform_ks = feats.get("uniform_ks_stat", 0.5)
    perm_e     = feats.get("mean_permutation_entropy", 0.5)
    
    # Complex but NOT uniform (uniform data is high entropy but synthetic)
    if perm_e > 0.70 and uniform_ks > 0.15:
        boost += 0.10
        reasons.append(f"High-complexity non-uniform signal ({perm_e:.2f}) (+0.10)")

    miss = feats.get("missing_pct", 0.0)
    miss_var = feats.get("missing_variance", 0.0)
    if 0.005 < miss < 0.15 and miss_var > 0:
        boost += 0.05
        reasons.append(f"Natural sparse missingness ({miss:.1%}) (+0.05)")

    if not is_ts and uniform_ks > 0.20:
        boost += 0.08
        reasons.append(f"Diverse value distribution (KS={uniform_ks:.2f}) suggests real (+0.08)")

    entropy_val = feats.get("mean_entropy", 0.0)
    if entropy_val > 4.5 and uniform_ks > 0.15:
        boost += 0.05
        reasons.append(f"Authentic Shanon entropy ({entropy_val:.2f}) (+0.05)")

    # -- PENALTY RULES (Prioritize detecting fakes) ----------------

    if perm_e < 0.20:
        penalty += 0.40
        reasons.append(f"Extreme sequence regularity (Entropy={perm_e:.2f}) -> synthetic artifact (-0.40)")
    elif perm_e < 0.35:
        penalty += 0.20
        reasons.append(f"High sequence regularity (Entropy={perm_e:.2f}) -> probable fake (-0.20)")

    dup = feats.get("duplicate_pct", 0.0)
    if dup == 0.0 and entropy_val < 2.0:
        penalty += 0.25
        reasons.append("Zero duplicates + very low entropy -> likely grid/fake (-0.25)")

    if uniform_ks < 0.05 and not is_ts:
        penalty += 0.25
        reasons.append(f"Near-uniform distribution (KS={uniform_ks:.3f}) -> synthetic (-0.25)")

    near_dup = feats.get("near_duplicate_ratio", 0.0)
    if near_dup > 0.40:
        penalty += 0.15
        reasons.append(f"High near-duplicate ratio ({near_dup:.0%}) -> synthetic artifact (-0.15)")

    # -- APPLY -----------------------------------------------------
    calibrated = prob + boost - penalty

    # Real signal floors (lowered slightly to not over-protect fakes)
    if is_ts and (lag1 > 0.4 or perm_e > 0.8) and calibrated < 0.55:
        calibrated = 0.55
        reasons.append("Applied floor (0.55) for high-conf temporal signals")
    elif not is_ts and entropy_val > 5.0 and uniform_ks > 0.3 and calibrated < 0.50:
        calibrated = 0.50
        reasons.append("Applied floor (0.50) for high-conf tabular signals")

    calibrated = float(max(0.01, min(0.99, calibrated)))
    return calibrated, reasons


# ──────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────

class DetectionPipeline:
    def __init__(self,
                 model_path_v2: str = "models/ensemble_v2.pkl",
                 model_path_v1: str = "models/meta_classifier.pkl"):

        self.classifier     = None
        self.use_ensemble   = False
        self._ci_lower      = None
        self._ci_upper      = None

        if _USE_ENSEMBLE:
            ens = _EnsembleCls(model_path_v2)
            if ens.is_trained:
                self.classifier   = ens
                self.use_ensemble = True
                print("[PIPELINE] Loaded ensemble v2 from disk.")
            else:
                print("[PIPELINE] Ensemble v2 not found. Training now (this may take a few minutes)...")
                self._retrain_ensemble(model_path_v2)

        # Legacy fallback
        if self.classifier is None and _HAVE_LEGACY:
            leg = _LegacyCls(model_path_v1)
            if leg.is_trained:
                self.classifier = leg
                print("[PIPELINE] Loaded legacy meta-classifier (v1).")
            else:
                from ml.train_real import train_robust_model as _train
                _train()
                self.classifier = _LegacyCls(model_path_v1)

        if self.classifier is None:
            raise RuntimeError("Could not load or train any classifier!")

    def _retrain_ensemble(self, path: str):
        try:
            from ml.train_real import train_robust_model
            train_robust_model()
            from ml.ensemble import KaggleEnsemble
            ens = KaggleEnsemble(path)
            if ens.is_trained:
                self.classifier   = ens
                self.use_ensemble = True
        except Exception as e:
            print(f"[warn] Ensemble training failed: {e}")

    # ── Public API ─────────────────────────────────────────────────
    def process_url(self, url: str) -> Tuple[float, dict, pd.DataFrame]:
        df = download_and_read_kaggle_dataset(url)
        return self._run_analysis(df)

    def process_file(self, file) -> Tuple[float, dict, pd.DataFrame]:
        df = process_upload(file)
        if len(df) > 50_000:
            df = df.sample(n=50_000, random_state=42)
        return self._run_analysis(df)

    def _run_analysis(self, df: pd.DataFrame) -> Tuple[float, dict, pd.DataFrame]:
        features = extract_features(df)
        if not features:
            raise ValueError("Dataset parsing failed: zero rows/columns or entirely unparseable!")

        # Model input: strip context_flags and any string keys
        model_input = {
            k: v for k, v in features.items()
            if k != "context_flags" and isinstance(v, (int, float))
        }

        raw_prob = float(self.classifier.predict(model_input))

        # Confidence interval (ensemble only)
        if self.use_ensemble and hasattr(self.classifier, "confidence_interval"):
            self._ci_lower, self._ci_upper = self.classifier.confidence_interval()
        else:
            delta = 0.07
            self._ci_lower = max(0.0, raw_prob - delta)
            self._ci_upper = min(1.0, raw_prob + delta)

        # Post-calibration
        calibrated, reasons = _calibrate(raw_prob, features)
        features["context_flags"]["calibration_reasons"] = reasons
        features["context_flags"]["raw_score"]           = round(raw_prob, 4)
        features["context_flags"]["ci_lower"]            = round(self._ci_lower, 3)
        features["context_flags"]["ci_upper"]            = round(self._ci_upper, 3)

        return calibrated, features, df
