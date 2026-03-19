"""
ensemble.py
──────────────────────────────────────────────────────────────────
Stacked Ensemble Classifier for KaggleVerifier v2.

Architecture:
  Layer 1 (base models):
    - XGBClassifier
    - RandomForestClassifier
    - IsolationForest (anomaly score as numeric feature)
  Layer 2 (meta-learner):
    - CalibratedClassifierCV(LogisticRegression)
      trained on OOF predictions from Layer-1

Saved as:  models/ensemble_v2.pkl
"""
import os
import sys
import math
import warnings
import json
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────
# DEFAULT PATHS
# ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "models/ensemble_v2.pkl"
DEFAULT_REPORT_PATH = "models/training_report.json"


class KaggleEnsemble:
    """
    Stacked ensemble for real/fake dataset classification.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.is_trained = False
        self.feature_names = None

        # Base models
        self.xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.06,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.75,
            gamma=0.15,
            reg_lambda=1.5,
            reg_alpha=0.1,
            scale_pos_weight=1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.iso = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )

        # Meta-learner (calibrated)
        self.meta = CalibratedClassifierCV(
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
            method="sigmoid",
            cv=5,
        )

        if os.path.exists(model_path):
            self.load()

    # ──────────────────────────────────────────────────────────────
    def _prepare_X(self, X_df: pd.DataFrame) -> np.ndarray:
        """Ensure columns are ordered and fill any missing."""
        X = X_df.copy()
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]
        return X.values.astype(float)

    # ──────────────────────────────────────────────────────────────
    def train(self, X_df: pd.DataFrame, y: pd.Series,
              report_path: str = DEFAULT_REPORT_PATH) -> dict:
        """
        Full training pipeline with out-of-fold (OOF) meta-learner training.
        Returns a metrics dict.
        """
        self.feature_names = list(X_df.columns)
        X = X_df.values.astype(float)
        y_arr = y.values

        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        print(f"Training ensemble on {len(X)} samples with {X.shape[1]} features...")

        # ── OOF predictions for meta-learner ──────────────────────
        oof_xgb  = np.zeros(len(X))
        oof_rf   = np.zeros(len(X))
        oof_iso  = np.zeros(len(X))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_arr)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr         = y_arr[tr_idx]

            # XGB
            _xgb = XGBClassifier(**{k: v for k, v in self.xgb.get_params().items()
                                     if k not in ("scale_pos_weight",)})
            _xgb.set_params(scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum() + 1e-5)
            _xgb.fit(X_tr, y_tr, verbose=False)
            oof_xgb[val_idx] = _xgb.predict_proba(X_val)[:, 1]

            # RF
            _rf = RandomForestClassifier(**self.rf.get_params())
            _rf.fit(X_tr, y_tr)
            oof_rf[val_idx] = _rf.predict_proba(X_val)[:, 1]

            # IsoForest – train on positives only (treat real=1 as inliers)
            _iso = IsolationForest(**self.iso.get_params())
            _iso.fit(X_tr[y_tr == 1])
            # Convert score_samples to [0,1] probability (inlier → high prob)
            raw_scores = _iso.score_samples(X_val)
            lo, hi = raw_scores.min(), raw_scores.max() + 1e-10
            oof_iso[val_idx] = (raw_scores - lo) / (hi - lo)

            print(f"  Fold {fold + 1}/{n_splits} done.")

        # ── Train final base models on full data ───────────────────
        pos_weight = (y_arr == 0).sum() / (y_arr == 1).sum() + 1e-5
        self.xgb.set_params(scale_pos_weight=pos_weight)
        self.xgb.fit(X, y_arr, verbose=False)
        self.rf.fit(X, y_arr)
        self.iso.fit(X[y_arr == 1])

        # ── Train meta-learner on OOF stack ───────────────────────
        meta_X = np.column_stack([oof_xgb, oof_rf, oof_iso])
        self.meta.fit(meta_X, y_arr)

        # ── Metrics ───────────────────────────────────────────────
        meta_preds_proba = self.meta.predict_proba(meta_X)[:, 1]
        meta_preds       = (meta_preds_proba >= 0.5).astype(int)
        auc   = float(roc_auc_score(y_arr, meta_preds_proba))
        acc   = float(accuracy_score(y_arr, meta_preds))
        report = classification_report(y_arr, meta_preds, output_dict=True)

        metrics = {
            "oof_auc":  round(auc, 4),
            "oof_acc":  round(acc, 4),
            "n_train":  int(len(X)),
            "n_features": int(X.shape[1]),
            "class_balance": {
                "real_count":  int((y_arr == 1).sum()),
                "fake_count":  int((y_arr == 0).sum()),
            },
            "classification_report": report,
        }

        print(f"\n✅ OOF AUC: {auc:.4f}  |  OOF Accuracy: {acc:.4f}")

        self.is_trained = True
        self.save()

        # Save JSON report
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Training report saved: {report_path}")

        return metrics

    # ──────────────────────────────────────────────────────────────
    def predict(self, features: dict) -> float:
        """
        Predict probability that the dataset is REAL (class 1).
        `features` is the raw feature dict from extract_features().
        """
        if not self.is_trained:
            raise ValueError("Ensemble is not trained. Call train() or load() first.")

        X_df = pd.DataFrame([features])
        X_arr = self._prepare_X(X_df)

        xgb_p  = self.xgb.predict_proba(X_arr)[:, 1][0]
        rf_p   = self.rf.predict_proba(X_arr)[:, 1][0]

        iso_raw = self.iso.score_samples(X_arr)[0]
        # Normalize against training range stored at save time
        iso_p = float(np.clip(
            (iso_raw - self._iso_lo) / (self._iso_hi - self._iso_lo + 1e-10),
            0.0, 1.0
        ))

        meta_X = np.array([[xgb_p, rf_p, iso_p]])
        prob   = float(self.meta.predict_proba(meta_X)[0, 1])

        # Store individual model probabilities for CI computation
        self._last_probs = np.array([xgb_p, rf_p, iso_p])

        return float(np.clip(prob, 0.0, 1.0))

    def confidence_interval(self) -> tuple:
        """
        Returns (lower, upper) confidence bounds based on cross-model variance.
        Must be called after predict().
        """
        if not hasattr(self, "_last_probs"):
            return None, None
        probs = self._last_probs
        std   = float(np.std(probs))
        mean  = float(np.mean(probs))
        return (
            round(float(np.clip(mean - 1.96 * std, 0.0, 1.0)), 3),
            round(float(np.clip(mean + 1.96 * std, 0.0, 1.0)), 3),
        )

    # ──────────────────────────────────────────────────────────────
    def save(self):
        # Store Isolation Forest score range for normalization in predict()
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            "xgb":          self.xgb,
            "rf":           self.rf,
            "iso":          self.iso,
            "meta":         self.meta,
            "feature_names": self.feature_names,
            "iso_lo":       getattr(self, "_iso_lo", -1.0),
            "iso_hi":       getattr(self, "_iso_hi",  0.0),
        }, self.model_path)
        print(f"Ensemble saved: {self.model_path}")

    def load(self):
        try:
            data = joblib.load(self.model_path)
            self.xgb           = data["xgb"]
            self.rf            = data["rf"]
            self.iso           = data["iso"]
            self.meta          = data["meta"]
            self.feature_names = data["feature_names"]
            self._iso_lo       = data.get("iso_lo", -1.0)
            self._iso_hi       = data.get("iso_hi",  0.0)
            self.is_trained    = True
            print(f"Ensemble loaded from {self.model_path}")
        except Exception as e:
            print(f"Could not load ensemble: {e}")
            self.is_trained = False

    # ──────────────────────────────────────────────────────────────
    def _calibrate_iso_range(self, X: np.ndarray):
        """Store the IsoForest score range from training data (real samples only)."""
        scores = self.iso.score_samples(X)
        self._iso_lo = float(scores.min())
        self._iso_hi = float(scores.max())
