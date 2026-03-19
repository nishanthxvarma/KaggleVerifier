import sys, os, traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from src.ml.features import extract_features
from src.ml.ensemble import KaggleEnsemble
from src.ml.train_real import (
    _gen_sensor_ar1, _get_sklearn_real, _gen_classification_variants,
    corrupt_dataset, _augment_real
)

print("Building mini training set...")
real_dfs = _get_sklearn_real()[:2] + _gen_sensor_ar1(3) + _gen_classification_variants(3)
X_features, y_labels = [], []

for df in real_dfs:
    f = extract_features(df)
    if f:
        X_features.append(f)
        y_labels.append(1)
    for aug in _augment_real(df, seeds=[42, 73]):
        af = extract_features(aug)
        if af:
            X_features.append(af)
            y_labels.append(1)

for df in real_dfs[:3]:
    for method in ["uniform", "marginal"]:
        try:
            fk = corrupt_dataset(df, method)
            ff = extract_features(fk)
            if ff:
                X_features.append(ff)
                y_labels.append(0)
        except Exception as e:
            print(f"  corrupt {method} failed: {e}")

X_df = pd.DataFrame(X_features).fillna(0)
for c in ["context_flags", "dataset_type"]:
    if c in X_df.columns:
        X_df.drop(columns=[c], inplace=True)

y = pd.Series(y_labels)
print(f"Dataset: {X_df.shape}, classes: {y.value_counts().to_dict()}")

try:
    ens = KaggleEnsemble("models/_mini_test.pkl")
    metrics = ens.train(X_df, y, report_path="models/_mini_report.json")
    ens._calibrate_iso_range(X_df[y == 1].values)
    ens.save()
    print(f"AUC={metrics['oof_auc']}, Acc={metrics['oof_acc']}")
    print("ENSEMBLE TRAINING: OK")
except Exception:
    print("ENSEMBLE TRAINING: FAILED")
    traceback.print_exc()
finally:
    for fp in ["models/_mini_test.pkl", "models/_mini_report.json"]:
        if os.path.exists(fp):
            os.remove(fp)
