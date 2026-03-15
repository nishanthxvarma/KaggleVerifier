import os
import sys
import pandas as pd
from typing import Tuple, Dict, Any

# Ensure correct relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.kaggle_api import download_and_read_kaggle_dataset, process_upload
from ml.features import extract_features
from ml.model import KaggleMetaClassifier

class DetectionPipeline:
    def __init__(self, model_path='models/meta_classifier.pkl'):
        # Load the meta-classifier
        # If it doesn't exist, bootstrap it immediately using real-world tabular distributions
        self.classifier = KaggleMetaClassifier(model_path)
        if not self.classifier.is_trained:
            from ml.train_real import train_robust_model
            train_robust_model()
            self.classifier.load()

    def process_url(self, url: str) -> Tuple[float, dict, pd.DataFrame]:
        df = download_and_read_kaggle_dataset(url)
        return self._run_analysis(df)

    def process_file(self, file) -> Tuple[float, dict, pd.DataFrame]:
        df = process_upload(file)
        # Limit rows for uploads to prevent memory crashes
        if len(df) > 50000:
            df = df.sample(n=50000, random_state=42)
        return self._run_analysis(df)

    def _run_analysis(self, df: pd.DataFrame) -> Tuple[float, dict, pd.DataFrame]:
        features = extract_features(df)
        if not features:
            raise ValueError("Dataset parsing failed: zero rows/columns or entirely unparseable!")

        probability_real = float(self.classifier.predict(features))
        
        # --- Post-Processing Calibration & Rule-Based Heuristics ---
        context = features.get('context_flags', {})
        
        # 1. Rule based boosting for natural anomalies that often cause false positives
        if 0.4 < probability_real < 0.8:
            if context.get('clustered_observations', False):
                probability_real = min(0.88, probability_real + 0.25)
            elif context.get('narrow_numeric_range', False) and context.get('benford_bypassed', False):
                probability_real = min(0.85, probability_real + 0.15)
                
        # 2. Soft constraint: perfectly clean rows shouldn't punish score if entropy is strong
        if features.get('missing_pct', 1.0) == 0.0 and features.get('mean_entropy', 0.0) > 4.5:
            probability_real = min(0.96, probability_real + 0.05)
            
        return probability_real, features, df
