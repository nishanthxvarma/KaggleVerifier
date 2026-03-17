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
        
        # --- Balanced Calibration ---
        # Instead of aggressive stretching, we use a softer sigmoid to maintain nuance
        # This prevents the "everything is 99.9%" issue while still keeping scores decisive.
        def sigmoid(x):
            return 1 / (1 + math.exp(-6.0 * (x - 0.5)))
        
        probability_real = sigmoid(probability_real)
        
        context = features.get('context_flags', {})
        
        # Apply subtle hints from context without forcing 99%
        if context.get('clustered_observations', False):
            probability_real = min(0.95, probability_real + 0.05)
        
        if features.get('missing_pct', 1.0) == 0.0 and features.get('mean_entropy', 0.0) > 4.5:
            probability_real = min(0.92, probability_real + 0.03)
            
        return probability_real, features, df
