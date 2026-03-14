import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys

# Append parent dir for imports if run as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import extract_features

class KaggleMetaClassifier:
    def __init__(self, model_path='models/meta_classifier.pkl'):
        self.model_path = model_path
        self.model = XGBClassifier(
            n_estimators=50, 
            learning_rate=0.02, 
            max_depth=2, 
            min_child_weight=3,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.5,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = None

        if os.path.exists(self.model_path):
            self.load()

    def load(self):
        try:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.is_trained = True
        except Exception as e:
            print(f"Error loading model: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, self.model_path)

    def train(self, X: pd.DataFrame, y: pd.Series):
        print(f"Training Meta-Classifier on {len(X)} samples...")
        self.feature_names = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Validation Accuracy: {acc:.2f}")
        print(classification_report(y_test, preds))
        
        self.is_trained = True
        self.save()
        return acc

    def predict(self, df_features: dict):
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        # Ensure correct order
        X = pd.DataFrame([df_features])
        # Fill missing features with 0 if any
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
                
        X = X[self.feature_names]
        
        prob = self.model.predict_proba(X)[0]
        # prob[1] is probability of class 1 (Real)
        return prob[1]

# Quick demo function to bootstrap a model if needed
def bootstrap_dummy_model():
    """Generates synthetic feature vectors to train a dummy baseline model immediately"""
    print("Bootstrapping baseline model...")
    # Fake real data features (e.g. low duplication, low kurtosis, typical benfords)
    real_feats = []
    fake_feats = []
    
    for _ in range(150):
        # Semi-real distribution (More overlapping noise)
        real_feats.append({
            'duplicate_pct': np.random.uniform(0, 0.1),
            'missing_pct': np.random.uniform(0, 0.5),
            'missing_variance': np.random.uniform(0, 0.2),
            'col_sanity_score': np.random.uniform(0, 0.05),
            'mean_cardinality': np.random.uniform(0.05, 0.95),
            'rounded_num_ratio': np.random.uniform(0, 0.4),
            'integer_col_ratio': np.random.uniform(0, 0.9),
            'mean_skewness': np.random.normal(1, 1),
            'mean_kurtosis': np.random.normal(2, 2),
            'uniform_ks_stat': np.random.uniform(0.1, 0.5),
            'mean_abs_correlation': np.random.uniform(0.05, 0.5),
            'benfords_law_mae': np.random.uniform(0, 0.12),
            'mean_entropy': np.random.uniform(1, 7),
            'outlier_fraction': np.random.uniform(0.01, 0.08),
            'mean_string_len_variance': np.random.uniform(2, 25)
        })
        
        # Fake data features (Overlapping heavily with real, harder to distinguish)
        fake_feats.append({
            'duplicate_pct': np.random.uniform(0.05, 0.4),
            'missing_pct': np.random.uniform(0.1, 0.8),
            'missing_variance': np.random.uniform(0.05, 0.4),
            'col_sanity_score': np.random.uniform(0, 0.2),
            'mean_cardinality': np.random.uniform(0.01, 0.6),
            'rounded_num_ratio': np.random.uniform(0.1, 0.8),
            'integer_col_ratio': np.random.uniform(0.1, 1.0),
            'mean_skewness': np.random.normal(0, 0.5),
            'mean_kurtosis': np.random.normal(0, 0.5),
            'uniform_ks_stat': np.random.uniform(0.0, 0.25),
            'mean_abs_correlation': np.random.uniform(0.0, 0.3),
            'benfords_law_mae': np.random.uniform(0.05, 0.25), 
            'mean_entropy': np.random.uniform(0.5, 4),
            'outlier_fraction': np.random.uniform(0.03, 0.15),
            'mean_string_len_variance': np.random.uniform(0, 8)
        })
        
    X_df = pd.DataFrame(real_feats + fake_feats)
    y_ser = pd.Series([1]*150 + [0]*150)
    
    classifier = KaggleMetaClassifier('models/meta_classifier.pkl')
    classifier.train(X_df, y_ser)
    print("Bootstrap complete. Saved to models/meta_classifier.pkl")

if __name__ == "__main__":
    bootstrap_dummy_model()
