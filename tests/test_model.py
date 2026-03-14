import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml.model import KaggleMetaClassifier, bootstrap_dummy_model

@pytest.fixture
def mock_model():
    model_path = "models/test_meta_classifier.pkl"
    # Ensure it's clean
    if os.path.exists(model_path):
         os.remove(model_path)
    
    classifier = KaggleMetaClassifier(model_path)
    yield classifier
    # Cleanup
    if os.path.exists(model_path):
         os.remove(model_path)

def test_model_training_and_prediction(mock_model):
    assert mock_model.is_trained == False
    
    # Train dummy data
    X = pd.DataFrame({
        'feat1': np.random.normal(0, 1, 100),
        'feat2': np.random.uniform(0, 5, 100),
    })
    # Target 1 or 0
    y = pd.Series(np.random.choice([0, 1], 100))
    
    acc = mock_model.train(X, y)
    assert acc >= 0.0 # Just verifying it returns score smoothly
    assert mock_model.is_trained == True
    
    # Check if file was saved
    assert os.path.exists("models/test_meta_classifier.pkl")
    
    # Predict
    prob = mock_model.predict({'feat1': 0.5, 'feat2': 2.5})
    assert 0.0 <= prob <= 1.0

def test_bootstrap():
    # Should run without crashing and generate the expected real model file
    bootstrap_dummy_model()
    assert os.path.exists("models/meta_classifier.pkl")
    classifier = KaggleMetaClassifier("models/meta_classifier.pkl")
    assert classifier.is_trained == True
