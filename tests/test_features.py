import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml.features import extract_features

def test_extract_features_empty():
    df = pd.DataFrame()
    feats = extract_features(df)
    assert not feats

def test_extract_features_basic():
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.uniform(0, 5, 100),
        'Cat': np.random.choice(['Yes', 'No'], 100)
    })
    feats = extract_features(df)
    
    # Must return dict with key metrics
    assert isinstance(feats, dict)
    assert 'duplicate_pct' in feats
    assert 'missing_pct' in feats
    assert 'mean_skewness' in feats
    assert 'benfords_law_mae' in feats
    
def test_extract_features_duplicates():
    df = pd.DataFrame({
        'A': [1, 1, 1, 2, 2],
        'B': [1, 1, 1, 3, 3] # Contains exact row duplicates
    })
    feats = extract_features(df)
    # Total rows = 5, duplicates = 3 (rows 1, 2 are dupe of 0, row 4 is dupe of 3)
    # 3/5 = 0.6 duplicate_pct exactly
    assert feats['duplicate_pct'] == 0.6
    assert feats['mean_cardinality'] == 0.4 # 2 uniques / 5 rows = 0.4

def test_missing_values():
    df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan],
        'B': [1, 2, np.nan, 4],
    })
    feats = extract_features(df)
    assert feats['missing_pct'] == 3/8
    # Missing variance between rows (0 missing, 1 missing, 1 missing, 1 missing) => vars > 0
    assert feats['missing_variance'] > 0

def test_string_sanity():
    df = pd.DataFrame({
        'Unnamed: 0': [1, 2],
        'col_55': [3, 4],
        'Valid': ['a', 'b']
    })
    feats = extract_features(df)
    assert feats['col_sanity_score'] == 2/3 # 2 weird columns
