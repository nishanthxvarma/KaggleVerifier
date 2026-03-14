import pandas as pd
import numpy as np
import scipy.stats as stats
import re
from collections import Counter
import math
from sklearn.ensemble import IsolationForest

def extract_features(df: pd.DataFrame) -> dict:
    """
    Extracts 20+ advanced statistical and ML-based features from a pandas DataFrame
    to determine if the dataset is real or synthetically generated.
    """
    if df is None or df.empty:
        return {}

    # Standardize types and fill some NAs for numeric processing
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(exclude=[np.number])
    
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        return {}

    features = {}

    # 1. Exact Duplicates %
    features['duplicate_pct'] = df.duplicated().mean()

    # 2. Missing Value %
    features['missing_pct'] = df.isna().mean().mean()

    # 3. Missing Value Variance (per row)
    row_missingness = df.isna().sum(axis=1) / n_cols
    features['missing_variance'] = row_missingness.var() if len(row_missingness) > 1 else 0.0

    # 4. Column Sanity (check for weird names like 'Unnamed: 0', 'col_1')
    suspicious_cols = sum(1 for c in df.columns if re.search(r'(unnamed|col_\d+|var_\d+)', str(c).lower()))
    features['col_sanity_score'] = suspicious_cols / n_cols if n_cols else 0.0

    # 5. Mean Cardinality
    # Low cardinality across all columns often means simulated categorical data
    cardinality_pcts = [df[c].nunique() / n_rows for c in df.columns]
    features['mean_cardinality'] = np.mean(cardinality_pcts)

    # 6. Rounded Numbers Ratio
    # Real continuous data rarely ends strictly in 00 or forms perfect integers frequently.
    if not num_df.empty:
        total_nums = 0
        rounded_nums = 0
        for c in num_df.columns:
            vals = num_df[c].dropna().astype(float)
            if len(vals) == 0: continue
            total_nums += len(vals)
            # Check if it ends in exactly .00 or modulus 10 == 0 if int
            rounded_nums += np.sum((vals % 10 == 0) | (vals % 1 == 0))
        features['rounded_num_ratio'] = rounded_nums / total_nums if total_nums else 0.0
    else:
        features['rounded_num_ratio'] = 0.0

    # 7. Integer vs Float Fraction
    if not num_df.empty:
        is_int_col = [1 if pd.api.types.is_integer_dtype(num_df[c]) else 0 for c in num_df.columns]
        features['integer_col_ratio'] = np.mean(is_int_col)
    else:
        features['integer_col_ratio'] = 0.0

    # Distribution Features
    skew_vals = []
    kurtosis_vals = []
    uniformity_ks = []
    
    if not num_df.empty:
        for c in num_df.columns:
            vals = num_df[c].dropna()
            if len(vals) > 3:
                skew_vals.append(vals.skew())
                kurtosis_vals.append(vals.kurtosis())
                
                # Check KS test against uniform distro (synthetic data generators sometimes default to Uniform)
                # Normalize values
                scaled = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
                d, _ = stats.kstest(scaled, 'uniform')
                uniformity_ks.append(d)

        features['mean_skewness'] = np.nanmean(skew_vals) if skew_vals else 0.0
        features['mean_kurtosis'] = np.nanmean(kurtosis_vals) if kurtosis_vals else 0.0
        features['uniform_ks_stat'] = np.nanmean(uniformity_ks) if uniformity_ks else 0.0
        features['uniform_ks_stat'] = 0.0 if math.isnan(features['uniform_ks_stat']) else features['uniform_ks_stat']
        features['mean_skewness'] = 0.0 if math.isnan(features['mean_skewness']) else features['mean_skewness']
        features['mean_kurtosis'] = 0.0 if math.isnan(features['mean_kurtosis']) else features['mean_kurtosis']
    else:
        features['mean_skewness'] = 0.0
        features['mean_kurtosis'] = 0.0
        features['uniform_ks_stat'] = 0.0

    # 13. Correlation Anomalies
    if num_df.shape[1] > 1:
        corr_matrix = num_df.corr().abs().values
        # Get flattened upper triangle without diagonal
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        features['mean_abs_correlation'] = np.nanmean(upper_tri) if len(upper_tri) else 0.0
        features['mean_abs_correlation'] = 0.0 if math.isnan(features['mean_abs_correlation']) else features['mean_abs_correlation']
    else:
        features['mean_abs_correlation'] = 0.0

    # 14. Benford's Law Deviation
    def benford_mae(series):
        # Extract first digit
        str_vals = series.dropna().astype(str)
        first_digits = str_vals.str.extract(r'([1-9])')[0].dropna().astype(int)
        if len(first_digits) == 0: return np.nan
        
        counts = first_digits.value_counts(normalize=True).sort_index()
        actual = np.zeros(9)
        for idx in counts.index:
            actual[idx-1] = counts[idx]
            
        expected = np.log10(1 + 1/np.arange(1, 10))
        return np.mean(np.abs(actual - expected))

    benford_maes = []
    if not num_df.empty:
        for c in num_df.columns:
            mae = benford_mae(num_df[c])
            if not math.isnan(mae):
                benford_maes.append(mae)
    features['benfords_law_mae'] = np.mean(benford_maes) if benford_maes else 0.0

    # 15. Mean Shannon Entropy
    def shannon_entropy(series):
        counts = series.value_counts(normalize=True)
        return -np.sum(counts * np.log2(counts + 1e-9))
        
    entropies = [shannon_entropy(df[c]) for c in df.columns]
    features['mean_entropy'] = np.mean(entropies) if entropies else 0.0

    # 16. Outlier Fraction (Isolation Forest)
    if not num_df.empty and len(num_df) > 10:
        # Fill na with mean for IsoForest
        num_filled = num_df.fillna(num_df.mean())
        # Use a small number of samples if dataset is huge, otherwise all
        sample_size = min(len(num_filled), 1000)
        sample = num_filled.sample(n=sample_size, random_state=42)
        
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(sample)
        # -1 indicates outlier
        outlier_ratio = np.mean(preds == -1)
        features['outlier_fraction'] = outlier_ratio
    else:
        features['outlier_fraction'] = 0.0
        
    # Text-specific features (17-20)
    str_vars = []
    if not cat_df.empty:
        for c in cat_df.columns:
            str_lens = cat_df[c].dropna().astype(str).str.len()
            if len(str_lens) > 0:
                str_vars.append(str_lens.var())
    features['mean_string_len_variance'] = np.nanmean(str_vars) if str_vars else 0.0
    features['mean_string_len_variance'] = 0.0 if math.isnan(features['mean_string_len_variance']) else features['mean_string_len_variance']

    # Normalize feature dictionary values against NaNs just in case
    for k, v in features.items():
        if pd.isna(v) or math.isnan(v):
            features[k] = 0.0

    return features

def evaluate_benfords_law(series: pd.Series):
    """Returns actual and expected distributions for UI visualization"""
    str_vals = series.dropna().astype(str)
    first_digits = str_vals.str.extract(r'([1-9])')[0].dropna().astype(int)
    
    actual = np.zeros(9)
    if len(first_digits) > 0:
        counts = first_digits.value_counts(normalize=True).sort_index()
        for idx in counts.index:
            actual[idx-1] = counts[idx]
            
    expected = np.log10(1 + 1/np.arange(1, 10))
    return list(range(1, 10)), actual, expected
