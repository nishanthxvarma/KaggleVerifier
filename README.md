# Data Verace

**Premium, AI-Powered, Ensemble-Based Dataset Authenticity Intelligence**

Data Verace determines whether a tabular dataset is genuinely real-world data or synthetically generated — with specialized intelligence for sensor, IoT, and time-series data.

---

## 🆕 What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Classifier | Single XGBoost | **XGBoost + RandomForest + IsoForest stacked ensemble** |
| Time-series awareness | ❌ | ✅ Auto-detected, specialized scoring path |
| Sensor/IoT detection | ❌ | ✅ AR(1), autocorrelation, ADF stationarity |
| Feature count | 15 | **30+** |
| Benford's Law | Always applied | ✅ Skipped automatically for bounded sensor domains |
| Post-calibration | Sigmoid only | ✅ **8 domain-aware rules** |
| Explainability | Basic tooltips | ✅ **Natural-language explanation panel** |
| Confidence interval | ❌ | ✅ Cross-model variance CI |
| Training diversity | 5 sklearn + shuffled | **130+ real datasets (sensor + tabular + IoT)** |
| Fake generator types | 7 | **10 (+ CTGAN-style marginal, perfect grid, block-missing)** |
| Test coverage | 2 files | **5 test files + end-to-end simulation** |

---

## 🏗️ Architecture

```
Upload CSV
     │
     ▼
Feature Extraction (features.py, timeseries_detector.py)
     │
     ├── Domain detection: tabular / time-series / sensor_iot
     ├── 30+ statistical + temporal features
     │   ├── Standard: entropy, skewness, kurtosis, Benford's, IsoForest
     │   └── Temporal: ADF/KPSS, autocorrelation, STL, permutation entropy,
     │                 Higuchi FD, spike fraction, noise level
     │
     ▼
Ensemble v2 (ensemble.py)
     │
     ├── XGBClassifier  ──┐
     ├── RandomForest   ──┼──► Meta LogisticRegression (calibrated)
     └── IsoForest      ──┘
                             │
                             ▼
                       Raw score [0..1]
                             │
                             ▼
     Domain-Adaptive Calibration (pipeline.py)
     │
     ├── BOOST: sensor noise, ADF stationarity, autocorrelation memory,
     │          seasonality, natural missingness, clustered observations
     └── PENALTY: low permutation entropy, near-uniform distributions,
                  high near-duplicates, suspiciously perfect data
                             │
                             ▼
                   Final Authenticity Score
```

---

## 🎯 Accuracy Targets

| Dataset Type | Expected Score |
|---|---|
| Real sensor/IoT (natural noise + timestamps) | **80 – 95%** |
| Real tabular (UCI, sklearn, survey data) | **65 – 90%** |
| Pure CTGAN/uniform synthetic | **< 35%** |
| Column-shuffled or perfectly gridded fake | **< 40%** |

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

On first launch, the system will automatically train the ensemble (takes ~90-120 seconds). Subsequent runs load the saved model from `models/ensemble_v2.pkl`.

### Run Tests

```bash
# All unit tests
python -m pytest tests/ -v

# End-to-end score simulation (shows accuracy on 6 dataset types)
python tests/simulate_scores.py
```

### Retrain Model

```bash
python src/ml/train_real.py
```

---

## 📁 Project Structure

```
KaggleVerifier/
├── app.py                         # Streamlit UI (v2)
├── requirements.txt
├── models/
│   ├── ensemble_v2.pkl            # Stacked ensemble (XGB + RF + IsoForest)
│   └── training_report.json       # OOF AUC + class balance report
├── src/
│   ├── core/
│   │   ├── pipeline.py            # Orchestration + calibration rules
│   │   └── kaggle_api.py          # File/URL ingest
│   ├── ml/
│   │   ├── features.py            # 30+ feature extractor
│   │   ├── timeseries_detector.py # Time-series / sensor domain detection
│   │   ├── ensemble.py            # Stacked ensemble model
│   │   ├── train_real.py          # Training pipeline
│   │   └── generator.py           # Synthetic data augmentation
│   └── ui/
│       ├── components.py          # UI widgets (badge, gauge, ACF, NL panel)
│       └── style.css
├── tests/
│   ├── test_features.py           # Feature extractor tests
│   ├── test_ensemble.py           # Ensemble model tests
│   ├── test_calibration.py        # Calibration rule tests
│   ├── test_model.py              # Legacy classifier tests
│   └── simulate_scores.py         # End-to-end accuracy simulation
└── data/
    ├── real/                      # Drop extra real CSVs here for training
    └── synthetic/
```

---

## 🧠 Domain Detection Logic

The system automatically identifies dataset type before scoring:

1. **Sensor/IoT** – timestamp column detected + monotone time index + lag-1 autocorrelation > 0.3
2. **Time-series** – timestamp column present or very high autocorrelation without timestamp
3. **Standard tabular** – everything else

Once the domain is identified:
- Sensor path: applies ADF/KPSS stationarity, STL decomposition, permutation entropy, Higuchi fractal dimension, spike analysis
- Tabular path: applies enhanced Benford's Law, Shapiro-Wilk normality, uniform KS test, correlation structure

---

## 📄 License

MIT License — for academic final-year project use.
