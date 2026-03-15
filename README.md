# KaggleVerifier – Detect Fake/Synthetic Datasets 🔎

**KaggleVerifier** is an advanced AI-powered web application built to detect whether a Kaggle dataset (or any uploaded CSV) is authentic/real data or synthetically generated/manipulated.

Designed with a premium fintech-inspired dark mode UI, this tool extracts over 20 advanced statistical, distributional, and machine learning features to assess data integrity. It ultimately uses an XGBoost meta-classifier to provide a binary verdict and a confidence score.

**Final Year Project 2026**  
**Developed by:** Nishanth

---

## 🚀 Key Features

* **Multi-Layered Analysis:** Evaluates 20+ features, including:
  * **Statistical Anomalies:** Duplicate row clustering, missing value variance, cardinality scores, rounded numbers ratios.
  * **Adaptive Context Rules:** Intelligently bypasses penalties for natural data patterns (e.g., permits high duplicates if data shows high entropy indicative of clustered sensor/biology measurements, or forgives Benford's deviation on narrow range survey metrics).
  * **Distribution Checks:** Skewness, Kurtosis, Correlation heatmap consistency.
  * **Advanced Checks:** Benford's Law Mean Absolute Error (MAE), Shannon Entropy, and Isolation Forest Outlier Fractions.
* **XGBoost Meta-Classifier:** A high-speed, scalable model dynamically trained on robust synthetically-corrupted real-world tabular data.
* **Sleek Glassmorphism UI:** Built gracefully with Streamlit, Plotly, custom CSS and dark mode defaults.
* **Universal Input:** Extract massive tables directly from Kaggle URLs securely or via CSV file upload.

---

## 🛠 Tech Stack

* **Frontend:** Streamlit, Custom HTML/CSS
* **Backend Pipeline:** Python, pandas, numpy, scipy
* **Machine Learning:** scikit-learn (IsolationForest), XGBoost
* **Visuals:** Plotly

---

## 📦 Installation & Setup

1. **Clone/Navigate to Project Folder:**
   ```bash
   cd KaggleVerifier
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle API (Optional):**
   To download directly from Kaggle URL, ensure you have exported your API variables or have a `kaggle.json` set up.
   ```bash
   export KAGGLE_USERNAME="your-username"
   export KAGGLE_KEY="your-api-key"
   ```

4. **Bootstrap Initial ML Model:**
   The app uses an accurate XGBoost meta-classifier that must be trained offline first. Execute:
   ```bash
   python src/ml/train_real.py
   ```
   *(This downloads standard tabular real-world datasets from Scikit-Learn, generates complex synthetic aberrations (duplication, uniform noise), and trains `models/meta_classifier.pkl` instantly with >90% validation accuracy)*

---

## 🏃‍♂️ Running the Application

Execute the Streamlit application:
```bash
streamlit run app.py
```
This will launch a local server typically at `http://localhost:8501`.

---

## 🧪 Testing

The codebase includes `pytest` integration. To verify core extraction logic features:

```bash
pytest tests/
```

---

## ☁️ Deployment Guide

KaggleVerifier is fully portable and stateless, making it incredibly easy to deploy completely free on cloud providers:

### Render.com or Railway
1. Push this repository to GitHub.
2. Link the repository to Render/Railway as a "Web Service".
3. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY` if using the URL fetcher.

### Streamlit Community Cloud
1. Push repository to GitHub.
2. Sign in to share.streamlit.io.
3. Deploy new app, select `app.py`.
4. In **Advanced Settings**, safely add your API keys.

---

### Limitations & Future Work
* Maximum local file size capped around 10MB/50k rows due to expensive Isolation Forest calculations.
* Complex text-heavy NLP datasets require heavier transformer models (e.g., `sentence-transformers`) which were omitted here to maintain pure local CPU runtime.

--- 
*Built with ❤️ for data integrity.*
