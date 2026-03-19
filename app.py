import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import time

from src.core.pipeline import DetectionPipeline
from src.ml.features import extract_features, evaluate_benfords_law
from src.ui.components import (
    render_metric_card,
    render_probability_gauge,
    plot_benfords_law,
    render_footer,
    render_dataset_type_badge,
    render_explanation_panel,
    plot_autocorrelation,
    render_timeseries_stats_table,
    render_verdict_badge,
)

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="KaggleVerifier – AI Dataset Authenticity",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load CSS ─────────────────────────────────────────────────────
try:
    with open("src/ui/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ── Pipeline singleton ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI ensemble v3…")
def get_pipeline(version_str="3.0.1"): # Cache buster
    return DetectionPipeline()

pipeline = get_pipeline()
# Diagnostic: Show model status in sidebar/expander hiddenly or just ensure it reloads
if st.sidebar.checkbox("Show Debug Diagnostics", value=False):
    st.sidebar.write(f"Pipeline Type: {'Ensemble v2' if pipeline.use_ensemble else 'Legacy v1'}")
    st.sidebar.write(f"Model Path Found: {os.path.exists('models/ensemble_v2.pkl')}")
    if st.session_state.results:
        st.sidebar.markdown("---")
        st.sidebar.write("**Live Analysis Telemetry:**")
        f = st.session_state.results["feats"]
        c = f.get("context_flags", {})
        st.sidebar.json({
            "is_ts": c.get("is_timeseries"),
            "dtype": c.get("dataset_type"),
            "ks_stat": f.get("uniform_ks_stat"),
            "entropy": f.get("mean_entropy"),
            "recon_err": f.get("reconstruction_error"),
            "grid": f.get("grid_density_score"),
            "raw_prob": c.get("raw_score")
        })


# ── Header ────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; font-size:3.5rem; letter-spacing:-2px; margin-bottom:0;'>"
    "Kaggle<span style='color:#38BDF8;'>Verifier</span> <sup style='font-size:1rem;color:#10B981;'>v3</sup>"
    "</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align:center; color:#94A3B8; font-weight:400; font-size:1.2rem; "
    "margin-top:0; margin-bottom:40px;'>"
    "Truthful Detector · Deep Pipeline · Unsupervised Manifold Analytics"
    "</h3>",
    unsafe_allow_html=True,
)

# ── State ─────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Process function ──────────────────────────────────────────────
def process_data(file=None):
    with st.spinner("Running ensemble ML pipeline + domain-adaptive analysis…"):
        try:
            prob, feats, df = pipeline.process_file(file)
            st.session_state.results = {
                "prob":  prob,
                "feats": feats,
                "df":    df,
            }
            time.sleep(0.5)
            st.success("✅ Analysis complete!")
        except Exception as e:
            st.error(f"Error during analysis: {e}")


# ── UPLOAD VIEW ───────────────────────────────────────────────────
if not st.session_state.results:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Upload Dataset (CSV)</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["csv"])
        st.markdown("---")
        analyze_btn = st.button("🚀 Start AI Analysis Pipeline", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if analyze_btn:
            if uploaded_file:
                process_data(file=uploaded_file)
                st.rerun()
            else:
                st.warning("Please upload a CSV file before analyzing.")


# ── RESULTS DASHBOARD ─────────────────────────────────────────────
else:
    with st.expander("🔄 Upload a different dataset"):
        new_file = st.file_uploader("Upload CSV (max 50k rows sampled)", type=["csv"])
        if st.button("Analyze New Dataset"):
            if new_file:
                process_data(file=new_file)
                st.rerun()
            else:
                st.warning("Please upload a CSV file.")

    st.markdown("<hr style='border:1px solid #334155; margin:20px 0;'>", unsafe_allow_html=True)

    res   = st.session_state.results
    prob  = res["prob"]
    feats = res["feats"]
    df    = res["df"]
    context = feats.get("context_flags", {})

    dtype   = context.get("dataset_type", "tabular")
    is_ts   = context.get("is_timeseries", False)
    ci_lo   = context.get("ci_lower", None)
    ci_hi   = context.get("ci_upper", None)

    # ── Hero Section ──────────────────────────────────────────────
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)

    # Dataset type badge (centered)
    cx1, cx2, cx3 = st.columns([1, 2, 1])
    with cx2:
        render_dataset_type_badge(context)
        render_verdict_badge(prob)

    # Gauge (centered)
    g1, g2, g3 = st.columns([1, 2, 1])
    with g2:
        render_probability_gauge(prob, ci_lo, ci_hi)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── AI Insight Callouts ───────────────────────────────────────
    if context.get("clustered_observations", False):
        st.info("💡 **AI Insight:** High duplicates + high entropy = natural clustered measurements (not synthetic inflation).")
    if context.get("narrow_numeric_range", False):
        st.info("💡 **AI Insight:** Narrow bounded columns detected (ratings/scores range). Integer penalties softened.")
    if feats.get("missing_pct", 1.0) == 0.0 and feats.get("mean_entropy", 0) > 4.5:
        st.success("✨ **AI Check:** Zero missing values + very high randomness → expert preprocessing, not synthetic generation.")
    if is_ts and dtype == "sensor_iot":
        lag1 = feats.get("mean_autocorr_lag1", 0)
        st.success(f"📡 **Sensor Mode Active:** Strong temporal structure detected (autocorrelation lag-1 = {lag1:.2f}). Sensor/IoT scoring rules applied.")

    # ── Explanation Panel ─────────────────────────────────────────
    render_explanation_panel(feats)

    st.markdown("<hr style='border:1px solid #334155; margin:40px 0;'>", unsafe_allow_html=True)

    # ── Metrics Cards ─────────────────────────────────────────────
    st.markdown("<h2>📊 Detailed ML Analytics</h2>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        dup_trend = "Natural cluster" if context.get("clustered_observations") else "Lower = more authentic"
        render_metric_card("Duplicate Rows", f"{feats.get('duplicate_pct',0):.1%}", trend=dup_trend)
    with m2:
        render_metric_card("Missing Volatility", f"{feats.get('missing_variance',0):.3f}", trend="High variance = natural")
    with m3:
        render_metric_card("Entropy Score", f"{feats.get('mean_entropy',0):.2f}", trend="Low entropy = generation artifact")
    with m4:
        render_metric_card("Outlier Fraction", f"{feats.get('outlier_fraction',0):.1%}", trend="Isolation Forest anomaly score")

    # Second row – time-series metrics (always shown, highlighted if sensor)
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        render_metric_card(
            "Permutation Entropy",
            f"{feats.get('mean_permutation_entropy', 0):.2f}",
            trend="< 0.4 = synthetic regularity",
            color="#A78BFA",
        )
    with m6:
        render_metric_card(
            "Autocorr Lag-1",
            f"{feats.get('mean_autocorr_lag1', 0):.2f}",
            trend="> 0.3 = real temporal memory",
            color="#38BDF8" if is_ts else "#64748B",
        )
    with m7:
        render_metric_card(
            "Near-Duplicate %",
            f"{feats.get('near_duplicate_ratio', 0):.1%}",
            trend="High → possible generator artifact",
            color="#F87171",
        )
    with m8:
        render_metric_card(
            "Higuchi Fractal Dim.",
            f"{feats.get('higuchi_fd', 1.5):.2f}",
            trend="~1.5–1.9 is typical for real signals",
            color="#34D399",
        )

    # ── Charts ────────────────────────────────────────────────────
    p1, p2 = st.columns([1, 1])

    with p1:
        if is_ts:
            st.markdown("### Autocorrelation Function")
            st.info("Real sensor signals show persistent autocorrelation. Synthetic data often has near-zero ACF.")
            plot_autocorrelation(df, context.get("ts_column"))
        else:
            st.markdown("### First-Digit Distribution (Benford's Law)")
            if context.get("benford_bypassed", True):
                st.info("Benford's Law skipped — sensor/bounded domain columns detected (not applicable).")
            else:
                st.info("Authentic financial & natural numeric data follows Benford's logarithmic distribution.")
                num_df = df.select_dtypes(include="number")
                if not num_df.empty:
                    best_col, best_var = num_df.columns[0], 0
                    for c in num_df.columns:
                        if num_df[c].var() > best_var and len(num_df[c].unique()) > 10:
                            best_var = num_df[c].var()
                            best_col = c
                    d, act, exp = evaluate_benfords_law(num_df[best_col])
                    plot_benfords_law(act, exp, d)
                else:
                    st.warning("No continuous numeric columns for Benford's Law analysis.")

    with p2:
        if is_ts:
            st.markdown("### Time-Series Diagnostic Metrics")
            st.info("Comprehensive temporal statistics used in the ensemble scoring. Values highlighted for sensor/IoT data.")
            render_timeseries_stats_table(feats)
        else:
            st.markdown("### Top AI Features Extracted")
            feature_display = {
                k: v for k, v in feats.items()
                if k != "context_flags"
                   and isinstance(v, (int, float))
                   and k not in ("is_timeseries", "is_sensor_iot")
            }
            df_f = pd.DataFrame([
                {"Indicator": k.replace("_", " ").capitalize(), "Value": round(float(v), 4)}
                for k, v in list(feature_display.items())[:12]
            ])
            import plotly.express as px2
            fig = px2.bar(df_f, x="Value", y="Indicator", orientation="h",
                          color="Value", color_continuous_scale="teal")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E2E8F0"},
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

render_footer()
