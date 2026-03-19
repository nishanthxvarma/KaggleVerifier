import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────
# Core UI Helpers
# ──────────────────────────────────────────────────────────────────

def render_metric_card(title: str, value: str, trend: str = None, color: str = "#38BDF8"):
    """Sleek glassmorphic metric block."""
    st.markdown(f"""
    <div class="glass-container">
        <div style="font-size: 0.9rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
        <div style="font-size: 2.2rem; font-weight: 700; color: {color}; margin: 8px 0;">{value}</div>
        {f'<div style="font-size: 0.85rem; color: #2DD4BF;">{trend}</div>' if trend else ''}
    </div>
    """, unsafe_allow_html=True)


def render_dataset_type_badge(context: dict):
    """Renders a coloured badge showing auto-detected dataset domain."""
    dtype = context.get("dataset_type", "tabular")
    ts    = context.get("is_timeseries", False)
    col   = context.get("ts_column")

    badge_map = {
        "sensor_iot":  ("📡", "Sensor / IoT Time-Series Detected", "#0EA5E9", "#0C4A6E"),
        "timeseries":  ("⏱️", "Sequential Time-Series Detected",   "#8B5CF6", "#2E1065"),
        "tabular":     ("📊", "Standard Tabular Data",              "#10B981", "#064E3B"),
    }
    icon, label, fg, bg = badge_map.get(dtype, badge_map["tabular"])
    col_hint = f"  ·  Timestamp column: <b>{col}</b>" if col else ""

    st.markdown(f"""
    <div style="display:inline-block; background:{bg}; border:1px solid {fg};
                border-radius:24px; padding:6px 20px; margin-bottom:18px;">
        <span style="color:{fg}; font-weight:700; font-size:0.97rem;">{icon}&nbsp; {label}</span>
        <span style="color:#94A3B8; font-size:0.85rem;">{col_hint}</span>
    </div>
    """, unsafe_allow_html=True)


def render_probability_gauge(prob: float, ci_lower: float = None, ci_upper: float = None):
    """Hero gauge with optional confidence interval subtitle."""
    color = "green" if prob > 0.6 else ("orange" if prob > 0.3 else "red")
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 54}},
        title={"text": "Likelihood of Being Authentic", "font": {"size": 20, "color": "#E2E8F0"}},
        gauge={
            "axis":      {"range": [None, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar":       {"color": color},
            "bgcolor":   "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(239, 68, 68, 0.2)"},
                {"range": [40, 70], "color": "rgba(245,158, 11, 0.2)"},
                {"range": [70,100], "color": "rgba( 16,185,129, 0.2)"},
            ],
            "threshold": {
                "line":      {"color": "white", "width": 3},
                "thickness": 0.75,
                "value":     prob * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"},
        height=360,
        margin=dict(l=20, r=20, t=50, b=5),
    )
    st.plotly_chart(fig, use_container_width=True)

    if ci_lower is not None and ci_upper is not None:
        st.markdown(
            f"<p style='text-align:center; color:#94A3B8; font-size:0.9rem; margin-top:-10px;'>"
            f"Confidence interval: <b style='color:#E2E8F0;'>{ci_lower*100:.0f}%</b> – "
            f"<b style='color:#E2E8F0;'>{ci_upper*100:.0f}%</b></p>",
            unsafe_allow_html=True,
        )


def render_explanation_panel(feats: dict):
    """
    Rule-based natural-language explanation of the verdict.
    Generates contextual sentences from the feature dict.
    """
    context  = feats.get("context_flags", {})
    dtype    = context.get("dataset_type", "tabular")
    is_ts    = context.get("is_timeseries", False)
    reasons  = context.get("calibration_reasons", [])
    raw      = context.get("raw_score", None)

    lines = []

    # Domain line
    if dtype == "sensor_iot":
        lines.append("🔬 **Sensor/IoT dataset** path selected — temporal and signal-continuity tests applied.")
    elif dtype == "timeseries":
        lines.append("⏱️ **Time-series dataset** path selected — autocorrelation and stationarity tests applied.")
    else:
        lines.append("📊 **Standard tabular** analysis applied.")

    # Key evidence lines
    lag1 = feats.get("mean_autocorr_lag1", 0.0)
    if lag1 > 0.3 and is_ts:
        lines.append(f"✅ Strong temporal memory detected (autocorr lag-1 = **{lag1:.2f}**), consistent with real sensor readings.")

    adf = feats.get("adf_p_value", 0.5)
    if is_ts and adf < 0.05:
        lines.append(f"✅ Signal is stationary (ADF p-value = **{adf:.4f}**) — common in calibrated sensors.")

    perm_e = feats.get("mean_permutation_entropy", 0.5)
    if perm_e > 0.7:
        lines.append(f"✅ High permutation entropy (**{perm_e:.2f}**) → complex natural disorder, not synthetic regularity.")
    elif perm_e < 0.35:
        lines.append(f"⚠️ Low permutation entropy (**{perm_e:.2f}**) → sequence may be too regular — possible generator artifact.")

    miss = feats.get("missing_pct", 0.0)
    if 0.005 < miss < 0.15:
        lines.append(f"✅ Natural missingness ({miss:.1%}) randomly distributed — typical of real-world collection processes.")

    dup = feats.get("duplicate_pct", 0.0)
    if dup > 0.3:
        lines.append(f"⚠️ High duplicate row rate ({dup:.0%}) — may indicate synthetic inflation or cheap data scraping.")

    benford = feats.get("benfords_law_mae", None)
    bypassed = context.get("benford_bypassed", True)
    if not bypassed and benford is not None:
        if benford < 0.04:
            lines.append(f"✅ First-digit distribution follows Benford's Law closely (MAE = **{benford:.4f}**).")
        elif benford > 0.10:
            lines.append(f"⚠️ Significant Benford's deviation (MAE = **{benford:.4f}**) — unusual for natural numeric data.")
    elif bypassed:
        lines.append("ℹ️ Benford's Law check skipped — bounded/sensor columns detected (inapplicable domain).")

    spikes = feats.get("spike_fraction", 0.0)
    if is_ts and 0.005 < spikes < 0.10:
        lines.append(f"✅ Natural spike pattern ({spikes:.1%} of readings) — consistent with real sensor anomalies.")

    # Calibration reasons
    if reasons:
        lines.append("")
        lines.append("**Score adjustments applied:**")
        for r in reasons:
            lines.append(f"- {r}")

    if raw is not None:
        lines.append(f"\n_Raw ensemble score before calibration: **{raw*100:.1f}%**_")

    with st.expander("🧠 AI Explanation & Evidence", expanded=True):
        for line in lines:
            if line == "":
                st.markdown("---")
            else:
                st.markdown(line)


def plot_benfords_law(actual: list, expected: list, digits: list):
    """Plotly bar chart: actual first-digit distribution vs Benford's Law."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=digits, y=actual, name="Actual Data",
        marker_color="#3B82F6", opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=digits, y=expected, name="Benford Expected",
        mode="lines+markers",
        line=dict(color="#2DD4BF", width=3),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title="Benford's Law First-Digit Check",
        xaxis_title="First Digit",
        yaxis_title="Frequency",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"},
        barmode="overlay",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_autocorrelation(df: pd.DataFrame, ts_col: str = None, n_lags: int = 40):
    """Autocorrelation plot for time-series data."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return
    col = num_df.columns[0]
    s   = num_df[col].dropna()
    if len(s) < 10:
        return

    try:
        from pandas.plotting import autocorrelation_plot as _acf
        # Compute manually for plotly
        acf_vals = [s.autocorr(lag=lag) for lag in range(1, min(n_lags + 1, len(s) // 2))]
        lags     = list(range(1, len(acf_vals) + 1))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=lags, y=acf_vals, marker_color="#8B5CF6", name="ACF"))
        fig.add_hline(y=0, line_color="white", line_width=1)
        fig.add_hline(y=1.96 / (len(s) ** 0.5), line_dash="dash", line_color="#F59E0B", line_width=1, annotation_text="95% CI")
        fig.add_hline(y=-1.96 / (len(s) ** 0.5), line_dash="dash", line_color="#F59E0B", line_width=1)
        fig.update_layout(
            title=f"Autocorrelation Function — {col}",
            xaxis_title="Lag",
            yaxis_title="ACF",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E2E8F0"},
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Autocorrelation chart temporarily unavailable.")


def render_timeseries_stats_table(feats: dict):
    """Mini-table of key time-series diagnostic metrics."""
    rows = [
        ("ADF Stationarity p-value",    feats.get("adf_p_value",        "—")),
        ("KPSS Stationarity p-value",   feats.get("kpss_p_value",       "—")),
        ("Autocorr Lag-1",              feats.get("mean_autocorr_lag1",  "—")),
        ("Autocorr Lag-5",              feats.get("mean_autocorr_lag5",  "—")),
        ("Seasonality Strength",        feats.get("seasonality_strength","—")),
        ("Residual Variance",           feats.get("residual_variance",   "—")),
        ("Permutation Entropy",         feats.get("mean_permutation_entropy","—")),
        ("Higuchi Fractal Dimension",   feats.get("higuchi_fd",          "—")),
        ("Spike Fraction",              feats.get("spike_fraction",      "—")),
        ("Noise Level Estimate",        feats.get("noise_level_estimate","—")),
    ]
    formatted = []
    for name, val in rows:
        try:
            formatted.append({"Metric": name, "Value": f"{float(val):.4f}"})
        except Exception:
            formatted.append({"Metric": name, "Value": str(val)})

    st.dataframe(
        pd.DataFrame(formatted),
        use_container_width=True,
        hide_index=True,
    )


def render_footer():
    st.markdown("""
    <div class="custom-footer">
        <b>KaggleVerifier v2</b> · Ensemble + Domain-Adaptive Authenticity Detection
    </div>
    """, unsafe_allow_html=True)
