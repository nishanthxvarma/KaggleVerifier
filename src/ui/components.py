import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────
# Core UI Helpers
# ──────────────────────────────────────────────────────────────────

def render_metric_card(title: str, value: str, trend: str = None, color: str = "#22D3EE"):
    """Sleek obsidian metric block with aqua highlight."""
    st.markdown(f"""
    <div class="glass-container">
        <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'Inter';">{title}</div>
        <div style="font-size: 2.22rem; font-weight: 700; color: {color}; margin: 8px 0; font-family: 'JetBrains Mono';">{value}</div>
        {f'<div style="font-size: 0.8rem; color: #38BDF8; opacity: 0.8;">{trend}</div>' if trend else ''}
    </div>
    """, unsafe_allow_html=True)


def render_verdict_badge(prob: float):
    """Deep-contrast branding verdict badge."""
    if prob > 0.75:
        label, color, bg = "VERIFIED AUTHENTIC", "#00FFFF", "rgba(0, 255, 255, 0.05)"
    elif prob > 0.60:
        label, color, bg = "PROBABLE AUTHENTICITY", "#22D3EE", "rgba(34, 211, 238, 0.05)"
    elif prob > 0.40:
        label, color, bg = "SIGNALS UNCERTAIN", "#94A3B8", "rgba(148, 163, 184, 0.05)"
    else:
        label, color, bg = "SYNTHETIC ARTIFACT DETECTED", "#EF4444", "rgba(239, 68, 68, 0.08)"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 24px; border-radius: 4px; border: 1px solid {color}; background: {bg}; margin-bottom: 30px;">
        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; font-family:'Inter'; font-weight:500;">Intelligence Verdict</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {color}; font-family:'Playfair Display'; letter-spacing:0.02em;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_dataset_type_badge(context: dict):
    """Renders a coloured badge showing auto-detected dataset domain."""
    dtype = context.get("dataset_type", "tabular")
    col   = context.get("ts_column")

    badge_map = {
        "sensor_iot":  ("📡", "Verified Sensor Stream",  "#22D3EE", "rgba(34, 211, 238, 0.05)"),
        "timeseries":  ("⏱️", "Sequential Analysis Active", "#38BDF8", "rgba(56, 189, 248, 0.05)"),
        "tabular":     ("📊", "Standard Tabular Matrix", "#F8FAFC", "rgba(255, 255, 255, 0.05)"),
    }
    icon, label, fg, bg = badge_map.get(dtype, badge_map["tabular"])
    col_hint = f"  ·  Source: <span style='font-family:monospace;'>{col}</span>" if col else ""

    st.markdown(f"""
    <div style="display:inline-block; background:{bg}; border:1px solid {fg};
                border-radius:4px; padding:6px 20px; margin-bottom:24px;">
        <span style="color:{fg}; font-weight:600; font-size:0.9rem; letter-spacing:0.05em; text-transform:uppercase;">{icon}&nbsp; {label}</span>
        <span style="color:#94A3B8; font-size:0.8rem;">{col_hint}</span>
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
    Richer NL explanation panel for v3 deep insights.
    """
    context  = feats.get("context_flags", {})
    dtype    = context.get("dataset_type", "tabular")
    is_ts    = context.get("is_timeseries", False)
    reasons  = context.get("calibration_reasons", [])
    raw      = context.get("raw_score", None)

    with st.expander("📝 Deep Analysis & Evidence", expanded=True):
        st.markdown(f"#### Why this verdict?")
        
        # Primary Indicators
        cols = st.columns(2)
        
        # Tabular Insights
        with cols[0]:
            ks = feats.get("uniform_ks_stat", 0.5)
            recon = feats.get("reconstruction_error", 0.5)
            st.write("**Tabular Structure**")
            if ks > 0.2:
                st.write(f"✅ Natural value distribution (KS={ks:.2f})")
            else:
                st.write(f"⚠️ Unusually uniform distribution (KS={ks:.2f})")
            
            if recon < 0.15:
                st.write(f"✅ Strong manifold fit (recon={recon:.3f})")
            else:
                st.write(f"⚠️ High reconstruction error (recon={recon:.3f})")

        # Temporal/Pattern Insights
        with cols[1]:
            st.write("**Data Regularity**")
            grid = feats.get("grid_density_score", 0.5)
            perm = feats.get("mean_permutation_entropy", 0.5)
            if grid > 0.3:
                st.write(f"✅ Natural value spacing (grid={grid:.2f})")
            else:
                st.write(f"⚠️ Perfect grid spacing detected (grid={grid:.2f})")
            
            if perm > 0.6:
                st.write(f"✅ Complex pattern (entropy={perm:.2f})")
            else:
                st.write(f"⚠️ High sequence regularity (entropy={perm:.2f})")

        st.markdown("---")
        
        # Calibration Reasons (All v3 rules listed)
        if reasons:
            st.write("**Calibration Signals:**")
            for r in reasons:
                st.markdown(f"- {r}")

        if raw is not None:
            st.markdown(f"<div style='font-size:0.7rem; color:#475569; margin-top:20px; font-family:monospace;'>"
                        f"Technical: Data Verace Core Intelligence Raw P: {raw*100:.1f}%</div>", 
                        unsafe_allow_html=True)


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
        <span>Data Verace</span> · Premium Data Intelligence · Built for Serious Analysts
        <br>
        <div style="font-size: 0.7rem; opacity: 0.6; margin-top: 8px; color: #64748B;">
            AI can make mistakes. Verify critical results.
        </div>
    </div>
    """, unsafe_allow_html=True)
