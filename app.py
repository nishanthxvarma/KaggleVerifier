import os
import sys

# Make importable from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import time
from src.core.pipeline import DetectionPipeline
from src.ml.features import evaluate_benfords_law
from src.ui.components import render_metric_card, render_probability_gauge, plot_benfords_law, render_footer

# Page Config
st.set_page_config(
    page_title="KaggleVerifier – Fake Dataset Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
try:
    with open("src/ui/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception as e:
    pass

def get_pipeline():
    return DetectionPipeline()

pipeline = get_pipeline()

# Title Section
st.markdown("<h1 style='text-align: center; font-size: 3.5rem; letter-spacing: -2px; margin-bottom: 0px;'>Kaggle<span style='color: #38BDF8;'>Verifier</span></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #94A3B8; font-weight: 400; font-size: 1.2rem; margin-top: 0px; margin-bottom: 40px;'>AI-Powered Fraud Detection for Cloud Datasets</h3>", unsafe_allow_html=True)

# State Management
if 'results' not in st.session_state:
    st.session_state.results = None

def process_data(file=None):
    with st.spinner("Initializing advanced ML feature extraction pipeline..."):
        try:
            prob, feats, df = pipeline.process_file(file)
            st.session_state.results = {
                'prob': prob,
                'feats': feats,
                'df': df
            }
            # Add a slight delay to simulate complex ML ops
            time.sleep(1)
            st.success("Analysis complete!")
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# UI Layout
if not st.session_state.results:
    # 1. Clean Centered Input View (When no results)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Upload Dataset (CSV)</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['csv'])
        st.markdown("---")
        analyze_btn = st.button("Start AI Analysis Pipeline", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze_btn:
            if uploaded_file:
                process_data(file=uploaded_file)
                st.rerun()
            else:
                st.warning("Please upload a CSV file before analyzing.")
else:
    # 2. Results Dashboard View
    # Provide a simple top bar to upload a new file, freeing up screen real estate for charts
    with st.expander("Upload a different dataset"):
        uploaded_file = st.file_uploader("Upload local CSV (max 10MB sampled)", type=['csv'])
        if st.button("Analyze New Dataset"):
            if uploaded_file:
                process_data(file=uploaded_file)
                st.rerun()
            else:
                st.warning("Please upload a CSV file.")
    
    st.markdown("<hr style='border: 1px solid #334155; margin: 20px 0;'>", unsafe_allow_html=True)

    res = st.session_state.results
    prob = res['prob']
    feats = res['feats']
    df = res['df']
    
    # Hero Gauge centered
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    # Centering the gauge properly using columns
    g1, g2, g3 = st.columns([1, 2, 1])
    with g2:
        render_probability_gauge(prob)
        
        # Big verdict token
        if prob > 0.6:
            verdict = "<span style='color: #10B981;'>AUTHENTIC DATASET</span>"
        elif prob > 0.35:
            verdict = "<span style='color: #F59E0B;'>SUSPICIOUS / SYNTHETIC ANOMALIES</span>"
        else:
            verdict = "<span style='color: #EF4444;'>LIKELY FAKE / TAMPERED</span>"
            
        st.markdown(f"<h2 style='text-align: center; margin-top: 0px;'>VERDICT: {verdict}</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Context tooltips below gauge
    context = feats.get('context_flags', {})
    if context.get('clustered_observations', False):
         st.info("💡 **AI Insight:** High duplicates detected, but strong dataset entropy indicates natural clustered data (e.g. repeated measurements) rather than synthetic inflation. Scoring rules relaxed.")
    if context.get('narrow_numeric_range', False):
         st.info("💡 **AI Insight:** Narrow numeric boundaries detected (e.g. surveys or ratings). Strict integer/rounded number penalties were bypassed.")
    if feats.get('missing_pct', 1.0) == 0.0 and feats.get('mean_entropy', 0.0) > 4.5:
         st.success("✨ **AI Check:** The dataset is perfectly clean (0 missing values), but high randomness points to flawless preprocessing rather than synthetic generation.")
         
    st.markdown("<hr style='border: 1px solid #334155; margin: 40px 0;'>", unsafe_allow_html=True)
    
    # Feature Analytics Dashboard
    st.markdown("<h2>Detailed ML Analytics</h2>", unsafe_allow_html=True)
    
    # Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        dup_trend = "Natural cluster detected" if context.get('clustered_observations', False) else "Lower is better for natural data"
        render_metric_card("Duplicate Rows", f"{feats['duplicate_pct']:.1%}", trend=dup_trend)
    with m2:
        render_metric_card("Missing Volatility", f"{feats['missing_variance']:.2f}", trend="Too uniform = fake")
    with m3:
        render_metric_card("Entropy Score", f"{feats['mean_entropy']:.2f}", trend="Low entropy = generation artifacts")
    with m4:
        render_metric_card("Outlier Fraction", f"{feats['outlier_fraction']:.1%}", trend="Isolation Forest Anomaly")

    # Interactive Plots
    p1, p2 = st.columns([1, 1])
    
    with p1:
        st.markdown("### First-Digit Deviation (Benford's Law)")
        st.info("Authentic financial and natural numerical datasets typically follow Benford's logarithmic distribution. Fake data often trends linearly or uniformly.")
        
        num_df = df.select_dtypes(include='number')
        if not num_df.empty:
            best_col = num_df.columns[0]
            best_var = 0
            for c in num_df.columns:
                if num_df[c].var() > best_var and len(num_df[c].unique()) > 10:
                    best_var = num_df[c].var()
                    best_col = c
            
            d, act, exp = evaluate_benfords_law(num_df[best_col])
            plot_benfords_law(act, exp, d)
        else:
            st.warning("No continuous numerical columns found to evaluate Benford's Law.")
            
    with p2:
        st.markdown("### Top AI Features Extracted")
        
        df_f = pd.DataFrame([{
            'Indicator': k.replace('_', ' ').capitalize(), 
            'Value': v
        } for k, v in list(feats.items())[:8]])
        
        import plotly.express as px
        fig = px.bar(df_f, x='Value', y='Indicator', orientation='h', color='Value', color_continuous_scale='teal')
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#E2E8F0"})
        st.plotly_chart(fig, use_container_width=True)

render_footer()
