import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def style_page():
    """Injects custom CSS for premium fintech look"""
    try:
        with open("src/ui/style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        # Fallback if run elsewhere
        st.warning("Could not load style.css")

def render_metric_card(title: str, value: str, trend: str = None, color: str = "#38BDF8"):
    """Renders a sleek glassmorphic metric block via custom HTML/CSS."""
    st.markdown(f"""
    <div class="glass-container">
        <div style="font-size: 0.9rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
        <div style="font-size: 2.2rem; font-weight: 700; color: {color}; margin: 8px 0;">{value}</div>
        {f'<div style="font-size: 0.85rem; color: #2DD4BF;">{trend}</div>' if trend else ''}
    </div>
    """, unsafe_allow_html=True)

def render_probability_gauge(prob: float):
    """Renders a main hero gauge for the probability of being real."""
    color = "green" if prob > 0.6 else ("orange" if prob > 0.3 else "red")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%"},
        title={'text': "Likelihood of Being Authentic", 'font': {'size': 20, 'color': '#E2E8F0'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'}],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': prob * 100}
        }
    ))
    # Dark mode layout
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#E2E8F0"},
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_benfords_law(actual: list, expected: list, digits: list):
    """Plotly bar chart comparison of first digit distribution vs benford's law."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=digits, y=actual, name='Actual Data Dist',
        marker_color='#3B82F6', opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=digits, y=expected, name='Expected (Benford)',
        mode='lines+markers', line=dict(color='#2DD4BF', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Benford's Law Deviation Check",
        xaxis_title="First Digit",
        yaxis_title="Frequency",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#E2E8F0"},
        barmode='overlay',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    # Add subtle grid
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_xaxes(showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)

def render_footer():
    st.markdown("""
    <div class="custom-footer">
        <b>KaggleVerifier</b>
    </div>
    """, unsafe_allow_html=True)
