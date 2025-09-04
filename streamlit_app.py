import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# =========================
# Page config
# =========================
st.set_page_config(page_title="Rockfall Risk Dashboard", layout="wide")

# =========================
# Styles & Hero
# =========================
st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 100%;}
      .hero {
        background: linear-gradient(to right, #141E30, #243B55);
        color: #fff; padding: 40px; border-radius: 0; text-align: center; margin-bottom: 20px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.3);
      }
      .hero h1 {font-size: 3em; font-weight: 700; margin: 0;}
      .hero p {font-size: 1.1em; margin-top: 10px; opacity: 0.95;}
      .card {background: rgba(255,255,255,0.06); color: #fff; padding: 20px; border-radius: 12px; margin: 20px;}
      .footer {text-align: center; color: gray; font-size: 0.9em; margin-top: 40px;}
      .glow {animation: glow 1s infinite alternate; border: 2px solid orange; padding:10px; border-radius:8px;}
      @keyframes glow {from {box-shadow: 0 0 5px orange;} to {box-shadow: 0 0 20px orange;}}
      .shake {animation: shake 0.5s; animation-iteration-count: infinite;}
      @keyframes shake {0% { transform: translate(1px, 1px);} 25% { transform: translate(-1px, -2px);} 50% { transform: translate(-3px, 0px);} 75% { transform: translate(3px, 2px);} 100% { transform: translate(1px, -1px);} }
    </style>
    <div class="hero">
      <h1>⛰️ Rockfall Prediction System</h1>
      <p>AI-powered risk monitoring for open-pit mines</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Load Model & Dataset
# =========================
FEATURES = ["slope_angle_deg","factor_of_safety","green_index","rainfall_mm_day","pore_pressure_kpa"]

loaded_model = None
df = None
try:
    loaded_model = joblib.load("ridge_reg.pkl")
except Exception:
    st.warning("⚠️ No trained model found. Please ensure 'ridge_reg.pkl' is available.")

try:
    df = pd.read_csv("Final_Dataset.csv")
except Exception:
    st.error("❌ Could not load dataset. Please ensure 'Final_Dataset.csv' is available in project folder.")

# =========================
# Gauge Function (Needle)
# =========================
def risk_gauge(value):
    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=[0.21, 0.04, 0.75], labels=["Low", "Moderate", "High"],
        marker=dict(colors=["green", "orange", "red"]),
        hole=0.5, direction="clockwise", textinfo="label", rotation=180, showlegend=False
    ))
    angle = 180 * value
    x = 0.5 + 0.4 * np.cos(np.radians(180 - angle))
    y = 0.5 + 0.4 * np.sin(np.radians(180 - angle))
    fig.add_shape(type="line", x0=0.5, y0=0.5, x1=x, y1=y, line=dict(color="black", width=4))
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0),
                      annotations=[dict(text=f"Risk Score: {value:.3f}", x=0.5, y=0.2, showarrow=False, font_size=16)])
    return fig

# =========================
# Sidebar Navigation
# =========================
menu = ["🏠 Home", "📊 Risk Prediction", "📂 Data Insights", "📈 Visual Reports"]
choice = st.sidebar.radio("Navigation", menu)

# =========================
# Home
# =========================
if choice == "🏠 Home":
    st.markdown("<div class='card'><h3>Welcome</h3><p>This dashboard lets engineers estimate rockfall risk with AI, explore datasets, and view analytics.</p></div>", unsafe_allow_html=True)
    if df is not None:
        st.metric("📊 Dataset Loaded", f"{df.shape[0]} rows, {df.shape[1]} columns")

# =========================
# Risk Prediction
# =========================
elif choice == "📊 Risk Prediction":
    st.markdown("<div class='card'><h3>Enter Mine Conditions</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        slope_angle_deg = st.number_input("⛰️ Slope Angle (deg)", 0.0, 90.0, 30.0)
        factor_of_safety = st.number_input("⚖ Factor of Safety", 0.0, 5.0, 1.2)
        green_index = st.number_input("🌿 Green Index", 0.0, 1.0, 0.75)
    with col2:
        rainfall_mm_day = st.number_input("🌧 Rainfall (mm/day)", 0.0, 500.0, 12.5)
        pore_pressure_kpa = st.number_input("💧 Pore Pressure (kPa)", 0.0, 100.0, 22.0)

    if st.button("🔍 Predict Risk", type="primary"):
        if loaded_model is None:
            st.error("Model not loaded.")
        else:
            input_df = pd.DataFrame([{FEATURES[0]: slope_angle_deg, FEATURES[1]: factor_of_safety,
                                      FEATURES[2]: green_index, FEATURES[3]: rainfall_mm_day,
                                      FEATURES[4]: pore_pressure_kpa}])[FEATURES]
            pred = float(loaded_model.predict(input_df)[0])
            fig = risk_gauge(pred)
            st.plotly_chart(fig, use_container_width=True)
            if pred <= 0.21:
                st.success(f"✅ Low Risk (score: {pred:.3f}) 🎉")
                st.snow()  # confetti effect
            elif pred <= 0.25:
                st.markdown(f"<div class='glow'>⚠️ Moderate Risk (score: {pred:.3f})</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='shake'>🚨 High Risk (score: {pred:.3f})</div>", unsafe_allow_html=True)

# =========================
# Data Insights
# =========================
elif choice == "📂 Data Insights" and df is not None:
    st.markdown("<div class='card'><h3>Dataset Explorer</h3></div>", unsafe_allow_html=True)
    st.subheader("🔎 Preview")
    st.dataframe(df.head())
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe(include="all").transpose())
    st.subheader("🔗 Correlation Heatmap")
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", origin="lower", title="Correlation")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# Visual Reports
# =========================
elif choice == "📈 Visual Reports" and df is not None:
    st.markdown("<div class='card'><h3>Visual Analytics</h3></div>", unsafe_allow_html=True)
    if "risk_score" in df.columns:
        st.subheader("📊 Risk Score Distribution")
        fig = px.histogram(df, x="risk_score", nbins=20, color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)
    if {"rainfall_mm_day", "risk_score", "slope_angle_deg"}.issubset(df.columns):
        st.subheader("🌧 Rainfall vs Risk")
        fig = px.scatter(df, x="rainfall_mm_day", y="risk_score", color="slope_angle_deg", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# Footer
# =========================
st.markdown("<div class='footer'>👨‍💻 Designed by <b>Team Name</b> | SIH 2025</div>", unsafe_allow_html=True)
