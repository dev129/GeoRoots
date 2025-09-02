import streamlit as st
import pandas as pd
import joblib

# Load trained model
loaded_model = joblib.load("ridge_reg.pkl")

# Title
st.title("Slope Stability Risk Prediction App")

st.markdown(
    """
    Enter the slope parameters below and get the predicted **Risk Score** and **Alert Level** 🚨
    """
) 

# Input fields
slope_angle_deg = st.number_input("Slope Angle (deg)", min_value=0.0, max_value=90.0, value=30.0)
factor_of_safety = st.number_input("Factor of Safety", min_value=0.0, max_value=5.0, value=1.2)
green_index = st.number_input("Green Index", min_value=0.0, max_value=1.0, value=0.75)
rainfall_mm_day = st.number_input("Rainfall (mm/day)", min_value=0.0, max_value=500.0, value=12.5)
pore_pressure_kpa = st.number_input("Pore Pressure (kPa)", min_value=0.0, max_value=100.0, value=22.0)

# Predict button
if st.button("Predict Risk"):
    # Prepare input
    input_df = pd.DataFrame([{
        "slope_angle_deg": slope_angle_deg,
        "factor_of_safety": factor_of_safety,
        "green_index": green_index,
        "rainfall_mm_day": rainfall_mm_day,
        "pore_pressure_kpa": pore_pressure_kpa
    }])

    # Prediction
    pred = loaded_model.predict(input_df)[0]

    # Alert level classification
    if pred < 0.21:
        level = "Low"
    elif 0.21 < pred <= 0.25:
        level = "Medium"
    else:
        level = "High"

    # Show results
    st.success(f"✅ Predicted Risk Score: {pred:.4f}")
    st.warning(f"⚠️ Alert Level: **{level}**")
