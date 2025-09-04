import streamlit as st
import pandas as pd
import joblib
import numpy as np
import cv2
from PIL import Image

# --- Function to calculate green index ---
def calculate_green_index(img):
    # Convert PIL image to OpenCV (BGR)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize image (reduce size for display) → width=600
    scale_width = 600
    h, w = img.shape[:2]
    scale_ratio = scale_width / w
    new_dim = (scale_width, int(h * scale_ratio))
    img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Vegetation HSV range
    lower = (72, 42, 24)
    upper = (108, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)

    # Vegetation percentage
    veg_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    veg_percent = (veg_pixels / total_pixels) * 100

    # Masked image (green parts visible)
    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # back to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return veg_percent, img_rgb, result


# --- Load trained model ---
loaded_model = joblib.load("ridge_reg.pkl")

# --- App Title ---
st.title("🌱 Slope Stability Risk Prediction App")

st.markdown("Upload slope parameters and an image to calculate **Green Index**, then predict **Risk Score** 🚨")

# --- Inputs ---
slope_angle_deg = st.number_input("⛰️ Slope Angle (deg)", min_value=0.0, max_value=90.0, value=30.0)
factor_of_safety = st.number_input("🛡️ Factor of Safety", min_value=0.0, max_value=5.0, value=1.2)

uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

# Default value
green_index = 0.0

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Calculate green index
    green_index, original_img, masked_img = calculate_green_index(image)

    # Display results
    st.subheader("🌿 Vegetation Analysis")
    st.metric(label="Vegetation Cover (%)", value=f"{green_index:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original Image", width='stretch')
    with col2:
        st.image(masked_img, caption="Vegetation Detected", width='stretch')


rainfall_mm_day = st.number_input("🌧️ Rainfall (mm/day)", min_value=0.0, max_value=500.0, value=12.5)
pore_pressure_kpa = st.number_input("💧 Pore Pressure (kPa)", min_value=0.0, max_value=100.0, value=22.0)

# --- Prediction ---
if st.button("🔮 Predict Risk"):
    input_df = pd.DataFrame([{
        "slope_angle_deg": slope_angle_deg,
        "factor_of_safety": factor_of_safety,
        "green_index": green_index,
        "rainfall_mm_day": rainfall_mm_day,
        "pore_pressure_kpa": pore_pressure_kpa
    }])

    pred = loaded_model.predict(input_df)[0]

    # Risk classification
    if pred < 0.21:
        level = "🟢 Low"
    elif 0.21 < pred <= 0.25:
        level = "🟡 Medium"
    else:
        level = "🔴 High"

    st.subheader("📊 Prediction Results")
    st.success(f"✅ Predicted Risk Score: {pred:.4f}")
    st.warning(f"⚠️ Alert Level: **{level}**")
