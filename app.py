import streamlit as st
import joblib
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# -------------------------
# Page config & common CSS
# -------------------------
st.set_page_config(page_title="GeoGuardians - Risk Predictor", layout="wide")

st.markdown(
    """
    <style>
    /* hide default header */
    header { visibility: hidden; }

    /* full-page background image + dark overlay for readability */
    [data-testid="stAppViewContainer"]{
      background-image: url("https://i0.wp.com/eos.org/wp-content/uploads/2025/01/landslide_xinjing_coal_mine_alxa_league_inner_mongolia_china_20230228_ssc2_rgb_flat_50cm_3840px_logo_wm-scaled.jpg?ssl=1");
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;
      position: relative;
    }
    [data-testid="stAppViewContainer"]::before{
      content: "";
      position: absolute;
      top:0; left:0;
      width:100%; height:100%;
      background: rgba(0,0,0,0.6);
      z-index: 0;
    }

    /* ensure Streamlit blocks are above the overlay */
    .block-container {
      position: relative;
      z-index: 1;
      padding-top: 24px;
      padding-left: 36px;
      padding-right: 36px;
      padding-bottom: 36px;
    }

    /* transparent navbar with blur */
    .title-bar {
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 12px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: fixed;
        top: 0; left: 0; width: 100%;
        z-index: 999;
        box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        font-family: "Segoe UI", Roboto, sans-serif;
    }

    .hamburger { cursor:pointer; margin-right:12px; }
    .hamburger div { width:22px; height:3px; background:#fff; margin:3px 0; border-radius:2px; }
    .title-text { color:#fff; font-weight:700; font-size:18px; text-align:center; flex-grow:1; }
    .contact-btn { background: linear-gradient(135deg,#22c55e,#16a34a); color:#fff; padding:8px 14px; border-radius:20px; text-decoration:none; }

    /* hero */
    .hero { margin-top:68px; height:36vh; background-image:url('https://i.imgur.com/YH9xX.jpg'); background-size:cover; background-position:center; display:flex; align-items:center; justify-content:center; position:relative; color:white; border-radius:8px; overflow:hidden; }
    .hero .overlay { position:absolute; inset:0; background: rgba(0,0,0,0.55); }
    .hero .hero-content { position:relative; z-index:2; text-align:center; padding: 1.5rem; }
    .hero h1 { margin:0; font-size:34px; font-weight:800; }
    .hero p { margin:8px 0 0; opacity:0.95; }

    /* inputs sizing */
    input[type="number"], input[type="text"] { height: 2.2rem; font-size:15px; }

    /* sections spacing */
    .section { background: rgba(255,255,255,0.03); border-radius:10px; padding: 14px; margin-top:18px; border:1px solid rgba(255,255,255,0.04); }
    .section h2 { color:#fff; margin:0 0 6px 0; }

    </style>
    """,
    unsafe_allow_html=True,
)

# --- Navbar + Hero (HTML) ---
st.markdown(
    """
    <div class="title-bar">
      <div style="display:flex; align-items:center;">
        <div class="hamburger" title="menu">
          <div></div><div></div><div></div>
        </div>
      </div>
      <div class="title-text">GeoRoots — Predict · Prevent · Protect</div>
      <button class="contact-btn" href="#">Get Started</button>
    </div>

    <div class="hero">
      <div class="overlay"></div>
      <div class="hero-content">
        <h1>GeoRoots</h1>
        <p>AI rockfall risk prediction using imagery & geotechnical inputs</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Image processing helper
# -------------------------
def calculate_green_percentage(pil_img, lower_h=35, upper_h=85, lower_s=40, lower_v=40):
    """
    Input: PIL Image (RGB)
    Returns: green_percent (0-100 float), mask_rgb (H,W,3), img_rgb (H,W,3)
    """
    img = np.array(pil_img)  # RGB
    h, w = img.shape[:2]
    max_width = 800
    if w > max_width:
        ratio = max_width / w
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

    # convert to HSV and compute mask
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([lower_h, lower_s, lower_v])
    upper = np.array([upper_h, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    green_pixels = int(np.count_nonzero(mask))
    total_pixels = mask.size if mask.size > 0 else 1
    green_percent = (green_pixels / total_pixels) * 100.0

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return green_percent, mask_rgb, img_rgb

# -------------------------
# Load model (joblib)
# -------------------------
MODEL_PATH = "ridge_reg.pkl"
model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model_load_error = e

# -------------------------
# Main layout: two columns
# -------------------------
st.markdown('<div class="section">', unsafe_allow_html=True)
left, right = st.columns([1, 1.05])

with left:
    st.subheader("Input Parameters")
    slope_angle_deg = st.number_input("Slope angle (deg)", min_value=0.0, max_value=90.0,
                                      value=30.0, step=0.1, format="%.2f")
    factor_of_safety = st.number_input("Factor of Safety", min_value=0.0, max_value=10.0,
                                       value=1.2, step=0.01, format="%.3f")
    rainfall_mm_day = st.number_input("Rainfall (mm/day)", min_value=0.0, max_value=5000.0,
                                      value=12.5, step=0.1, format="%.2f")
    pore_pressure_kpa = st.number_input("Pore Pressure (kPa)", min_value=0.0, max_value=5000.0,
                                        value=22.0, step=0.1, format="%.2f")

    # small spacing and placeholder for Predict button (directly below pore pressure)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    predict_placeholder = st.empty()

    # validation helper
    def is_valid_number(x):
        return x is not None and isinstance(x, (int, float)) and not np.isnan(x)

    inputs_valid = all(map(is_valid_number, [slope_angle_deg, factor_of_safety, rainfall_mm_day, pore_pressure_kpa]))

with right:
    st.subheader("Upload image for Green Index")
    uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    preview_col1, preview_col2 = st.columns(2)
    preview_image = None
    mask_rgb = None
    green_percent = 0.0

    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file).convert("RGB")
            # default HSV thresholds (kept simple to match your original flow)
            h_min, h_max, s_min, v_min = 35, 85, 40, 40

            green_percent, mask_rgb, preview_image = calculate_green_percentage(
                pil_img, lower_h=h_min, upper_h=h_max, lower_s=s_min, lower_v=v_min
            )

            # display metric & images
            st.metric("Vegetation Cover (%)", f"{green_percent:.2f}%")
            preview_col1.image(preview_image, caption="Original image", width='stretch')
            preview_col2.image(mask_rgb, caption="Detected green areas", width='stretch')

        except Exception as e:
            st.error(f"Error processing image: {e}")
            uploaded_file = None
            green_percent = 0.0

# Determine readiness for prediction (matching your original rule)
ready_to_predict = (uploaded_file is not None) and inputs_valid and (model is not None) and (model_load_error is None)

# Show disabled or enabled Predict button directly below pore pressure (left column)
with predict_placeholder.container():
    if model is None:
        st.button("Predict Risk (model not loaded)", disabled=True)
    else:
        if uploaded_file is None:
            st.button("Predict Risk (upload image to enable)", disabled=True)
        elif not inputs_valid:
            st.button("Predict Risk (fill all inputs)", disabled=True)
        else:
            # active button
            if st.button("Predict Risk"):
                # Build input dataframe (green_index passed as fraction 0-1)
                green_index_frac = (green_percent / 100.0) if green_percent is not None else 0.0
                input_df = pd.DataFrame([{
                    "slope_angle_deg": float(slope_angle_deg),
                    "factor_of_safety": float(factor_of_safety),
                    "green_index": float(green_index_frac),
                    "rainfall_mm_day": float(rainfall_mm_day),
                    "pore_pressure_kpa": float(pore_pressure_kpa)
                }])

                # Run prediction
                try:
                    pred = model.predict(input_df)[0]
                    # classification thresholds (as you used previously)
                    if pred < 0.21:
                        alert = "🟢 Low"
                    elif 0.21 < pred <= 0.25:
                        alert = "🟡 Medium"
                    else:
                        alert = "🔴 High"

                    # Notification and results
                    st.success(f"✅ Predicted Risk Score: {pred:.4f}")
                    st.warning(f"⚠️ Alert Level: {alert}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)
