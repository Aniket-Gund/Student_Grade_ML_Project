# app.py
"""
Student Score Predictor — UI inspired by provided screenshot.
Place these files next to this script:
 - student_scores (1).csv
 - Student_model.pkl  (optional - fallback training will be used if missing)

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------- Page config --------------------
st.set_page_config(page_title="Student Score Prediction", layout="wide")

# -------------------- Custom CSS to mimic screenshot --------------------
st.markdown(
    """
    <style>
    /* page background + font sizes */
    .stApp {
        background-color: #0b0c0d;
        color: #e6eef8;
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* big title */
    .big-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .subtitle {
        color: #cbd5e1;
        margin-top: 0px;
        margin-bottom: 20px;
        font-size: 16px;
    }

    /* input container look: make number_input taller and pill-shaped */
    .stNumberInput>div>div>input {
        height: 46px !important;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        background-color: #222428 !important;
        color: #fff !important;
        border: 1px solid #2f3136 !important;
    }
    /* wide button style */
    .predict-btn {
        background: linear-gradient(90deg,#2a3340,#1f2933) !important;
        color: white !important;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
        font-weight: 600 !important;
    }

    /* green result box */
    .result-box {
        background: #0f3d2e !important;
        border-radius: 8px !important;
        padding: 18px !important;
        color: #d7f6e8 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        border: 1px solid rgba(255,255,255,0.03) !important;
    }

    /* card-like form area */
    .control-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.03);
    }

    /* footer small text */
    .footer {
        color: #99a3b3;
        font-size: 13px;
    }

    /* remove streamlit default header padding */
    .block-container {
        padding-top: 28px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Title / header --------------------
col1, col2 = st.columns([7, 1])
with col1:
    st.markdown('<div class="big-title">🎓 Student Score Prediction App by Aniket</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict student exam scores based on study habits and attendance.</div>', unsafe_allow_html=True)

# -------------------- Files & data checks --------------------
DATA_FILE = "student_scores (1).csv"
MODEL_FILE = "Student_model.pkl"

if not os.path.exists(DATA_FILE):
    st.error(f"Missing CSV file: {DATA_FILE} — place it in the same folder as this script.")
    st.stop()

# load csv
try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# expected schema
expected = ["Hours_Studied", "Attendance", "Assignments_Submitted", "Score"]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"CSV missing required columns: {missing}. Expected: {expected}")
    st.stop()

df = df[expected].dropna().reset_index(drop=True)
if df.shape[0] == 0:
    st.error("CSV contains no valid rows after dropping NA.")
    st.stop()

# -------------------- Layout: two columns (left input, right preview/result) --------------------
left, right = st.columns([6, 4])

with left:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)

    # Hours Studied (styled big)
    hours = st.number_input("📘 Hours Studied", min_value=0.0, value=float(df["Hours_Studied"].median()), step=0.5, format="%.2f")
    st.markdown("<br>", unsafe_allow_html=True)

    # Attendance
    attendance = st.number_input("🗓️ Attendance (%)", min_value=0.0, max_value=100.0, value=float(df["Attendance"].median()), step=1.0, format="%.2f")
    st.markdown("<br>", unsafe_allow_html=True)

    # Assignments Submitted
    assignments = st.number_input("📝 Assignments Submitted", min_value=0.0, value=float(df["Assignments_Submitted"].median()), step=1.0, format="%.2f")
    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button (styled)
    predict_clicked = st.button("🧠 Predict Score", key="predict_button")
    st.markdown("</div>", unsafe_allow_html=True)

    # Add small spacer
    st.markdown("<br>", unsafe_allow_html=True)

    # Footer small text
    st.markdown('<div class="footer">Built with ❤️ using Streamlit and Machine Learning</div>', unsafe_allow_html=True)

with right:
    # Show input preview card
    st.markdown('<div style="background:transparent; padding:0 8px;">', unsafe_allow_html=True)
    st.subheader("Input Values")
    input_df = pd.DataFrame([{
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Assignments_Submitted": assignments
    }])
    st.table(input_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # Result area placeholder
    result_placeholder = st.empty()

# -------------------- Model loading / fallback training --------------------
model = None
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
    except Exception:
        # fail silently and fallback
        model = None

if model is None:
    # train a fallback regressor
    X = df[["Hours_Studied", "Attendance", "Assignments_Submitted"]]
    y = df["Score"]
    if len(X) >= 3:
        # safe train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        except Exception:
            X_train, y_train = X, y
    else:
        X_train, y_train = X, y

    model = Pipeline([("reg", RandomForestRegressor(n_estimators=120, random_state=42))])
    model.fit(X_train, y_train)

# -------------------- Prediction action --------------------
if predict_clicked:
    try:
        pred_raw = model.predict(input_df)[0]
        # clamp & format
        try:
            pred_val = float(pred_raw)
            if pred_val < 0:
                pred_val = 0.0
            if pred_val > 100:
                pred_val = 100.0
            pred_text = f"{pred_val:.2f} / 100"
        except Exception:
            pred_text = str(pred_raw)
    except Exception as e:
        result_placeholder.error(f"Prediction failed: {e}")
    else:
        # show green result box
        result_placeholder.markdown(
            f'<div class="result-box">Predicted Score: {pred_text}</div>',
            unsafe_allow_html=True
        )
