# app.py
"""
Centered UI Student Score Predictor — Styled & Clean
"""

import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------- Page Setup ----------
st.set_page_config(page_title="Student Score Prediction", layout="wide")

# ---------- Custom CSS for Center Alignment ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0c0d;
        color: #e6eef8;
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    .center-container {
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }

    .big-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .subtitle {
        color: #cbd5e1;
        margin-top: -10px;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .input-card {
        background: rgba(255,255,255,0.02);
        border-radius: 14px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.04);
        margin-bottom: 20px;
    }

    /* Centering number inputs */
    .stNumberInput > div > div > input {
        height: 46px !important;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        background-color: #222428 !important;
        color: #fff !important;
        border: 1px solid #2f3136 !important;
        text-align: center;
    }

    /* Predict Button */
    .center-button {
        display: flex;
        justify-content: center;
    }

    .predict-btn {
        background: linear-gradient(90deg,#2a3340,#1f2933) !important;
        color: white !important;
        padding: 10px 25px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
        font-weight: 600 !important;
        width: 60%;
    }

    /* Green result box centered */
    .result-box {
        background: #0f3d2e !important;
        border-radius: 10px !important;
        padding: 18px !important;
        color: #d7f6e8 !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        border: 1px solid rgba(255,255,255,0.03) !important;
        text-align: center;
        margin-top: 18px;
    }

    .footer {
        color: #99a3b3;
        margin-top: 30px;
        font-size: 13px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Title ----------
st.markdown('<div class="center-container big-title">🎓 Student Score Prediction App by Aniket</div>', unsafe_allow_html=True)
st.markdown('<div class="center-container subtitle">Predict student exam scores based on study habits and attendance.</div>', unsafe_allow_html=True)

# ---------- Load Files ----------
DATA_FILE = "student_scores (1).csv"
MODEL_FILE = "Student_model.pkl"

if not os.path.exists(DATA_FILE):
    st.error(f"CSV file '{DATA_FILE}' not found.")
    st.stop()

df = pd.read_csv(DATA_FILE)

required = ["Hours_Studied", "Attendance", "Assignments_Submitted", "Score"]
missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"CSV missing columns: {missing}")
    st.stop()

df = df.dropna().reset_index(drop=True)

# ---------- Input Card Centered ----------
st.markdown('<div class="center-container">', unsafe_allow_html=True)
st.markdown('<div class="input-card">', unsafe_allow_html=True)

hours = st.number_input("📘 Hours Studied", min_value=0.0, value=float(df["Hours_Studied"].median()), step=0.5)
attendance = st.number_input("🗓️ Attendance (%)", min_value=0.0, max_value=100.0, value=float(df["Attendance"].median()), step=1.0)
assignments = st.number_input("📝 Assignments Submitted", min_value=0.0, value=float(df["Assignments_Submitted"].median()), step=1.0)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Predict Button ----------
col_btn = st.columns([3, 6, 3])[1]  # center the button
with col_btn:
    predict_clicked = st.button("🧠 Predict Score", key="predict_btn")

result_placeholder = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Load or Train Model ----------
model = None
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
    except:
        model = None

if model is None:
    X = df[["Hours_Studied", "Attendance", "Assignments_Submitted"]]
    y = df["Score"]

    if len(X) > 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        except:
            X_train, y_train = X, y
    else:
        X_train, y_train = X, y

    model = Pipeline([("reg", RandomForestRegressor(n_estimators=120, random_state=42))])
    model.fit(X_train, y_train)

# ---------- Prediction ----------
if predict_clicked:
    input_df = pd.DataFrame([{
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Assignments_Submitted": assignments
    }])

    try:
        pred_raw = model.predict(input_df)[0]
        pred_val = max(0.0, min(100.0, float(pred_raw)))
        pred_text = f"{pred_val:.2f} %"
    except Exception as e:
        result_placeholder.error(f"Prediction error: {e}")
    else:
        result_html = f'<div class="center-container result-box">Predicted Score: {pred_text}</div>'
        result_placeholder.markdown(result_html, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown('<div class="footer">Built with ❤️ using Streamlit and Machine Learning</div>', unsafe_allow_html=True)
