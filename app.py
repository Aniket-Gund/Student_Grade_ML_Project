# app.py
import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Score Predictor", layout="centered")
st.title("Student Score Predictor")

DATA_FILE = "student_scores (1).csv"
MODEL_FILE = "Student_model.pkl"

# Check files
if not os.path.exists(DATA_FILE):
    st.error(f"Missing CSV file: {DATA_FILE}")
    st.stop()

# Load CSV
try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Failed to read {DATA_FILE}: {e}")
    st.stop()

# Required columns
required = ["Hours_Studied", "Attendance", "Assignments_Submitted", "Score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"CSV missing required columns: {missing}")
    st.stop()

# Keep only required and drop NA
df = df[required].dropna().reset_index(drop=True)
if df.shape[0] == 0:
    st.error("CSV contains no usable rows after dropping NA.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Enter Student Details")
hs_default = float(df["Hours_Studied"].median())
att_default = float(df["Attendance"].median())
asgmt_default = float(df["Assignments_Submitted"].median())

hours = st.sidebar.number_input("Hours Studied", min_value=0.0, value=hs_default, step=0.5, format="%.2f")
attendance = st.sidebar.number_input("Attendance (percentage)", min_value=0.0, max_value=100.0, value=att_default, step=1.0, format="%.2f")
assignments = st.sidebar.number_input("Assignments Submitted", min_value=0.0, value=asgmt_default, step=1.0, format="%.2f")

input_df = pd.DataFrame([{
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Assignments_Submitted": assignments
}])

# Load model if present, otherwise train a silent fallback regressor
model = None
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
    except Exception:
        model = None

if model is None:
    # Train a simple fallback regressor on the CSV (no extra output)
    X = df[["Hours_Studied", "Attendance", "Assignments_Submitted"]]
    y = df["Score"]
    # handle tiny datasets safely
    if len(X) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
    model = Pipeline([("reg", RandomForestRegressor(n_estimators=100, random_state=42))])
    model.fit(X_train, y_train)

# Show input values
st.subheader("Input Values")
st.write(input_df)

# Predict button
if st.button("Predict Score"):
    try:
        pred = model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        st.subheader("Predicted Score")
        # If score is float, round to 2 decimals for display
        try:
            display_val = round(float(pred), 2)
        except Exception:
            display_val = str(pred)
        st.success(display_val)
