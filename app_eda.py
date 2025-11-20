# app_eda.py
"""
Streamlit EDA showcase for student_scores (1).csv
Reads: /mnt/data/student_scores (1).csv
Expected columns: Hours_Studied, Attendance, Assignments_Submitted, Score
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Page configuration & CSS to match your ML app look
st.set_page_config(page_title="Student EDA", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0b0c0d; color: #e6eef8; }
    .center-container { max-width: 880px; margin-left:auto; margin-right:auto; text-align:center; }
    .big-title { font-size:42px; font-weight:800; margin-bottom:4px; }
    .subtitle { color:#cbd5e1; margin-bottom:18px; }
    .control-card { background: rgba(255,255,255,0.02); border-radius:12px; padding:18px; border:1px solid rgba(255,255,255,0.03); }
    .footer { color:#99a3b3; font-size:13px; margin-top:18px; text-align:center; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="center-container big-title">📊 Student Score EDA by Aniket</div>', unsafe_allow_html=True)
st.markdown('<div class="center-container subtitle">Interactive exploratory data analysis — play with charts and download cleaned data.</div>', unsafe_allow_html=True)

# ---------- Load data (use exact uploaded local path) ----------
DATA_PATH = "/mnt/data/student_scores (1).csv"
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Unable to read data at {DATA_PATH}: {e}")
    st.stop()

# Validate columns
expected = ["Hours_Studied", "Attendance", "Assignments_Submitted", "Score"]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"CSV missing expected columns: {missing}. Expected: {expected}")
    st.stop()

# Basic cleaning
df = df[expected].copy()
df = df.dropna().reset_index(drop=True)

# ---------- Sidebar controls ----------
st.sidebar.header("EDA Controls")
n_preview = st.sidebar.slider("Rows to preview", min_value=5, max_value=min(200, len(df)), value=10)
hist_var = st.sidebar.selectbox("Histogram variable", options=expected[:-1], index=0)
hist_bins = st.sidebar.slider("Histogram bins", 5, 80, 20)
box_var = st.sidebar.selectbox("Boxplot variable", options=expected[:-1], index=0)
scatter_x = st.sidebar.selectbox("Scatter X", options=expected[:-1], index=0)
scatter_y = st.sidebar.selectbox("Scatter Y", options=expected[:-1], index=1)
show_corr = st.sidebar.checkbox("Show correlation heatmap", value=True)
download_clean = st.sidebar.button("Download cleaned CSV")

# ---------- Top-level metrics ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Avg Hours", f"{df['Hours_Studied'].mean():.2f}")
col3.metric("Avg Attendance", f"{df['Attendance'].mean():.2f}%")
col4.metric("Avg Score", f"{df['Score'].mean():.2f}")

# ---------- Main layout ----------
st.markdown("<div class='control-card'>", unsafe_allow_html=True)

# Preview
st.subheader("Dataset preview")
st.dataframe(df.head(n_preview))

# Summary stats
st.subheader("Summary statistics")
st.table(df.describe().T[['count','mean','std','min','25%','50%','75%','max']])

# Histogram (Plotly for interactivity)
st.subheader(f"Distribution of {hist_var}")
fig_hist = px.histogram(df, x=hist_var, nbins=hist_bins, marginal="box", template="plotly_dark", title=f"{hist_var} distribution")
fig_hist.update_layout(height=420)
st.plotly_chart(fig_hist, use_container_width=True)

# Boxplot (Seaborn/Matplotlib)
st.subheader(f"Boxplot of {box_var}")
fig, ax = plt.subplots(figsize=(6,2.8))
sns.boxplot(x=df[box_var], ax=ax, color="#1f77b4")
ax.set_xlabel(box_var)
st.pyplot(fig, clear_figure=True)

# Scatter with trendline
st.subheader(f"Scatter: {scatter_x} vs {scatter_y}")
fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y, trendline="ols", template="plotly_dark", height=420)
st.plotly_chart(fig_scatter, use_container_width=True)

# Correlation heatmap
if show_corr:
    st.subheader("Correlation heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, ax=ax)
    st.pyplot(fig, clear_figure=True)

# Simple feature engineering playground
st.subheader("Feature engineering quick checks")
if st.button("Add interaction: Hours * Attendance"):
    df_mod = df.copy()
    df_mod["Hours_x_Attendance"] = df_mod["Hours_Studied"] * df_mod["Attendance"]
    st.write("Added `Hours_x_Attendance` — top 5 rows:")
    st.dataframe(df_mod.head())

# Download cleaned CSV
if download_clean:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Click to download cleaned CSV", data=buf.getvalue().encode('utf-8'), file_name="student_scores_cleaned.csv", mime="text/csv")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer">Built with ❤️ using Streamlit — export it as a shareable app.</div>', unsafe_allow_html=True)
