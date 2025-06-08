import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

base_dir = os.path.dirname(__file__)
data_path = lambda file: os.path.join(base_dir, "../data", file)

# Load data
try:
    submission = pd.read_csv(data_path("submission.csv"))
    test = pd.read_csv(data_path("test.csv"), parse_dates=["Date"])  # test.csv with Date column
    merged = pd.merge(test, submission, on="Id")
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔎 Filter")
store_id = st.sidebar.selectbox("Select Store", sorted(merged["Store"].unique()))

# Filter by store only (no date filtering)
filtered = merged[merged["Store"] == store_id]

st.title("🏪 Retail Sales Prediction Dashboard")
st.markdown(f"Visualizing predictions for **Store {store_id}**")

# Line chart of predicted sales over time for selected store
st.subheader("📈 Sales Over Time")
st.line_chart(filtered.set_index("Date")["Sales"])

# Sales summary stats
st.subheader("🧮 Sales Summary Stats")
st.dataframe(filtered[["Date", "Sales"]].describe())

# Visual Gallery
st.subheader("📊 Visual Insights")

col1, col2 = st.columns(2)

with col1:
    st.image(data_path("daily_sales_plot.png"), caption="Predicted Daily Sales", use_container_width=True)
    st.image(data_path("dayofweek_plot.png"), caption="Avg Sales by Day of Week", use_container_width=True)

with col2:
    st.image(data_path("top_stores_plot.png"), caption="Top 20 Stores by Sales", use_container_width=True)
    st.image(data_path("feature_importance_plot.png"), caption="Feature Importance", use_container_width=True)

# Raw data toggle
with st.expander("📄 Show raw data"):
    st.dataframe(filtered)
