import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
base_dir = os.path.dirname(__file__)
submission = pd.read_csv(os.path.join(base_dir, "../data/submission.csv"))
test = pd.read_csv(os.path.join(base_dir, "../data/test_with_date.csv"), parse_dates=["Date"])
merged = pd.merge(test, submission, on="Id")

# Sidebar
st.sidebar.header("Store & Date Filter")
store_id = st.sidebar.selectbox("Select Store", sorted(merged["Store"].unique()))
date_range = st.sidebar.date_input("Select Date Range", [merged["Date"].min(), merged["Date"].max()])

# Filter
filtered = merged[(merged["Store"] == store_id) &
                  (merged["Date"] >= pd.to_datetime(date_range[0])) &
                  (merged["Date"] <= pd.to_datetime(date_range[1]))]

# Title
st.title(f"📊 Rossmann Sales Predictions – Store {store_id}")

# Line chart
st.line_chart(filtered.set_index("Date")["Sales"])

# Daily statistics
st.subheader("📈 Sales Summary")
st.dataframe(filtered[["Date", "Sales"]].describe())

# Raw data toggle
if st.checkbox("Show raw data"):
    st.dataframe(filtered)