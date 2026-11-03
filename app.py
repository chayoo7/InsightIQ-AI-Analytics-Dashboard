import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="InsightIQ",
    page_icon="📊",
    layout="wide"
)

st.title("📊 InsightIQ")
st.subheader("AI-Powered Business Analytics Platform")
st.caption("Phase 1: Interactive Sales Dashboard")

# Example dataset
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Sales": [12000, 15000, 14000, 21000, 25000, 29000],
    "Profit": [2400, 3500, 3100, 5200, 6100, 7300]
}

df = pd.DataFrame(data)

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"${df['Sales'].sum():,}")
col2.metric("Total Profit", f"${df['Profit'].sum():,}")
col3.metric("Months Analyzed", len(df))

# Interactive graph
fig = px.line(
    df,
    x="Month",
    y=["Sales", "Profit"],
    markers=True,
    title="Sales and Profit Trends"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Sales Data")
st.dataframe(df, use_container_width=True)
