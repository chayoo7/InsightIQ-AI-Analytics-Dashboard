from pathlib import Path
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st
import hashlib
import json

from ai_analyst import prepare_business_context, ask_ai

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="InsightIQ | Business Analytics",
    page_icon="📊",
    layout="wide",
)

REQUIRED_COLUMNS = [
    "order_date", "region", "category", "sales", "profit"
]
SAMPLE_FILE = Path(__file__).parent / "data" / "sales.csv"


# --------------------------------------------------
# DATA VALIDATION AND CLEANING
# --------------------------------------------------
def clean_data(raw_df):
    df = raw_df.copy()

   
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )

    missing = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    original_rows = len(df)

    
    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce", format="%Y-%m-%d"
    )

    for col in ["sales", "profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    
    for col in ["region", "category"]:
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""]

    
    if "order_id" in df.columns:
        df["order_id"] = df["order_id"].astype("string").str.strip()
        df["order_id"] = df["order_id"].replace("", pd.NA)

    df = df.sort_values("order_date").reset_index(drop=True)

    removed_rows = original_rows - len(df)
    return df, removed_rows


@st.cache_data
def load_sample_data():
    return pd.read_csv(SAMPLE_FILE)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📊 InsightIQ")
st.caption("Business Analytics and Data Intelligence Platform")
st.divider()


# --------------------------------------------------
# SIDEBAR: DATA SOURCE
# --------------------------------------------------
st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload a sales CSV file",
    type=["csv"],
    help="Upload CSV data with dates, regions, categories, sales and profit."
)

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
        st.sidebar.success("CSV uploaded successfully!")
        source_name = uploaded_file.name
    else:
        raw_df = load_sample_data()
        st.sidebar.info("Using sample sales data.")
        source_name = "sales.csv"

    df, removed_rows = clean_data(raw_df)

except (ValueError, pd.errors.ParserError, UnicodeError, OSError) as exc:
    st.error(f"Unable to load or validate dataset: {exc}")
    st.info(
        "Required columns: order_date, region, category, sales, profit. "
        "Dates must use YYYY-MM-DD."
    )
    st.stop()

if df.empty:
    st.error("There are no valid transactions in this dataset.")
    st.stop()

st.sidebar.caption(f"Dataset: {source_name}")

if removed_rows:
    st.warning(
        f"Data quality: {removed_rows} row(s) were excluded because "
        "required values were missing or invalid."
    )
else:
    st.success("Data quality check passed: all rows are valid.")


# --------------------------------------------------
# SIDEBAR: INTERACTIVE FILTERS
# --------------------------------------------------
st.sidebar.divider()
st.sidebar.header("Filters")

min_date = df["order_date"].min().date()
max_date = df["order_date"].max().date()

date_range = st.sidebar.date_input(
    "Order date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

available_regions = sorted(df["region"].unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Regions",
    options=available_regions,
    default=available_regions,
)

available_categories = sorted(df["category"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Categories",
    options=available_categories,
    default=available_categories,
)

if len(date_range) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

filtered = df[
    (df["order_date"].dt.date >= start_date)
    & (df["order_date"].dt.date <= end_date)
    & (df["region"].isin(selected_regions))
    & (df["category"].isin(selected_categories))
].copy()

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()


# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------
total_revenue = filtered["sales"].sum()
total_profit = filtered["profit"].sum()

if "order_id" in filtered.columns:
    
    known_orders = filtered["order_id"].dropna().nunique()
    missing_ids = filtered["order_id"].isna().sum()
    total_orders = known_orders + missing_ids
else:
    
    total_orders = len(filtered)

profit_margin = (
    (total_profit / total_revenue) * 100
    if total_revenue != 0
    else None
)

st.subheader("Business Performance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Revenue", f"${total_revenue:,.2f}")
c2.metric("Total Profit", f"${total_profit:,.2f}")
c3.metric("Total Orders", f"{total_orders:,}")
c4.metric(
    "Profit Margin",
    f"{profit_margin:.2f}%" if profit_margin is not None else "N/A"
)

st.caption(
    "Revenue and profit are summed from the filtered rows. "
    "Orders are counted by distinct order_id when available; "
    "otherwise each row is treated as one transaction."
)

st.divider()


# --------------------------------------------------
# MONTHLY SALES AND PROFIT TRENDS
# --------------------------------------------------
monthly = (
    filtered.assign(
        month=filtered["order_date"].dt.to_period("M").dt.to_timestamp()
    )
    .groupby("month", as_index=False)[["sales", "profit"]]
    .sum()
)

st.subheader("Revenue and Profit Trends")

trend_fig = px.line(
    monthly,
    x="month",
    y=["sales", "profit"],
    markers=True,
    labels={
        "month": "Month",
        "value": "Amount ($)",
        "variable": "Metric",
    },
    title="Monthly Business Performance",
)

st.plotly_chart(trend_fig, use_container_width=True)


# --------------------------------------------------
# REGION AND CATEGORY PERFORMANCE
# --------------------------------------------------
left, right = st.columns(2)

regional = (
    filtered.groupby("region", as_index=False)[["sales", "profit"]]
    .sum()
    .sort_values("sales", ascending=False)
)

category = (
    filtered.groupby("category", as_index=False)[["sales", "profit"]]
    .sum()
    .sort_values("sales", ascending=False)
)

with left:
    st.subheader("Revenue by Region")

    region_fig = px.bar(
        regional,
        x="region",
        y="sales",
        color="region",
        title="Regional Revenue",
        labels={"sales": "Revenue ($)", "region": "Region"},
    )

    st.plotly_chart(region_fig, use_container_width=True)

with right:
    st.subheader("Revenue by Category")

    category_fig = px.bar(
        category,
        x="category",
        y="sales",
        color="category",
        title="Category Performance",
        labels={"sales": "Revenue ($)", "category": "Category"},
    )

    st.plotly_chart(category_fig, use_container_width=True)


# --------------------------------------------------
# BUSINESS INSIGHTS
# --------------------------------------------------
st.divider()
st.subheader("💡 Automated Business Insights")

top_region = regional.iloc[0]
top_category = category.iloc[0]

st.write(
    f"**Top region:** {top_region['region']} generated "
    f"${top_region['sales']:,.2f} in revenue."
)

st.write(
    f"**Top category:** {top_category['category']} generated "
    f"${top_category['sales']:,.2f} in revenue."
)

st.write(
    f"**Overall profit:** ${total_profit:,.2f} across "
    f"{len(filtered):,} valid transaction rows."
)

st.caption(
    "These insights are calculated from your data. "
    "AI-generated explanations will be added in Phase 3."
)


# --------------------------------------------------
# DATA EXPLORER AND EXPORT
# --------------------------------------------------
st.divider()
st.subheader("Transaction Explorer")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Filtered Data",
    data=csv_data,
    file_name="insightiq_filtered_sales.csv",
    mime="text/csv",
)

with st.expander("View Data Quality Summary"):
    st.write(f"Original rows: {len(raw_df):,}")
    st.write(f"Valid rows: {len(df):,}")
    st.write(f"Excluded rows: {removed_rows:,}")
    st.write(f"Rows matching current filters: {len(filtered):,}")

st.caption("InsightIQ | Phase 2 — Advanced Data Analytics")

# --------------------------------------------------
# PHASE 3 — AI DATA ANALYST CHATBOT
# --------------------------------------------------

st.divider()
st.subheader("🤖 InsightIQ AI Data Analyst")

st.write(
    "Ask questions about your filtered sales data. "
    "The AI will explain insights using verified business metrics."
)


try:
    business_context = prepare_business_context(filtered)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()


context_json = json.dumps(
    business_context, sort_keys=True
)

context_id = hashlib.sha256(
    context_json.encode("utf-8")
).hexdigest()


if st.session_state.get("ai_context_id") != context_id:
    st.session_state.ai_context_id = context_id
    st.session_state.ai_messages = []

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []


for message in st.session_state.ai_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


st.caption("Try asking:")
st.info(
    "Which region has the highest revenue?  |  "
    "Which category is most profitable?  |  "
    "What recommendations can improve profitability?"
)


question = st.chat_input(
    "Ask InsightIQ about your sales data..."
)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.ai_messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        try:
            api_key = st.secrets["OPENAI_API_KEY"]

            with st.spinner("Analyzing your business data..."):
                answer = ask_ai(
                    question=question,
                    context=business_context,
                    api_key=api_key
                )

            st.markdown(answer)

        except KeyError:
            answer = (
                "OpenAI API key not found. "
                "Add OPENAI_API_KEY to "
                ".streamlit/secrets.toml and restart Streamlit."
            )
            st.error(answer)

        except Exception as exc:
            
            answer = (
                "The AI request could not be completed. "
                "Check your API key, billing, model access, "
                "and internet connection."
            )
            st.error(answer)
            print(f"AI request error: {type(exc).__name__}")

    st.session_state.ai_messages.append({
        "role": "assistant",
        "content": answer
    })


if st.button("Clear AI Conversation"):
    st.session_state.ai_messages = []
    st.rerun()
