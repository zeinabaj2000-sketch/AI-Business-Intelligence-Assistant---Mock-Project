
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DEXCR | AI Business Intelligence",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #f6f7fb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    .hero {
        background: linear-gradient(120deg, #111827, #312e81);
        color: white;
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        min-height: 115px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
    }

    .metric-value {
        color: #111827;
        font-size: 27px;
        font-weight: 700;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 2. DATA LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


def read_uploaded_file(uploaded_file):
    """Read CSV or Excel data into a pandas DataFrame."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    raise ValueError("Please upload a CSV or XLSX file.")


def load_sample_data():
    """Load the original sales dataset if it exists."""
    path = BASE_DIR / "sales_transactions.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


# --------------------------------------------------
# 3. AUTOMATIC DATA PROFILING
# --------------------------------------------------

def profile_data(df):
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    date_candidates = []

    for col in df.columns:
        name = str(col).lower()

        if any(word in name for word in
               ["date", "datetime", "timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce")

            if parsed.notna().mean() >= 0.5:
                date_candidates.append(col)

    category_columns = []

    for col in df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns:
        unique_count = df[col].nunique(dropna=True)
        if 2 <= unique_count <= min(50, max(2, len(df) // 2)):
            category_columns.append(col)

    return numeric_columns, date_candidates, category_columns


def guess_measure(numeric_columns):
    """Guess a useful numeric column without assuming all data is sales."""
    priorities = [
        "net_sales", "revenue", "sales", "amount",
        "profit", "income", "quantity", "units",
        "cost", "price", "value",
    ]

    for word in priorities:
        for col in numeric_columns:
            if word in str(col).lower():
                return col

    return numeric_columns[0] if numeric_columns else None


def format_number(value):
    if pd.isna(value):
        return "N/A"

    if abs(value) >= 1_000_000:
        return f"{value:,.2f}"

    return f"{value:,.2f}"


# --------------------------------------------------
# 4. SIDEBAR AND NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ◈ DEXCR")
    st.caption("Your data. Any question.")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "My Data",
            "Data Explorer",
            "AI Analyst",
        ],
    )

    st.divider()
    st.caption("Universal business intelligence")


# --------------------------------------------------
# 5. DATASET SELECTION
# --------------------------------------------------

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "sample_mode" not in st.session_state:
    st.session_state.sample_mode = True

with st.sidebar:
    st.markdown("### Active dataset")

    data_choice = st.radio(
        "Choose data",
        ["My uploaded data", "Sample sales data"],
        horizontal=False,
    )

if data_choice == "My uploaded data":
    df = st.session_state.uploaded_df
    dataset_name = st.session_state.uploaded_name or "Uploaded dataset"
else:
    df = load_sample_data()
    dataset_name = "Sample sales transactions"

has_data = df is not None and not df.empty


# --------------------------------------------------
# 6. OVERVIEW
# --------------------------------------------------

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <h1>Business Intelligence, without the busywork.</h1>
            <p>
            Upload a dataset. Discover what it contains.
            Explore patterns and turn raw data into decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Platform</div>'
            '<div class="metric-value">DEXCR</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Supported formats</div>'
            '<div class="metric-value">CSV · XLSX</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Analysis engine</div>'
            '<div class="metric-value">Dynamic</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Start analyzing your data")

    st.write(
        "DEXCR is designed for sales, finance, operations, "
        "marketing, customer, survey, and other tabular datasets."
    )

    st.info(
        "Open **My Data** in the sidebar to upload a dataset. "
        "The current version automatically profiles the columns "
        "and creates an exploratory analysis."
    )

    if has_data:
        st.success(
            f"Active dataset: {dataset_name} "
            f"({len(df):,} rows, {len(df.columns):,} columns)"
        )


# --------------------------------------------------
# 7. UPLOAD AND PROFILE DATA
# --------------------------------------------------

elif page == "My Data":
    st.title("My Data")
    st.write(
        "Upload a dataset from your computer. "
        "DEXCR will inspect its structure and adapt the analysis "
        "to the columns available."
    )

    uploaded_file = st.file_uploader(
        "Choose a dataset",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel XLSX file.",
    )

    if uploaded_file is not None:
        try:
            new_df = read_uploaded_file(uploaded_file)

            if new_df.empty:
                st.error("The uploaded file contains no data.")
            else:
                st.session_state.uploaded_df = new_df
                st.session_state.uploaded_name = uploaded_file.name
                st.session_state.sample_mode = False

                st.success(
                    f"Loaded {uploaded_file.name}: "
                    f"{len(new_df):,} rows and "
                    f"{len(new_df.columns):,} columns."
                )
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

    df = st.session_state.uploaded_df
    dataset_name = st.session_state.uploaded_name

    if df is not None and not df.empty:
        st.divider()
        st.subheader("Dataset profile")

        numeric_cols, date_cols, category_cols = profile_data(df)

        a, b, c, d = st.columns(4)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", f"{len(df.columns):,}")
        c.metric("Numeric columns", len(numeric_cols))
        d.metric("Date candidates", len(date_cols))

        st.markdown("#### Column types")

        schema = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Detected type": [
                str(df[col].dtype) for col in df.columns
            ],
            "Unique values": [
                df[col].nunique(dropna=True) for col in df.columns
            ],
            "Missing values": [
                int(df[col].isna().sum()) for col in df.columns
            ],
            "Missing %": [
                round(df[col].isna().mean() * 100, 2)
                for col in df.columns
            ],
        })

        st.dataframe(schema, use_container_width=True)

        st.markdown("#### Data preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("#### Data quality checks")

        duplicate_rows = int(df.duplicated().sum())
        missing_cells = int(df.isna().sum().sum())

        q1, q2, q3 = st.columns(3)
        q1.metric("Duplicate rows", f"{duplicate_rows:,}")
        q2.metric("Missing cells", f"{missing_cells:,}")
        q3.metric(
            "Complete cells",
            f"{df.size - missing_cells:,}"
        )

        st.download_button(
            "Download uploaded data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="dexcr_export.csv",
            mime="text/csv",
        )

        st.caption(
            "Uploaded data is kept in the active app session; "
            "this code does not save it into your GitHub repository."
        )
    else:
        st.info(
            "No uploaded dataset yet. Upload a CSV or XLSX file above."
        )


# --------------------------------------------------
# 8. DYNAMIC DATA EXPLORER
# --------------------------------------------------

elif page == "Data Explorer":
    st.title("Data Explorer")

    if not has_data:
        st.warning(
            "No active dataset. Upload data under My Data, "
            "or select Sample sales data in the sidebar."
        )
    else:
        numeric_cols, date_cols, category_cols = profile_data(df)

        st.caption(
            f"Dataset: {dataset_name} · "
            f"{len(df):,} rows · {len(df.columns)} columns"
        )

        search_text = st.text_input(
            "Search column names",
            placeholder="Try revenue, date, customer..."
        )

        shown_cols = [
            col for col in df.columns
            if search_text.lower() in str(col).lower()
        ]

        st.dataframe(
            df[shown_cols].head(500),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Explore a numeric measure")

        if numeric_cols:
            measure = st.selectbox(
                "Choose a numeric column",
                numeric_cols,
                index=numeric_cols.index(guess_measure(numeric_cols))
                if guess_measure(numeric_cols) in numeric_cols else 0,
            )

            values = pd.to_numeric(
                df[measure], errors="coerce"
            ).dropna()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", format_number(values.sum()))
            m2.metric("Average", format_number(values.mean()))
            m3.metric("Median", format_number(values.median()))
            m4.metric("Distinct values", f"{values.nunique():,}")

            st.plotly_chart(
                px.histogram(
                    df,
                    x=measure,
                    title=f"Distribution of {measure}",
                    template="plotly_white",
                ),
                use_container_width=True,
            )

            if category_cols:
                group_col = st.selectbox(
                    "Compare by category",
                    category_cols,
                )

                grouped = (
                    df.groupby(group_col, dropna=False)[measure]
                    .sum()
                    .reset_index()
                    .sort_values(measure, ascending=False)
                    .head(20)
                )

                st.plotly_chart(
                    px.bar(
                        grouped,
                        x=group_col,
                        y=measure,
                        title=f"{measure} by {group_col}",
                        template="plotly_white",
                    ),
                    use_container_width=True,
                )
                st.dataframe(grouped, use_container_width=True)
        else:
            st.info(
                "No numeric columns were detected. You can still "
                "inspect the data and review its column types."
            )

        if date_cols and numeric_cols:
            st.divider()
            st.subheader("Trend analysis")

            date_col = st.selectbox("Date column", date_cols)
            trend_measure = st.selectbox(
                "Measure to track",
                numeric_cols,
                key="trend_measure",
            )

            trend_df = df[[date_col, trend_measure]].copy()
            trend_df[date_col] = pd.to_datetime(
                trend_df[date_col], errors="coerce"
            )
            trend_df[trend_measure] = pd.to_numeric(
                trend_df[trend_measure], errors="coerce"
            )
            trend_df = trend_df.dropna()

            if not trend_df.empty:
                trend_df["Period"] = (
                    trend_df[date_col].dt.to_period("M").astype(str)
                )
                monthly = (
                    trend_df.groupby("Period")[trend_measure]
                    .sum()
                    .reset_index()
                )

                st.plotly_chart(
                    px.line(
                        monthly,
                        x="Period",
                        y=trend_measure,
                        markers=True,
                        title=f"{trend_measure} over time",
                        template="plotly_white",
                    ),
                    use_container_width=True,
                )


# --------------------------------------------------
# 9. INITIAL AI ANALYST
# --------------------------------------------------

elif page == "AI Analyst":
    st.title("AI Analyst")
    st.write(
        "Ask questions about the active dataset. "
        "This initial version calculates answers with Python; "
        "a language model will be connected in a later step."
    )

    if not has_data:
        st.warning(
            "Upload a dataset under My Data or select sample data "
            "in the sidebar before asking questions."
        )
    else:
        numeric_cols, date_cols, category_cols = profile_data(df)

        if not numeric_cols:
            st.info(
                "This dataset has no detected numeric columns. "
                "Try a question about the number of records or "
                "the available columns."
            )

        question = st.text_input(
            "Ask a question",
            placeholder="e.g. What is the average order value?"
        )

        if st.button("Analyze question", type="primary"):
            q = question.lower().strip()

            if not q:
                st.warning("Please enter a question.")
            elif any(word in q for word in [
                "how many rows", "number of rows",
                "record count", "how many records",
            ]):
                st.metric("Number of records", f"{len(df):,}")
            elif "column" in q and any(
                word in q for word in ["which", "what", "list"]
            ):
                st.write("Available columns:")
                st.write(list(df.columns))
            elif numeric_cols:
                measure = guess_measure(numeric_cols)

                for col in numeric_cols:
                    if str(col).lower() in q:
                        measure = col
                        break

                values = pd.to_numeric(
                    df[measure], errors="coerce"
                ).dropna()

                if any(word in q for word in [
                    "average", "mean", "typical"
                ]):
                    st.success(
                        f"Average {measure}: "
                        f"{format_number(values.mean())}"
                    )
                elif any(word in q for word in [
                    "median", "middle"
                ]):
                    st.success(
                        f"Median {measure}: "
                        f"{format_number(values.median())}"
                    )
                elif any(word in q for word in [
                    "highest", "maximum", "max", "largest"
                ]):
                    st.success(
                        f"Highest {measure}: "
                        f"{format_number(values.max())}"
                    )
                elif any(word in q for word in [
                    "lowest", "minimum", "min", "smallest"
                ]):
                    st.success(
                        f"Lowest {measure}: "
                        f"{format_number(values.min())}"
                    )
                elif any(word in q for word in [
                    "count", "how many"
                ]):
                    st.success(
                        f"Non-empty values in {measure}: "
                        f"{values.count():,}"
                    )
                else:
                    st.success(
                        f"Total {measure}: "
                        f"{format_number(values.sum())}"
                    )

                st.caption(
                    f"Calculated from column '{measure}'. "
                    "Check the selected measure before relying on "
                    "the result for a business decision."
                )
            else:
                st.write("Available columns:", list(df.columns))

        st.divider()
        st.markdown("#### Questions to try")
        st.markdown(
            "- How many rows are in this dataset?\n"
            "- What is the average of a numeric column?\n"
            "- What is the highest value?\n"
            "- Which columns are available?"
        )
