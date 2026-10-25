from pathlib import Path
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional LangChain integration
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ============================================================
# 1. PAGE SETUP AND DEXCR DESIGN
# ============================================================
st.set_page_config(
    page_title="DEXCR | Make your data make sense",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #182238;
    --muted: #737d91;
    --purple: #6658df;
    --purple-dark: #4e42bf;
    --line: #e7eaf2;
    --surface: #ffffff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7f8fc;
    color: var(--ink);
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #17192f;
    border-right: 1px solid #292b43;
}

[data-testid="stSidebar"] * {
    color: #f7f7ff;
}

[data-testid="stSidebar"] .stCaption {
    color: #b9bdd5;
}

[data-testid="stSidebar"] hr {
    border-color: #34364f;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #20223a;
    padding: 12px;
    border: 1px solid #383a56;
    border-radius: 12px;
}

/* Brand */
.brand-mark {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.065em;
    color: white;
    line-height: 1.1;
}

.brand-caption {
    color: #b9bdd5;
    font-size: 0.82rem;
    margin-top: 7px;
    line-height: 1.5;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #17192f 0%, #302c68 58%, #6658df 100%);
    color: white;
    padding: 42px 42px;
    border-radius: 24px;
    margin: 0 0 26px 0;
    box-shadow: 0 14px 35px rgba(40, 36, 100, 0.12);
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    right: -75px;
    top: -95px;
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 50%;
    box-shadow: 0 0 0 30px rgba(255,255,255,0.035),
                0 0 0 65px rgba(255,255,255,0.025);
    pointer-events: none;
}

.hero .eyebrow {
    color: #d4d0ff;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero h1 {
    color: white !important;
    font-size: clamp(2.3rem, 4vw, 3.4rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.055em;
    margin: 0 0 10px 0;
}

.hero .tagline {
    color: white;
    font-size: clamp(1.15rem, 2vw, 1.45rem);
    font-weight: 600;
    margin-bottom: 12px;
}

.hero p {
    color: #e5e6ff;
    font-size: 1rem;
    line-height: 1.75;
    max-width: 720px;
    margin: 0;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 19px 21px;
    box-shadow: 0 4px 14px rgba(24, 34, 56, 0.025);
}

div[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-size: 0.86rem;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: var(--ink);
    font-weight: 800;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    background: transparent;
    border-radius: 10px 10px 0 0;
    padding: 0 18px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: var(--purple);
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
    transition: all 0.15s ease;
}

.stButton > button[kind="primary"] {
    background: var(--purple);
    border: 1px solid var(--purple);
}

.stButton > button[kind="primary"]:hover {
    background: var(--purple-dark);
    border-color: var(--purple-dark);
}

/* Inputs and tables */
.stTextArea textarea,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 10px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
}

.section-description {
    color: var(--muted);
    margin-top: -8px;
    margin-bottom: 20px;
}

.welcome-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 25px;
}

footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
    }

    .hero {
        padding: 28px 22px;
        border-radius: 17px;
    }
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. SETTINGS, FILES, AND DATA HELPERS
# ============================================================
def setting(name, default=None):
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def read_file(file):
    filename = file.name.lower()
    file.seek(0)

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(file, low_memory=False)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(
                file,
                encoding="latin-1",
                low_memory=False,
            )

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raise ValueError("Please upload a CSV, XLSX, or XLS file.")


def load_sample_data():
    """Load the optional sample CSV stored beside app.py."""
    sample_path = BASE_DIR / "sales_transactions.csv"

    if not sample_path.exists():
        raise FileNotFoundError(
            "sales_transactions.csv was not found in the app repository."
        )

    return pd.read_csv(sample_path, low_memory=False)


def numeric_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def norm(value):
    value = str(value).strip().lower()
    value = re.sub(r"[_./\\\-]+", " ", value)
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


ALIASES = {
    "revenue": [
        "revenue", "sales", "sales amount", "sales value",
        "net sales", "total revenue", "turnover", "amount",
        "total amount", "line total"
    ],
    "profit": [
        "profit", "net profit", "gross profit",
        "profit amount", "margin amount"
    ],
    "cost": [
        "cost", "total cost", "expense", "expenses",
        "cogs", "cost of goods sold"
    ],
    "quantity": [
        "quantity", "qty", "units", "units sold",
        "quantity sold", "volume"
    ],
    "date": [
        "date", "order date", "transaction date",
        "invoice date", "created at", "timestamp", "period", "month"
    ],
    "product": [
        "product", "product name", "item", "item name", "sku", "service"
    ],
    "customer": [
        "customer", "customer name", "customer id", "client", "buyer"
    ],
    "region": [
        "region", "area", "territory", "country",
        "city", "location", "branch"
    ],
    "category": [
        "category", "product category", "department", "segment", "product type"
    ],
    "order": [
        "order id", "order number", "transaction id",
        "invoice id", "receipt id"
    ],
}


def detect_types(df):
    numeric, categorical, dates = [], [], []

    for col in df.columns:
        s = df[col]
        name = norm(col)

        if pd.api.types.is_datetime64_any_dtype(s):
            dates.append(col)

        elif any(
            x in name.split()
            for x in ["date", "datetime", "timestamp", "period"]
        ):
            parsed = pd.to_datetime(s, errors="coerce")

            if len(s) and parsed.notna().mean() >= 0.5:
                dates.append(col)
            elif pd.api.types.is_numeric_dtype(s):
                numeric.append(col)
            else:
                categorical.append(col)

        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)

        else:
            converted = numeric_series(s)

            if len(s.dropna()) and converted.notna().mean() >= 0.85:
                numeric.append(col)
            else:
                parsed = pd.to_datetime(s, errors="coerce")

                if (
                    len(s.dropna()) >= 3
                    and parsed.notna().mean() >= 0.9
                    and s.nunique(dropna=True) > 2
                ):
                    dates.append(col)
                else:
                    categorical.append(col)

    return numeric, categorical, dates


def infer_roles(df):
    numeric, categorical, dates = detect_types(df)
    roles = {}

    for role, aliases in ALIASES.items():
        matches = []

        for col in df.columns:
            name = norm(col)

            score = max([
                100 if name == norm(alias)
                else 80 if norm(alias) in name
                else 0
                for alias in aliases
            ] or [0])

            if role in {"revenue", "profit", "cost", "quantity"} and col not in numeric:
                score = 0

            if role == "date" and col not in dates:
                score = 0

            if role not in {
                "revenue", "profit", "cost", "quantity", "date"
            } and col in numeric:
                score = min(score, 40)

            if role == "revenue" and any(
                x in name for x in ["profit", "cost", "margin"]
            ):
                score = 0

            if score:
                matches.append((score, col))

        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            roles[role] = matches[0][1]

    if dates and "date" not in roles:
        roles["date"] = dates[0]

    return roles


def automatic_review(df):
    numeric, categorical, dates = detect_types(df)
    roles = infer_roles(df)

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    insights = [
        f"Dataset contains **{len(df):,} rows** and **{len(df.columns):,} columns**.",
        f"Missing cells: **{missing:,}** ({missing / max(df.size, 1) * 100:.2f}%).",
        f"Fully duplicated rows: **{duplicates:,}**.",
        f"Detected {len(numeric)} numeric, {len(categorical)} categorical, and {len(dates)} date-like columns.",
    ]

    if roles:
        insights.append(
            "Likely business fields: "
            + ", ".join(f"`{k}` → `{v}`" for k, v in roles.items())
            + "."
        )

    for col in numeric[:4]:
        s = numeric_series(df[col]).dropna()

        if len(s):
            insights.append(
                f"`{col}` — total **{s.sum():,.2f}**, "
                f"average **{s.mean():,.2f}**, median **{s.median():,.2f}**."
            )

    if roles.get("date"):
        ds = pd.to_datetime(
            df[roles["date"]],
            errors="coerce",
        ).dropna()

        if len(ds):
            insights.append(
                f"Date coverage: **{ds.min().date()}** to **{ds.max().date()}**."
            )

    return {
        "numeric": numeric,
        "categorical": categorical,
        "dates": dates,
        "roles": roles,
        "insights": insights,
    }


def schema_for_llm(df):
    schema = []

    for col in df.columns:
        s = df[col]

        schema.append({
            "column": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
            "examples": [
                str(v)[:80]
                for v in s.dropna().head(3).tolist()
            ],
        })

    return schema


# ============================================================
# 3. BUSINESS ADVISOR: EVIDENCE-BASED OPPORTUNITY ANALYSIS
# ============================================================
def business_opportunity_analysis(df):
    """
    Calculate potential business opportunities from available data.

    This function calculates evidence using Pandas. The LLM interprets
    the evidence later; it does not invent or calculate the metrics.
    """
    review = automatic_review(df)
    roles = review["roles"]

    revenue_col = roles.get("revenue")
    profit_col = roles.get("profit")
    quantity_col = roles.get("quantity")
    date_col = roles.get("date")

    dimensions = []

    for role in ["product", "category", "region", "customer"]:
        col = roles.get(role)

        if col and col in df.columns and col not in dimensions:
            dimensions.append(col)

    result = {
        "answer": "",
        "table": None,
        "chart": None,
        "data": df,
        "evidence_tables": {},
        "key_findings": [],
        "available_fields": {
            "revenue": revenue_col,
            "profit": profit_col,
            "quantity": quantity_col,
            "date": date_col,
            "dimensions": dimensions,
        },
    }

    if not revenue_col:
        result["answer"] = (
            "I couldn't confidently identify a revenue or sales-value column. "
            "I can still inspect your data, but I can't responsibly identify "
            "revenue-growth priorities without a suitable revenue measure. "
            "Check the column names or rename the relevant field to something "
            "like Revenue, Sales, or Sales Amount."
        )

        if dimensions:
            result["key_findings"].append(
                "Potential grouping fields were detected: "
                + ", ".join(str(c) for c in dimensions)
                + "."
            )

        return result

    revenue = numeric_series(df[revenue_col])
    valid_revenue = revenue.dropna()

    if valid_revenue.empty:
        result["answer"] = (
            f"The column `{revenue_col}` was identified as revenue, "
            "but it contains no usable numeric values."
        )
        return result

    total_revenue = float(valid_revenue.sum())
    average_revenue = float(valid_revenue.mean())
    median_revenue = float(valid_revenue.median())

    result["key_findings"].append(
        f"Total of the identified revenue field `{revenue_col}`: "
        f"{total_revenue:,.2f} across {len(valid_revenue):,} non-missing rows."
    )
    result["key_findings"].append(
        f"Average row-level value for `{revenue_col}`: {average_revenue:,.2f}; "
        f"median: {median_revenue:,.2f}. These are row-level values, "
        "not necessarily average order values."
    )

    summary = {
        "Metric": ["Total revenue", "Average row-level revenue", "Median row-level revenue"],
        "Value": [total_revenue, average_revenue, median_revenue],
    }
    result["evidence_tables"]["Revenue summary"] = pd.DataFrame(summary)

    if profit_col:
        profit = numeric_series(df[profit_col])
        valid_profit = profit.dropna()

        if len(valid_profit):
            total_profit = float(valid_profit.sum())
            result["key_findings"].append(
                f"Total of the identified profit field `{profit_col}`: "
                f"{total_profit:,.2f}."
            )

            result["evidence_tables"]["Profit summary"] = pd.DataFrame({
                "Metric": ["Total profit", "Average row-level profit"],
                "Value": [
                    total_profit,
                    float(valid_profit.mean()),
                ],
            })

            # Revenue and profit may be on different rows or currencies;
            # this is a simple dataset-level ratio, not a verified accounting margin.
            if total_revenue != 0:
                result["key_findings"].append(
                    f"Dataset-level profit-to-revenue ratio: "
                    f"{total_profit / total_revenue * 100:.2f}%, "
                    "calculated from the identified fields. Confirm the "
                    "fields use consistent periods, currencies, and accounting definitions."
                )

    if quantity_col:
        quantity = numeric_series(df[quantity_col])
        valid_quantity = quantity.dropna()

        if len(valid_quantity):
            result["key_findings"].append(
                f"Total of the identified quantity field `{quantity_col}`: "
                f"{valid_quantity.sum():,.2f}."
            )

    # Compare performance across useful business dimensions.
    for dimension in dimensions:
        work = pd.DataFrame({
            "_dimension": df[dimension].astype("string").fillna("(Missing)"),
            "_revenue": revenue,
        })

        if profit_col:
            work["_profit"] = numeric_series(df[profit_col])

        if quantity_col:
            work["_quantity"] = numeric_series(df[quantity_col])

        aggregations = {"_revenue": "sum"}

        if profit_col:
            aggregations["_profit"] = "sum"

        if quantity_col:
            aggregations["_quantity"] = "sum"

        grouped = (
            work.groupby("_dimension", dropna=False)
            .agg(aggregations)
            .reset_index()
            .rename(columns={
                "_dimension": dimension,
                "_revenue": "Revenue",
                "_profit": "Profit",
                "_quantity": "Quantity",
            })
        )

        grouped = grouped.sort_values(
            "Revenue",
            ascending=False,
        )

        result["evidence_tables"][f"Revenue by {dimension}"] = grouped.head(30)

        nonmissing_grouped = grouped[
            grouped[dimension] != "(Missing)"
        ]

        if not nonmissing_grouped.empty:
            top = nonmissing_grouped.iloc[0]
            result["key_findings"].append(
                f"Highest-revenue `{dimension}` in this dataset: "
                f"{top[dimension]} with revenue of {top['Revenue']:,.2f}."
            )

            bottom = nonmissing_grouped.sort_values(
                "Revenue",
                ascending=True,
            ).iloc[0]

            if len(nonmissing_grouped) > 1:
                result["key_findings"].append(
                    f"Lowest-revenue `{dimension}` among the observed groups: "
                    f"{bottom[dimension]} with revenue of {bottom['Revenue']:,.2f}. "
                    "Low revenue alone does not establish that this group has "
                    "untapped potential; group size and business context matter."
                )

    # Analyze trends and recent performance where a usable date column exists.
    if date_col:
        work = pd.DataFrame({
            "_date": pd.to_datetime(df[date_col], errors="coerce"),
            "_revenue": revenue,
        }).dropna()

        if not work.empty:
            work["_period"] = work["_date"].dt.to_period("M").dt.to_timestamp()

            monthly = (
                work.groupby("_period")["_revenue"]
                .sum()
                .reset_index()
                .rename(columns={
                    "_period": "Month",
                    "_revenue": "Revenue",
                })
                .sort_values("Month")
            )

            result["evidence_tables"]["Monthly revenue"] = monthly.tail(36)

            if len(monthly) >= 2:
                latest = float(monthly.iloc[-1]["Revenue"])
                previous = float(monthly.iloc[-2]["Revenue"])

                if previous != 0:
                    change_pct = (latest - previous) / abs(previous) * 100

                    result["key_findings"].append(
                        f"Revenue in the latest observed month "
                        f"({monthly.iloc[-1]['Month']:%Y-%m}) was {latest:,.2f}, "
                        f"compared with {previous:,.2f} in the preceding observed "
                        f"month, a change of {change_pct:+.2f}%."
                    )
                else:
                    result["key_findings"].append(
                        f"Revenue in the latest observed month was {latest:,.2f}; "
                        "the preceding observed month had zero revenue, so a "
                        "percentage change cannot be calculated."
                    )

                # Compare the last three observed months with the previous
                # three observed months. These are observed periods, not
                # necessarily consecutive calendar months.
                if len(monthly) >= 6:
                    recent = float(monthly.tail(3)["Revenue"].sum())
                    previous_three = float(
                        monthly.iloc[-6:-3]["Revenue"].sum()
                    )

                    if previous_three != 0:
                        recent_change = (
                            (recent - previous_three)
                            / abs(previous_three)
                            * 100
                        )

                        result["key_findings"].append(
                            f"Revenue across the latest three observed months "
                            f"was {recent:,.2f}, compared with "
                            f"{previous_three:,.2f} across the preceding three "
                            f"observed months ({recent_change:+.2f}%)."
                        )

            result["chart"] = px.line(
                monthly.tail(24),
                x="Month",
                y="Revenue",
                markers=True,
                title="Observed monthly revenue",
            )

    if not dimensions and not date_col:
        result["key_findings"].append(
            "No suitable product, category, region, customer, or date field "
            "was detected for segmentation or trend analysis. The current "
            "dataset supports a basic revenue summary, but more dimensions "
            "would help identify actionable growth opportunities."
        )

    result["answer"] = (
        "## Business opportunity analysis\n\n"
        "I analyzed the available revenue data and looked for useful breakdowns "
        "by business dimension and time period. The evidence below is based on "
        "the uploaded dataset. It identifies patterns to investigate, not "
        "guaranteed causes or guaranteed revenue gains.\n\n"
        + "\n".join(f"- {finding}" for finding in result["key_findings"])
    )

    return result


def business_advice_with_llm(question, result):
    """
    Ask the LLM to interpret pre-calculated evidence and prioritize actions.
    The model must distinguish observed facts from hypotheses.
    """
    llm = get_llm()

    evidence_tables = {}

    for name, table in result.get("evidence_tables", {}).items():
        if isinstance(table, pd.DataFrame):
            evidence_tables[name] = table.head(30).to_dict(
                orient="records"
            )

    payload = {
        "business_question": question,
        "calculated_findings": result.get("key_findings", []),
        "available_fields": result.get("available_fields", {}),
        "evidence_tables": evidence_tables,
    }

    response = llm.invoke([
        SystemMessage(content="""
You are DEXCR's Business Advisor, an evidence-based business intelligence analyst.

The user wants to improve business performance. You have been given findings and
tables calculated from their actual dataset. Use these as your evidence.

Your job:
1. Answer the user's business question directly.
2. Prioritize up to three actions, ranking them by how strongly the data supports
   investigating them. Explain why each action deserves attention.
3. Cite the exact figures or group names from the supplied evidence when relevant.
4. Explain the possible business mechanism: for example, improving product mix,
   investigating a declining region, retaining customers, or reviewing pricing.
5. Separate observed facts from hypotheses. The data may show where performance
   differs without explaining why it differs.
6. Do not invent numbers, columns, customers, products, causes, or expected revenue
   gains. Do not promise a particular financial outcome.
7. If the dataset lacks profit, customer, product, region, date, or other relevant
   fields, state the limitation and explain what additional data would help.
8. Do not assume that the lowest-performing group is automatically the best
   growth opportunity. Consider size, trend, profitability, and available evidence.
9. End with a short list of practical next steps or questions to validate the
   recommendation.

Use this format:

### Recommendation
A concise direct answer.

### What the data tells us
The strongest relevant findings with supporting numbers.

### Priorities to investigate
Ranked actions. For each, state why, the supporting evidence, and what to check.

### What we cannot conclude yet
Important data limitations and hypotheses that require validation.

### Recommended next step
One concrete next analysis or business experiment.

Be clear, specific, and concise. If no revenue column was detected, do not pretend
that a revenue-growth analysis was completed.
"""),
        HumanMessage(
            content=json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
        ),
    ])

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    return str(content)


# ============================================================
# 4. LANGCHAIN LLM: QWEN OR OPENAI
# ============================================================
def get_llm():
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError(
            "LangChain is missing. Add langchain-core and langchain-openai to requirements.txt."
        )

    provider = str(setting("LLM_PROVIDER", "qwen")).lower()

    if provider == "openai":
        api_key = setting("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY in your Streamlit Secrets."
            )

        return ChatOpenAI(
            model=setting("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=0,
            timeout=45,
            max_retries=2,
        )

    if provider == "qwen":
        api_key = setting("LLM_API_KEY") or setting("DASHSCOPE_API_KEY")

        if not api_key:
            raise ValueError(
                "Set LLM_API_KEY in Streamlit Secrets."
            )

        return ChatOpenAI(
            model=setting("LLM_MODEL", "qwen-plus"),
            api_key=api_key,
            base_url=setting(
                "LLM_BASE_URL",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            ),
            temperature=0,
            timeout=45,
            max_retries=2,
        )

    raise ValueError("LLM_PROVIDER must be 'qwen' or 'openai'.")


def get_llm_plan(question, df):
    llm = get_llm()
    review = automatic_review(df)

    system = """
You are DEXCR, a business intelligence question interpreter.

Translate the question into ONE JSON object that describes an analysis.
Never calculate results. Use only exact column names from the supplied schema.
Do not return Python, SQL, or markdown.

Operations:
- business_advisor: strategic business questions asking what to focus on,
  how to increase revenue or profit, improve performance, find opportunities,
  or understand what action to take
- profile: dataset summary
- columns: list columns
- missing: missing-data analysis
- duplicates: duplicate-row count
- show_rows: show sample rows
- aggregate: aggregate one metric, optionally by group
- rank: rank an entity by an aggregated metric
- trend: summarize a metric over time
- forecast: basic time-series forecast
- distribution: distribution of a column
- correlation: correlations among numeric columns
- filter: filter rows
- unknown: unsupported or underspecified request

Use business_advisor for questions such as:
"What should I focus on to increase revenue?"
"How can I improve sales?"
"Where are my biggest business opportunities?"
"How can I improve profitability?"
"What should I do to grow my business?"
"What actions should I take based on this data?"

Do NOT map strategic questions to a simple rank or aggregate just because they
mention revenue or sales. Use business_advisor when the user asks for advice,
priorities, opportunities, or actions.

Aggregations: sum, mean, median, min, max, count, nunique, std.
Filter operators: eq, ne, gt, gte, lt, lte, contains, between.
Best-selling means quantity/units if available. Highest revenue uses revenue/sales.
Most profitable requires a profit column; never substitute revenue.
For trend/forecast, select a real date column and numeric metric.

Return keys:
operation, metric_column, entity_column, aggregation, group_by, direction,
top_n, date_column, frequency, periods, column, columns, filters, answer_note.

For filters, return a list of objects:
{"column": "...", "operator": "...", "value": "..."}.
"""

    request = {
        "question": question,
        "columns": schema_for_llm(df),
        "detected_roles": {
            k: str(v)
            for k, v in review["roles"].items()
        },
    }

    response = llm.bind(
        response_format={"type": "json_object"}
    ).invoke([
        SystemMessage(content=system),
        HumanMessage(
            content=json.dumps(
                request,
                ensure_ascii=False,
                default=str,
            )
        ),
    ])

    content = response.content

    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    plan = json.loads(content)

    if not isinstance(plan, dict):
        raise ValueError("The model did not return a JSON object.")

    return plan


def explain_with_llm(question, plan, result):
    try:
        llm = get_llm()
        table = result.get("table")

        records = (
            table.head(20).to_dict(orient="records")
            if isinstance(table, pd.DataFrame)
            else None
        )

        payload = {
            "question": question,
            "plan": plan,
            "calculated_result": result.get("answer", ""),
            "table_preview": records,
        }

        response = llm.invoke([
            SystemMessage(content="""You are DEXCR, a business intelligence analyst.
Explain the supplied Python-calculated result in clear business language.
Lead with the answer, then explain its meaning and offer one useful next question.
Use only supplied evidence. Never invent numbers or claim causation from correlation.
Keep it concise."""),
            HumanMessage(
                content=json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                )
            ),
        ])

        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        return str(content)

    except Exception as exc:
        return (
            f"AI explanation could not be generated "
            f"({type(exc).__name__}: {exc}). "
            "The calculated result is still available below."
        )


# ============================================================
# 5. SAFE PLAN VALIDATION
# ============================================================
OPERATIONS = {
    "business_advisor",
    "profile",
    "columns",
    "missing",
    "duplicates",
    "show_rows",
    "aggregate",
    "rank",
    "trend",
    "forecast",
    "distribution",
    "correlation",
    "filter",
    "unknown",
}

AGGS = {
    "sum", "mean", "median", "min", "max",
    "count", "nunique", "std",
}

OPS = {
    "eq", "ne", "gt", "gte", "lt", "lte",
    "contains", "between",
}


def validate_plan(plan, df):
    if not isinstance(plan, dict):
        raise ValueError("Analysis plan is not a dictionary.")

    p = dict(plan)
    p["operation"] = str(
        p.get("operation", "unknown")
    ).lower()

    if p["operation"] not in OPERATIONS:
        p["operation"] = "unknown"

    p["aggregation"] = str(
        p.get("aggregation", "sum")
    ).lower()

    if p["aggregation"] not in AGGS:
        p["aggregation"] = "sum"

    p["direction"] = (
        "asc"
        if str(p.get("direction", "desc")).lower() == "asc"
        else "desc"
    )

    try:
        p["top_n"] = max(
            1,
            min(int(p.get("top_n", 10)), 100),
        )
    except (ValueError, TypeError):
        p["top_n"] = 10

    try:
        p["periods"] = max(
            1,
            min(int(p.get("periods", 6)), 24),
        )
    except (ValueError, TypeError):
        p["periods"] = 6

    p["frequency"] = str(
        p.get("frequency", "month")
    ).lower()

    if p["frequency"] not in {
        "day", "week", "month", "quarter", "year"
    }:
        p["frequency"] = "month"

    for key in [
        "metric_column", "entity_column",
        "date_column", "column",
    ]:
        value = p.get(key)

        if value not in df.columns:
            p[key] = None

    for key in ["group_by", "columns"]:
        value = p.get(key, [])

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            value = []

        p[key] = [
            x for x in value
            if x in df.columns
        ]

    filters = p.get("filters", [])
    p["filters"] = []

    if isinstance(filters, list):
        for f in filters[:10]:
            if (
                not isinstance(f, dict)
                or f.get("column") not in df.columns
            ):
                continue

            op = str(
                f.get("operator", "eq")
            ).lower()

            if op not in OPS:
                continue

            p["filters"].append({
                "column": f["column"],
                "operator": op,
                "value": f.get("value"),
            })

    return p


# ============================================================
# 6. LOCAL INTERPRETER FALLBACK
# ============================================================
def local_plan(question, df):
    q = norm(question)
    review = automatic_review(df)
    roles = review["roles"]
    numeric = review["numeric"]
    categorical = review["categorical"]
    dates = review["dates"]

    plan = {
        "operation": "unknown",
        "metric_column": None,
        "entity_column": None,
        "aggregation": "sum",
        "group_by": [],
        "direction": "desc",
        "top_n": 10,
        "date_column": roles.get("date"),
        "frequency": "month",
        "periods": 6,
        "column": None,
        "columns": [],
        "filters": [],
        "answer_note": "",
    }

    def has(*words):
        return any(norm(word) in q for word in words)

    mentioned = next(
        (
            col for col in df.columns
            if norm(col) and norm(col) in q
        ),
        None,
    )

    metric = (
        roles.get("revenue")
        or roles.get("quantity")
        or (numeric[0] if numeric else None)
    )

    if has("profit", "profitable", "profitability"):
        metric = roles.get("profit")
    elif has(
        "quantity", "units sold", "volume",
        "best selling", "bestselling",
    ):
        metric = roles.get("quantity") or metric
    elif has("cost", "expenses"):
        metric = roles.get("cost") or metric

    plan["metric_column"] = metric

    entity = roles.get("product")

    if has("customer", "client", "buyer"):
        entity = roles.get("customer") or entity
    elif has(
        "region", "country", "city",
        "territory", "branch",
    ):
        entity = roles.get("region") or entity
    elif has("category", "department", "segment"):
        entity = roles.get("category") or entity

    plan["entity_column"] = entity

    # Strategic questions should use the Business Advisor, not just a ranking.
    business_advisor_phrases = [
        "increase revenue",
        "grow revenue",
        "grow my business",
        "increase sales",
        "improve sales",
        "improve revenue",
        "improve profitability",
        "increase profit",
        "grow profit",
        "business opportunity",
        "business opportunities",
        "what should i focus on",
        "what should we focus on",
        "what should i do",
        "what should we do",
        "what actions",
        "recommendation",
        "recommendations",
        "how can i improve",
        "how can we improve",
        "how can i increase",
        "how can we increase",
        "how do i increase",
        "how do we increase",
        "where should i focus",
        "where should we focus",
        "prioritize",
        "priorities",
        "action plan",
        "business strategy",
        "growth strategy",
        "growth opportunities",
        "improve performance",
        "improve business performance",
        "increase my revenue",
        "increase our revenue",
        "increase our sales",
        "increase my sales",
        "help me grow",
        "help us grow",
    ]

    is_business_advisor_question = any(
        phrase in q
        for phrase in business_advisor_phrases
    )

    if is_business_advisor_question:
        plan["operation"] = "business_advisor"

    elif has(
        "column names", "list columns",
        "what columns", "available columns",
    ):
        plan["operation"] = "columns"

    elif has(
        "missing", "null values",
        "data quality", "incomplete data",
    ):
        plan["operation"] = "missing"

    elif has("duplicate", "duplicates"):
        plan["operation"] = "duplicates"

    elif has(
        "profile", "overview", "summarize dataset",
        "summary of dataset", "analyze my data",
        "analyse my data",
    ):
        plan["operation"] = "profile"

    elif has("correlation", "correlated", "relationship between"):
        plan["operation"] = "correlation"
        plan["columns"] = numeric

    elif has(
        "forecast", "predict", "future",
        "next month", "next year",
    ):
        plan["operation"] = "forecast"
        plan["date_column"] = (
            roles.get("date")
            or (dates[0] if dates else None)
        )

        nums = re.findall(r"\b(\d{1,2})\b", q)

        if nums:
            plan["periods"] = min(
                max(int(nums[0]), 1),
                24,
            )

        for freq, words in {
            "day": ["daily", "by day"],
            "week": ["weekly", "by week"],
            "quarter": ["quarterly", "quarter"],
            "year": ["yearly", "annual", "by year"],
        }.items():
            if has(*words):
                plan["frequency"] = freq

    elif has(
        "trend", "over time", "by month",
        "monthly", "by year", "sales over time",
    ):
        plan["operation"] = "trend"
        plan["date_column"] = (
            roles.get("date")
            or (dates[0] if dates else None)
        )

        if has("weekly", "by week"):
            plan["frequency"] = "week"
        elif has("quarter", "quarterly"):
            plan["frequency"] = "quarter"
        elif has("yearly", "by year", "annual"):
            plan["frequency"] = "year"
        elif has("daily", "by day"):
            plan["frequency"] = "day"

    elif has("distribution", "histogram", "spread of"):
        plan["operation"] = "distribution"
        plan["column"] = mentioned or metric

    elif has(
        "show rows", "show records",
        "preview", "sample rows", "first rows",
    ):
        plan["operation"] = "show_rows"

    elif has(
        "top", "highest", "best", "most",
        "rank", "leading", "lowest", "least",
    ):
        plan["operation"] = "rank"

        if has("lowest", "least", "smallest", "worst"):
            plan["direction"] = "asc"

        nums = re.findall(r"\btop\s+(\d+)", q)

        if nums:
            plan["top_n"] = min(
                max(int(nums[0]), 1),
                100,
            )

    elif has("average", "mean"):
        plan["operation"] = "aggregate"
        plan["aggregation"] = "mean"

    elif has("median"):
        plan["operation"] = "aggregate"
        plan["aggregation"] = "median"

    elif has("count", "how many"):
        plan["operation"] = "aggregate"
        plan["aggregation"] = "count"

    elif has("total", "sum", "overall"):
        plan["operation"] = "aggregate"
        plan["aggregation"] = "sum"

    elif mentioned:
        plan["operation"] = "distribution"
        plan["column"] = mentioned

    if mentioned and mentioned in numeric:
        plan["metric_column"] = mentioned

    if mentioned and mentioned in categorical:
        plan["entity_column"] = mentioned

        if plan["operation"] == "unknown":
            plan["operation"] = "distribution"
            plan["column"] = mentioned

    if plan["operation"] == "aggregate" and entity:
        plan["group_by"] = [entity]

    if plan["operation"] == "unknown":
        plan["answer_note"] = (
            "I couldn't confidently map that question to an analysis. "
            "Try asking for total revenue, top products, monthly trends, "
            "missing values, a forecast, or a business recommendation such "
            "as 'What should I focus on to increase revenue?'"
        )

    return validate_plan(plan, df)


# ============================================================
# 7. FILTERS AND ANALYSIS EXECUTION
# ============================================================
def apply_filters(df, filters):
    result = df.copy()

    for f in filters:
        col, op, value = (
            f["column"],
            f["operator"],
            f["value"],
        )
        s = result[col]

        try:
            if op in {"gt", "gte", "lt", "lte", "between"}:
                ns = numeric_series(s)

                if op == "gt":
                    mask = ns > float(value)
                elif op == "gte":
                    mask = ns >= float(value)
                elif op == "lt":
                    mask = ns < float(value)
                elif op == "lte":
                    mask = ns <= float(value)
                else:
                    bounds = (
                        value
                        if isinstance(value, list)
                        else str(value).split(",")
                    )

                    if len(bounds) != 2:
                        continue

                    mask = ns.between(
                        float(bounds[0]),
                        float(bounds[1]),
                    )

            elif op == "contains":
                mask = s.astype("string").str.contains(
                    str(value),
                    case=False,
                    na=False,
                    regex=False,
                )

            elif op == "eq":
                mask = (
                    s.astype("string").str.lower()
                    == str(value).lower()
                )

            elif op == "ne":
                mask = (
                    s.astype("string").str.lower()
                    != str(value).lower()
                )

            else:
                continue

            result = result.loc[mask.fillna(False)]

        except (ValueError, TypeError):
            continue

    return result


def execute_plan(df, plan):
    op = plan["operation"]
    data = apply_filters(df, plan.get("filters", []))

    result = {
        "answer": "",
        "table": None,
        "chart": None,
        "data": data,
    }

    # New strategic analysis operation.
    if op == "business_advisor":
        return business_opportunity_analysis(data)

    if op == "unknown":
        result["answer"] = (
            plan.get("answer_note")
            or "I could not determine the right analysis. Please rephrase the question."
        )

    elif op == "columns":
        result["answer"] = (
            f"The dataset has {len(df.columns)} columns."
        )
        result["table"] = pd.DataFrame({
            "Column": df.columns,
            "Data type": [
                str(df[c].dtype)
                for c in df.columns
            ],
        })

    elif op == "profile":
        review = automatic_review(data)
        result["answer"] = "\n\n".join(review["insights"])
        result["table"] = pd.DataFrame({
            "Column": data.columns,
            "Data type": [
                str(data[c].dtype)
                for c in data.columns
            ],
            "Missing": [
                int(data[c].isna().sum())
                for c in data.columns
            ],
            "Unique": [
                int(data[c].nunique(dropna=True))
                for c in data.columns
            ],
        })

    elif op == "missing":
        table = pd.DataFrame({
            "Column": data.columns,
            "Missing": [
                int(data[c].isna().sum())
                for c in data.columns
            ],
            "Missing %": [
                round(data[c].isna().mean() * 100, 2)
                for c in data.columns
            ],
        })

        result["table"] = table.sort_values(
            "Missing %",
            ascending=False,
        )
        result["answer"] = (
            f"{int(data.isna().sum().sum()):,} missing cells "
            f"across {len(data):,} rows."
        )
        result["chart"] = px.bar(
            result["table"],
            x="Column",
            y="Missing %",
            title="Missing values by column",
        )

    elif op == "duplicates":
        n = int(data.duplicated().sum())
        result["answer"] = (
            f"Found **{n:,} fully duplicated rows** "
            "in the current dataset."
        )
        result["table"] = data[
            data.duplicated(keep=False)
        ].head(100)

    elif op == "show_rows":
        result["answer"] = (
            f"Showing up to 100 of {len(data):,} rows."
        )
        result["table"] = data.head(100)

    elif op == "filter":
        result["answer"] = (
            f"Filters returned {len(data):,} rows "
            f"out of {len(df):,}."
        )
        result["table"] = data.head(500)

    elif op == "aggregate":
        metric = plan.get("metric_column")

        if not metric:
            result["answer"] = (
                "I couldn't identify a numeric measure for this aggregation."
            )

        elif plan["aggregation"] == "count":
            if plan.get("group_by"):
                result["table"] = (
                    data.groupby(
                        plan["group_by"],
                        dropna=False,
                    )[metric]
                    .count()
                    .reset_index(name="count")
                )
            else:
                result["answer"] = (
                    f"Non-missing `{metric}` values: "
                    f"{data[metric].count():,}."
                )

        else:
            values = numeric_series(data[metric])

            if plan.get("group_by"):
                work = data[plan["group_by"]].copy()
                work["_metric"] = values

                agg = getattr(
                    work.groupby(
                        plan["group_by"],
                        dropna=False,
                    )["_metric"],
                    plan["aggregation"],
                    None,
                )

                if agg is None:
                    result["answer"] = "Unsupported aggregation."
                else:
                    result["table"] = agg().reset_index(
                        name=f"{plan['aggregation']}_{metric}"
                    )
                    result["table"] = result["table"].sort_values(
                        result["table"].columns[-1],
                        ascending=False,
                    )
                    result["chart"] = px.bar(
                        result["table"].head(30),
                        x=plan["group_by"][0],
                        y=result["table"].columns[-1],
                        title=(
                            f"{plan['aggregation'].title()} of {metric} "
                            f"by {plan['group_by'][0]}"
                        ),
                    )
            else:
                val = getattr(
                    values,
                    plan["aggregation"],
                )()
                result["answer"] = (
                    f"{plan['aggregation'].title()} of `{metric}`: "
                    f"**{val:,.2f}**."
                )

    elif op == "rank":
        entity = plan.get("entity_column")
        metric = plan.get("metric_column")

        if not entity or not metric:
            result["answer"] = (
                "I couldn't identify both an entity (such as product) "
                "and a measure (such as sales or quantity)."
            )
        else:
            work = data[[entity]].copy()
            work["_metric"] = numeric_series(data[metric])

            grouped = (
                work.groupby(
                    entity,
                    dropna=False,
                )["_metric"]
                .agg(plan["aggregation"])
                .reset_index(
                    name=f"{plan['aggregation']}_{metric}"
                )
            )

            grouped = grouped.sort_values(
                grouped.columns[-1],
                ascending=plan["direction"] == "asc",
            ).head(plan["top_n"])

            result["table"] = grouped
            label = (
                "lowest"
                if plan["direction"] == "asc"
                else "highest"
            )
            result["answer"] = (
                f"Showing the {label}-ranked `{entity}` values "
                f"by {plan['aggregation']} of `{metric}`."
            )
            result["chart"] = px.bar(
                grouped,
                x=entity,
                y=grouped.columns[-1],
                title=f"{label.title()} {entity} by {metric}",
            )

    elif op in {"trend", "forecast"}:
        date_col = plan.get("date_column")
        metric = plan.get("metric_column")

        if not date_col or not metric:
            result["answer"] = (
                "A time analysis needs a date column and a numeric measure. "
                "I couldn't identify both."
            )
        else:
            work = pd.DataFrame({
                "_date": pd.to_datetime(
                    data[date_col],
                    errors="coerce",
                ),
                "_value": numeric_series(data[metric]),
            }).dropna()

            if work.empty:
                result["answer"] = (
                    "No valid date/metric rows were available for time analysis."
                )
            else:
                freq_map = {
                    "day": "D",
                    "week": "W",
                    "month": "MS",
                    "quarter": "QS",
                    "year": "YS",
                }
                freq = freq_map[plan["frequency"]]
                work = work.set_index("_date").sort_index()

                if plan["aggregation"] == "mean":
                    series = work["_value"].resample(freq).mean()
                elif plan["aggregation"] == "count":
                    series = work["_value"].resample(freq).count()
                else:
                    series = work["_value"].resample(freq).sum()

                series = series.dropna()
                history = series.rename("actual").reset_index()
                history = history.rename(
                    columns={"_date": "period"}
                )

                if op == "trend":
                    result["table"] = history
                    result["answer"] = (
                        f"Trend for `{metric}` grouped by "
                        f"{plan['frequency']} ({len(history)} periods)."
                    )
                    result["chart"] = px.line(
                        history,
                        x="period",
                        y="actual",
                        markers=True,
                        title=(
                            f"{plan['frequency'].title()} "
                            f"{metric} trend"
                        ),
                    )

                elif len(series) < 2:
                    result["answer"] = (
                        "At least two non-empty time periods are needed "
                        "for a basic forecast."
                    )

                else:
                    y = series.to_numpy(dtype=float)
                    x = np.arange(len(y), dtype=float)
                    slope, intercept = np.polyfit(x, y, 1)

                    future_x = np.arange(
                        len(y),
                        len(y) + plan["periods"],
                        dtype=float,
                    )

                    predicted = intercept + slope * future_x

                    if (y >= 0).all():
                        predicted = np.maximum(predicted, 0)

                    last_date = series.index[-1]
                    future_dates = []

                    for i in range(1, plan["periods"] + 1):
                        if freq == "D":
                            future_dates.append(
                                last_date + pd.Timedelta(days=i)
                            )
                        elif freq == "W":
                            future_dates.append(
                                last_date + pd.Timedelta(weeks=i)
                            )
                        elif freq == "MS":
                            future_dates.append(
                                last_date + pd.offsets.MonthBegin(i)
                            )
                        elif freq == "QS":
                            future_dates.append(
                                last_date + pd.offsets.QuarterBegin(i)
                            )
                        else:
                            future_dates.append(
                                last_date + pd.offsets.YearBegin(i)
                            )

                    forecast = pd.DataFrame({
                        "period": future_dates,
                        "forecast": predicted,
                    })

                    result["table"] = forecast
                    result["answer"] = (
                        f"Basic linear trend forecast for `{metric}` over "
                        f"the next {plan['periods']} {plan['frequency']} "
                        "period(s). This is a simple baseline, not a guarantee."
                    )

                    fig = px.line(
                        history,
                        x="period",
                        y="actual",
                        markers=True,
                        title=(
                            f"{metric}: historical values "
                            "and baseline forecast"
                        ),
                    )
                    fig.add_scatter(
                        x=forecast["period"],
                        y=forecast["forecast"],
                        mode="lines+markers",
                        name="Forecast",
                    )
                    result["chart"] = fig

    elif op == "distribution":
        col = plan.get("column") or plan.get("metric_column")

        if not col:
            result["answer"] = (
                "Please name a column to inspect its distribution."
            )

        elif col in detect_types(data)[0]:
            values = numeric_series(data[col]).dropna()

            if len(values):
                result["answer"] = (
                    f"Distribution of `{col}`: {len(values):,} "
                    f"non-missing values; mean {values.mean():,.2f}; "
                    f"median {values.median():,.2f}."
                )
            else:
                result["answer"] = "No numeric values found."

            result["chart"] = px.histogram(
                pd.DataFrame({col: values}),
                x=col,
                title=f"Distribution of {col}",
            )

        else:
            result["table"] = (
                data[col].astype("string")
                .value_counts(dropna=False)
                .head(50)
                .rename_axis(col)
                .reset_index(name="count")
            )
            result["answer"] = (
                f"Most frequent values in `{col}`."
            )
            result["chart"] = px.bar(
                result["table"].head(25),
                x=col,
                y="count",
                title=f"Top values in {col}",
            )

    elif op == "correlation":
        cols = [
            c for c in plan.get("columns", [])
            if c in data.columns
        ]

        if len(cols) < 2:
            cols = detect_types(data)[0]

        if len(cols) < 2:
            result["answer"] = (
                "At least two numeric columns are needed for correlation."
            )
        else:
            corr = data[cols].apply(numeric_series).corr()
            result["table"] = corr.reset_index(names="Column")
            result["answer"] = (
                "Correlation matrix for numeric columns. "
                "Correlation shows association, not causation."
            )
            result["chart"] = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Correlation matrix",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )

    return result


# ============================================================
# 8. SIDEBAR AND DATA SOURCE
# ============================================================
st.sidebar.markdown("""
<div style="padding: 12px 0 22px 0;">
    <div class="brand-mark">DEXCR<span style="color:#a79cff;">.</span></div>
    <div class="brand-caption">Make your data make sense.</div>
    <div class="brand-caption" style="margin-top:12px;">
        Your business intelligence workspace
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Your data")
st.sidebar.caption("Upload a dataset to start exploring.")

uploaded = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)

use_sample = st.sidebar.checkbox(
    "Use sales_transactions.csv if available",
    value=True,
)

df = None
source_name = None

if uploaded is not None:
    try:
        df = read_file(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.sidebar.error(f"Could not read file: {exc}")

elif use_sample and (BASE_DIR / "sales_transactions.csv").exists():
    try:
        df = load_sample_data()
        source_name = "sales_transactions.csv"
    except Exception as exc:
        st.sidebar.error(f"Could not read sample data: {exc}")


# ============================================================
# 9. HERO / WELCOME
# ============================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR DATA. YOUR DECISIONS.</div>
    <h1>DEXCR</h1>
    <div class="tagline">Make your data make sense.</div>
    <p>
        Your AI-powered business intelligence workspace. Explore your numbers,
        uncover trends, and ask questions in plain English to turn raw data
        into clearer business decisions.
    </p>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.markdown("### Welcome to your data workspace")
    st.markdown(
        "Start by uploading a CSV or Excel file from the sidebar. "
        "DEXCR will profile your dataset and help you explore it."
    )

    st.markdown("#### What you can do with DEXCR")
    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("**📈 Explore performance**")
            st.caption(
                "Understand sales, revenue, profit, and product performance."
            )

        with st.container(border=True):
            st.markdown("**🔎 Check data quality**")
            st.caption(
                "Find missing values, duplicates, and unusual distributions."
            )

    with c2:
        with st.container(border=True):
            st.markdown("**💬 Ask questions in plain English**")
            st.caption(
                "Ask business questions instead of writing analysis code."
            )

        with st.container(border=True):
            st.markdown("**🎯 Get business recommendations**")
            st.caption(
                "Identify patterns and prioritize opportunities using "
                "evidence from your data."
            )

    st.info(
        "You can also add `sales_transactions.csv` to the same repository "
        "as `app.py` to use it as the optional sample dataset."
    )
    st.stop()

if df.empty:
    st.error("The uploaded dataset has no rows.")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]

review = automatic_review(df)
numeric_cols = review["numeric"]
categorical_cols = review["categorical"]
date_cols = review["dates"]

st.caption(
    f"**Current dataset:** {source_name}  ·  "
    f"{len(df):,} rows  ·  {len(df.columns):,} columns"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total rows", f"{len(df):,}")
m2.metric("Total columns", f"{len(df.columns):,}")
m3.metric("Numeric fields", len(numeric_cols))
m4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")


# ============================================================
# 10. MAIN NAVIGATION
# ============================================================
overview_tab, data_tab, explorer_tab, analyst_tab = st.tabs([
    "Overview",
    "My Data",
    "Explore Columns",
    "Ask DEXCR",
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with overview_tab:
    st.subheader("Your data at a glance")
    st.markdown(
        '<p class="section-description">'
        'A quick health check of your dataset and its key metrics.'
        '</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### Automatic data review")

        for insight in review["insights"]:
            st.markdown(f"- {insight}")

    if numeric_cols:
        st.markdown("#### Explore a key metric")

        selected_metric = st.selectbox(
            "Choose a numeric column to summarize",
            numeric_cols,
            key="overview_metric",
        )

        vals = numeric_series(df[selected_metric]).dropna()

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Total",
            f"{vals.sum():,.2f}" if len(vals) else "N/A",
        )
        c2.metric(
            "Average",
            f"{vals.mean():,.2f}" if len(vals) else "N/A",
        )
        c3.metric(
            "Median",
            f"{vals.median():,.2f}" if len(vals) else "N/A",
        )

        st.plotly_chart(
            px.histogram(
                pd.DataFrame({selected_metric: vals}),
                x=selected_metric,
                title=f"Distribution of {selected_metric}",
            ),
            use_container_width=True,
        )


# ============================================================
# TAB 2: MY DATA
# ============================================================
with data_tab:
    st.subheader("Your dataset")
    st.markdown(
        '<p class="section-description">'
        'Preview records, inspect column quality, and export your data.'
        '</p>',
        unsafe_allow_html=True,
    )

    with st.expander("Preview records", expanded=True):
        st.dataframe(
            df.head(500),
            use_container_width=True,
        )

    st.markdown("#### Column profile")

    profile = pd.DataFrame({
        "Column": df.columns,
        "Data type": [
            str(df[c].dtype)
            for c in df.columns
        ],
        "Missing": [
            int(df[c].isna().sum())
            for c in df.columns
        ],
        "Missing %": [
            round(df[c].isna().mean() * 100, 2)
            for c in df.columns
        ],
        "Unique": [
            int(df[c].nunique(dropna=True))
            for c in df.columns
        ],
    })

    st.dataframe(profile, use_container_width=True)

    st.download_button(
        "Download current data as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="dexcr_data.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 3: EXPLORE COLUMNS
# ============================================================
with explorer_tab:
    st.subheader("Explore a column")
    st.markdown(
        '<p class="section-description">'
        'Inspect a field’s statistics, distribution, and most common values.'
        '</p>',
        unsafe_allow_html=True,
    )

    selected_col = st.selectbox(
        "Choose a column",
        list(df.columns),
        key="explorer_col",
    )

    info1, info2, info3 = st.columns(3)
    info1.metric("Data type", str(df[selected_col].dtype))
    info2.metric(
        "Missing values",
        f"{int(df[selected_col].isna().sum()):,}",
    )
    info3.metric(
        "Unique values",
        f"{int(df[selected_col].nunique(dropna=True)):,}",
    )

    if selected_col in numeric_cols:
        values = numeric_series(df[selected_col]).dropna()

        st.markdown("#### Descriptive statistics")
        st.dataframe(
            values.describe().to_frame("Statistics"),
            use_container_width=True,
        )

        st.plotly_chart(
            px.histogram(
                pd.DataFrame({selected_col: values}),
                x=selected_col,
                title=f"Distribution of {selected_col}",
            ),
            use_container_width=True,
        )

    else:
        counts = (
            df[selected_col].astype("string")
            .value_counts(dropna=False)
            .head(50)
        )

        count_df = counts.rename_axis(
            selected_col
        ).reset_index(name="count")

        st.markdown("#### Most common values")
        st.dataframe(count_df, use_container_width=True)

        st.plotly_chart(
            px.bar(
                count_df.head(25),
                x=selected_col,
                y="count",
                title=f"Most frequent values in {selected_col}",
            ),
            use_container_width=True,
        )


# ============================================================
# TAB 4: ASK DEXCR
# ============================================================
def set_analyst_question(question):
    """
    Update the widget's session-state value from a button callback.

    Streamlit runs callbacks before the script reruns, avoiding the
    StreamlitWidgetAlreadyInstantiatedError from changing a widget key
    after the widget has been instantiated.
    """
    st.session_state["analyst_question"] = question


with analyst_tab:
    st.subheader("Ask DEXCR")
    st.markdown(
        '<p class="section-description">'
        'Ask a business question in plain English. DEXCR can calculate '
        'metrics, explore trends, and help prioritize business opportunities.'
        '</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### What would you like to understand?")

        question = st.text_area(
            "Your question",
            placeholder=(
                "For example: What should I focus on to increase revenue?"
            ),
            height=105,
            key="analyst_question",
            label_visibility="collapsed",
        )

        st.markdown("**Need inspiration? Try asking:**")

        suggested_questions = [
            "What should I focus on to increase revenue?",
            "What are my top products by revenue?",
            "Show monthly sales trends.",
            "How can I improve profitability?",
            "Which regions generate the most revenue?",
            "Find missing values and duplicates.",
        ]

        suggestions = st.columns(3)

        for i, suggestion in enumerate(suggested_questions):
            with suggestions[i % 3]:
                st.button(
                    suggestion,
                    key=f"suggest_{i}",
                    use_container_width=True,
                    on_click=set_analyst_question,
                    args=(suggestion,),
                )

        use_llm = st.toggle(
            "Use AI to understand and explain my question",
            value=True,
        )

        provider = str(
            setting("LLM_PROVIDER", "qwen")
        ).lower()

        if provider == "openai":
            has_key = bool(setting("OPENAI_API_KEY"))
        else:
            has_key = bool(
                setting("LLM_API_KEY")
                or setting("DASHSCOPE_API_KEY")
            )

        if use_llm and not LANGCHAIN_AVAILABLE:
            st.warning(
                "LangChain isn't installed in this environment. "
                "The local question interpreter will be used."
            )
        elif use_llm and not has_key:
            st.info(
                "No LLM API key configured. DEXCR will use its local "
                "interpreter until you add a key in Streamlit Secrets."
            )

        run = st.button(
            "Analyze question",
            type="primary",
            disabled=not question.strip(),
            use_container_width=True,
        )

    if run:
        plan = None

        if use_llm and LANGCHAIN_AVAILABLE and has_key:
            with st.spinner(
                "Understanding your question with the LLM..."
            ):
                try:
                    plan = get_llm_plan(
                        question.strip(),
                        df,
                    )
                    plan = validate_plan(plan, df)

                    st.session_state["last_plan_source"] = (
                        f"LLM ({provider})"
                    )

                except Exception as exc:
                    st.warning(
                        "LLM planning failed; using the local interpreter "
                        "instead. "
                        f"Details: {type(exc).__name__}: {exc}"
                    )

        if plan is None:
            plan = local_plan(question.strip(), df)
            st.session_state["last_plan_source"] = "Local fallback"

        st.markdown("### Analysis results")

        st.caption(
            f"**Question interpretation:** `{plan['operation']}`  ·  "
            f"**Method:** {st.session_state['last_plan_source']}"
        )

        if plan.get("answer_note"):
            st.info(plan["answer_note"])

        with st.spinner("Calculating from your data..."):
            result = execute_plan(df, plan)

        # Render the calculated findings.
        with st.container(border=True):
            st.markdown("#### Result")
            st.markdown(result.get("answer", ""))

        # Business Advisor: use a dedicated strategy-oriented LLM prompt.
        if plan["operation"] == "business_advisor":
            if (
                use_llm
                and LANGCHAIN_AVAILABLE
                and has_key
                and result.get("key_findings")
            ):
                with st.spinner(
                    "Evaluating opportunities and prioritizing actions..."
                ):
                    try:
                        advice = business_advice_with_llm(
                            question.strip(),
                            result,
                        )

                        st.markdown("### DEXCR's recommendations")

                        with st.container(border=True):
                            st.markdown(advice)

                    except Exception as exc:
                        st.warning(
                            "The calculated business findings are available, "
                            "but AI recommendations could not be generated. "
                            f"Details: {type(exc).__name__}: {exc}"
                        )

                        st.info(
                            "Check your API key, provider configuration, "
                            "and model access to enable AI recommendations."
                        )
            else:
                st.markdown("### Business Advisor: next steps")

                if not result.get("available_fields", {}).get("revenue"):
                    st.markdown(
                        "1. Identify which column represents revenue or sales.\n"
                        "2. If possible, include transaction dates and product, "
                        "category, customer, or region fields.\n"
                        "3. Re-run the analysis to identify actionable "
                        "opportunities."
                    )
                else:
                    st.markdown(
                        "The findings above are calculated from your data. "
                        "To get a prioritized, natural-language action plan, "
                        "enable AI and configure your Qwen or OpenAI API key."
                    )

                    st.markdown(
                        "- Investigate the strongest and weakest groups "
                        "identified in the results.\n"
                        "- Compare recent performance with earlier periods "
                        "if date data is available.\n"
                        "- Validate possible causes with additional business "
                        "information before investing in an intervention."
                    )

        # For regular analytical questions, retain the existing explanation.
        elif (
            use_llm
            and LANGCHAIN_AVAILABLE
            and has_key
            and plan["operation"] != "unknown"
        ):
            with st.spinner("Writing a business explanation..."):
                explanation = explain_with_llm(
                    question.strip(),
                    plan,
                    result,
                )

            st.markdown("#### Business interpretation")
            st.markdown(explanation)

        if result.get("chart") is not None:
            st.plotly_chart(
                result["chart"],
                use_container_width=True,
            )

        # Display supporting tables for the Business Advisor.
        if plan["operation"] == "business_advisor":
            evidence_tables = result.get("evidence_tables", {})

            if evidence_tables:
                st.markdown("### Supporting evidence")

                for table_name, table in evidence_tables.items():
                    with st.expander(table_name):
                        st.dataframe(
                            table,
                            use_container_width=True,
                        )

                        st.download_button(
                            f"Download {table_name} as CSV",
                            table.to_csv(index=False).encode("utf-8"),
                            file_name=(
                                re.sub(
                                    r"[^a-zA-Z0-9_-]+",
                                    "_",
                                    table_name.lower(),
                                )
                                + ".csv"
                            ),
                            mime="text/csv",
                            key=(
                                "download_business_"
                                + re.sub(
                                    r"[^a-zA-Z0-9_-]+",
                                    "_",
                                    table_name.lower(),
                                )
                            ),
                        )

        elif isinstance(result.get("table"), pd.DataFrame):
            st.markdown("#### Detailed results")
            st.dataframe(
                result["table"],
                use_container_width=True,
            )

            st.download_button(
                "Download result as CSV",
                result["table"].to_csv(index=False).encode("utf-8"),
                file_name="dexcr_analysis_result.csv",
                mime="text/csv",
                key="download_analysis_result",
            )

        if plan.get("filters"):
            st.caption(
                f"Applied {len(plan['filters'])} filter(s)."
            )


# ============================================================
# 11. FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "DEXCR · Make your data make sense. "
    "Verify important business decisions against source data. "
    "Recommendations identify opportunities to investigate, not guaranteed "
    "results. Forecasts are baseline estimates, not guarantees."
)
