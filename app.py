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
            return pd.read_csv(file, encoding="latin-1", low_memory=False)

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

        elif any(x in name.split() for x in ["date", "datetime", "timestamp", "period"]):
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
        ds = pd.to_datetime(df[roles["date"]], errors="coerce").dropna()

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
            "examples": [str(v)[:80] for v in s.dropna().head(3).tolist()],
        })

    return schema


# ============================================================
# 3. LANGCHAIN LLM: QWEN OR OPENAI
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
            raise ValueError("Set OPENAI_API_KEY in your Streamlit Secrets.")

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
            raise ValueError("Set LLM_API_KEY in Streamlit Secrets.")

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
Translate the question into ONE JSON object that describes an analysis. Never calculate results.
Use only exact column names from the supplied schema. Do not return Python, SQL, or markdown.

Operations:
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
        "detected_roles": {k: str(v) for k, v in review["roles"].items()},
    }

    response = llm.bind(
        response_format={"type": "json_object"}
    ).invoke([
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(request, ensure_ascii=False, default=str)),
    ])

    content = response.content

    if isinstance(content, list):
        content = "".join(
            x.get("text", "")
            for x in content
            if isinstance(x, dict)
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
                content=json.dumps(payload, ensure_ascii=False, default=str)
            ),
        ])

        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "\n".join(
                x.get("text", "")
                for x in content
                if isinstance(x, dict)
            )

        return str(content)

    except Exception as exc:
        return (
            f"AI explanation could not be generated "
            f"({type(exc).__name__}: {exc}). "
            "The calculated result is still available below."
        )


# ============================================================
# 4. SAFE PLAN VALIDATION
# ============================================================
OPERATIONS = {
    "profile", "columns", "missing", "duplicates", "show_rows",
    "aggregate", "rank", "trend", "forecast", "distribution",
    "correlation", "filter", "unknown",
}
AGGS = {"sum", "mean", "median", "min", "max", "count", "nunique", "std"}
OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "between"}


def validate_plan(plan, df):
    if not isinstance(plan, dict):
        raise ValueError("Analysis plan is not a dictionary.")

    p = dict(plan)
    p["operation"] = str(p.get("operation", "unknown")).lower()

    if p["operation"] not in OPERATIONS:
        p["operation"] = "unknown"

    p["aggregation"] = str(p.get("aggregation", "sum")).lower()

    if p["aggregation"] not in AGGS:
        p["aggregation"] = "sum"

    p["direction"] = (
        "asc"
        if str(p.get("direction", "desc")).lower() == "asc"
        else "desc"
    )

    try:
        p["top_n"] = max(1, min(int(p.get("top_n", 10)), 100))
    except (ValueError, TypeError):
        p["top_n"] = 10

    try:
        p["periods"] = max(1, min(int(p.get("periods", 6)), 24))
    except (ValueError, TypeError):
        p["periods"] = 6

    p["frequency"] = str(p.get("frequency", "month")).lower()

    if p["frequency"] not in {"day", "week", "month", "quarter", "year"}:
        p["frequency"] = "month"

    for key in ["metric_column", "entity_column", "date_column", "column"]:
        value = p.get(key)

        if value not in df.columns:
            p[key] = None

    for key in ["group_by", "columns"]:
        value = p.get(key, [])

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            value = []

        p[key] = [x for x in value if x in df.columns]

    filters = p.get("filters", [])
    p["filters"] = []

    if isinstance(filters, list):
        for f in filters[:10]:
            if not isinstance(f, dict) or f.get("column") not in df.columns:
                continue

            op = str(f.get("operator", "eq")).lower()

            if op not in OPS:
                continue

            p["filters"].append({
                "column": f["column"],
                "operator": op,
                "value": f.get("value"),
            })

    return p


# ============================================================
# 5. LOCAL INTERPRETER FALLBACK
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
        return any(norm(w) in q for w in words)

    mentioned = next(
        (c for c in df.columns if norm(c) and norm(c) in q),
        None,
    )

    metric = (
        roles.get("revenue")
        or roles.get("quantity")
        or (numeric[0] if numeric else None)
    )

    if has("profit", "profitable"):
        metric = roles.get("profit")
    elif has("quantity", "units sold", "volume", "best selling", "bestselling"):
        metric = roles.get("quantity") or metric
    elif has("cost", "expenses"):
        metric = roles.get("cost") or metric

    plan["metric_column"] = metric

    entity = roles.get("product")

    if has("customer", "client", "buyer"):
        entity = roles.get("customer") or entity
    elif has("region", "country", "city", "territory", "branch"):
        entity = roles.get("region") or entity
    elif has("category", "department", "segment"):
        entity = roles.get("category") or entity

    plan["entity_column"] = entity

    if has("column names", "list columns", "what columns", "available columns"):
        plan["operation"] = "columns"

    elif has("missing", "null values", "data quality", "incomplete data"):
        plan["operation"] = "missing"

    elif has("duplicate", "duplicates"):
        plan["operation"] = "duplicates"

    elif has("profile", "overview", "summarize dataset",
             "summary of dataset", "analyze my data", "analyse my data"):
        plan["operation"] = "profile"

    elif has("correlation", "correlated", "relationship between"):
        plan["operation"] = "correlation"
        plan["columns"] = numeric

    elif has("forecast", "predict", "future", "next month", "next year"):
        plan["operation"] = "forecast"
        plan["date_column"] = roles.get("date") or (dates[0] if dates else None)

        nums = re.findall(r"\b(\d{1,2})\b", q)

        if nums:
            plan["periods"] = min(max(int(nums[0]), 1), 24)

        for freq, words in {
            "day": ["daily", "by day"],
            "week": ["weekly", "by week"],
            "quarter": ["quarterly", "quarter"],
            "year": ["yearly", "annual", "by year"],
        }.items():
            if has(*words):
                plan["frequency"] = freq

    elif has("trend", "over time", "by month", "monthly", "by year", "sales over time"):
        plan["operation"] = "trend"
        plan["date_column"] = roles.get("date") or (dates[0] if dates else None)

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

    elif has("show rows", "show records", "preview", "sample rows", "first rows"):
        plan["operation"] = "show_rows"

    elif has("top", "highest", "best", "most", "rank", "leading", "lowest", "least"):
        plan["operation"] = "rank"

        if has("lowest", "least", "smallest", "worst"):
            plan["direction"] = "asc"

        nums = re.findall(r"\btop\s+(\d+)", q)

        if nums:
            plan["top_n"] = min(max(int(nums[0]), 1), 100)

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
            "missing values, or a forecast."
        )

    return validate_plan(plan, df)


# ============================================================
# 6. FILTERS AND ANALYSIS EXECUTION
# ============================================================
def apply_filters(df, filters):
    result = df.copy()

    for f in filters:
        col, op, value = f["column"], f["operator"], f["value"]
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
                    bounds = value if isinstance(value, list) else str(value).split(",")

                    if len(bounds) != 2:
                        continue

                    mask = ns.between(float(bounds[0]), float(bounds[1]))

            elif op == "contains":
                mask = s.astype("string").str.contains(
                    str(value), case=False, na=False, regex=False
                )

            elif op == "eq":
                mask = s.astype("string").str.lower() == str(value).lower()

            elif op == "ne":
                mask = s.astype("string").str.lower() != str(value).lower()

            else:
                continue

            result = result.loc[mask.fillna(False)]

        except (ValueError, TypeError):
            continue

    return result


def execute_plan(df, plan):
    op = plan["operation"]
    data = apply_filters(df, plan.get("filters", []))
    result = {"answer": "", "table": None, "chart": None, "data": data}

    if op == "unknown":
        result["answer"] = (
            plan.get("answer_note")
            or "I could not determine the right analysis. Please rephrase the question."
        )

    elif op == "columns":
        result["answer"] = f"The dataset has {len(df.columns)} columns."
        result["table"] = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(df[c].dtype) for c in df.columns],
        })

    elif op == "profile":
        review = automatic_review(data)
        result["answer"] = "\n\n".join(review["insights"])
        result["table"] = pd.DataFrame({
            "Column": data.columns,
            "Data type": [str(data[c].dtype) for c in data.columns],
            "Missing": [int(data[c].isna().sum()) for c in data.columns],
            "Unique": [int(data[c].nunique(dropna=True)) for c in data.columns],
        })

    elif op == "missing":
        table = pd.DataFrame({
            "Column": data.columns,
            "Missing": [int(data[c].isna().sum()) for c in data.columns],
            "Missing %": [round(data[c].isna().mean() * 100, 2) for c in data.columns],
        })

        result["table"] = table.sort_values("Missing %", ascending=False)
        result["answer"] = (
            f"{int(data.isna().sum().sum()):,} missing cells "
            f"across {len(data):,} rows."
        )
        result["chart"] = px.bar(
            result["table"], x="Column", y="Missing %",
            title="Missing values by column",
        )

    elif op == "duplicates":
        n = int(data.duplicated().sum())
        result["answer"] = f"Found **{n:,} fully duplicated rows** in the current dataset."
        result["table"] = data[data.duplicated(keep=False)].head(100)

    elif op == "show_rows":
        result["answer"] = f"Showing up to 100 of {len(data):,} rows."
        result["table"] = data.head(100)

    elif op == "filter":
        result["answer"] = f"Filters returned {len(data):,} rows out of {len(df):,}."
        result["table"] = data.head(500)

    elif op == "aggregate":
        metric = plan.get("metric_column")

        if not metric:
            result["answer"] = "I couldn't identify a numeric measure for this aggregation."

        elif plan["aggregation"] == "count":
            if plan.get("group_by"):
                result["table"] = (
                    data.groupby(plan["group_by"], dropna=False)[metric]
                    .count().reset_index(name="count")
                )
            else:
                result["answer"] = f"Non-missing `{metric}` values: {data[metric].count():,}."

        else:
            values = numeric_series(data[metric])

            if plan.get("group_by"):
                work = data[plan["group_by"]].copy()
                work["_metric"] = values

                agg = getattr(
                    work.groupby(plan["group_by"], dropna=False)["_metric"],
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
                        result["table"].columns[-1], ascending=False
                    )
                    result["chart"] = px.bar(
                        result["table"].head(30),
                        x=plan["group_by"][0],
                        y=result["table"].columns[-1],
                        title=f"{plan['aggregation'].title()} of {metric} by {plan['group_by'][0]}",
                    )
            else:
                val = getattr(values, plan["aggregation"])()
                result["answer"] = (
                    f"{plan['aggregation'].title()} of `{metric}`: **{val:,.2f}**."
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
                work.groupby(entity, dropna=False)["_metric"]
                .agg(plan["aggregation"])
                .reset_index(name=f"{plan['aggregation']}_{metric}")
            )

            grouped = grouped.sort_values(
                grouped.columns[-1],
                ascending=plan["direction"] == "asc",
            ).head(plan["top_n"])

            result["table"] = grouped
            label = "lowest" if plan["direction"] == "asc" else "highest"
            result["answer"] = (
                f"Showing the {label}-ranked `{entity}` values "
                f"by {plan['aggregation']} of `{metric}`."
            )
            result["chart"] = px.bar(
                grouped, x=entity, y=grouped.columns[-1],
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
                "_date": pd.to_datetime(data[date_col], errors="coerce"),
                "_value": numeric_series(data[metric]),
            }).dropna()

            if work.empty:
                result["answer"] = "No valid date/metric rows were available for time analysis."
            else:
                freq_map = {
                    "day": "D", "week": "W", "month": "MS",
                    "quarter": "QS", "year": "YS",
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
                history = history.rename(columns={"_date": "period"})

                if op == "trend":
                    result["table"] = history
                    result["answer"] = (
                        f"Trend for `{metric}` grouped by {plan['frequency']} "
                        f"({len(history)} periods)."
                    )
                    result["chart"] = px.line(
                        history, x="period", y="actual", markers=True,
                        title=f"{plan['frequency'].title()} {metric} trend",
                    )

                elif len(series) < 2:
                    result["answer"] = (
                        "At least two non-empty time periods are needed for a basic forecast."
                    )

                else:
                    y = series.to_numpy(dtype=float)
                    x = np.arange(len(y), dtype=float)
                    slope, intercept = np.polyfit(x, y, 1)
                    future_x = np.arange(
                        len(y), len(y) + plan["periods"], dtype=float
                    )

                    predicted = intercept + slope * future_x
                    if (y >= 0).all():
                        predicted = np.maximum(predicted, 0)

                    last_date = series.index[-1]
                    future_dates = []

                    for i in range(1, plan["periods"] + 1):
                        if freq == "D":
                            future_dates.append(last_date + pd.Timedelta(days=i))
                        elif freq == "W":
                            future_dates.append(last_date + pd.Timedelta(weeks=i))
                        elif freq == "MS":
                            future_dates.append(last_date + pd.offsets.MonthBegin(i))
                        elif freq == "QS":
                            future_dates.append(last_date + pd.offsets.QuarterBegin(i))
                        else:
                            future_dates.append(last_date + pd.offsets.YearBegin(i))

                    forecast = pd.DataFrame({
                        "period": future_dates,
                        "forecast": predicted,
                    })

                    result["table"] = forecast
                    result["answer"] = (
                        f"Basic linear trend forecast for `{metric}` over the next "
                        f"{plan['periods']} {plan['frequency']} period(s). "
                        "This is a simple baseline, not a guarantee."
                    )

                    fig = px.line(
                        history, x="period", y="actual", markers=True,
                        title=f"{metric}: historical values and baseline forecast",
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
            result["answer"] = "Please name a column to inspect its distribution."

        elif col in detect_types(data)[0]:
            values = numeric_series(data[col]).dropna()

            if len(values):
                result["answer"] = (
                    f"Distribution of `{col}`: {len(values):,} non-missing values; "
                    f"mean {values.mean():,.2f}; median {values.median():,.2f}."
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
                .value_counts(dropna=False).head(50)
                .rename_axis(col).reset_index(name="count")
            )
            result["answer"] = f"Most frequent values in `{col}`."
            result["chart"] = px.bar(
                result["table"].head(25),
                x=col, y="count",
                title=f"Top values in {col}",
            )

    elif op == "correlation":
        cols = [c for c in plan.get("columns", []) if c in data.columns]

        if len(cols) < 2:
            cols = detect_types(data)[0]

        if len(cols) < 2:
            result["answer"] = "At least two numeric columns are needed for correlation."
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
# 7. SIDEBAR AND DATA SOURCE
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
# 8. HERO / WELCOME
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
            st.caption("Understand sales, revenue, profit, and product performance.")
        with st.container(border=True):
            st.markdown("**🔎 Check data quality**")
            st.caption("Find missing values, duplicates, and unusual distributions.")

    with c2:
        with st.container(border=True):
            st.markdown("**💬 Ask questions in plain English**")
            st.caption("Ask business questions instead of writing analysis code.")
        with st.container(border=True):
            st.markdown("**🔮 Explore trends and forecasts**")
            st.caption("Review historical trends and simple baseline forecasts.")

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
# 9. MAIN NAVIGATION
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
        c1.metric("Total", f"{vals.sum():,.2f}" if len(vals) else "N/A")
        c2.metric("Average", f"{vals.mean():,.2f}" if len(vals) else "N/A")
        c3.metric("Median", f"{vals.median():,.2f}" if len(vals) else "N/A")

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
        st.dataframe(df.head(500), use_container_width=True)

    st.markdown("#### Column profile")

    profile = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
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
    info2.metric("Missing values", f"{int(df[selected_col].isna().sum()):,}")
    info3.metric("Unique values", f"{int(df[selected_col].nunique(dropna=True)):,}")

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
            .value_counts(dropna=False).head(50)
        )

        count_df = counts.rename_axis(selected_col).reset_index(name="count")

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
with analyst_tab:
    st.subheader("Ask DEXCR")
    st.markdown(
        '<p class="section-description">'
        'Ask a business question in plain English and get an analysis based on your data.'
        '</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### What would you like to understand?")

        question = st.text_area(
            "Your question",
            placeholder=(
                "For example: Which 10 products generated the most revenue?"
            ),
            height=105,
            key="analyst_question",
            label_visibility="collapsed",
        )

        st.markdown("**Need inspiration? Try asking:**")

        suggestions = st.columns(3)
        suggested_questions = [
            "What are my top products by revenue?",
            "Show monthly sales trends.",
            "Find missing values and duplicates.",
        ]

        for col, suggestion in zip(suggestions, suggested_questions):
            with col:
                if st.button(
                    suggestion,
                    key=f"suggest_{suggestion}",
                    use_container_width=True,
                ):
                    st.session_state["analyst_question"] = suggestion
                    st.rerun()

        use_llm = st.toggle(
            "Use AI to understand and explain my question",
            value=True,
        )

        provider = str(setting("LLM_PROVIDER", "qwen")).lower()

        if provider == "openai":
            has_key = bool(setting("OPENAI_API_KEY"))
        else:
            has_key = bool(
                setting("LLM_API_KEY") or setting("DASHSCOPE_API_KEY")
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
            with st.spinner("Understanding your question with the LLM..."):
                try:
                    plan = get_llm_plan(question.strip(), df)
                    plan = validate_plan(plan, df)
                    st.session_state["last_plan_source"] = f"LLM ({provider})"

                except Exception as exc:
                    st.warning(
                        "LLM planning failed; using the local interpreter instead. "
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

        with st.container(border=True):
            st.markdown("#### Result")
            st.markdown(result.get("answer", ""))

            if (
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

        if isinstance(result.get("table"), pd.DataFrame):
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
            st.caption(f"Applied {len(plan['filters'])} filter(s).")


# ============================================================
# 10. FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "DEXCR · Make your data make sense. "
    "Verify important business decisions against source data. "
    "Forecasts are baseline estimates, not guarantees."
)
