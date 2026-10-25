
from pathlib import Path
import json
import os
import re
import urllib.error
import urllib.request
from difflib import get_close_matches

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DEXCR | AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp { background: #f6f7fb; }
[data-testid="stSidebar"] { background: #111827; }
[data-testid="stSidebar"] * { color: #f9fafb; }
.hero {
    background: linear-gradient(120deg, #111827, #312e81);
    color: white;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 22px;
}
.hero h1 { color: white; margin-bottom: 8px; }
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5e7eb;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. FILE LOADING
# ============================================================

def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, low_memory=False)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file, encoding="latin-1", low_memory=False
            )

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Upload a CSV, XLSX, or XLS file.")


def load_sample_data():
    path = BASE_DIR / "sales_transactions.csv"
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


# ============================================================
# 3. GENERAL UTILITIES
# ============================================================

def normalize_name(value):
    value = str(value).replace("\ufeff", "").strip().lower()
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    value = re.sub(r"[_./\\\-]+", " ", value)
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def numeric_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def display_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    return str(value)


def make_schema(df):
    return pd.DataFrame({
        "Column": [str(c) for c in df.columns],
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [
            round(df[c].isna().mean() * 100, 2) for c in df.columns
        ],
    })


def detect_column_types(df):
    numeric, categorical, dates = [], [], []

    for col in df.columns:
        s = df[col]
        name = normalize_name(col)

        if pd.api.types.is_datetime64_any_dtype(s):
            dates.append(col)
            continue

        date_name = any(x in name.split() for x in [
            "date", "datetime", "timestamp", "time",
            "created", "updated", "period"
        ])

        if date_name and not pd.api.types.is_numeric_dtype(s):
            parsed = pd.to_datetime(s, errors="coerce")
            if len(s) and parsed.notna().mean() >= 0.5:
                dates.append(col)
                continue

        if pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
            continue

        # Recognize numeric values stored as text, e.g. "$1,250.50".
        converted = numeric_series(s)
        if len(s.dropna()) and converted.notna().mean() >= 0.85:
            numeric.append(col)
            continue

        # Recognize date values even if the column has an unusual name.
        parsed = pd.to_datetime(s, errors="coerce")
        if (
            len(s.dropna()) >= 3
            and parsed.notna().mean() >= 0.9
            and s.nunique(dropna=True) > 2
        ):
            dates.append(col)
            continue

        if s.nunique(dropna=True) <= min(100, max(20, len(s) // 2)):
            categorical.append(col)

    return numeric, categorical, dates


# ============================================================
# 4. SEMANTIC COLUMN UNDERSTANDING
# ============================================================

# These are semantic hints, not mandatory column names.
# The application also supports names it has never seen before.

ROLE_ALIASES = {
    "product": [
        "product", "product name", "productname", "product title",
        "item", "item name", "item description", "item_description",
        "drug", "drug name", "medicine", "medication", "sku",
        "stock keeping unit", "article", "part name", "service name",
        "material description", "product description"
    ],
    "quantity": [
        "quantity", "qty", "quantity sold", "qty sold", "units",
        "units sold", "unit sold", "volume", "volume sold",
        "number of units", "sales quantity", "ordered quantity"
    ],
    "revenue": [
        "revenue", "sales", "sales amount", "sales value", "net sales",
        "gross sales", "net revenue", "total revenue", "turnover",
        "amount", "total amount", "line total", "invoice amount",
        "total sales", "selling price", "extended price"
    ],
    "profit": [
        "profit", "net profit", "gross profit", "profit amount",
        "margin amount", "operating profit", "contribution margin"
    ],
    "cost": [
        "cost", "unit cost", "total cost", "purchase cost",
        "expense", "expenses", "cost amount", "cogs",
        "cost of goods sold"
    ],
    "date": [
        "date", "order date", "transaction date", "sales date",
        "invoice date", "created at", "created date", "timestamp",
        "month", "period", "time", "date sold", "delivery date"
    ],
    "customer": [
        "customer", "customer name", "customer id", "client",
        "client name", "buyer", "account", "account id",
        "patient", "patient id"
    ],
    "region": [
        "region", "area", "territory", "country", "city",
        "location", "market", "branch", "store", "warehouse"
    ],
    "category": [
        "category", "product category", "product type", "department",
        "segment", "class", "group", "therapeutic class"
    ],
    "order": [
        "order id", "order number", "transaction id",
        "transaction number", "invoice id", "invoice number",
        "receipt id", "purchase id"
    ],
}


def semantic_score(column, aliases):
    name = normalize_name(column)
    if not name:
        return 0

    best = 0
    for alias in aliases:
        a = normalize_name(alias)

        if name == a:
            best = max(best, 100)
        elif a and re.search(r"\b" + re.escape(a) + r"\b", name):
            best = max(best, 85)
        elif a and (a in name or name in a):
            best = max(best, 65)
        else:
            match = get_close_matches(name, [a], n=1, cutoff=0.82)
            if match:
                best = max(best, 55)

    # Avoid confusing price with sales/revenue or cost with profit.
    if "profit" in name and aliases is ROLE_ALIASES["revenue"]:
        best = 0
    if "cost" in name and aliases is ROLE_ALIASES["revenue"]:
        best = min(best, 20)
    if "price" in name and aliases is ROLE_ALIASES["revenue"]:
        best = min(best, 50)

    return best


def infer_semantic_columns(df):
    """Return likely meanings of columns in this specific dataset."""
    numeric, categorical, dates = detect_column_types(df)
    roles = {}

    for role, aliases in ROLE_ALIASES.items():
        candidates = []

        for col in df.columns:
            score = semantic_score(col, aliases)

            if role in ["quantity", "revenue", "profit", "cost"]:
                if col not in numeric:
                    # Numeric text is also allowed.
                    if numeric_series(df[col]).notna().mean() < 0.85:
                        score = 0

            if role == "date" and col not in dates:
                # Only give a small boost to date-looking names.
                score = min(score, 35)

            if role in ["product", "customer", "region", "category", "order"]:
                if col in numeric and role not in ["order"]:
                    # IDs can be numeric, but raw numeric measures
                    # should not be treated as entity labels by default.
                    if df[col].nunique(dropna=True) > max(100, len(df) * 0.5):
                        score -= 10

            if score > 0:
                candidates.append((score, col))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates and candidates[0][0] >= 55:
            roles[role] = candidates[0][1]

    # If a column is date-like, prefer it for the date role.
    if dates and "date" not in roles:
        roles["date"] = dates[0]

    # A categorical column with many repeated values can be a product
    # name even when its name is unfamiliar.
    if "product" not in roles:
        likely_entities = []
        for col in df.columns:
            s = df[col]
            if (
                col not in numeric
                and col not in dates
                and 1 < s.nunique(dropna=True) <= min(5000, len(df) * 0.8)
            ):
                name = normalize_name(col)
                if any(k in name for k in [
                    "item", "product", "drug", "medicine", "sku",
                    "material", "article", "description"
                ]):
                    likely_entities.append(col)
        if likely_entities:
            roles["product"] = likely_entities[0]

    return roles


def get_setting(name, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# ============================================================
# 5. AUTOMATIC DATA REVIEW
# ============================================================

def automatic_review(df):
    numeric_cols, category_cols, date_cols = detect_column_types(df)
    roles = infer_semantic_columns(df)
    insights = []

    missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    insights.append(
        f"The dataset contains **{len(df):,} rows** and "
        f"**{len(df.columns):,} columns**."
    )

    if missing:
        insights.append(
            f"**{missing:,} cells are missing** "
            f"({missing / max(df.size, 1) * 100:.2f}% of all cells)."
        )
    else:
        insights.append("No missing cells were detected.")

    if duplicate_rows:
        insights.append(
            f"**{duplicate_rows:,} fully duplicated rows** were found."
        )
    else:
        insights.append("No fully duplicated rows were detected.")

    insights.append(
        f"Detected {len(numeric_cols)} numeric, "
        f"{len(category_cols)} categorical, and "
        f"{len(date_cols)} date-like columns."
    )

    if roles:
        role_text = ", ".join(
            f"{role}: `{col}`" for role, col in roles.items()
        )
        insights.append(f"Likely business fields — {role_text}.")

    # Automatic high-level numeric summaries.
    if numeric_cols:
        candidates = []
        for col in numeric_cols:
            values = numeric_series(df[col]).dropna()
            if len(values) and values.nunique() > 1:
                candidates.append((col, values))

        for col, values in candidates[:3]:
            insights.append(
                f"**{col}**: total {display_number(values.sum())}, "
                f"average {display_number(values.mean())}, "
                f"median {display_number(values.median())}."
            )

    # Automatically rank likely products if a product and quantity
    # or revenue field can be identified.
    entity = roles.get("product")
    metric = roles.get("quantity") or roles.get("revenue")

    if entity and metric and metric in df.columns:
        values = numeric_series(df[metric])
        work = pd.DataFrame({
            "entity": df[entity],
            "metric": values,
        }).dropna()

        if not work.empty:
            ranking = (
                work.groupby("entity", dropna=False)["metric"]
                .sum()
                .sort_values(ascending=False)
            )

            if not ranking.empty:
                insights.append(
                    f"Highest-ranked `{entity}` by total `{metric}`: "
                    f"**{ranking.index[0]}** "
                    f"({display_number(ranking.iloc[0])})."
                )

    # Automatically identify the time span if a date field exists.
    date_col = roles.get("date")
    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        valid_dates = parsed.dropna()
        if not valid_dates.empty:
            insights.append(
                f"Date coverage: **{valid_dates.min().date()}** to "
                f"**{valid_dates.max().date()}** using `{date_col}`."
            )

    return {
        "numeric": numeric_cols,
        "categorical": category_cols,
        "dates": date_cols,
        "roles": roles,
        "insights": insights,
    }


# ============================================================
# 6. OPTIONAL LLM QUESTION INTERPRETER
# ============================================================

ALLOWED_OPERATIONS = {
    "rank", "aggregate", "trend", "distribution", "correlation",
    "profile", "missing", "duplicates", "show_rows", "columns",
    "filter", "forecast", "unknown"
}

ALLOWED_AGGREGATIONS = {
    "sum", "mean", "median", "min", "max", "count", "nunique", "std"
}

ALLOWED_OPERATORS = {
    "eq", "ne", "gt", "gte", "lt", "lte", "contains", "between"
}


def get_llm_plan(question, df):
    api_key = get_setting("OPENAI_API_KEY")
    if not api_key:
        return None, "No OPENAI_API_KEY is configured."

    base_url = get_setting(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    ).rstrip("/")
    model = get_setting("OPENAI_MODEL", "gpt-4o-mini")

    profile = automatic_review(df)
    schema = [
        {
            "name": str(c),
            "dtype": str(df[c].dtype),
            "unique_values": int(df[c].nunique(dropna=True)),
            "missing_values": int(df[c].isna().sum()),
        }
        for c in df.columns
    ]

    system_prompt = """
You are DEXCR's analytical question interpreter.
Return one valid JSON object only. Never return Python, SQL,
executable code, or invented results. Python will calculate the answer.

The user's dataset may use unfamiliar column names, camelCase,
abbreviations, or synonyms. Use the column names and semantic hints
to map the user's words to actual dataset columns.

Available operations:
- rank: rank entities. Fields: entity_column, metric_column,
  aggregation, direction, top_n.
- aggregate: calculate a measure. Fields: metric_column,
  aggregation, group_by (list).
- trend: aggregate a measure over time. Fields: date_column,
  metric_column, aggregation, frequency (day/week/month/quarter/year).
- forecast: forecast a numeric measure over time. Fields: date_column,
  metric_column, aggregation, frequency, periods.
- distribution: inspect a column. Field: column.
- correlation: correlation matrix. Field: columns (list).
- profile, missing, duplicates, show_rows, columns, filter, unknown.

Allowed aggregation: sum, mean, median, min, max, count, nunique, std.
Allowed filter operators: eq, ne, gt, gte, lt, lte, contains, between.
A filter has column, operator, value.

Rules:
- Best-selling product means quantity/units sold when available.
  If unavailable, use record count as a clearly disclosed proxy.
- Highest-revenue product means revenue/sales amount, not profit.
- Most-profitable product means a profit field, not revenue.
- "Sales" can mean revenue, but do not assume a column named sales
  exists; use semantic hints.
- Product may mean productName, drug_name, item_description, SKU, etc.
- If the question asks for predictions/forecasting and a date column
  plus numeric measure exist, use forecast.
- If the required field does not exist, use unknown and explain.
- Use only actual column names listed in the schema.
- Do not claim that a calculation has already been performed.
- Forecasting is an estimate, not a guarantee.
- Return valid JSON only.

JSON shape:
{
 "operation":"rank",
 "entity_column":null,
 "metric_column":null,
 "aggregation":"sum",
 "direction":"desc",
 "top_n":10,
 "group_by":[],
 "date_column":null,
 "frequency":"month",
 "periods":6,
 "column":null,
 "columns":[],
 "filters":[],
 "answer_note":""
}
"""

    user_prompt = {
        "question": question,
        "dataset_schema": schema,
        "detected_semantic_columns": {
            k: str(v) for k, v in profile["roles"].items()
        },
        "numeric_columns": [str(c) for c in profile["numeric"]],
        "categorical_columns": [str(c) for c in profile["categorical"]],
        "date_columns": [str(c) for c in profile["dates"]],
    }

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
        "response_format": {"type": "json_object"},
    }

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))

        plan = json.loads(result["choices"][0]["message"]["content"])
        if not isinstance(plan, dict):
            raise ValueError("The model did not return a JSON object.")
        return plan, None

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return None, f"LLM API HTTP {exc.code}: {detail[:400]}"
    except Exception as exc:
        return None, f"LLM interpretation failed: {exc}"


# ============================================================
# 7. LOCAL QUESTION INTERPRETER (NO API KEY REQUIRED)
# ============================================================

def local_question_plan(question, df):
    q = normalize_name(question)
    profile = automatic_review(df)
    roles = profile["roles"]
    numeric = profile["numeric"]
    categories = profile["categorical"]
    dates = profile["dates"]

    plan = {
        "operation": "unknown",
        "entity_column": None,
        "metric_column": None,
        "aggregation": "sum",
        "direction": "desc",
        "top_n": 10,
        "group_by": [],
        "date_column": roles.get("date"),
        "frequency": "month",
        "periods": 6,
        "column": None,
        "columns": [],
        "filters": [],
        "answer_note": "",
    }

    def has(*phrases):
        return any(normalize_name(p) in q for p in phrases)

    mentioned = None
    for col in df.columns:
        col_name = normalize_name(col)
        if col_name and col_name in q:
            mentioned = col
            break

    # Interpret semantic references such as "product" even if the
    # literal column is productName or drug_name.
    entity = roles.get("product")
    if has("customer", "client", "buyer"):
        entity = roles.get("customer") or entity
    elif has("region", "country", "city", "territory", "branch"):
        entity = roles.get("region") or entity
    elif has("category", "product type", "department"):
        entity = roles.get("category") or entity

    if has("list columns", "what columns", "available columns",
           "which columns", "column names"):
        plan["operation"] = "columns"
        return plan

    if has("missing values", "missing data", "null values",
           "incomplete data", "data quality"):
        plan["operation"] = "missing"
        return plan

    if has("duplicate rows", "duplicate records", "how many duplicates"):
        plan["operation"] = "duplicates"
        return plan

    if has("profile dataset", "describe dataset", "summarize dataset",
           "dataset overview", "review my data", "analyze my data",
           "analyse my data", "initial analysis"):
        plan["operation"] = "profile"
        return plan

    if has("correlation", "correlated", "relationship between"):
        plan["operation"] = "correlation"
        plan["columns"] = [str(c) for c in numeric]
        return plan

    if has("forecast", "forecasting", "predict", "prediction",
           "future", "next month", "next 6 months", "next year"):
        plan["operation"] = "forecast"
        plan["date_column"] = roles.get("date") or (dates[0] if dates else None)
        plan["metric_column"] = (
            roles.get("revenue") or roles.get("quantity")
            or (numeric[0] if numeric else None)
        )
        if has("profit", "profitable"):
            plan["metric_column"] = roles.get("profit")
        if has("quantity", "units sold", "volume"):
            plan["metric_column"] = roles.get("quantity")
        if has("cost", "expenses"):
            plan["metric_column"] = roles.get("cost")
        if has("weekly", "by week", "weekly forecast"):
            plan["frequency"] = "week"
        elif has("daily", "by day", "daily forecast"):
            plan["frequency"] = "day"
        elif has("quarter", "quarterly"):
            plan["frequency"] = "quarter"
        elif has("year", "yearly", "annually"):
            plan["frequency"] = "year"
        nums = re.findall(r"\b(\d{1,2})\b", q)
        if nums:
            plan["periods"] = min(max(int(nums[0]), 1), 24)
        return plan

    if has("best selling", "best selling product", "most selling",
           "sells the most", "top selling", "most popular product",
           "most ordered product", "top products", "rank products",
           "best performing product"):
        plan["operation"] = "rank"
        plan["entity_column"] = entity
        plan["metric_column"] = roles.get("quantity")
        plan["aggregation"] = "sum"

        if has("revenue", "sales value", "sales amount", "highest sales"):
            plan["metric_column"] = roles.get("revenue")
        elif has("profit", "profitable"):
            plan["metric_column"] = roles.get("profit")
        elif has("orders", "number of transactions", "most transactions"):
            plan["metric_column"] = roles.get("order")
            plan["aggregation"] = "count"

        if plan["metric_column"] is None:
            plan["aggregation"] = "count"
            plan["answer_note"] = (
                "No quantity column was detected. Ranking by row count "
                "instead; this is a proxy, not confirmed units sold."
            )
        return plan

    if has("most profitable", "highest profit", "top profit", "profit by product"):
        plan["operation"] = "rank"
        plan["entity_column"] = entity
        plan["metric_column"] = roles.get("profit")
        if plan["metric_column"] is None:
            plan["operation"] = "unknown"
            plan["answer_note"] = (
                "I could not identify a profit column. A revenue or sales "
                "column is not the same as profit."
            )
        return plan

    if has("highest revenue", "most revenue", "highest sales",
           "most sales", "revenue by product", "sales by product",
           "which product generated"):
        plan["operation"] = "rank"
        plan["entity_column"] = entity
        plan["metric_column"] = roles.get("revenue")
        return plan

    if has("trend", "over time", "by month", "by year", "monthly",
           "daily", "weekly", "quarterly", "growth over time",
           "changed month", "month by month"):
        plan["operation"] = "trend"
        plan["date_column"] = roles.get("date") or (dates[0] if dates else None)
        plan["metric_column"] = (
            mentioned if mentioned in numeric
            else roles.get("revenue") or roles.get("quantity")
            or (numeric[0] if numeric else None)
        )
        if has("profit"):
            plan["metric_column"] = roles.get("profit")
        if has("quantity", "units"):
            plan["metric_column"] = roles.get("quantity")
        if has("daily", "by day"):
            plan["frequency"] = "day"
        elif has("weekly", "by week"):
            plan["frequency"] = "week"
        elif has("quarter", "quarterly"):
            plan["frequency"] = "quarter"
        elif has("year", "yearly", "annually"):
            plan["frequency"] = "year"
        return plan

    if has("average", "mean", "typical"):
        plan["operation"] = "aggregate"
        plan["metric_column"] = (
            mentioned if mentioned in numeric else roles.get("revenue")
        )
        plan["aggregation"] = "mean"
        return plan

    if has("median", "middle value"):
        plan["operation"] = "aggregate"
        plan["metric_column"] = (
            mentioned if mentioned in numeric else roles.get("revenue")
        )
        plan["aggregation"] = "median"
        return plan

    if has("total", "sum of", "overall revenue", "overall sales",
           "how much", "total amount"):
        plan["operation"] = "aggregate"
        if has("profit"):
            plan["metric_column"] = roles.get("profit")
        elif has("quantity", "units"):
            plan["metric_column"] = roles.get("quantity")
        elif has("cost", "expenses"):
            plan["metric_column"] = roles.get("cost")
        else:
            plan["metric_column"] = roles.get("revenue")
        return plan

    if has("distribution", "histogram", "spread of", "breakdown"):
        plan["operation"] = "distribution"
        plan["column"] = mentioned or (
            categories[0] if categories else
            numeric[0] if numeric else None
        )
        return plan

    if has("top", "bottom", "rank", "ranking", "highest by", "lowest by"):
        plan["operation"] = "rank"
        plan["entity_column"] = (
            mentioned if mentioned in categories else entity
        )
        plan["metric_column"] = (
            roles.get("profit") if "profit" in q
            else roles.get("quantity") if has("quantity", "units")
            else roles.get("revenue")
        )
        if has("bottom", "lowest"):
            plan["direction"] = "asc"
        nums = re.findall(r"\b(\d{1,3})\b", q)
        if nums:
            plan["top_n"] = min(max(int(nums[0]), 1), 100)
        return plan

    if has("how many rows", "number of rows", "how many records",
           "record count", "how many transactions"):
        plan["operation"] = "aggregate"
        plan["aggregation"] = "count"
        return plan

    if mentioned is not None and has("show", "display", "compare",
                                     "comparison", "distribution"):
        plan["operation"] = "distribution"
        plan["column"] = mentioned
        return plan

    return plan


# ============================================================
# 8. PLAN VALIDATION
# ============================================================

def validate_plan(plan, df):
    if not isinstance(plan, dict):
        raise ValueError("The analysis plan is not a valid object.")

    plan = dict(plan)
    operation = str(plan.get("operation", "unknown")).lower()
    if operation not in ALLOWED_OPERATIONS:
        operation = "unknown"
    plan["operation"] = operation

    def resolve(value):
        if value is None:
            return None
        target = normalize_name(value)

        for col in df.columns:
            if normalize_name(col) == target:
                return col

        # Permit a close match for camelCase / minor naming differences.
        normalized = [normalize_name(c) for c in df.columns]
        matches = get_close_matches(target, normalized, n=1, cutoff=0.88)
        if matches:
            return df.columns[normalized.index(matches[0])]

        # Map semantic words to inferred columns.
        roles = infer_semantic_columns(df)
        semantic = {
            "product": "product",
            "item": "product",
            "drug": "product",
            "medicine": "product",
            "quantity": "quantity",
            "units": "quantity",
            "revenue": "revenue",
            "sales": "revenue",
            "profit": "profit",
            "customer": "customer",
            "region": "region",
            "category": "category",
            "date": "date",
        }
        if target in semantic:
            return roles.get(semantic[target])
        return None

    for field in ["metric_column", "entity_column", "date_column", "column"]:
        if plan.get(field) is not None:
            resolved = resolve(plan[field])
            if resolved is None:
                raise ValueError(
                    f"Could not map '{plan[field]}' to a column. "
                    f"Available columns: {', '.join(map(str, df.columns[:30]))}"
                )
            plan[field] = resolved

    for field in ["group_by", "columns"]:
        resolved = []
        for value in plan.get(field) or []:
            col = resolve(value)
            if col is None:
                raise ValueError(f"Unknown column: {value}")
            resolved.append(col)
        plan[field] = resolved

    aggregation = str(plan.get("aggregation", "sum")).lower()
    plan["aggregation"] = (
        aggregation if aggregation in ALLOWED_AGGREGATIONS else "sum"
    )

    direction = str(plan.get("direction", "desc")).lower()
    plan["direction"] = "asc" if direction in [
        "asc", "ascending", "bottom", "lowest"
    ] else "desc"

    for field, default, maximum in [
        ("top_n", 10, 100),
        ("periods", 6, 24),
    ]:
        try:
            plan[field] = min(max(int(plan.get(field, default)), 1), maximum)
        except (TypeError, ValueError):
            plan[field] = default

    frequency = str(plan.get("frequency", "month")).lower()
    plan["frequency"] = frequency if frequency in [
        "day", "week", "month", "quarter", "year"
    ] else "month"

    filters = []
    for item in plan.get("filters") or []:
        if not isinstance(item, dict):
            continue
        col = resolve(item.get("column"))
        op = str(item.get("operator", "eq")).lower()
        if col is None:
            raise ValueError(f"Filter uses an unknown column: {item.get('column')}")
        if op not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported filter operator: {op}")
        filters.append({
            "column": col, "operator": op, "value": item.get("value")
        })
    plan["filters"] = filters
    return plan


# ============================================================
# 9. FILTERING
# ============================================================

def apply_filters(df, filters):
    result = df.copy()

    for item in filters:
        col, op, value = item["column"], item["operator"], item.get("value")
        s = result[col]

        if op == "contains":
            mask = s.astype("string").str.contains(
                str(value), case=False, na=False, regex=False
            )
            result = result[mask]
            continue

        if op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("A between filter requires [start, end].")
            left, right = value
            if pd.api.types.is_numeric_dtype(s):
                vals = numeric_series(s)
                result = result[vals.between(float(left), float(right))]
            else:
                result = result[s.astype(str).between(str(left), str(right))]
            continue

        if op in ["gt", "gte", "lt", "lte"]:
            if pd.api.types.is_numeric_dtype(s):
                left = numeric_series(s)
                right = float(value)
            else:
                left = pd.to_datetime(s, errors="coerce")
                right = pd.to_datetime(value, errors="coerce")
                if pd.isna(right):
                    raise ValueError(
                        f"Cannot compare '{value}' with {col}; "
                        "expected a numeric value or date."
                    )

            if op == "gt":
                mask = left > right
            elif op == "gte":
                mask = left >= right
            elif op == "lt":
                mask = left < right
            else:
                mask = left <= right

            result = result[mask.fillna(False)]
            continue

        if op in ["eq", "ne"]:
            if pd.api.types.is_numeric_dtype(s):
                left = numeric_series(s)
                try:
                    right = float(value)
                except (ValueError, TypeError):
                    left = s.astype(str)
                    right = str(value)
            else:
                left, right = s.astype(str), str(value)

            mask = left == right
            if op == "ne":
                mask = ~mask
            result = result[mask.fillna(False)]

    return result


# ============================================================
# 10. FORECASTING
# ============================================================

def build_forecast(df, date_col, metric_col, frequency="month",
                   periods=6, aggregation="sum"):
    """
    Simple transparent baseline forecast:
    1. Aggregate the measure into regular time periods.
    2. Fit a linear trend.
    3. Estimate a rough uncertainty band from historical residuals.

    This is a baseline forecast, not a guarantee or a causal model.
    """

    if date_col not in df.columns or metric_col not in df.columns:
        raise ValueError("Forecast needs a valid date and measure column.")

    work = df[[date_col, metric_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[metric_col] = numeric_series(work[metric_col])
    work = work.dropna()

    if work.empty:
        raise ValueError("No valid date and numeric measure pairs were found.")

    freq_map = {
        "day": "D", "week": "W", "month": "MS",
        "quarter": "QS", "year": "YS"
    }
    freq = freq_map.get(frequency, "MS")

    work = work.set_index(date_col).sort_index()
    if aggregation == "mean":
        series = work[metric_col].resample(freq).mean()
    elif aggregation == "median":
        series = work[metric_col].resample(freq).median()
    elif aggregation == "count":
        series = work[metric_col].resample(freq).count()
    else:
        series = work[metric_col].resample(freq).sum()

    series = series.dropna()
    if len(series) < 4:
        raise ValueError(
            f"Only {len(series)} historical periods are available. "
            "At least 4 are required for this baseline forecast; "
            "more history generally produces a more reliable estimate."
        )

    # Remove periods with no observations from the fitting input.
    y = series.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residuals = y - fitted

    # Use a rough residual standard deviation as an uncertainty band.
    error = float(np.std(residuals, ddof=1)) if len(y) > 2 else 0.0
    future_x = np.arange(len(y), len(y) + periods, dtype=float)
    predictions = intercept + slope * future_x

    # Future period labels.
    last_date = series.index[-1]
    future_index = pd.date_range(
        start=last_date, periods=periods + 1, freq=freq
    )[1:]

    forecast = pd.DataFrame({
        "Period": future_index,
        "Forecast": predictions,
        "Lower estimate": predictions - 1.96 * error,
        "Upper estimate": predictions + 1.96 * error,
    })

    history = pd.DataFrame({
        "Period": series.index,
        "Actual": y,
    })

    return history, forecast, error


# ============================================================
# 11. ANALYSIS ENGINE
# ============================================================

def execute_plan(original_df, plan):
    df = apply_filters(original_df, plan.get("filters", []))
    op = plan["operation"]
    note = plan.get("answer_note", "")

    result = {"answer": "", "table": None, "chart": None}

    if op == "unknown":
        result["answer"] = note or (
            "I couldn't confidently interpret that question. Try specifying "
            "the measure (such as revenue, profit, or quantity) and the "
            "entity (such as product, customer, or region)."
        )
        return result

    if op == "columns":
        result["answer"] = (
            f"The dataset has {len(df):,} rows and {len(df.columns)} columns."
        )
        result["table"] = make_schema(df)
        return result

    if op == "profile":
        review = automatic_review(df)
        result["answer"] = "\n\n".join(f"- {x}" for x in review["insights"])
        result["table"] = make_schema(df)
        return result

    if op == "missing":
        table = make_schema(df)[["Column", "Missing values", "Missing %"]]
        table = table.sort_values("Missing values", ascending=False)
        result["answer"] = (
            f"Found {int(df.isna().sum().sum()):,} missing cells."
        )
        result["table"] = table
        result["chart"] = {
            "kind": "bar", "x": "Column", "y": "Missing values",
            "title": "Missing values by column"
        }
        return result

    if op == "duplicates":
        result["answer"] = (
            f"Found {int(df.duplicated().sum()):,} duplicate rows "
            f"out of {len(df):,} records."
        )
        return result

    if op in ["show_rows", "filter"]:
        result["answer"] = f"{len(df):,} records match the current conditions."
        result["table"] = df.head(100)
        return result

    if op == "aggregate":
        metric = plan.get("metric_column")
        agg = plan.get("aggregation", "sum")
        groups = plan.get("group_by") or []

        if metric is None:
            if agg == "count":
                result["answer"] = f"Total row count: {len(df):,}."
                return result
            raise ValueError("I couldn't identify a measure column.")

        work = df.copy()
        if agg != "count":
            work[metric] = numeric_series(work[metric])
            if work[metric].notna().sum() == 0:
                raise ValueError(f"'{metric}' has no usable numeric values.")

        if groups:
            if agg == "count":
                table = work.groupby(groups, dropna=False).size().reset_index(
                    name="record_count"
                )
                value_col = "record_count"
            else:
                table = work.groupby(groups, dropna=False)[metric].agg(
                    agg
                ).reset_index()
                value_col = metric

            table = table.sort_values(value_col, ascending=False)
            result["answer"] = (
                f"Calculated {agg} of `{metric}` grouped by "
                f"{', '.join(map(str, groups))}."
            )
            result["table"] = table.head(100)
            if len(groups) == 1:
                result["chart"] = {
                    "kind": "bar", "x": groups[0], "y": value_col,
                    "title": f"{agg.title()} of {metric} by {groups[0]}"
                }
            return result

        if agg == "count":
            value = int(work[metric].count())
        else:
            values = numeric_series(work[metric]).dropna()
            if values.empty:
                raise ValueError(f"No numeric values found in '{metric}'.")
            value = getattr(values, agg)()

        result["answer"] = f"{agg.title()} of `{metric}`: **{display_number(value)}**."
        return result

    if op == "rank":
        entity = plan.get("entity_column")
        metric = plan.get("metric_column")
        agg = plan.get("aggregation", "sum")

        if entity is None:
            raise ValueError(
                "I couldn't identify the entity to rank. Check the column "
                "schema or ask about a specific product/customer/region."
            )

        if metric is None or agg == "count":
            table = df.groupby(entity, dropna=False).size().reset_index(
                name="record_count"
            )
            value_col = "record_count"
            measure = "record count"
        else:
            values = numeric_series(df[metric])
            work = pd.DataFrame({entity: df[entity], metric: values})
            table = work.groupby(entity, dropna=False)[metric].agg(
                agg
            ).reset_index()
            value_col = metric
            measure = f"{agg} of {metric}"

        table = table.sort_values(
            value_col, ascending=plan.get("direction") == "asc"
        ).head(plan.get("top_n", 10))

        if table.empty:
            raise ValueError("No usable records were available to rank.")

        winner = table.iloc[0]
        result["answer"] = (
            f"Top-ranked **{winner[entity]}** by {measure}: "
            f"**{display_number(winner[value_col])}**."
        )
        if note:
            result["answer"] += f"\n\nNote: {note}"
        result["table"] = table
        result["chart"] = {
            "kind": "bar", "x": entity, "y": value_col,
            "title": f"Top {len(table)} by {measure}"
        }
        return result

    if op == "trend":
        date_col, metric = plan.get("date_column"), plan.get("metric_column")
        if date_col is None or metric is None:
            raise ValueError("Trend analysis needs a date and a numeric measure.")

        work = df[[date_col, metric]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work[metric] = numeric_series(work[metric])
        work = work.dropna()
        if work.empty:
            raise ValueError("No valid date/measure pairs were found.")

        freq_map = {
            "day": "D", "week": "W", "month": "MS",
            "quarter": "QS", "year": "YS"
        }
        freq = freq_map.get(plan.get("frequency", "month"), "MS")
        series = work.set_index(date_col)[metric].resample(freq).agg(
            plan.get("aggregation", "sum")
        ).dropna()

        table = pd.DataFrame({
            "Period": series.index,
            metric: series.values,
        })
        result["answer"] = (
            f"Calculated {plan.get('aggregation', 'sum')} of `{metric}` "
            f"by {plan.get('frequency', 'month')}. "
            f"Historical periods: {len(table)}."
        )
        result["table"] = table
        result["chart"] = {
            "kind": "line", "x": "Period", "y": metric,
            "title": f"{metric} over time"
        }
        return result

    if op == "forecast":
        date_col = plan.get("date_column")
        metric = plan.get("metric_column")
        if date_col is None or metric is None:
            raise ValueError(
                "Forecasting requires a date column and a numeric measure. "
                "DEXCR could not identify both in this dataset."
            )

        history, forecast, error = build_forecast(
            df,
            date_col,
            metric,
            frequency=plan.get("frequency", "month"),
            periods=plan.get("periods", 6),
            aggregation=plan.get("aggregation", "sum"),
        )

        result["answer"] = (
            f"Created a baseline {plan.get('frequency', 'month')} forecast "
            f"for `{metric}` over the next {len(forecast)} periods. "
            f"The approximate historical residual standard deviation is "
            f"{display_number(error)}. Forecasts are estimates, not guarantees."
        )
        result["table"] = forecast
        result["chart"] = {
            "kind": "forecast",
            "history": history,
            "title": f"Historical and forecast values for {metric}"
        }
        return result

    if op == "distribution":
        col = plan.get("column")
        if col is None:
            raise ValueError("Specify the column you want to explore.")

        if pd.api.types.is_numeric_dtype(df[col]) or (
            numeric_series(df[col]).notna().mean() >= 0.85
        ):
            values = numeric_series(df[col]).dropna()
            result["answer"] = (
                f"`{col}` — mean: {display_number(values.mean())}; "
                f"median: {display_number(values.median())}; "
                f"minimum: {display_number(values.min())}; "
                f"maximum: {display_number(values.max())}."
            )
            result["chart"] = {
                "kind": "histogram", "x": col, "title": f"Distribution of {col}"
            }
        else:
            table = (
                df[col].astype("string").fillna("(Missing)")
                .value_counts().head(30).rename_axis(col)
                .reset_index(name="record_count")
            )
            result["answer"] = f"Most common values in `{col}`."
            result["table"] = table
            result["chart"] = {
                "kind": "bar", "x": col, "y": "record_count",
                "title": f"Distribution of {col}"
            }
        return result

    if op == "correlation":
        cols = plan.get("columns") or []
        usable = []
        for col in cols:
            vals = numeric_series(df[col])
            if vals.notna().sum() > 1:
                usable.append(col)

        if len(usable) < 2:
            usable = [
                c for c in df.columns
                if numeric_series(df[c]).notna().mean() >= 0.85
            ]

        if len(usable) < 2:
            raise ValueError("Correlation requires at least two numeric columns.")

        work = df[usable].copy()
        for col in usable:
            work[col] = numeric_series(work[col])
        matrix = work.corr(numeric_only=True)
        result["answer"] = (
            "Pearson correlations between numeric variables. "
            "Correlation does not establish causation."
        )
        result["table"] = matrix.reset_index(names="Column")
        return result

    raise ValueError(f"Unsupported analysis operation: {op}")


# ============================================================
# 12. SESSION STATE AND DATASET SELECTION
# ============================================================

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

with st.sidebar:
    st.markdown("## ◈ DEXCR")
    st.caption("AI-powered data intelligence")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        ["Overview", "My Data", "Data Explorer", "AI Analyst"]
    )

    st.divider()

    has_uploaded = (
        st.session_state.uploaded_df is not None
        and not st.session_state.uploaded_df.empty
    )
    choices = ["Sample sales data"]
    if has_uploaded:
        choices.insert(0, "My uploaded data")

    data_choice = st.radio("ACTIVE DATASET", choices, index=0)

if data_choice == "My uploaded data" and has_uploaded:
    df = st.session_state.uploaded_df
    dataset_name = st.session_state.uploaded_name
else:
    df = load_sample_data()
    dataset_name = "Sample sales transactions"

has_data = df is not None and not df.empty


# ============================================================
# 13. OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown("""
    <div class="hero">
        <h1>DEXCR</h1>
        <h3>Ask your data better questions.</h3>
        <p>Upload a dataset, review its quality, discover patterns,
        ask questions, and forecast suitable time series.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Platform", "DEXCR")
    c2.metric("Supported files", "CSV · XLSX · XLS")
    c3.metric("Active dataset", "Ready" if has_data else "No data")

    st.subheader("Analytics workspace")
    st.markdown("""
    - **My Data:** upload a dataset and automatically review its structure.
    - **Data Explorer:** inspect columns, numeric summaries, and trends.
    - **AI Analyst:** ask questions using natural language.
    - **Forecasting:** estimate future values when sufficient date history exists.
    """)

    if has_data:
        st.success(
            f"Active dataset: {dataset_name} — "
            f"{len(df):,} rows and {len(df.columns)} columns."
        )
        review = automatic_review(df)
        st.subheader("Automatic data review")
        for insight in review["insights"]:
            st.markdown(f"- {insight}")

    if not get_setting("OPENAI_API_KEY"):
        st.info(
            "Local interpretation is enabled. Add an LLM API key in "
            "Streamlit Secrets for more flexible natural-language questions."
        )


# ============================================================
# 14. MY DATA: UPLOAD + AUTOMATIC REVIEW
# ============================================================

elif page == "My Data":
    st.title("My Data")
    st.write(
        "Upload a dataset from any domain. DEXCR will profile it and "
        "infer likely business meanings from its column names and values."
    )

    uploaded_file = st.file_uploader(
        "Choose your dataset",
        type=["csv", "xlsx", "xls"],
        key="dataset_uploader",
    )

    if uploaded_file is not None:
        current_signature = (
            uploaded_file.name, uploaded_file.size
        )
        if st.session_state.get("upload_signature") != current_signature:
            try:
                new_df = read_uploaded_file(uploaded_file)
                if new_df.empty:
                    st.error("The uploaded file contains no rows.")
                else:
                    # Remove completely empty columns and normalize headers.
                    new_df.columns = [
                        str(c).strip() for c in new_df.columns
                    ]
                    new_df = new_df.loc[
                        :, ~new_df.columns.astype(str).str.match(r"^Unnamed:")
                    ]

                    if new_df.columns.duplicated().any():
                        st.warning(
                            "Duplicate column names were detected. "
                            "Please rename them in the source file before analysis."
                        )
                    else:
                        st.session_state.uploaded_df = new_df
                        st.session_state.uploaded_name = uploaded_file.name
                        st.session_state.upload_signature = current_signature
                        st.rerun()
            except Exception as exc:
                st.error(f"Could not read this file: {exc}")

    if has_data:
        review = automatic_review(df)

        st.divider()
        st.subheader(f"Automatic review: {dataset_name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Numeric fields", len(review["numeric"]))
        c4.metric("Date fields", len(review["dates"]))

        st.markdown("### Initial findings")
        for insight in review["insights"]:
            st.markdown(f"- {insight}")

        st.markdown("### Detected column meanings")
        if review["roles"]:
            st.dataframe(
                pd.DataFrame([
                    {"Meaning": role.title(), "Detected column": col}
                    for role, col in review["roles"].items()
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No familiar semantic roles were detected. "
                "You can still inspect columns and ask about exact column names."
            )

        st.markdown("### Column schema")
        st.dataframe(make_schema(df), use_container_width=True)

        st.markdown("### Data preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("### Data quality")
        a, b, c = st.columns(3)
        missing = int(df.isna().sum().sum())
        a.metric("Missing cells", f"{missing:,}")
        b.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        c.metric("Complete cells", f"{df.size - missing:,}")

        # Forecast controls appear automatically when a date and numeric
        # measure are available.
        if review["dates"] and review["numeric"]:
            st.divider()
            st.markdown("### Quick forecast")
            st.caption(
                "Choose a date and measure. Forecasting needs enough distinct "
                "historical periods; otherwise DEXCR will explain the limitation."
            )

            fc1, fc2 = st.columns(2)
            date_col = fc1.selectbox(
                "Date field", review["dates"], key="quick_fc_date"
            )
            metric_options = review["numeric"]
            preferred = review["roles"].get("revenue")
            default_metric_index = (
                metric_options.index(preferred) if preferred in metric_options else 0
            )
            metric_col = fc2.selectbox(
                "Measure to forecast",
                metric_options,
                index=default_metric_index,
                key="quick_fc_metric",
            )
            fc3, fc4 = st.columns(2)
            frequency = fc3.selectbox(
                "Forecast frequency",
                ["month", "week", "day", "quarter", "year"],
                index=0,
                key="quick_fc_frequency",
            )
            periods = fc4.selectbox(
                "Periods to forecast",
                [3, 6, 12, 18, 24],
                index=1,
                key="quick_fc_periods",
            )

            if st.button("Generate forecast", type="primary"):
                try:
                    history, forecast, error = build_forecast(
                        df, date_col, metric_col, frequency, periods, "sum"
                    )
                    st.success(
                        f"Forecast generated. Historical residual standard "
                        f"deviation: {display_number(error)}."
                    )
                    st.dataframe(forecast, use_container_width=True)

                    hist_plot = history.copy()
                    hist_plot["Type"] = "Historical"
                    hist_plot = hist_plot.rename(columns={"Actual": "Value"})

                    future_plot = forecast[["Period", "Forecast"]].copy()
                    future_plot["Type"] = "Forecast"
                    future_plot = future_plot.rename(columns={"Forecast": "Value"})

                    combined = pd.concat(
                        [hist_plot[["Period", "Value", "Type"]],
                         future_plot[["Period", "Value", "Type"]]],
                        ignore_index=True,
                    )
                    st.plotly_chart(
                        px.line(
                            combined, x="Period", y="Value", color="Type",
                            markers=True, title=f"{metric_col}: history and forecast",
                            template="plotly_white",
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "This is a simple linear-trend baseline. It does not "
                        "automatically model seasonality, promotions, market "
                        "changes, or causal drivers."
                    )
                except Exception as exc:
                    st.warning(str(exc))

        st.download_button(
            "Download active dataset as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="dexcr_export.csv",
            mime="text/csv",
        )
        st.caption(
            "The upload is kept in the active Streamlit session; this code "
            "does not write your uploaded dataset into the GitHub repository."
        )
    else:
        st.info("Upload a CSV or Excel file to start your analysis.")


# ============================================================
# 15. DATA EXPLORER
# ============================================================

elif page == "Data Explorer":
    st.title("Data Explorer")

    if not has_data:
        st.warning("Upload a dataset first.")
    else:
        review = automatic_review(df)
        st.caption(
            f"{dataset_name} · {len(df):,} rows · {len(df.columns)} columns"
        )

        search = st.text_input(
            "Search column names", placeholder="Type a column name..."
        )
        visible = [
            c for c in df.columns
            if normalize_name(search) in normalize_name(c)
        ]
        st.dataframe(df[visible].head(500), use_container_width=True)

        if review["numeric"]:
            st.divider()
            st.subheader("Numeric measure")
            measure = st.selectbox("Select a measure", review["numeric"])
            values = numeric_series(df[measure]).dropna()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", display_number(values.sum()))
            c2.metric("Average", display_number(values.mean()))
            c3.metric("Median", display_number(values.median()))
            c4.metric("Distinct values", f"{values.nunique():,}")

            st.plotly_chart(
                px.histogram(
                    x=values, title=f"Distribution of {measure}",
                    labels={"x": measure}, template="plotly_white"
                ),
                use_container_width=True,
            )

            if review["categorical"]:
                group_col = st.selectbox(
                    "Break down by category", review["categorical"]
                )
                work = pd.DataFrame({
                    group_col: df[group_col],
                    measure: numeric_series(df[measure]),
                })
                grouped = (
                    work.groupby(group_col, dropna=False)[measure]
                    .sum().reset_index()
                    .sort_values(measure, ascending=False).head(25)
                )
                st.plotly_chart(
                    px.bar(
                        grouped, x=group_col, y=measure,
                        title=f"{measure} by {group_col}",
                        template="plotly_white",
                    ),
                    use_container_width=True,
                )
                st.dataframe(grouped, use_container_width=True)

        if review["dates"] and review["numeric"]:
            st.divider()
            st.subheader("Time trend")
            date_col = st.selectbox("Date field", review["dates"], key="de_date")
            trend_measure = st.selectbox(
                "Measure", review["numeric"], key="de_measure"
            )
            trend = pd.DataFrame({
                "Date": pd.to_datetime(df[date_col], errors="coerce"),
                trend_measure: numeric_series(df[trend_measure]),
            }).dropna()
            if not trend.empty:
                trend["Period"] = trend["Date"].dt.to_period("M").astype(str)
                monthly = trend.groupby("Period")[trend_measure].sum().reset_index()
                st.plotly_chart(
                    px.line(
                        monthly, x="Period", y=trend_measure, markers=True,
                        title=f"{trend_measure} over time",
                        template="plotly_white",
                    ),
                    use_container_width=True,
                )


# ============================================================
# 16. AI ANALYST
# ============================================================

elif page == "AI Analyst":
    st.title("AI Analyst")
    st.write(
        "Ask a question about the active dataset. DEXCR maps the meaning "
        "of your words to columns, validates the plan, and calculates "
        "results in Python."
    )

    if not has_data:
        st.warning("Upload a dataset before asking questions.")
    else:
        review = automatic_review(df)

        with st.expander("Detected column meanings"):
            if review["roles"]:
                st.json({k: str(v) for k, v in review["roles"].items()})
            st.dataframe(make_schema(df), use_container_width=True)

        question = st.text_area(
            "What would you like to know?",
            placeholder=(
                "Examples:\n"
                "• What is my best-selling product?\n"
                "• Which drug generated the most revenue?\n"
                "• What is the total quantity sold by productName?\n"
                "• Which region has the highest profit?\n"
                "• Forecast revenue for the next 6 months.\n"
                "• What data-quality issues should I investigate?"
            ),
            height=140,
        )

        if st.button("Analyze question", type="primary"):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    plan, llm_error = get_llm_plan(question.strip(), df)

                    if plan is None:
                        plan = local_question_plan(question.strip(), df)
                        st.caption(
                            "Using local interpretation. Configure an LLM API "
                            "key for broader language understanding."
                        )
                        if llm_error and get_setting("OPENAI_API_KEY"):
                            with st.expander("LLM connection details"):
                                st.write(llm_error)

                    plan = validate_plan(plan, df)

                    with st.expander("View analysis plan"):
                        st.json(plan)

                    result = execute_plan(df, plan)
                    st.divider()
                    st.subheader("Answer")
                    st.markdown(result["answer"])

                    if result["table"] is not None:
                        st.dataframe(result["table"], use_container_width=True)

                    chart = result["chart"]
                    if chart:
                        if chart["kind"] == "bar":
                            st.plotly_chart(
                                px.bar(
                                    result["table"],
                                    x=chart["x"], y=chart["y"],
                                    title=chart["title"],
                                    template="plotly_white",
                                ),
                                use_container_width=True,
                            )
                        elif chart["kind"] == "line":
                            st.plotly_chart(
                                px.line(
                                    result["table"],
                                    x=chart["x"], y=chart["y"],
                                    markers=True, title=chart["title"],
                                    template="plotly_white",
                                ),
                                use_container_width=True,
                            )
                        elif chart["kind"] == "histogram":
                            st.plotly_chart(
                                px.histogram(
                                    df, x=chart["x"], title=chart["title"],
                                    template="plotly_white",
                                ),
                                use_container_width=True,
                            )
                        elif chart["kind"] == "forecast":
                            history = chart["history"].copy()
                            history["Type"] = "Historical"
                            history = history.rename(columns={"Actual": "Value"})

                            forecast = result["table"][["Period", "Forecast"]].copy()
                            forecast["Type"] = "Forecast"
                            forecast = forecast.rename(columns={"Forecast": "Value"})

                            combined = pd.concat([
                                history[["Period", "Value", "Type"]],
                                forecast[["Period", "Value", "Type"]],
                            ], ignore_index=True)

                            st.plotly_chart(
                                px.line(
                                    combined, x="Period", y="Value",
                                    color="Type", markers=True,
                                    title=chart["title"],
                                    template="plotly_white",
                                ),
                                use_container_width=True,
                            )

                    st.caption(
                        "Review the selected measure, aggregation, and assumptions "
                        "before using the results for decisions."
                    )

                except Exception as exc:
                    st.error(
                        "DEXCR could not complete this analysis. "
                        "The required field may be missing, the question may be "
                        "ambiguous, or the operation may not be supported."
                    )
                    st.write(str(exc))

        st.divider()
        st.subheader("Try a question")
        examples = [
            "What is my best-selling product?",
            "Which product generated the most revenue?",
            "Which product is the most profitable?",
            "Show the top 10 products by quantity sold.",
            "What is the average revenue?",
            "What is the total profit?",
            "Which region has the highest sales?",
            "Show the distribution of my product categories.",
            "How have sales changed month by month?",
            "Forecast revenue for the next 6 months.",
            "Which columns contain missing values?",
            "How many duplicate records are there?",
            "Which numeric variables are correlated?",
        ]
        for example in examples:
            st.markdown(f"- {example}")


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()
st.caption(
    "DEXCR · General-purpose data analytics · "
    "Python performs calculations on the active dataset."
)
