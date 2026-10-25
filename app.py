
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
# 1. PAGE CONFIGURATION AND DESIGN
# ============================================================

st.set_page_config(
    page_title="DEXCR | AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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
    }

    .muted {
        color: #6b7280;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. FILE LOADING
# ============================================================

def read_uploaded_file(uploaded_file):
    """Read a CSV or Excel file."""

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file,
                encoding="latin-1",
            )

    if filename.endswith(".xlsx"):
        uploaded_file.seek(0)
        return pd.read_excel(
            uploaded_file,
            engine="openpyxl",
        )

    raise ValueError(
        "Unsupported file. Upload a CSV or XLSX file."
    )


def load_sample_data():
    """Load an optional sample sales dataset."""

    path = BASE_DIR / "sales_transactions.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


# ============================================================
# 3. GENERAL DATA UTILITIES
# ============================================================

def normalize_name(value):
    """Normalize column names and question text for matching."""

    value = str(value).lower().strip()
    value = re.sub(r"[_\-/]+", " ", value)
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def numeric_series(series):
    """Convert common numeric strings into numeric values."""

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def display_number(value):
    """Format numbers for presentation."""

    if pd.isna(value):
        return "N/A"

    if isinstance(value, (int, np.integer)):
        return f"{value:,}"

    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"

    return str(value)


def find_column(keywords, columns):
    """Find a column by a list of prioritized name hints."""

    normalized_columns = {
        normalize_name(col): col for col in columns
    }

    for keyword in keywords:
        key = normalize_name(keyword)

        if key in normalized_columns:
            return normalized_columns[key]

    for keyword in keywords:
        key = normalize_name(keyword)

        for normalized, original in normalized_columns.items():
            if key and key in normalized:
                return original

    return None


def detect_column_types(df):
    """Detect numeric, categorical, and date-like columns."""

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = []
    date_columns = []

    for col in df.columns:
        series = df[col]

        if (
            pd.api.types.is_datetime64_any_dtype(series)
        ):
            date_columns.append(col)
            continue

        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            name = normalize_name(col)

            looks_like_date = any(
                word in name.split()
                for word in [
                    "date", "datetime", "timestamp",
                    "created", "updated",
                ]
            )

            if looks_like_date:
                parsed = pd.to_datetime(
                    series, errors="coerce"
                )

                if len(series) and parsed.notna().mean() >= 0.5:
                    date_columns.append(col)
                    continue

        if (
            not pd.api.types.is_numeric_dtype(series)
            and series.nunique(dropna=True) <= 100
        ):
            categorical_columns.append(col)

    return numeric_columns, categorical_columns, date_columns


def make_schema(df):
    """Create a readable schema table."""

    return pd.DataFrame({
        "Column": [str(c) for c in df.columns],
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Unique values": [
            int(df[c].nunique(dropna=True)) for c in df.columns
        ],
        "Missing values": [
            int(df[c].isna().sum()) for c in df.columns
        ],
        "Missing %": [
            round(df[c].isna().mean() * 100, 2)
            for c in df.columns
        ],
    })


# ============================================================
# 4. OPTIONAL LLM CONNECTION
# ============================================================

def get_setting(name, default=None):
    """Read a setting from environment variables or Streamlit Secrets."""

    value = os.getenv(name)

    if value:
        return value

    try:
        value = st.secrets.get(name, default)
        return value
    except Exception:
        return default


def get_llm_plan(question, df):
    """
    Ask an LLM to interpret the question and return a JSON plan.

    The LLM selects an allowed analysis operation.
    Python performs the actual calculation.
    The uploaded dataset itself is NOT sent to the LLM.
    """

    api_key = get_setting("OPENAI_API_KEY")

    if not api_key:
        return None, (
            "No LLM API key is configured. "
            "Using the local question parser instead."
        )

    base_url = get_setting(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    ).rstrip("/")

    model = get_setting(
        "OPENAI_MODEL",
        "gpt-4o-mini",
    )

    numeric_cols, category_cols, date_cols = (
        detect_column_types(df)
    )

    schema = []

    for col in df.columns:
        schema.append({
            "name": str(col),
            "dtype": str(df[col].dtype),
            "unique_values": int(df[col].nunique(dropna=True)),
            "examples": [
                str(x)[:80]
                for x in df[col].dropna().head(3).tolist()
            ],
        })

    system_prompt = """
You are the question interpreter for DEXCR, a general-purpose
data analytics application.

Interpret the user's business or analytical question and return
ONLY a valid JSON object. Never write Python code, SQL, formulas
to execute, or executable expressions.

The dataset schema and a few example values are supplied.
Use only the actual column names from that schema.

Available operations:

1. "rank"
   Rank entities such as products, customers, regions, categories,
   or suppliers. Parameters:
   entity_column, metric_column, aggregation, direction, top_n.

2. "aggregate"
   Calculate sum, mean, median, min, max, count, distinct count,
   or standard deviation. Parameters:
   metric_column, aggregation, group_by (list of columns).

3. "trend"
   Analyze a measure over time. Parameters:
   date_column, metric_column, aggregation, frequency
   (day, week, month, quarter, year).

4. "distribution"
   Show the distribution of a numeric or categorical column.
   Parameters: column.

5. "correlation"
   Calculate correlations between numeric columns.
   Parameters: columns (list of numeric column names).

6. "profile"
   Summarize the dataset structure and data types.

7. "missing"
   Analyze missing values.

8. "duplicates"
   Analyze duplicate rows.

9. "show_rows"
   Display records, optionally applying filters.

10. "columns"
    List the available columns.

11. "filter"
    Filter records using conditions and show the matching rows.

12. "unknown"
    Use when the question cannot be answered from the schema
    or the intended calculation is genuinely ambiguous.

Return this structure:

{
  "operation": "rank",
  "entity_column": "product_name",
  "metric_column": "quantity_sold",
  "aggregation": "sum",
  "direction": "desc",
  "top_n": 10,
  "group_by": [],
  "date_column": null,
  "frequency": "month",
  "column": null,
  "columns": [],
  "filters": [],
  "answer_note": "Brief explanation of the intended analysis"
}

Allowed aggregation values:
sum, mean, median, min, max, count, nunique, std.

Allowed filter operators:
eq, ne, gt, gte, lt, lte, contains, between.

Each filter must contain:
{"column": "actual_column_name",
 "operator": "eq",
 "value": "value"}

Important rules:

- "Best-selling" normally means total units/quantity sold, if such a
  column exists. If not, use transaction count and explain the proxy.
- "Highest revenue" means rank by the relevant revenue/sales amount.
- "Most profitable" means rank by a profit measure if available.
- "Most frequent" or "most orders" means record/order count unless
  the schema indicates a more appropriate measure.
- "Average", "median", "total", "trend", "growth", "comparison",
  "correlation", "top", "bottom", "highest", and "lowest" are
  analytical intents, not literal column names.
- Distinguish revenue from profit and units from transaction count.
- For grouped comparisons, use the requested category as group_by.
- Use only columns that exist in the supplied schema.
- Never invent a metric or claim that a result has been calculated.
- If a question needs a column that does not exist, use "unknown"
  and explain what is missing.
- If the user asks for an explanation or summary, choose the closest
  valid analysis operation and provide a concise answer_note.
- Do not infer medical or business causation from descriptive data.
- Return valid JSON only.
"""

    user_prompt = {
        "question": question,
        "dataset_schema": schema,
        "numeric_columns": [str(c) for c in numeric_cols],
        "categorical_columns": [str(c) for c in category_cols],
        "date_columns": [str(c) for c in date_cols],
    }

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(user_prompt),
            },
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
        with urllib.request.urlopen(
            request, timeout=45
        ) as response:
            result = json.loads(response.read().decode("utf-8"))

        content = result["choices"][0]["message"]["content"]
        plan = json.loads(content)

        if not isinstance(plan, dict):
            raise ValueError("The model did not return a JSON object.")

        return plan, None

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return None, (
            f"The LLM API returned HTTP {exc.code}. "
            f"Check your API key, model, and provider settings. "
            f"Details: {detail[:500]}"
        )

    except Exception as exc:
        return None, (
            f"LLM interpretation failed: {exc}"
        )


# ============================================================
# 5. SAFE LOCAL QUESTION PARSER
# ============================================================

def local_question_plan(question, df):
    """
    A limited fallback parser for common analytical questions.

    It does not provide the flexible language understanding of an LLM.
    """

    q = normalize_name(question)
    numeric_cols, category_cols, date_cols = detect_column_types(df)

    product_col = find_column(
        [
            "product name", "product", "item name", "item",
            "service name", "service", "sku", "customer name",
            "customer", "supplier", "region", "category",
        ],
        df.columns,
    )

    quantity_col = find_column(
        [
            "quantity sold", "units sold", "qty sold",
            "quantity", "units", "volume sold",
        ],
        numeric_cols,
    )

    revenue_col = find_column(
        [
            "net revenue", "total revenue", "revenue",
            "net sales", "total sales", "sales amount",
            "gross sales", "sales", "amount",
        ],
        numeric_cols,
    )

    profit_col = find_column(
        [
            "gross profit", "net profit", "profit",
            "profit amount",
        ],
        numeric_cols,
    )

    plan = {
        "operation": "unknown",
        "entity_column": None,
        "metric_column": None,
        "aggregation": "sum",
        "direction": "desc",
        "top_n": 10,
        "group_by": [],
        "date_column": None,
        "frequency": "month",
        "column": None,
        "columns": [],
        "filters": [],
        "answer_note": "",
    }

    if any(x in q for x in [
        "list columns", "what columns", "available columns",
        "which columns", "column names",
    ]):
        plan["operation"] = "columns"
        return plan

    if any(x in q for x in [
        "missing values", "missing data", "null values",
        "incomplete data", "data quality",
    ]):
        plan["operation"] = "missing"
        return plan

    if any(x in q for x in [
        "duplicate rows", "duplicate records",
        "how many duplicates",
    ]):
        plan["operation"] = "duplicates"
        return plan

    if any(x in q for x in [
        "describe dataset", "profile dataset",
        "summarize dataset", "dataset overview",
        "overview of the data",
    ]):
        plan["operation"] = "profile"
        return plan

    if any(x in q for x in [
        "correlation", "correlated", "relationship between",
    ]):
        plan["operation"] = "correlation"
        plan["columns"] = [str(c) for c in numeric_cols]
        return plan

    # Find a column explicitly mentioned by the user.
    mentioned_col = None

    for col in df.columns:
        if normalize_name(col) and normalize_name(col) in q:
            mentioned_col = col
            break

    # Fuzzy matching helps with small spelling differences.
    if mentioned_col is None:
        candidates = [
            normalize_name(c) for c in df.columns
        ]
        matches = get_close_matches(
            q,
            candidates,
            n=1,
            cutoff=0.65,
        )
        if matches:
            mentioned_col = df.columns[candidates.index(matches[0])]

    if any(x in q for x in [
        "best selling", "best selling product",
        "most selling", "sells the most", "top selling",
        "top selling product", "most popular product",
        "best performing product", "top products",
        "rank products", "most ordered product",
    ]):
        plan["operation"] = "rank"
        plan["entity_column"] = product_col
        plan["metric_column"] = quantity_col
        plan["aggregation"] = "sum"

        if any(x in q for x in [
            "revenue", "sales value", "highest sales amount",
        ]):
            plan["metric_column"] = revenue_col

        if any(x in q for x in [
            "profit", "profitable",
        ]):
            plan["metric_column"] = profit_col

        if plan["metric_column"] is None:
            plan["aggregation"] = "count"
            plan["answer_note"] = (
                "No quantity measure was detected. "
                "The ranking will use record count instead."
            )

        return plan

    if any(x in q for x in [
        "most profitable", "highest profit",
        "highest gross profit", "top profit",
    ]):
        plan["operation"] = "rank"
        plan["entity_column"] = product_col
        plan["metric_column"] = profit_col
        plan["aggregation"] = "sum"
        return plan

    if any(x in q for x in [
        "trend", "over time", "by month", "by year",
        "monthly", "daily", "weekly", "quarterly",
        "growth over time", "how has it changed",
    ]):
        plan["operation"] = "trend"
        plan["date_column"] = (
            mentioned_col if mentioned_col in date_cols
            else (date_cols[0] if date_cols else None)
        )
        plan["metric_column"] = (
            revenue_col if revenue_col is not None
            else (numeric_cols[0] if numeric_cols else None)
        )

        if "year" in q or "annually" in q:
            plan["frequency"] = "year"
        elif "quarter" in q:
            plan["frequency"] = "quarter"
        elif "week" in q or "weekly" in q:
            plan["frequency"] = "week"
        elif "day" in q or "daily" in q:
            plan["frequency"] = "day"

        return plan

    if any(x in q for x in [
        "highest", "maximum", "largest", "lowest",
        "minimum", "smallest",
    ]) and mentioned_col in numeric_cols:
        plan["operation"] = "aggregate"
        plan["metric_column"] = mentioned_col
        plan["aggregation"] = (
            "min" if any(x in q for x in [
                "lowest", "minimum", "smallest",
            ]) else "max"
        )
        return plan

    if any(x in q for x in [
        "average", "mean", "typical",
    ]):
        plan["operation"] = "aggregate"
        plan["metric_column"] = (
            mentioned_col if mentioned_col in numeric_cols
            else revenue_col
        )
        if plan["metric_column"] is None and numeric_cols:
            plan["metric_column"] = numeric_cols[0]
        plan["aggregation"] = "mean"
        return plan

    if any(x in q for x in [
        "median", "middle value",
    ]):
        plan["operation"] = "aggregate"
        plan["metric_column"] = (
            mentioned_col if mentioned_col in numeric_cols
            else revenue_col
        )
        if plan["metric_column"] is None and numeric_cols:
            plan["metric_column"] = numeric_cols[0]
        plan["aggregation"] = "median"
        return plan

    if any(x in q for x in [
        "total", "sum of", "overall revenue",
        "overall sales", "how much",
    ]):
        plan["operation"] = "aggregate"
        plan["metric_column"] = (
            profit_col if "profit" in q
            else revenue_col
        )

        if plan["metric_column"] is None:
            plan["metric_column"] = mentioned_col

        if plan["metric_column"] is None and numeric_cols:
            plan["metric_column"] = numeric_cols[0]

        plan["aggregation"] = "sum"
        return plan

    if any(x in q for x in [
        "distribution", "histogram", "spread of",
    ]):
        plan["operation"] = "distribution"
        plan["column"] = (
            mentioned_col if mentioned_col is not None
            else (
                numeric_cols[0] if numeric_cols
                else (
                    category_cols[0] if category_cols else None
                )
            )
        )
        return plan

    if any(x in q for x in [
        "top", "bottom", "rank", "ranking",
        "highest by", "lowest by",
    ]):
        plan["operation"] = "rank"
        plan["entity_column"] = (
            mentioned_col if mentioned_col in category_cols
            else product_col
        )
        plan["metric_column"] = revenue_col
        plan["aggregation"] = "sum"

        if any(x in q for x in ["bottom", "lowest"]):
            plan["direction"] = "asc"

        numbers = re.findall(r"\b(\d{1,3})\b", q)
        if numbers:
            plan["top_n"] = min(max(int(numbers[0]), 1), 100)

        return plan

    if any(x in q for x in [
        "how many rows", "number of rows",
        "how many records", "record count",
        "how many transactions",
    ]):
        plan["operation"] = "aggregate"
        plan["aggregation"] = "count"
        return plan

    if mentioned_col is not None and any(x in q for x in [
        "show", "display", "distribution", "breakdown",
        "compare", "comparison", "by category",
    ]):
        plan["operation"] = "distribution"
        plan["column"] = mentioned_col
        return plan

    return plan


# ============================================================
# 6. VALIDATE THE LLM'S ANALYSIS PLAN
# ============================================================

ALLOWED_OPERATIONS = {
    "rank", "aggregate", "trend", "distribution",
    "correlation", "profile", "missing", "duplicates",
    "show_rows", "columns", "filter", "unknown",
}

ALLOWED_AGGREGATIONS = {
    "sum", "mean", "median", "min", "max",
    "count", "nunique", "std",
}

ALLOWED_OPERATORS = {
    "eq", "ne", "gt", "gte", "lt", "lte",
    "contains", "between",
}


def validate_plan(plan, df):
    """Validate operation names and resolve actual dataset columns."""

    if not isinstance(plan, dict):
        raise ValueError("The analysis plan must be a JSON object.")

    operation = str(plan.get("operation", "unknown")).lower()

    if operation not in ALLOWED_OPERATIONS:
        raise ValueError("The requested analysis operation is unsupported.")

    plan["operation"] = operation

    def resolve(value):
        if value is None:
            return None

        normalized = normalize_name(value)

        for col in df.columns:
            if normalize_name(col) == normalized:
                return col

        # Do not silently use a vaguely similar column for LLM plans.
        return None

    for field in [
        "metric_column", "entity_column",
        "date_column", "column",
    ]:
        value = plan.get(field)

        if value is not None:
            resolved = resolve(value)

            if resolved is None:
                raise ValueError(
                    f"Column '{value}' does not exist in the dataset. "
                    "Please review the available columns."
                )

            plan[field] = resolved

    resolved_groups = []

    for value in plan.get("group_by") or []:
        col = resolve(value)

        if col is None:
            raise ValueError(
                f"Grouping column '{value}' does not exist."
            )

        resolved_groups.append(col)

    plan["group_by"] = resolved_groups

    resolved_columns = []

    for value in plan.get("columns") or []:
        col = resolve(value)

        if col is None:
            raise ValueError(
                f"Column '{value}' does not exist."
            )

        resolved_columns.append(col)

    plan["columns"] = resolved_columns

    aggregation = str(
        plan.get("aggregation", "sum")
    ).lower()

    if aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError("Unsupported aggregation method.")

    plan["aggregation"] = aggregation

    direction = str(plan.get("direction", "desc")).lower()

    plan["direction"] = (
        "asc" if direction in ["asc", "ascending", "bottom", "lowest"]
        else "desc"
    )

    try:
        plan["top_n"] = min(
            max(int(plan.get("top_n", 10)), 1),
            100,
        )
    except (TypeError, ValueError):
        plan["top_n"] = 10

    validated_filters = []

    for item in plan.get("filters") or []:
        if not isinstance(item, dict):
            continue

        col = resolve(item.get("column"))
        operator = str(item.get("operator", "eq")).lower()

        if col is None:
            raise ValueError(
                f"A filter refers to an unknown column: "
                f"{item.get('column')}"
            )

        if operator not in ALLOWED_OPERATORS:
            raise ValueError(
                f"Unsupported filter operator: {operator}"
            )

        validated_filters.append({
            "column": col,
            "operator": operator,
            "value": item.get("value"),
        })

    plan["filters"] = validated_filters

    return plan


# ============================================================
# 7. FILTER DATA SAFELY
# ============================================================

def apply_filters(df, filters):
    """Apply validated filter conditions to a copy of the data."""

    result = df.copy()

    for condition in filters:
        col = condition["column"]
        operator = condition["operator"]
        value = condition.get("value")
        series = result[col]

        if operator == "contains":
            result = result[
                series.astype(str).str.contains(
                    str(value),
                    case=False,
                    na=False,
                    regex=False,
                )
            ]
            continue

        if operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(
                    "A between filter requires two values."
                )

            left, right = value

            if pd.api.types.is_numeric_dtype(series):
                values = numeric_series(series)
                result = result[
                    values.between(
                        float(left), float(right), inclusive="both"
                    )
                ]
            else:
                result = result[
                    series.astype(str).between(
                        str(left), str(right), inclusive="both"
                    )
                ]
            continue

        if operator in ["gt", "gte", "lt", "lte"]:
            if pd.api.types.is_numeric_dtype(series):
                left = numeric_series(series)
                right = float(value)
            else:
                left = pd.to_datetime(series, errors="coerce")
                right = pd.to_datetime(value, errors="coerce")

                if pd.isna(right):
                    raise ValueError(
                        f"Cannot interpret '{value}' as a date."
                    )

            if operator == "gt":
                mask = left > right
            elif operator == "gte":
                mask = left >= right
            elif operator == "lt":
                mask = left < right
            else:
                mask = left <= right

            result = result[mask.fillna(False)]
            continue

        if operator in ["eq", "ne"]:
            if pd.api.types.is_numeric_dtype(series):
                try:
                    compare_value = float(value)
                    left = numeric_series(series)
                except (TypeError, ValueError):
                    compare_value = value
                    left = series
            else:
                left = series.astype(str)
                compare_value = str(value)

            mask = left == compare_value

            if operator == "ne":
                mask = ~mask

            result = result[mask.fillna(False)]

    return result


# ============================================================
# 8. EXECUTE THE ANALYSIS WITH PYTHON
# ============================================================

def execute_plan(original_df, plan):
    """
    Execute only approved analytical operations.

    This function does not execute model-generated code.
    """

    df = apply_filters(
        original_df,
        plan.get("filters", []),
    )

    operation = plan["operation"]
    note = plan.get("answer_note", "")

    result = {
        "answer": "",
        "table": None,
        "chart": None,
    }

    if operation == "unknown":
        result["answer"] = (
            note or
            "I couldn't confidently translate that question into "
            "a supported analysis. Try rephrasing it or specifying "
            "the measure and category you want to analyze."
        )
        return result

    if operation == "columns":
        result["answer"] = (
            f"The dataset contains {len(df.columns)} columns "
            f"and {len(df):,} rows."
        )
        result["table"] = pd.DataFrame({
            "Column": [str(c) for c in df.columns],
            "Data type": [str(df[c].dtype) for c in df.columns],
        })
        return result

    if operation == "profile":
        result["answer"] = (
            f"The dataset contains {len(df):,} rows and "
            f"{len(df.columns)} columns. "
            f"It has {int(df.isna().sum().sum()):,} missing cells "
            f"and {int(df.duplicated().sum()):,} duplicate rows."
        )
        result["table"] = make_schema(df)
        return result

    if operation == "missing":
        missing = pd.DataFrame({
            "Column": [str(c) for c in df.columns],
            "Missing values": [
                int(df[c].isna().sum()) for c in df.columns
            ],
            "Missing %": [
                round(df[c].isna().mean() * 100, 2)
                for c in df.columns
            ],
        }).sort_values("Missing values", ascending=False)

        result["answer"] = (
            f"There are {int(df.isna().sum().sum()):,} missing "
            "cells in the filtered dataset."
        )
        result["table"] = missing
        result["chart"] = {
            "kind": "bar",
            "x": "Column",
            "y": "Missing values",
            "title": "Missing values by column",
        }
        return result

    if operation == "duplicates":
        duplicates = int(df.duplicated().sum())
        result["answer"] = (
            f"Found {duplicates:,} duplicate rows "
            f"out of {len(df):,} records."
        )
        return result

    if operation == "show_rows":
        result["answer"] = (
            f"Showing up to 100 matching records "
            f"from {len(df):,} records."
        )
        result["table"] = df.head(100)
        return result

    if operation == "filter":
        result["answer"] = (
            f"{len(df):,} records match the specified conditions."
        )
        result["table"] = df.head(100)
        return result

    if operation == "aggregate":
        metric = plan.get("metric_column")
        aggregation = plan.get("aggregation", "sum")
        group_by = plan.get("group_by") or []

        if metric is None:
            if aggregation == "count":
                result["answer"] = (
                    f"The dataset contains {len(df):,} rows."
                )
                return result

            raise ValueError(
                "I need a measure column to perform this calculation."
            )

        if aggregation != "count":
            if not pd.api.types.is_numeric_dtype(df[metric]):
                values = numeric_series(df[metric])

                if values.notna().sum() == 0:
                    raise ValueError(
                        f"Column '{metric}' does not contain usable "
                        "numeric values for this calculation."
                    )

                df = df.copy()
                df[metric] = values

        if group_by:
            if aggregation == "count":
                table = (
                    df.groupby(group_by, dropna=False)
                    .size()
                    .reset_index(name="record_count")
                )
                value_col = "record_count"
            else:
                table = (
                    df.groupby(group_by, dropna=False)[metric]
                    .agg(aggregation)
                    .reset_index()
                )
                value_col = metric

            table = table.sort_values(
                value_col, ascending=False
            )

            result["answer"] = (
                f"Calculated {aggregation} of '{metric}' "
                f"grouped by {', '.join(map(str, group_by))}."
            )
            result["table"] = table.head(100)
            result["chart"] = {
                "kind": "bar",
                "x": str(group_by[0]),
                "y": value_col,
                "title": f"{aggregation.title()} of {metric} by {group_by[0]}",
            }
            return result

        if aggregation == "count":
            value = int(df[metric].count())
        else:
            values = numeric_series(df[metric]).dropna()

            if values.empty:
                raise ValueError(
                    f"No usable numeric values in '{metric}'."
                )

            value = getattr(values, aggregation)()

        result["answer"] = (
            f"{aggregation.title()} of '{metric}': "
            f"{display_number(value)}."
        )
        return result

    if operation == "rank":
        entity = plan.get("entity_column")
        metric = plan.get("metric_column")
        aggregation = plan.get("aggregation", "sum")
        top_n = plan.get("top_n", 10)

        if entity is None:
            raise ValueError(
                "I couldn't identify which entity to rank, such as "
                "a product, customer, region, or category."
            )

        if metric is None:
            ranking = (
                df.groupby(entity, dropna=False)
                .size()
                .reset_index(name="record_count")
            )
            value_col = "record_count"
            measure_description = "record count"
        elif aggregation == "count":
            ranking = (
                df.groupby(entity, dropna=False)[metric]
                .count()
                .reset_index(name="record_count")
            )
            value_col = "record_count"
            measure_description = f"count of {metric}"
        else:
            values = numeric_series(df[metric])

            work = df[[entity]].copy()
            work[metric] = values

            ranking = (
                work.groupby(entity, dropna=False)[metric]
                .agg(aggregation)
                .reset_index()
            )

            value_col = metric
            measure_description = f"{aggregation} of {metric}"

        ranking = ranking.sort_values(
            value_col,
            ascending=(plan.get("direction") == "asc"),
        ).head(top_n)

        if ranking.empty:
            raise ValueError(
                "No usable records were available to rank."
            )

        winner = ranking.iloc[0]

        result["answer"] = (
            f"Top-ranked {entity}: **{winner[entity]}**, "
            f"with {display_number(winner[value_col])} "
            f"({measure_description})."
        )

        result["table"] = ranking

        result["chart"] = {
            "kind": "bar",
            "x": str(entity),
            "y": value_col,
            "title": f"Top {top_n}: {measure_description} by {entity}",
        }

        return result

    if operation == "trend":
        date_col = plan.get("date_column")
        metric = plan.get("metric_column")
        aggregation = plan.get("aggregation", "sum")
        frequency = plan.get("frequency", "month")

        if date_col is None or metric is None:
            raise ValueError(
                "Trend analysis requires a date column and a measure."
            )

        work = df[[date_col, metric]].copy()
        work[date_col] = pd.to_datetime(
            work[date_col], errors="coerce"
        )
        work[metric] = numeric_series(work[metric])
        work = work.dropna()

        if work.empty:
            raise ValueError(
                "No valid date and numeric measure pairs were found."
            )

        frequency_map = {
            "day": "D",
            "week": "W",
            "month": "M",
            "quarter": "Q",
            "year": "Y",
        }

        period_frequency = frequency_map.get(
            frequency, "M"
        )

        work["Period"] = (
            work[date_col]
            .dt.to_period(period_frequency)
            .astype(str)
        )

        trend = (
            work.groupby("Period")[metric]
            .agg(aggregation)
            .reset_index()
        )

        result["answer"] = (
            f"Calculated {aggregation} of '{metric}' by "
            f"{frequency}. The table contains "
            f"{len(trend)} time periods."
        )

        result["table"] = trend
        result["chart"] = {
            "kind": "line",
            "x": "Period",
            "y": metric,
            "title": f"{metric} over time ({frequency})",
        }

        return result

    if operation == "distribution":
        col = plan.get("column")

        if col is None:
            raise ValueError(
                "Specify which column's distribution you want."
            )

        if pd.api.types.is_numeric_dtype(df[col]):
            result["answer"] = (
                f"Distribution of '{col}'. "
                f"Mean: {display_number(numeric_series(df[col]).mean())}; "
                f"median: {display_number(numeric_series(df[col]).median())}."
            )

            result["chart"] = {
                "kind": "histogram",
                "x": str(col),
                "title": f"Distribution of {col}",
            }

        else:
            counts = (
                df[col].astype("string")
                .fillna("(Missing)")
                .value_counts()
                .head(30)
                .rename_axis(str(col))
                .reset_index(name="record_count")
            )

            result["answer"] = (
                f"Showing the most common values in '{col}'."
            )
            result["table"] = counts
            result["chart"] = {
                "kind": "bar",
                "x": str(col),
                "y": "record_count",
                "title": f"Distribution of {col}",
            }

        return result

    if operation == "correlation":
        cols = plan.get("columns") or []

        if len(cols) < 2:
            cols = [
                c for c in df.select_dtypes(include="number").columns
            ]

        if len(cols) < 2:
            raise ValueError(
                "Correlation requires at least two numeric columns."
            )

        matrix = df[cols].corr(numeric_only=True)

        result["answer"] = (
            "Calculated Pearson correlations between the selected "
            "numeric columns. Correlation alone does not establish causation."
        )
        result["table"] = matrix.reset_index(names="Column")

        return result

    raise ValueError(
        "This analysis operation is not implemented yet."
    )


# ============================================================
# 9. SIDEBAR AND DATASET SELECTION
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
        [
            "Overview",
            "My Data",
            "Data Explorer",
            "AI Analyst",
        ],
    )

    st.divider()

    has_uploaded = (
        st.session_state.uploaded_df is not None
        and not st.session_state.uploaded_df.empty
    )

    options = ["Sample sales data"]

    if has_uploaded:
        options.insert(0, "My uploaded data")

    default_index = 0 if has_uploaded else 0

    data_choice = st.radio(
        "ACTIVE DATASET",
        options,
        index=default_index,
    )

if data_choice == "My uploaded data" and has_uploaded:
    df = st.session_state.uploaded_df
    dataset_name = st.session_state.uploaded_name
else:
    df = load_sample_data()
    dataset_name = "Sample sales transactions"

has_data = df is not None and not df.empty


# ============================================================
# 10. OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <h1>DEXCR</h1>
            <h3>Ask your data better questions.</h3>
            <p>
            Upload a dataset, explore its structure, uncover patterns,
            and calculate answers to business questions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Platform", "DEXCR")
    col2.metric("File formats", "CSV · XLSX")
    col3.metric(
        "Current dataset",
        "Ready" if has_data else "No data",
    )

    st.subheader("Your analytics workspace")

    st.markdown(
        """
        - **My Data:** upload and profile your own dataset.
        - **Data Explorer:** inspect columns and create charts.
        - **AI Analyst:** ask analytical questions in natural language.
        """
    )

    if has_data:
        st.success(
            f"Active dataset: {dataset_name} — "
            f"{len(df):,} rows and {len(df.columns)} columns."
        )

    if not get_setting("OPENAI_API_KEY"):
        st.warning(
            "The LLM connection is not configured. The AI Analyst "
            "will use its limited local question parser until you "
            "add an API key in Streamlit Secrets."
        )


# ============================================================
# 11. MY DATA
# ============================================================

elif page == "My Data":
    st.title("My Data")
    st.write(
        "Upload a CSV or Excel dataset. DEXCR does not require "
        "specific business column names."
    )

    uploaded_file = st.file_uploader(
        "Choose your dataset",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:
        try:
            new_df = read_uploaded_file(uploaded_file)

            if new_df.empty:
                st.error("The uploaded file contains no data.")
            else:
                st.session_state.uploaded_df = new_df
                st.session_state.uploaded_name = uploaded_file.name
                st.success(
                    f"Loaded {uploaded_file.name}: "
                    f"{len(new_df):,} rows and "
                    f"{len(new_df.columns)} columns."
                )
                st.rerun()

        except Exception as exc:
            st.error(f"Could not read the file: {exc}")

    if has_data:
        numeric_cols, category_cols, date_cols = (
            detect_column_types(df)
        )

        st.divider()
        st.subheader(f"Dataset profile: {dataset_name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Numeric columns", len(numeric_cols))
        c4.metric("Date columns", len(date_cols))

        st.subheader("Column schema")
        st.dataframe(
            make_schema(df),
            use_container_width=True,
        )

        st.subheader("Data preview")
        st.dataframe(
            df.head(100),
            use_container_width=True,
        )

        st.subheader("Data quality")

        a, b, c = st.columns(3)
        missing_cells = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        a.metric("Missing cells", f"{missing_cells:,}")
        b.metric("Duplicate rows", f"{duplicate_rows:,}")
        c.metric(
            "Complete cells",
            f"{df.size - missing_cells:,}",
        )

        st.download_button(
            "Download dataset as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="dexcr_export.csv",
            mime="text/csv",
        )

        st.caption(
            "Uploaded data is held in the active Streamlit session. "
            "This application does not save it to GitHub."
        )
    else:
        st.info(
            "Upload a dataset above. If no sample file is present, "
            "the workspace will remain empty until you upload data."
        )


# ============================================================
# 12. DATA EXPLORER
# ============================================================

elif page == "Data Explorer":
    st.title("Data Explorer")

    if not has_data:
        st.warning("Upload a dataset before exploring it.")
    else:
        numeric_cols, category_cols, date_cols = (
            detect_column_types(df)
        )

        st.caption(
            f"{dataset_name} · {len(df):,} rows · "
            f"{len(df.columns)} columns"
        )

        search = st.text_input(
            "Search column names",
            placeholder="Type a column name...",
        )

        visible_cols = [
            col for col in df.columns
            if normalize_name(search) in normalize_name(col)
        ]

        st.dataframe(
            df[visible_cols].head(500),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Explore a measure")

        if numeric_cols:
            measure = st.selectbox(
                "Numeric column",
                numeric_cols,
            )

            values = numeric_series(df[measure]).dropna()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", display_number(values.sum()))
            c2.metric("Average", display_number(values.mean()))
            c3.metric("Median", display_number(values.median()))
            c4.metric("Distinct values", f"{values.nunique():,}")

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
                    "Break down by category",
                    category_cols,
                )

                work = df[[group_col]].copy()
                work[measure] = numeric_series(df[measure])

                grouped = (
                    work.groupby(group_col, dropna=False)[measure]
                    .sum()
                    .reset_index()
                    .sort_values(measure, ascending=False)
                    .head(25)
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

                st.dataframe(
                    grouped,
                    use_container_width=True,
                )
        else:
            st.info("No numeric columns were detected.")

        if date_cols and numeric_cols:
            st.divider()
            st.subheader("Time trend")

            date_col = st.selectbox(
                "Date column",
                date_cols,
            )
            trend_measure = st.selectbox(
                "Measure",
                numeric_cols,
                key="explorer_trend_measure",
            )

            trend = df[[date_col, trend_measure]].copy()
            trend[date_col] = pd.to_datetime(
                trend[date_col], errors="coerce"
            )
            trend[trend_measure] = numeric_series(
                trend[trend_measure]
            )
            trend = trend.dropna()

            if not trend.empty:
                trend["Period"] = (
                    trend[date_col]
                    .dt.to_period("M")
                    .astype(str)
                )

                monthly = (
                    trend.groupby("Period")[trend_measure]
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


# ============================================================
# 13. AI ANALYST
# ============================================================

elif page == "AI Analyst":
    st.title("AI Analyst")

    st.write(
        "Ask questions about your dataset using different wording. "
        "DEXCR interprets the question, validates the requested "
        "columns, and calculates the answer with Python."
    )

    if not has_data:
        st.warning("Upload a dataset before asking questions.")
    else:
        numeric_cols, category_cols, date_cols = (
            detect_column_types(df)
        )

        with st.expander("Inspect available columns"):
            st.dataframe(
                make_schema(df),
                use_container_width=True,
            )

        question = st.text_area(
            "What would you like to know?",
            placeholder=(
                "Examples:\n"
                "• What is my best-selling product?\n"
                "• Which region generates the most revenue?\n"
                "• How have sales changed month over month?\n"
                "• What is the average profit by category?\n"
                "• Are there any missing values?\n"
                "• Which numeric variables are correlated?"
            ),
            height=130,
        )

        run_analysis = st.button(
            "Analyze question",
            type="primary",
        )

        if run_analysis:
            if not question.strip():
                st.warning("Enter a question first.")

            else:
                try:
                    plan, llm_error = get_llm_plan(
                        question.strip(), df
                    )

                    if plan is None:
                        st.caption(
                            "Using local question interpretation. "
                            "For more flexible questions, configure "
                            "an LLM API key in Streamlit Secrets."
                        )

                        plan = local_question_plan(
                            question.strip(), df
                        )

                        if llm_error:
                            with st.expander(
                                "LLM connection details"
                            ):
                                st.write(llm_error)

                    plan = validate_plan(plan, df)

                    with st.expander(
                        "View the analysis plan"
                    ):
                        st.json(plan)

                    result = execute_plan(df, plan)

                    st.divider()
                    st.subheader("Answer")
                    st.markdown(result["answer"])

                    if result["table"] is not None:
                        st.dataframe(
                            result["table"],
                            use_container_width=True,
                        )

                    chart = result["chart"]

                    if chart:
                        if chart["kind"] == "bar":
                            st.plotly_chart(
                                px.bar(
                                    result["table"],
                                    x=chart["x"],
                                    y=chart["y"],
                                    title=chart["title"],
                                    template="plotly_white",
                                ),
                                use_container_width=True,
                            )

                        elif chart["kind"] == "line":
                            st.plotly_chart(
                                px.line(
                                    result["table"],
                                    x=chart["x"],
                                    y=chart["y"],
                                    markers=True,
                                    title=chart["title"],
                                    template="plotly_white",
                                ),
                                use_container_width=True,
                            )

                        elif chart["kind"] == "histogram":
                            st.plotly_chart(
                                px.histogram(
                                    df,
                                    x=chart["x"],
                                    title=chart["title"],
                                    template="plotly_white",
                                ),
                                use_container_width=True,
                            )

                    st.caption(
                        "Check the selected measure and aggregation "
                        "before using results for important decisions."
                    )

                except Exception as exc:
                    st.error(
                        "DEXCR could not complete this analysis. "
                        "It may need a different column, a clearer "
                        "question, or a supported operation."
                    )
                    st.write(str(exc))

        st.divider()
        st.subheader("Example questions")

        examples = [
            "What is my best-selling product?",
            "Which product generated the most revenue?",
            "Which product is the most profitable?",
            "Show the top 10 products by quantity sold.",
            "What is the average revenue?",
            "What is the total profit?",
            "Which region has the highest sales?",
            "Show me sales by product category.",
            "How have sales changed month by month?",
            "Which columns contain missing values?",
            "How many duplicate records are there?",
            "Which numeric variables are correlated?",
        ]

        for example in examples:
            st.markdown(f"- {example}")


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "DEXCR · General-purpose data analytics · "
    "Calculations are performed in Python on the active dataset."
)
