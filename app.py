
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# 1. APP CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="DEXCR | AI Business Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("DEXCR")
st.subheader("AI Business Intelligence Assistant")
st.write(
    "Explore sales, financial performance, and business trends "
    "using natural-language questions."
)

BASE_DIR = Path(__file__).resolve().parent


# =====================================================
# 2. LOAD THE DATASETS
# =====================================================

@st.cache_data
def load_data():
    filenames = [
        "sales_transactions.csv",
        "daily_drivers.csv",
        "monthly_financials.csv",
        "data_dictionary.csv"
    ]

    missing = [
        name for name in filenames
        if not (BASE_DIR / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing CSV files: " + ", ".join(missing)
        )

    sales = pd.read_csv(BASE_DIR / filenames[0])
    daily = pd.read_csv(BASE_DIR / filenames[1])
    financials = pd.read_csv(BASE_DIR / filenames[2])
    dictionary = pd.read_csv(BASE_DIR / filenames[3])

    sales["Transaction_Date"] = pd.to_datetime(
        sales["Transaction_Date"], errors="coerce"
    )

    daily["Date"] = pd.to_datetime(
        daily["Date"], errors="coerce"
    )

    financials["Year_Month"] = pd.to_datetime(
        financials["Year_Month"].astype(str),
        errors="coerce"
    ).dt.to_period("M")

    return sales, daily, financials, dictionary


try:
    sales, daily, financials, dictionary = load_data()
except Exception as error:
    st.error("The datasets could not be loaded.")
    st.exception(error)
    st.info(
        "Upload all four CSV files to the same GitHub repository "
        "folder as app.py."
    )
    st.stop()


# =====================================================
# 3. BUSINESS ANALYSIS FUNCTIONS
# =====================================================

def numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def business_kpis():
    columns = [
        "Gross_Revenue_USD",
        "Cost_of_Goods_Sold_USD",
        "Gross_Profit_USD",
        "Total_OpEx_USD",
        "EBITDA_USD",
        "EBIT_USD",
        "Net_Income_USD"
    ]

    result = {}

    for column in columns:
        if column in financials.columns:
            result[column] = round(
                float(numeric(financials[column]).sum()), 2
            )

    revenue = result.get("Gross_Revenue_USD", 0)
    gross_profit = result.get("Gross_Profit_USD", 0)
    net_income = result.get("Net_Income_USD", 0)

    result["Gross_Margin_Pct"] = (
        round(gross_profit / revenue * 100, 2)
        if revenue else None
    )

    result["Net_Margin_Pct"] = (
        round(net_income / revenue * 100, 2)
        if revenue else None
    )

    return result


def sales_analysis(group_by="Product_Category"):
    allowed = [
        "Product_Category",
        "Sales_Channel",
        "Customer_Segment",
        "Store_ID",
        "Payment_Method",
        "Sales_Rep_Name"
    ]

    if group_by not in allowed:
        raise ValueError("Unsupported sales grouping.")

    result = (
        sales.groupby(group_by, dropna=False)
        .agg(
            Transactions=("Transaction_ID", "count"),
            Units_Sold=("Quantity_Sold", "sum"),
            Net_Sales_USD=("Net_Sales_USD", "sum")
        )
        .reset_index()
        .sort_values("Net_Sales_USD", ascending=False)
    )

    result["Net_Sales_USD"] = result["Net_Sales_USD"].round(2)
    return result


def revenue_trend():
    result = (
        financials.groupby("Year_Month", as_index=False)
        .agg(
            Gross_Revenue_USD=("Gross_Revenue_USD", "sum"),
            EBITDA_USD=("EBITDA_USD", "sum"),
            Net_Income_USD=("Net_Income_USD", "sum")
        )
        .sort_values("Year_Month")
    )

    result["Year_Month"] = result["Year_Month"].astype(str)

    result["Revenue_Change_Pct"] = (
        result["Gross_Revenue_USD"]
        .pct_change()
        .mul(100)
        .round(2)
    )

    return result


# =====================================================
# 4. DISPLAY THE FINANCIAL DASHBOARD
# =====================================================

st.header("Financial Overview")

kpis = business_kpis()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Gross Revenue",
    f"${kpis.get('Gross_Revenue_USD', 0):,.0f}"
)

c2.metric(
    "Gross Profit",
    f"${kpis.get('Gross_Profit_USD', 0):,.0f}"
)

c3.metric(
    "EBITDA",
    f"${kpis.get('EBITDA_USD', 0):,.0f}"
)

c4.metric(
    "Net Income",
    f"${kpis.get('Net_Income_USD', 0):,.0f}"
)

with st.expander("View all financial KPIs"):
    st.json(kpis)


# =====================================================
# 5. EXPLORE SALES
# =====================================================

st.header("Sales Analysis")

dimension = st.selectbox(
    "Analyze sales by",
    [
        "Product_Category",
        "Sales_Channel",
        "Customer_Segment",
        "Store_ID",
        "Payment_Method",
        "Sales_Rep_Name"
    ]
)

sales_results = sales_analysis(dimension)

st.dataframe(sales_results, use_container_width=True)

st.bar_chart(
    sales_results.set_index(dimension)["Net_Sales_USD"]
)


# =====================================================
# 6. MONTHLY FINANCIAL TRENDS
# =====================================================

st.header("Monthly Financial Trends")

trend = revenue_trend()

st.line_chart(
    trend.set_index("Year_Month")[
        ["Gross_Revenue_USD", "EBITDA_USD", "Net_Income_USD"]
    ]
)

st.dataframe(trend, use_container_width=True)


# =====================================================
# 7. DATASET EXPLORER
# =====================================================

st.header("Dataset Explorer")

dataset_name = st.selectbox(
    "Choose a dataset",
    [
        "Sales Transactions",
        "Daily Drivers",
        "Monthly Financials",
        "Data Dictionary"
    ]
)

datasets = {
    "Sales Transactions": sales,
    "Daily Drivers": daily,
    "Monthly Financials": financials,
    "Data Dictionary": dictionary
}

selected_data = datasets[dataset_name]

st.write(f"**Rows:** {len(selected_data):,}")
st.write("Preview of the dataset:")
st.dataframe(selected_data.head(100), use_container_width=True)


# =====================================================
# 8. AI BUSINESS ASSISTANT
# =====================================================

st.header("Ask DEXCR")

st.caption(
    "Example questions: Which channel has the highest sales? "
    "What are our financial KPIs? How has revenue changed?"
)

@st.cache_resource(show_spinner="Loading the AI model...")
def load_llm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    model.eval()
    return tokenizer, model


def generate_text(messages, max_new_tokens=220):
    tokenizer, model = load_llm()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    import torch

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()


def select_tool(question):
    tools_description = """
    business_kpis: overall financial KPIs. Arguments {}
    sales_analysis: compare sales. Allowed group_by:
    Product_Category, Sales_Channel, Customer_Segment, Store_ID,
    Payment_Method, Sales_Rep_Name.
    Example {"group_by":"Sales_Channel"}
    revenue_trend: monthly revenue and profitability trends. Arguments {}
    """

    messages = [
        {
            "role": "system",
            "content": (
                "Choose one tool for the user's business question. "
                "Return only JSON with keys tool_name and arguments. "
                "Do not answer the question. If unsupported, return "
                '{"tool_name":"unsupported","arguments":{}}.'
            )
        },
        {
            "role": "user",
            "content": (
                f"Tools:\n{tools_description}\n"
                f"Question: {question}\n"
                'Return JSON, for example: '
                '{"tool_name":"business_kpis","arguments":{}}'
            )
        }
    ]

    raw = generate_text(messages, 120)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)

    if not match:
        return {"tool_name": "unsupported", "arguments": {}}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"tool_name": "unsupported", "arguments": {}}


def execute_tool(decision):
    tool_name = decision.get("tool_name")
    arguments = decision.get("arguments", {})

    if tool_name == "business_kpis":
        return business_kpis()

    if tool_name == "revenue_trend":
        return revenue_trend().to_dict(orient="records")

    if tool_name == "sales_analysis":
        group_by = arguments.get(
            "group_by", "Product_Category"
        )

        allowed = [
            "Product_Category",
            "Sales_Channel",
            "Customer_Segment",
            "Store_ID",
            "Payment_Method",
            "Sales_Rep_Name"
        ]

        if group_by not in allowed:
            raise ValueError("Invalid sales grouping.")

        return sales_analysis(group_by).to_dict(orient="records")

    raise ValueError(
        "Unsupported question. Try asking about financial KPIs, "
        "sales comparisons, or revenue trends."
    )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your business data...")

if question:
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing your question..."):
                decision = select_tool(question)

                if decision.get("tool_name") == "unsupported":
                    answer = (
                        "I couldn't identify a suitable analysis tool. "
                        "Try asking about financial KPIs, sales by channel "
                        "or category, or monthly revenue trends."
                    )
                else:
                    actual_results = execute_tool(decision)

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a business intelligence analyst. "
                                "Explain the supplied results clearly. "
                                "Use only the provided data for numerical "
                                "claims. Do not invent figures or causes."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n"
                                f"Results from Python:\n"
                                f"{json.dumps(actual_results, default=str)}\n"
                                "Explain the findings briefly."
                            )
                        }
                    ]

                    answer = generate_text(messages, 250)

                    with st.expander("View underlying Python results"):
                        st.json(actual_results)

                    st.caption(
                        f"Analysis tool selected: {decision['tool_name']}"
                    )

            st.markdown(answer)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as error:
            st.error(
                "The assistant encountered an error. "
                "Check the Streamlit deployment logs."
            )
            st.exception(error)
