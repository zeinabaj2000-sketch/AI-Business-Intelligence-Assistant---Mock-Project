m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Numeric columns", len(numeric_cols))
m4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

overview_tab, data_tab, explorer_tab, analyst_tab = st.tabs(
    ["Overview", "My Data", "Data Explorer", "AI Analyst"]
)

with overview_tab:
    st.subheader("Automatic data review")
    for insight in review["insights"]:
        st.markdown(f"- {insight}")
    if numeric_cols:
        selected_metric = st.selectbox("Numeric column to summarize", numeric_cols, key="overview_metric")
        vals = numeric_series(df[selected_metric]).dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"{vals.sum():,.2f}" if len(vals) else "N/A")
        c2.metric("Average", f"{vals.mean():,.2f}" if len(vals) else "N/A")
        c3.metric("Median", f"{vals.median():,.2f}" if len(vals) else "N/A")
        st.plotly_chart(px.histogram(pd.DataFrame({selected_metric: vals}), x=selected_metric,
                                     title=f"Distribution of {selected_metric}"), use_container_width=True)

with data_tab:
    st.subheader("Dataset preview")
    st.dataframe(df.head(500), use_container_width=True)
    st.subheader("Column profile")
    profile = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(profile, use_container_width=True)
    st.download_button("Download current data as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="dexcr_data.csv", mime="text/csv")

with explorer_tab:
    st.subheader("Explore a column")
    selected_col = st.selectbox("Choose a column", list(df.columns), key="explorer_col")
    st.write("**Data type:**", df[selected_col].dtype)
    st.write("**Missing values:**", int(df[selected_col].isna().sum()))
    st.write("**Unique values:**", int(df[selected_col].nunique(dropna=True)))
    if selected_col in numeric_cols:
        values = numeric_series(df[selected_col]).dropna()
        st.write(values.describe().to_frame("Statistics"))
        st.plotly_chart(px.histogram(pd.DataFrame({selected_col: values}), x=selected_col,
                                     title=f"Distribution of {selected_col}"), use_container_width=True)
    else:
        counts = df[selected_col].astype("string").value_counts(dropna=False).head(50)
        count_df = counts.rename_axis(selected_col).reset_index(name="count")
        st.dataframe(count_df, use_container_width=True)
        st.plotly_chart(px.bar(count_df.head(25), x=selected_col, y="count",
                               title=f"Most frequent values in {selected_col}"), use_container_width=True)

with analyst_tab:
    st.subheader("Ask DEXCR")
    st.write("Ask a business question about the uploaded dataset.")
    question = st.text_area(
        "Your question",
        placeholder="Example: Which 10 products generated the most revenue?",
        height=90,
        key="analyst_question",
    )
    use_llm = st.toggle("Use LLM for question understanding and explanation", value=True)
    provider = str(setting("LLM_PROVIDER", "qwen")).lower()
    has_key = bool(setting("OPENAI_API_KEY")) if provider == "openai" else bool(setting("LLM_API_KEY") or setting("DASHSCOPE_API_KEY"))
    if use_llm and not LANGCHAIN_AVAILABLE:
        st.warning("LangChain isn't installed in this environment. The local question interpreter will be used.")
    elif use_llm and not has_key:
        st.info("No LLM API key configured. The app will use its local interpreter until you add Streamlit Secrets.")
    run = st.button("Analyze question", type="primary", disabled=not question.strip())

    if run:
        plan = None
        if use_llm and LANGCHAIN_AVAILABLE and has_key:
            with st.spinner("Understanding your question with the LLM..."):
                try:
                    plan = get_llm_plan(question.strip(), df)
                    plan = validate_plan(plan, df)
                    st.session_state["last_plan_source"] = f"LLM ({provider})"
                except Exception as exc:
                    st.warning(f"LLM planning failed; using local interpreter instead. Details: {type(exc).__name__}: {exc}")
        if plan is None:
            plan = local_plan(question.strip(), df)
            st.session_state["last_plan_source"] = "Local fallback"
        st.markdown(f"**Question interpretation:** `{plan['operation']}` · **Source:** {st.session_state['last_plan_source']}")
        if plan.get("answer_note"):
            st.info(plan["answer_note"])
        with st.spinner("Calculating from your data..."):
            result = execute_plan(df, plan)
        st.markdown("### Result")
        st.markdown(result.get("answer", ""))
        if use_llm and LANGCHAIN_AVAILABLE and has_key and plan["operation"] != "unknown":
            with st.spinner("Writing a business explanation..."):
                explanation = explain_with_llm(question.strip(), plan, result)
            st.markdown("### Business interpretation")
            st.markdown(explanation)
        if result.get("chart") is not None:
            st.plotly_chart(result["chart"], use_container_width=True)
        if isinstance(result.get("table"), pd.DataFrame):
            st.dataframe(result["table"], use_container_width=True)
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
# 8. FOOTER
# ============================================================
st.markdown("---")
st.caption("DEXCR is an analytics assistant. Verify important business decisions against source data. Forecasts are baseline estimates, not guarantees.")
