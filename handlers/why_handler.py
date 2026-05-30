import os
import re
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from data_loader import get_schema, run_query

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def clean_sql(text):
    text = re.sub(r"```sql", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()


def detect_intent(user_input: str) -> str:
    """Return 'best' if the user is asking about top/good performers, 'worst' otherwise."""
    q = user_input.lower()
    best_signals = [
        "performing well", "performing best", "best performing", "best region",
        "top performing", "top region", "highest", "leading", "strongest",
        "doing well", "most profitable", "most sales", "which is best",
        "which region is best", "what is performing", "which is performing"
    ]
    worst_signals = [
        "underperform", "worst", "lowest", "down", "declining", "struggling",
        "poor", "weak", "least", "loss", "problem", "issue", "concern",
        "why is", "why are", "what's wrong", "what is wrong"
    ]
    for s in best_signals:
        if s in q:
            return "best"
    for s in worst_signals:
        if s in q:
            return "worst"
    # Default: if "well", "good", "great" appear → best
    if any(w in q for w in ["well", "good", "great", "strong"]):
        return "best"
    return "worst"


def get_comparison_table(user_input: str, intent: str):
    """Pull a multi-KPI comparison table; sort best-first or worst-first based on intent."""
    q = user_input.lower()
    if "region" in q:
        group_col = "region"
    elif "category" in q:
        group_col = "category"
    elif "segment" in q:
        group_col = "segment"
    elif "state" in q:
        group_col = "state"
    else:
        group_col = "region"

    order = "DESC" if intent == "best" else "ASC"

    sql = f"""
        SELECT
            {group_col},
            '$' || CAST(ROUND(SUM(sales)/1000, 1) AS TEXT) || 'K'          AS "Total Sales",
            '$' || CAST(ROUND(SUM(profit)/1000, 1) AS TEXT) || 'K'         AS "Total Profit",
            CAST(ROUND(SUM(profit)*100.0/SUM(sales), 1) AS TEXT) || '%'    AS "Profit Margin",
            CAST(ROUND(AVG(discount)*100, 1) AS TEXT) || '%'                AS "Avg Discount"
        FROM superstore
        GROUP BY {group_col}
        ORDER BY SUM(profit) {order}
    """
    df, err = run_query(sql)
    if err or df is None:
        return None, group_col
    df.columns = [group_col.title(), "Total Sales", "Total Profit", "Profit Margin", "Avg Discount"]
    return df, group_col


def get_drill_down(group_col: str, focus_group: str):
    """Get category breakdown for the focus group."""
    sql = f"""
        SELECT
            category,
            '$' || CAST(ROUND(SUM(sales)/1000, 1) AS TEXT) || 'K'   AS "Sales",
            '$' || CAST(ROUND(SUM(profit)/1000, 1) AS TEXT) || 'K'  AS "Profit",
            CAST(ROUND(SUM(profit)*100.0/SUM(sales), 1) AS TEXT) || '%' AS "Margin"
        FROM superstore
        WHERE {group_col} = '{focus_group}'
        GROUP BY category
        ORDER BY SUM(profit) DESC
    """
    df, err = run_query(sql)
    if err or df is None:
        return None
    df.columns = ["Category", "Sales", "Profit", "Margin"]
    return df


LAYOUT = dict(
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    font=dict(family="Inter", color="#475569"),
    title_font=dict(size=14, color="#0f172a"),
    margin=dict(t=48, b=40, l=16, r=16),
    xaxis=dict(showgrid=False, linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
)
PALETTE = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]


def build_chart(user_input: str, group_col: str, intent: str):
    """Choose the most relevant chart type based on the user's question."""
    q = user_input.lower()
    order = "DESC" if intent == "best" else "ASC"

    # ── SCATTER: correlation / relationship / discount vs profit ────────────
    if any(w in q for w in ["correlation", "relationship", "scatter",
                              "discount vs", "vs profit", "vs sales"]):
        sql = f"""
            SELECT {group_col},
                   AVG(discount)*100  AS avg_discount,
                   SUM(profit)        AS total_profit,
                   SUM(sales)         AS total_sales
            FROM superstore GROUP BY {group_col}
        """
        df, err = run_query(sql)
        if err or df is None or df.empty:
            return None
        fig = px.scatter(
            df, x="avg_discount", y="total_profit",
            text=group_col, color=group_col,
            title=f"Discount vs Profit by {group_col.title()}",
            labels={"avg_discount": "Avg Discount (%)", "total_profit": "Total Profit ($)"},
            color_discrete_sequence=PALETTE,
            trendline="ols" if len(df) > 3 else None,
        )
        fig.update_traces(marker=dict(size=14, line=dict(width=1.5, color="#ffffff")),
                          textposition="top center")
        fig.update_layout(**LAYOUT)
        return fig

    # ── LINE: trend over time ───────────────────────────────────────────────
    if any(w in q for w in ["trend", "over time", "monthly", "yearly", "by month", "by year"]):
        sql = """
            SELECT strftime('%Y-%m', order_date) AS month,
                   SUM(sales)  AS total_sales,
                   SUM(profit) AS total_profit
            FROM superstore GROUP BY month ORDER BY month
        """
        df, err = run_query(sql)
        if err or df is None or df.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["month"], y=df["total_sales"], name="Sales",
                                  line=dict(color="#2563eb", width=2.5),
                                  fill="tozeroy", fillcolor="rgba(37,99,235,0.07)"))
        fig.add_trace(go.Scatter(x=df["month"], y=df["total_profit"], name="Profit",
                                  line=dict(color="#10b981", width=2.5)))
        fig.update_layout(title="Sales & Profit Trend Over Time",
                          hovermode="x unified", **LAYOUT,
                          legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
        return fig

    # ── DONUT: share / proportion / mix ────────────────────────────────────
    if any(w in q for w in ["share", "proportion", "mix", "composition", "split", "breakdown"]):
        metric = "profit" if "profit" in q else "sales"
        sql = f"""
            SELECT {group_col}, SUM({metric}) AS value
            FROM superstore GROUP BY {group_col} ORDER BY value DESC
        """
        df, err = run_query(sql)
        if err or df is None or df.empty:
            return None
        fig = px.pie(df, names=group_col, values="value",
                     title=f"{metric.title()} Share by {group_col.title()}",
                     hole=0.48, color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker=dict(line=dict(color="#ffffff", width=2)))
        fig.update_layout(**LAYOUT,
                          legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
        return fig

    # ── GROUPED BAR: compare metrics side by side ──────────────────────────
    if any(w in q for w in ["compare", "vs", "versus", "sales and profit"]):
        sql = f"""
            SELECT {group_col},
                   SUM(sales)  AS total_sales,
                   SUM(profit) AS total_profit
            FROM superstore GROUP BY {group_col} ORDER BY total_sales {order}
        """
        df, err = run_query(sql)
        if err or df is None or df.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Total Sales",  x=df[group_col], y=df["total_sales"],
                             marker_color="#2563eb",
                             text=[f"${v/1000:.0f}K" for v in df["total_sales"]],
                             textposition="outside"))
        fig.add_trace(go.Bar(name="Total Profit", x=df[group_col], y=df["total_profit"],
                             marker_color="#10b981",
                             text=[f"${v/1000:.0f}K" for v in df["total_profit"]],
                             textposition="outside"))
        fig.update_layout(barmode="group", title=f"Sales vs Profit by {group_col.title()}",
                          **LAYOUT,
                          legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
        return fig

    # ── DEFAULT: bar chart highlighting the focus group ─────────────────────
    metric = "total_profit" if any(w in q for w in ["profit", "margin", "loss"]) else "total_sales"
    label  = "Profit ($)" if metric == "total_profit" else "Sales ($)"
    title  = f"{'Profit' if metric == 'total_profit' else 'Sales'} by {group_col.title()}"

    sql = f"""
        SELECT {group_col},
               SUM(sales)  AS total_sales,
               SUM(profit) AS total_profit
        FROM superstore GROUP BY {group_col} ORDER BY {metric} {order}
    """
    df, err = run_query(sql)
    if err or df is None or df.empty:
        return None

    focus_val = df[metric].max() if intent == "best" else df[metric].min()
    colors = ["#2563eb" if v == focus_val else "#93c5fd" for v in df[metric]]

    fig = go.Figure(go.Bar(
        x=df[group_col], y=df[metric],
        marker_color=colors,
        text=[f"${v/1000:.0f}K" for v in df[metric]],
        textposition="outside"
    ))
    fig.update_layout(title=title, xaxis_title=group_col.title(), yaxis_title=label,
                      showlegend=False, **LAYOUT)
    return fig


def handle_why_request(user_input: str, selected_kpi: str = "sales"):
    schema = get_schema()
    intent = detect_intent(user_input)

    comparison_df, group_col = get_comparison_table(user_input, intent)
    focus_group = comparison_df.iloc[0][group_col.title()] if comparison_df is not None else "Unknown"
    drill_df = get_drill_down(group_col, focus_group)

    # Build data context for AI
    context = ""
    if comparison_df is not None:
        context += f"All {group_col}s comparison (sorted {'best' if intent == 'best' else 'worst'} first):\n"
        context += f"{comparison_df.to_string(index=False)}\n\n"
    if drill_df is not None:
        context += f"Category breakdown for {focus_group}:\n{drill_df.to_string(index=False)}\n\n"

    # Display SQL shown in expander
    order_dir = "DESC" if intent == "best" else "ASC"
    comparison_sql = f"""SELECT
    {group_col},
    ROUND(SUM(sales) / 1000, 1)                  AS total_sales_k,
    ROUND(SUM(profit) / 1000, 1)                 AS total_profit_k,
    ROUND(SUM(profit) * 100.0 / SUM(sales), 1)  AS profit_margin_pct,
    ROUND(AVG(discount) * 100, 1)                AS avg_discount_pct
FROM superstore
GROUP BY {group_col}
ORDER BY total_profit_k {order_dir}"""

    sql_queries_used = [comparison_sql]

    # Decide prompt language based on intent
    if intent == "best":
        summary_instruction = f"One clear sentence identifying {focus_group} as the best/top performer with its {selected_kpi} number"
        findings_instruction = "What makes this group a top performer — with specific numbers"
        root_causes_label = "key_drivers"
        root_causes_instruction = "Key drivers of strong performance with data points"
        recommendations_instruction = "Actions other groups can take to replicate this success"
        benchmark_instruction = f"One sentence comparing {focus_group} to the next best group and by how much it leads"
        section_label = "best_performer"
    else:
        summary_instruction = f"One clear sentence identifying {focus_group} as the worst performer with its {selected_kpi} number"
        findings_instruction = "What is going wrong — with specific numbers"
        root_causes_label = "root_causes"
        root_causes_instruction = "Root causes of underperformance with specific data points"
        recommendations_instruction = "Specific actions to improve performance"
        benchmark_instruction = "One sentence about the best performing group and what makes them successful"
        section_label = "worst_performer"

    answer_prompt = f"""
You are a senior data analyst. A business stakeholder asked: "{user_input}"
KPI focus: {selected_kpi}

Here is the actual data:
{context}

Respond ONLY with a valid JSON object. No text before or after. No markdown.

{{
  "summary": "{summary_instruction}",
  "{section_label}": "{focus_group}",
  "findings": [
    {{
      "title": "Short title (e.g. Highest Sales Revenue)",
      "explanation": "2-3 sentences with specific numbers. {findings_instruction}"
    }},
    {{
      "title": "Second finding title",
      "explanation": "2-3 sentences with specific numbers"
    }},
    {{
      "title": "Third finding title",
      "explanation": "2-3 sentences with specific numbers"
    }},
    {{
      "title": "Fourth finding title",
      "explanation": "2-3 sentences with specific numbers"
    }}
  ],
  "{root_causes_label}": [
    "{root_causes_instruction} 1",
    "Point 2 with data",
    "Point 3 with data",
    "Point 4 with data"
  ],
  "recommendations": [
    "{recommendations_instruction} 1",
    "Action 2",
    "Action 3",
    "Action 4"
  ],
  "benchmark": "{benchmark_instruction}"
}}

Rules:
- Use ONLY real numbers from the data provided above
- Every finding must cite actual figures
- Keep "summary" to one sentence
- Return ONLY the JSON — no prose outside it
"""
    answer_resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.1
    )
    raw = answer_resp.choices[0].message.content.strip()
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)

    structured = {}
    if json_match:
        try:
            structured = json.loads(json_match.group())
        except json.JSONDecodeError:
            structured = {"summary": raw}

    # Normalise key so app.py always reads "worst_performer" or "best_performer" as focus_group
    if "best_performer" in structured:
        structured["worst_performer"] = structured["best_performer"]
    if "root_causes" not in structured and "key_drivers" in structured:
        structured["root_causes"] = structured["key_drivers"]

    structured["_intent"] = intent  # pass intent to app.py for label changes

    chart = build_chart(user_input, group_col, intent)
    raw_df, _ = run_query(f"SELECT * FROM superstore WHERE {group_col} = '{focus_group}' LIMIT 500")

    return structured, comparison_df, drill_df, chart, sql_queries_used, raw_df
