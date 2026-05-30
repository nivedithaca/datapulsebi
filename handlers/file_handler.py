import os
import pandas as pd
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def handle_file(df: pd.DataFrame, filename: str):
    rows, cols = df.shape
    col_names = list(df.columns)
    null_info = df.isnull().sum()
    null_cols = null_info[null_info > 0].to_dict()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    summary_for_ai = f"""
File: {filename}
Rows: {rows}
Columns ({cols}): {col_names}
Null values: {null_cols}
Numeric columns: {numeric_cols}
Sample data (first 5 rows):
{df.head().to_string()}

Basic stats:
{df.describe().to_string()}
"""

    prompt = f"""
You are a senior data analyst. Analyze this dataset and provide a clear, business-friendly summary.

{summary_for_ai}

Write a summary with:
1. What this dataset contains (1-2 sentences)
2. Top 3-5 key observations (use bullet points, include actual numbers)
3. Any data quality issues found (nulls, outliers, anomalies)
4. One business recommendation based on the data

Keep it concise and avoid technical jargon. Write as if presenting to a business stakeholder.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    insight = response.choices[0].message.content.strip()

    chart = None
    if numeric_cols:
        try:
            top_col = numeric_cols[0]
            text_cols = df.select_dtypes(include="object").columns.tolist()
            if text_cols:
                group_col = text_cols[0]
                chart_data = df.groupby(group_col)[top_col].sum().reset_index()
                chart_data = chart_data.sort_values(top_col, ascending=False).head(10)
                chart = px.bar(
                    chart_data,
                    x=group_col,
                    y=top_col,
                    title=f"{top_col} by {group_col}",
                    color=top_col,
                    color_continuous_scale="Blues"
                )
        except Exception:
            chart = None

    return insight, chart
