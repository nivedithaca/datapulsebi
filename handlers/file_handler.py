import os
import re
import pandas as pd
import plotly.express as px
from groq import Groq
from utils.config import get_groq_api_key
from utils.chart_generator import smart_chart

client = Groq(api_key=get_groq_api_key())


def _find_col(df: pd.DataFrame, keywords: list[str], exclude_id: bool = False):
    """Return first column whose name contains any keyword (case-insensitive).
       If exclude_id=True, skip columns that look like ID/code fields."""
    id_hints = ["_id", "id_", " id", "id ", "_code", "code_", "_key", "key_", "_num", "num_"]
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in col.lower():
                if exclude_id:
                    col_l = col.lower()
                    # Skip if column name ends with 'id' or contains id hint
                    if col_l == "id" or col_l.endswith("id") or any(h in col_l for h in id_hints):
                        continue
                return col
    return None


def _find_text_col(df: pd.DataFrame, keywords: list[str]):
    """Return first TEXT (object/string) column matching any keyword, skipping ID-like columns."""
    id_hints = ["_id", "id_", "_code", "code_", "_key", "_num"]
    for kw in keywords:
        for col in df.columns:
            col_l = col.lower()
            if kw.lower() in col_l and df[col].dtype == object:
                if col_l == "id" or col_l.endswith("id") or any(h in col_l for h in id_hints):
                    continue
                return col
    return None


def _best_text_col(df: pd.DataFrame):
    """Return best text column for grouping — skip IDs, prefer low-cardinality categorical columns."""
    id_hints = ["_id", "id_", "_code", "code_", "_key", "_num", "_no", "no_"]
    text_cols = df.select_dtypes(include="object").columns.tolist()
    candidates = []
    for col in text_cols:
        col_l = col.lower()
        if col_l == "id" or col_l.endswith("id") or any(h in col_l for h in id_hints):
            continue
        n_unique = df[col].nunique()
        candidates.append((col, n_unique))
    if not candidates:
        return None
    # Prefer columns with 2–100 unique values (true categories) over high-cardinality strings
    cats = [(c, n) for c, n in candidates if 2 <= n <= 100]
    if cats:
        return min(cats, key=lambda x: x[1])[0]   # fewest unique = most categorical
    return candidates[0][0]


ID_HINTS = ["_id", "id_", "_code", "code_", "_key", "_num", "_no", "no_"]


def _is_id_col(col: str) -> bool:
    col_l = col.lower().strip()
    return col_l == "id" or col_l.endswith("id") or any(h in col_l for h in ID_HINTS)


def _numeric_non_id_cols(df: pd.DataFrame) -> list:
    """All numeric columns that are NOT ID/code fields."""
    return [c for c in df.select_dtypes(include="number").columns if not _is_id_col(c)]


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    """Strip $, £, €, commas, spaces then convert to float."""
    if series.dtype == object:
        series = series.astype(str).str.replace(r'[\$£€,\s]', '', regex=True)
    return pd.to_numeric(series, errors="coerce")


def _compute_revenue(df: pd.DataFrame):
    """
    Try to find or compute a revenue/total-value column.
    Priority: explicit revenue col → price×qty → highest-mean numeric col.
    Always returns (col_name, df) — df may have a new '_revenue' column.
    """
    df = df.copy()

    # 1. Explicit named revenue/sales/total column
    for kw in ["revenue", "total_revenue", "total_sales", "net_sales", "gross_sales",
               "sales", "total_amount", "net_amount", "gross_amount", "amount", "total"]:
        for col in df.columns:
            if _is_id_col(col):
                continue
            if kw == col.lower().replace(" ", "_") or kw in col.lower():
                cleaned = _to_numeric_clean(df[col])
                if cleaned.notna().sum() > len(df) * 0.5:   # >50% valid values
                    df[col] = cleaned
                    return col, df

    # 2. Compute price × quantity
    price_col = None
    qty_col   = None
    for col in df.columns:
        col_l = col.lower()
        if _is_id_col(col):
            continue
        if price_col is None and any(k in col_l for k in ["unit_price", "price", "cost", "rate", "fee", "fare", "amount"]):
            price_col = col
        if qty_col is None and any(k in col_l for k in ["transaction_qty", "quantity", "qty", "units", "volume", "sold", "pieces", "count"]):
            qty_col = col

    if price_col and qty_col and price_col != qty_col:
        p = _to_numeric_clean(df[price_col])
        q = _to_numeric_clean(df[qty_col])
        rev = p * q
        if rev.notna().sum() > len(df) * 0.5:
            df["_revenue"] = rev
            return "_revenue", df

    # 3. Fallback: numeric col with the largest mean (most likely a $ amount, not a count)
    num_cols = _numeric_non_id_cols(df)
    if num_cols:
        best = max(num_cols, key=lambda c: _to_numeric_clean(df[c]).mean())
        df[best] = _to_numeric_clean(df[best])
        return best, df

    return None, df


def _answer_from_file(df: pd.DataFrame, question: str):
    """
    Answer a specific business question directly from the uploaded df using pandas.
    Returns (result_df, chart, answered: bool).
    """
    q = question.lower()

    # Detect N (top/bottom count)
    n_match = re.search(r'\b(\d+)\b', q)
    n = int(n_match.group(1)) if n_match else None
    is_top    = any(w in q for w in ["top", "best", "highest", "most", "largest", "maximum", "max", "generate"])
    is_bottom = any(w in q for w in ["bottom", "worst", "lowest", "least", "smallest", "minimum", "min"])

    df = df.copy()

    # ── Detect metric + aggregation function ─────────────────────────────────
    metric_col  = None
    agg_fn      = "sum"    # default: sum for monetary, overridden below for rates

    # Explicit "how many orders / transactions / count"
    if re.search(r'\b(how many|number of|count of)\b.*(order|transaction|purchase|record|row)', q):
        metric_col = None   # signal to use COUNT
        agg_fn     = "count"

    # Rating / score / review → mean
    elif any(w in q for w in ["rating", "rated", "review", "score", "star", "satisfaction"]):
        metric_col = _find_col(df, ["rating", "review", "score", "star", "satisfaction"])
        agg_fn = "mean"

    # Discount → mean
    elif any(w in q for w in ["discount", "markdown"]):
        metric_col = _find_col(df, ["discount", "markdown", "rebate"])
        agg_fn = "mean"

    # Quantity / units sold → sum
    elif any(w in q for w in ["quantity", "units sold", "qty", "volume", "items sold"]):
        metric_col = _find_col(df, ["quantity", "qty", "units", "volume", "sold", "pieces"])
        agg_fn = "sum"

    # Profit / margin → sum
    elif any(w in q for w in ["profit", "margin", "net income", "earnings"]):
        metric_col = _find_col(df, ["profit", "margin", "net_income", "earnings"])
        agg_fn = "sum"

    # Price / cost → mean (average price per group)
    elif any(w in q for w in ["average price", "avg price", "mean price", "price per", "cost per"]):
        metric_col = _find_col(df, ["price", "unit_price", "cost", "rate"])
        agg_fn = "mean"

    # Revenue / sales / income / "generates" / "earns" → computed revenue, sum
    elif any(w in q for w in ["revenue", "sales", "income", "earning", "generate", "earn",
                               "amount", "value", "worth", "spend", "spent"]):
        metric_col, df = _compute_revenue(df)
        agg_fn = "sum"

    # Orders/transactions WITHOUT "how many" → treat as revenue
    elif any(w in q for w in ["order", "transaction", "purchase", "bought", "sold"]):
        metric_col, df = _compute_revenue(df)
        agg_fn = "sum"

    # Default fallback → revenue
    if metric_col is None and agg_fn != "count":
        metric_col, df = _compute_revenue(df)

    # ── Detect group-by / dimension column ────────────────────────────────────
    group_col = None
    if any(w in q for w in ["product type", "product category", "item type"]):
        group_col = _find_text_col(df, ["product_type", "product_category", "type", "category", "item_type"])
    elif any(w in q for w in ["product", "item", "sku"]):
        group_col = _find_text_col(df, ["product_name", "product", "name", "title", "item", "sku", "description"])
    elif any(w in q for w in ["category", "department", "type", "genre"]):
        group_col = _find_text_col(df, ["category", "department", "type", "genre", "class"])
    elif any(w in q for w in ["customer", "buyer", "user", "client"]):
        group_col = _find_text_col(df, ["customer_name", "customer", "buyer", "user", "client", "name"])
    elif any(w in q for w in ["region", "state", "city", "country", "location", "area", "store"]):
        group_col = _find_text_col(df, ["region", "state", "city", "country", "location", "area", "store"])
    elif any(w in q for w in ["brand", "seller", "vendor", "supplier", "manufacturer"]):
        group_col = _find_text_col(df, ["brand", "seller", "vendor", "supplier", "manufacturer"])
    elif any(w in q for w in ["channel", "platform", "source", "medium"]):
        group_col = _find_text_col(df, ["channel", "platform", "source", "medium"])

    # Fallback: best categorical text column (avoids ID columns)
    if group_col is None:
        group_col = _best_text_col(df)

    if not group_col:
        return None, None, False
    if agg_fn != "count" and not metric_col:
        return None, None, False

    try:
        # ── Aggregation ───────────────────────────────────────────────────────
        if agg_fn == "count":
            agg = df.groupby(group_col).size().reset_index(name="Order Count")
            display_metric = "Order Count"
        else:
            df[metric_col] = _to_numeric_clean(df[metric_col])

            if metric_col == "_revenue":
                display_metric = "Total Revenue ($)"
            elif agg_fn == "mean":
                display_metric = f"Avg {metric_col.replace('_', ' ').title()}"
            else:
                display_metric = f"Total {metric_col.replace('_', ' ').title()}"

            grouped = df.groupby(group_col)[metric_col]
            agg_series = grouped.mean() if agg_fn == "mean" else grouped.sum()
            agg = agg_series.reset_index().rename(columns={metric_col: display_metric})

        # Clean group column display name
        group_display = group_col.replace("_", " ").title()
        agg = agg.rename(columns={group_col: group_display})

        ascending   = is_bottom and not is_top
        total_rows  = len(agg)

        if n:
            agg["_rn"] = agg[display_metric].rank(method="first", ascending=ascending).astype(int)
            result = agg[agg["_rn"] <= n].drop(columns=["_rn"]).sort_values(display_metric, ascending=ascending)
        else:
            result = agg.sort_values(display_metric, ascending=ascending)

        result = result.reset_index(drop=True)

        # Build a rich chart
        chart = _build_file_chart(result, group_display, display_metric, question, n, total_rows)
        # Meta: how many were requested vs available
        meta = {"requested": n, "available": total_rows, "returned": len(result)}
        return result, chart, True, meta

    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None, False, {}


def _build_file_chart(result, group_col, metric_col, title, n_requested, total_available):
    """Build a polished Plotly bar chart for file query results."""
    import plotly.graph_objects as go

    if result is None or result.empty:
        return None

    try:
        df_sorted = result.sort_values(metric_col, ascending=True)  # ascending for hbar
        n = len(df_sorted)

        # Gradient: darkest = highest value
        bar_colors = [f"rgba(37,99,235,{0.35 + 0.65 * i / max(n - 1, 1):.2f})" for i in range(n)]

        is_money = any(k in metric_col.lower() for k in ["revenue", "sales", "profit", "amount", "($)"])

        text_labels = [
            f"${v:,.0f}" if is_money else f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
            for v in df_sorted[metric_col]
        ]

        fig = go.Figure(go.Bar(
            x=df_sorted[metric_col],
            y=df_sorted[group_col],
            orientation="h",
            marker_color=bar_colors,
            marker_line_width=0,
            text=text_labels,
            textposition="outside",
            textfont=dict(size=12, color="#0f172a", family="Inter, sans-serif"),
            hovertemplate=f"<b>%{{y}}</b><br>{metric_col}: %{{x:,.2f}}<extra></extra>",
        ))

        shown = len(result)
        display_title = title if len(title) < 60 else title[:57] + "…"
        if n_requested and total_available < n_requested:
            display_title += f" (all {total_available} available)"

        max_val = df_sorted[metric_col].max()
        fig.update_layout(
            title=dict(text=display_title, font=dict(size=14, color="#0f172a", family="Inter, sans-serif")),
            xaxis=dict(
                title=metric_col,
                showgrid=True, gridcolor="#f1f5f9",
                linecolor="#e2e8f0",
                tickfont=dict(color="#64748b", size=11),
                tickformat="$,.0f" if is_money else ",",
                range=[0, max_val * 1.22],   # extra space for labels
            ),
            yaxis=dict(
                showgrid=False, linecolor="#e2e8f0",
                tickfont=dict(color="#1e293b", size=12),
                automargin=True,
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color="#475569"),
            margin=dict(t=56, b=40, l=16, r=80),
            hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_family="Inter"),
            height=max(300, n * 52 + 100),   # dynamic height based on row count
        )
        return fig
    except Exception:
        return None


def handle_file(df: pd.DataFrame, filename: str, user_question: str = ""):
    rows, cols  = df.shape
    col_names   = list(df.columns)
    null_info = df.isnull().sum()
    null_cols = null_info[null_info > 0].to_dict()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols    = df.select_dtypes(include="object").columns.tolist()

    # ── 1. Try to answer the specific question from the file first ─────────────
    if user_question:
        ans = _answer_from_file(df, user_question)
        specific_df, specific_chart, answered = ans[0], ans[1], ans[2]
        query_meta = ans[3] if len(ans) > 3 else {}
    else:
        specific_df, specific_chart, answered, query_meta = None, None, False, {}

    # ── 2. Build compact revenue-aware summary for AI ────────────────────────
    # Compute revenue column once for the summary
    rev_col, df_rev = _compute_revenue(df)

    # Revenue by best categorical column (for AI context — not transaction counts)
    revenue_by_group = ""
    try:
        best_cat = _best_text_col(df)
        if best_cat and rev_col:
            rev_grp = (
                df_rev.groupby(best_cat)[rev_col]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            revenue_by_group = f"\nTop 5 {best_cat} by Revenue:\n" + "\n".join(
                f"  {k}: {v:,.2f}" for k, v in rev_grp.items()
            )
    except Exception:
        pass

    col_ranges = []
    for c in [rev_col] + [c for c in _numeric_non_id_cols(df) if c != rev_col][:5]:
        if c and c != "_revenue":
            try:
                s = _to_numeric_clean(df[c])
                col_ranges.append(f"  {c}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}")
            except Exception:
                pass

    question_context = (
        f"\nUser's specific question: {user_question}\n"
        f"Focus your observations on revenue and business value — NOT transaction counts.\n"
    ) if user_question else ""

    summary_for_ai = f"""File: {filename}
Shape: {rows:,} rows × {cols} columns
Columns: {col_names}
Null counts (non-zero only): {null_cols if null_cols else 'None'}
Revenue column used: {rev_col if rev_col != '_revenue' else 'computed (unit_price × quantity)'}
Numeric column ranges:
{chr(10).join(col_ranges) if col_ranges else 'None'}
{revenue_by_group}
"""

    prompt = f"""You are a senior data analyst. Analyze this uploaded dataset.
{question_context}
{summary_for_ai}

Write a summary with:
1. What this dataset contains (1-2 sentences)
2. Top 3-5 key observations (bullet points with actual numbers)
3. Any data quality issues (nulls, outliers)
4. One business recommendation

Be concise. Max 200 words.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    insight = response.choices[0].message.content.strip()

    # ── 3. Fallback generic chart if specific query didn't produce one ─────────
    chart = specific_chart
    if chart is None and numeric_cols and text_cols:
        try:
            group_col  = text_cols[0]
            metric_col = numeric_cols[0]
            chart_data = (
                df.groupby(group_col)[metric_col]
                .sum()
                .reset_index()
                .sort_values(metric_col, ascending=False)
                .head(15)
            )
            chart = smart_chart(chart_data, title=f"{metric_col} by {group_col}", question="")
        except Exception:
            chart = None

    return insight, chart, specific_df, specific_chart, answered, query_meta
