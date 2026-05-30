import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Columns that are IDs/codes — never use as chart metrics
SKIP_COLS = {
    "row_id", "row id", "order_id", "customer_id", "product_id",
    "postal_code", "country", "city", "order_id", "ship_date"
}

# Preferred metric columns in priority order
METRIC_PRIORITY = ["sales", "profit", "quantity", "discount"]

# Shared clean layout defaults
LAYOUT_BASE = dict(
    font=dict(family="Inter, sans-serif", size=12, color="#475569"),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(t=48, b=40, l=16, r=16),
    title_font=dict(size=14, color="#0f172a", family="Inter, sans-serif"),
    hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_family="Inter"),
)

AXIS_CLEAN = dict(
    xaxis=dict(showgrid=False, linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
)

# Brand palette
BLUE       = "#2563eb"
BLUE_LIGHT = "#93c5fd"
GREEN      = "#10b981"
RED        = "#ef4444"
PALETTE    = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
SEQUENTIAL = [[0, "#dbeafe"], [0.5, "#93c5fd"], [1.0, "#1d4ed8"]]


def get_best_metric(numeric_cols):
    for preferred in METRIC_PRIORITY:
        for col in numeric_cols:
            if preferred in col.lower():
                return col
    for col in numeric_cols:
        if col.lower() not in SKIP_COLS:
            return col
    return numeric_cols[0] if numeric_cols else None


def detect_chart_type(question: str, df: pd.DataFrame) -> str:
    q = question.lower()
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c.lower() not in SKIP_COLS]
    text_cols    = df.select_dtypes(include="object").columns.tolist()
    date_cols    = [c for c in df.columns if any(x in c.lower() for x in ["date", "month", "year", "time", "day"])]
    num_cats     = df[text_cols[0]].nunique() if text_cols else 0

    # TIME SERIES → Line
    if date_cols and any(w in q for w in ["over time", "trend", "monthly", "yearly", "daily", "weekly",
                                           "by month", "by year", "by date", "by week"]):
        return "line"

    # RANKING / TOP-N → Horizontal bar
    if any(w in q for w in ["top", "bottom", "highest", "lowest", "ranking",
                              "best", "worst", "least", "most", "leading"]):
        return "hbar"

    # COMPARISON of multiple metrics → Grouped bar
    if any(w in q for w in ["compare", "vs", "versus", "difference", "between"]) and len(numeric_cols) >= 2:
        return "grouped_bar"

    # DISTRIBUTION → Histogram
    if any(w in q for w in ["distribution", "spread", "range", "histogram", "frequency"]):
        return "histogram"

    # CORRELATION → Scatter
    if len(numeric_cols) >= 2 and any(w in q for w in ["correlation", "relationship", "scatter"]):
        return "scatter"

    # PROPORTION / SHARE / SPLIT (few categories) → Donut
    if text_cols and 2 <= num_cats <= 6 and any(w in q for w in [
        "by", "per", "each", "share", "breakdown", "split", "proportion",
        "segment", "region", "category", "mix", "composition"
    ]):
        return "donut"

    # RAW ORDER DATA (many rows, has date + sales) → Summary bar
    if len(df) > 50 and date_cols and "sales" in [c.lower() for c in df.columns]:
        return "raw_summary"

    # MANY CATEGORIES → Horizontal bar
    if text_cols and num_cats > 8:
        return "hbar"

    # FEW CATEGORIES → Donut
    if text_cols and 2 <= num_cats <= 6:
        return "donut"

    return "vbar"


def _apply_base(fig, title):
    fig.update_layout(title=title, **LAYOUT_BASE, **AXIS_CLEAN)
    return fig


def smart_chart(df: pd.DataFrame, title: str = "", question: str = ""):
    try:
        if df is None or df.empty:
            return None

        all_numeric  = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in all_numeric if c.lower() not in SKIP_COLS]
        text_cols    = [c for c in df.select_dtypes(include="object").columns if c.lower() not in SKIP_COLS]
        date_cols    = [c for c in df.columns if any(x in c.lower() for x in ["order_date", "date", "month", "year"])]

        if not numeric_cols:
            return None

        y_col       = get_best_metric(numeric_cols)
        chart_type  = detect_chart_type(question or title, df)
        chart_title = title or question or f"{y_col.replace('_', ' ').title()} Analysis"

        # ── RAW SUMMARY ────────────────────────────────────────────────────────
        if chart_type == "raw_summary":
            cat_col    = next((c for c in df.columns if "category" in c.lower() and "sub" not in c.lower()), None)
            sales_col  = next((c for c in df.columns if "sales" in c.lower()), None)
            profit_col = next((c for c in df.columns if "profit" in c.lower()), None)

            if cat_col and sales_col:
                agg = df.groupby(cat_col).agg(
                    Total_Sales=(sales_col, "sum"),
                    Total_Profit=(profit_col, "sum") if profit_col else (sales_col, "count")
                ).reset_index().sort_values("Total_Sales", ascending=False)

                fig = go.Figure()
                fig.add_trace(go.Bar(name="Total Sales",  x=agg[cat_col], y=agg["Total_Sales"],
                                     marker_color=BLUE,  text=[f"${v:,.0f}" for v in agg["Total_Sales"]],
                                     textposition="outside"))
                if profit_col:
                    fig.add_trace(go.Bar(name="Total Profit", x=agg[cat_col], y=agg["Total_Profit"],
                                         marker_color=GREEN, text=[f"${v:,.0f}" for v in agg["Total_Profit"]],
                                         textposition="outside"))
                fig.update_layout(
                    barmode="group",
                    xaxis_title="Category", yaxis_title="Amount ($)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    **LAYOUT_BASE, **AXIS_CLEAN, title=chart_title
                )
                return fig

        # ── LINE CHART ─────────────────────────────────────────────────────────
        if chart_type == "line" and date_cols:
            x_col = date_cols[0]
            try:
                df = df.copy()
                col_vals = df[x_col].dropna().astype(str)

                # Detect if column contains year-only (e.g. "2014") or year-month (e.g. "2014-07")
                is_year_only   = col_vals.str.match(r'^\d{4}$').all()
                is_year_month  = col_vals.str.match(r'^\d{4}-\d{2}$').all()

                if is_year_only:
                    # Keep as string labels — no datetime conversion needed
                    df[x_col] = df[x_col].astype(str)
                    df = df.sort_values(x_col)
                elif is_year_month:
                    df[x_col] = pd.to_datetime(df[x_col], format="%Y-%m")
                    df = df.sort_values(x_col)
                    df[x_col] = df[x_col].dt.strftime("%b %Y")   # "Jan 2014"
                else:
                    df[x_col] = pd.to_datetime(df[x_col], infer_datetime_format=True, errors="coerce")
                    df = df.dropna(subset=[x_col]).sort_values(x_col)
                    if len(df) > 100:
                        df["_period"] = df[x_col].dt.to_period("M").dt.to_timestamp()
                        df = df.groupby("_period")[y_col].sum().reset_index()
                        df.columns = [x_col, y_col]
                        df[x_col] = pd.to_datetime(df[x_col]).dt.strftime("%b %Y")

                fig = px.line(df, x=x_col, y=y_col, title=chart_title, markers=True,
                              labels={x_col: x_col.replace("_", " ").title(),
                                      y_col: y_col.replace("_", " ").title()})
                fig.update_traces(line_color=BLUE, line_width=2.5, marker_size=8,
                                  marker_color=BLUE,
                                  fill="tozeroy", fillcolor="rgba(37,99,235,0.08)")
                fig.update_layout(hovermode="x unified", **LAYOUT_BASE, **AXIS_CLEAN, title=chart_title)
                return fig
            except Exception:
                pass

        # ── DONUT ──────────────────────────────────────────────────────────────
        if chart_type == "donut" and text_cols:
            x_col = text_cols[0]
            fig = px.pie(df, names=x_col, values=y_col, title=chart_title, hole=0.48,
                         color_discrete_sequence=PALETTE)
            fig.update_traces(
                textposition="inside", textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
                marker=dict(line=dict(color="#ffffff", width=2))
            )
            fig.update_layout(
                legend_title=x_col.replace("_", " ").title(),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                **LAYOUT_BASE, title=chart_title
            )
            return fig

        # ── HORIZONTAL BAR ─────────────────────────────────────────────────────
        if chart_type == "hbar" and text_cols:
            x_col    = text_cols[0]
            df_sorted = df.sort_values(y_col, ascending=True).tail(15)
            # Gradient: highest value = darkest blue
            n = len(df_sorted)
            bar_colors = [f"rgba(37,99,235,{0.35 + 0.65 * i / max(n - 1, 1):.2f})" for i in range(n)]
            fig = go.Figure(go.Bar(
                x=df_sorted[y_col], y=df_sorted[x_col],
                orientation="h",
                marker_color=bar_colors,
                text=[f"${v:,.0f}" if any(k in y_col.lower() for k in ["sales","profit","revenue","amount","($)"])
                      else f"{v:,.0f}" for v in df_sorted[y_col]],
                textposition="outside",
            ))
            fig.update_layout(
                xaxis_title=y_col.replace("_", " ").title(),
                yaxis_title="",
                **LAYOUT_BASE, **AXIS_CLEAN, title=chart_title
            )
            return fig

        # ── GROUPED BAR ────────────────────────────────────────────────────────
        if chart_type == "grouped_bar" and text_cols and len(numeric_cols) >= 2:
            x_col = text_cols[0]
            fig = go.Figure()
            for i, col in enumerate(numeric_cols[:4]):
                fig.add_trace(go.Bar(
                    name=col.replace("_", " ").title(),
                    x=df[x_col], y=df[col],
                    marker_color=PALETTE[i % len(PALETTE)]
                ))
            fig.update_layout(
                barmode="group",
                xaxis_title=x_col.replace("_", " ").title(), yaxis_title="Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                **LAYOUT_BASE, **AXIS_CLEAN, title=chart_title
            )
            return fig

        # ── SCATTER ────────────────────────────────────────────────────────────
        if chart_type == "scatter" and len(numeric_cols) >= 2:
            x_col     = numeric_cols[0]
            y_col2    = numeric_cols[1]
            color_col = text_cols[0] if text_cols else None
            fig = px.scatter(
                df, x=x_col, y=y_col2, color=color_col,
                title=chart_title,
                trendline="ols" if len(df) > 5 else None,
                color_discrete_sequence=PALETTE,
                labels={x_col: x_col.replace("_", " ").title(),
                        y_col2: y_col2.replace("_", " ").title()},
                opacity=0.75
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color="#ffffff")))
            fig.update_layout(**LAYOUT_BASE, **AXIS_CLEAN, title=chart_title)
            return fig

        # ── HISTOGRAM ──────────────────────────────────────────────────────────
        if chart_type == "histogram":
            fig = px.histogram(df, x=y_col, title=chart_title, nbins=20,
                               color_discrete_sequence=[BLUE],
                               labels={y_col: y_col.replace("_", " ").title()})
            fig.update_traces(marker_line_color="#ffffff", marker_line_width=1)
            fig.update_layout(bargap=0.06, **LAYOUT_BASE, **AXIS_CLEAN, title=chart_title)
            return fig

        # ── DEFAULT VERTICAL BAR ───────────────────────────────────────────────
        if text_cols:
            x_col     = text_cols[0]
            df_sorted = df.sort_values(y_col, ascending=False)
            n         = len(df_sorted)
            bar_colors = [f"rgba(37,99,235,{1.0 - 0.5 * i / max(n - 1, 1):.2f})" for i in range(n)]

            fig = go.Figure(go.Bar(
                x=df_sorted[x_col], y=df_sorted[y_col],
                marker_color=bar_colors,
                text=[f"${v:,.0f}" if any(k in y_col.lower() for k in ["sales","profit","revenue","($)"])
                      else f"{v:,.0f}" for v in df_sorted[y_col]],
                textposition="outside",
            ))
            fig.update_layout(
                xaxis_title=x_col.replace("_", " ").title(),
                yaxis_title=y_col.replace("_", " ").title(),
                **LAYOUT_BASE, **AXIS_CLEAN, title=chart_title
            )
            return fig

    except Exception:
        return None

    return None
