import os
import re
import sqlparse
import pandas as pd
from groq import Groq
from data_loader import get_schema, run_query
from utils.chart_generator import smart_chart
from utils.config import get_groq_api_key

client = Groq(api_key=get_groq_api_key())

# SAFE USPS codes — NOT common English words, match any case anywhere
SAFE_USPS = {
    "AK": "Alaska", "AZ": "Arizona", "CA": "California", "CT": "Connecticut",
    "FL": "Florida", "IA": "Iowa", "IL": "Illinois", "KS": "Kansas",
    "KY": "Kentucky", "MD": "Maryland", "MI": "Michigan", "MN": "Minnesota",
    "MT": "Montana", "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NV": "Nevada", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

# AMBIGUOUS USPS codes — clash with common English words
# Only match when written in ALL CAPS (user clearly intended an abbreviation)
AMBIGUOUS_USPS = {
    "AL": "Alabama", "AR": "Arkansas", "CO": "Colorado", "DE": "Delaware",
    "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IN": "Indiana",
    "LA": "Louisiana", "MA": "Massachusetts", "ME": "Maine", "MO": "Missouri",
    "MS": "Mississippi", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "UT": "Utah",
}

# Globally recognized shorthands (3+ chars) — always safe to match, any case
ALIAS_MAP = {
    # California
    "cal": "California", "cali": "California", "calif": "California",
    # New York
    "new york city": "New York", "nyc": "New York", "n.y.": "New York",
    # Texas
    "tex": "Texas",
    # Florida
    "fla": "Florida",
    # Pennsylvania
    "penn": "Pennsylvania", "penna": "Pennsylvania",
    # Washington
    "wash": "Washington",
    # Illinois
    "ill": "Illinois",
    # Michigan
    "mich": "Michigan",
    # Virginia
    "virg": "Virginia",
    # Tennessee
    "tenn": "Tennessee",
    # Colorado
    "colo": "Colorado",
    # Arizona
    "ariz": "Arizona",
    # Minnesota
    "minn": "Minnesota",
    # Wisconsin
    "wisc": "Wisconsin",
    # Massachusetts
    "mass": "Massachusetts",
    # Connecticut
    "conn": "Connecticut",
    # Regions
    "western": "West", "eastern": "East", "southern": "South",
    "central": "Central", "northern": "North", "midwest": "Central",
    "northeast": "East", "southeast": "South", "southwest": "West", "northwest": "West",
    # Categories / Departments
    "technologies": "Technology", "electronics": "Technology",
    "furnishings": "Furniture",
    "office supply": "Office Supplies", "stationery": "Office Supplies",
    "dept": "category", "department": "category", "departments": "category",
    "product type": "category", "product category": "category",
    # Sub-category aliases
    "sub category": "sub_category", "subcategory": "sub_category",
    "sub-category": "sub_category", "product line": "sub_category",
    # Segments
    "businesses": "Corporate", "corp": "Corporate",
    "consumers": "Consumer", "retail": "Consumer",
    # Ship mode
    "shipping mode": "ship_mode", "shipping method": "ship_mode",
    "delivery mode": "ship_mode",
}


# Semantic phrase rewrites — user intent → clear SQL-friendly phrasing
# Sorted longest first so multi-word phrases match before single words
SEMANTIC_MAP = [
    # "sales count / number of sales / sales amount" → total sales
    (r'\bsales\s+count\b',          "total sales (SUM of sales)"),
    (r'\bcount\s+of\s+sales\b',     "total sales (SUM of sales)"),
    (r'\bnumber\s+of\s+sales\b',    "total sales (SUM of sales)"),
    (r'\bsales\s+amount\b',         "total sales (SUM of sales)"),
    (r'\bsales\s+total\b',          "total sales (SUM of sales)"),
    (r'\btotal\s+revenue\b',        "total sales (SUM of sales)"),
    (r'\brevenue\b',                 "total sales (SUM of sales)"),
    # "profit count / number of profit" → total profit
    (r'\bprofit\s+count\b',         "total profit (SUM of profit)"),
    (r'\bcount\s+of\s+profit\b',    "total profit (SUM of profit)"),
    (r'\bnumber\s+of\s+profit\b',   "total profit (SUM of profit)"),
    # "order count / number of orders / how many orders" → count of orders
    (r'\border\s+count\b',          "number of orders (COUNT of rows)"),
    (r'\bcount\s+of\s+orders\b',    "number of orders (COUNT of rows)"),
    (r'\bnumber\s+of\s+orders\b',   "number of orders (COUNT of rows)"),
    (r'\bhow\s+many\s+orders\b',    "number of orders (COUNT of rows)"),
    # "customer count / number of customers"
    (r'\bcustomer\s+count\b',       "number of unique customers (COUNT DISTINCT customer_id)"),
    (r'\bnumber\s+of\s+customers\b',"number of unique customers (COUNT DISTINCT customer_id)"),
    (r'\bhow\s+many\s+customers\b', "number of unique customers (COUNT DISTINCT customer_id)"),
    # "quantity count / units sold"
    (r'\bquantity\s+count\b',       "total quantity (SUM of quantity)"),
    (r'\bunits\s+sold\b',           "total quantity (SUM of quantity)"),
    (r'\bitems\s+sold\b',           "total quantity (SUM of quantity)"),
    # "avg / average discount"
    (r'\bavg\s+discount\b',         "average discount (AVG of discount)"),
    (r'\baverage\s+discount\b',     "average discount (AVG of discount)"),
]


def normalize_input(text: str) -> str:
    """Replace abbreviations, aliases, and semantic phrases with clear SQL intent."""
    result = text

    # 1. Semantic rewrites first (business phrases → SQL intent)
    for pattern, replacement in SEMANTIC_MAP:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # 2. Longer aliases (3+ chars) — always safe, any case
    for alias, full in sorted(ALIAS_MAP.items(), key=lambda x: -len(x[0])):
        result = re.sub(rf'\b{re.escape(alias)}\b', full, result, flags=re.IGNORECASE)

    # 3. Safe USPS codes — not common English words, match any case
    for code, full in SAFE_USPS.items():
        result = re.sub(rf'\b{re.escape(code)}\b', full, result, flags=re.IGNORECASE)

    # 4. Ambiguous USPS codes — only match when ALL CAPS
    #    e.g. "IN" → Indiana, but "in" (preposition) stays unchanged
    for code, full in AMBIGUOUS_USPS.items():
        result = re.sub(rf'\b{re.escape(code)}\b', full, result)  # no IGNORECASE

    return result


def format_sql(sql: str) -> str:
    """Format SQL query to be readable with proper indentation."""
    try:
        formatted = sqlparse.format(
            sql,
            reindent=True,
            keyword_case='upper',
            identifier_case='lower',
            strip_comments=True,
            indent_width=4
        )
        return formatted.strip()
    except Exception:
        return sql.strip()


def handle_sql_request(user_input: str):
    schema = get_schema()
    normalized_input = normalize_input(user_input)

    # Show the user what was normalized for transparency
    entity_note = ""
    if normalized_input != user_input:
        entity_note = f"\nNote: The user typed '{user_input}'. This has been interpreted as: '{normalized_input}'"

    prompt = f"""
You are a SQL expert. Convert the user's question into a valid SQLite SQL query.

Database schema:
{schema}
{entity_note}

Rules:
- Use only the table and columns listed above
- Always use lowercase column names
- Return only the SQL query, nothing else
- No markdown, no explanation, no backticks
- ALWAYS include the GROUP BY column in the SELECT statement when aggregating
- Example: SELECT segment, SUM(sales) FROM superstore GROUP BY segment

LIMIT RULES — CRITICAL:
- NEVER use LIMIT under any circumstances — not for "top N", not for "bottom N", not ever
- ALWAYS use ROW_NUMBER() window function to rank and filter results
- "top N" (no grouping) → ROW_NUMBER() OVER (ORDER BY metric DESC) AS rn, then WHERE rn <= N
- "bottom N" (no grouping) → ROW_NUMBER() OVER (ORDER BY metric ASC) AS rn, then WHERE rn <= N
- "top N by [dimension]" → ROW_NUMBER() OVER (PARTITION BY dimension ORDER BY metric DESC) AS rn, then WHERE rn <= N
- If the user does NOT mention any number → return all rows, no filtering on rn
- Examples:
  - "top 10 customers by sales" →
    WITH ranked AS (
      SELECT customer_id, customer_name, segment, region, state, city,
             SUM(sales) AS total_sales,
             ROW_NUMBER() OVER (ORDER BY SUM(sales) DESC) AS rn
      FROM superstore GROUP BY customer_id, customer_name, segment, region, state, city
    )
    SELECT customer_id, customer_name, segment, region, state, city, total_sales
    FROM ranked WHERE rn <= 10
  - "top 5 customers by segment" →
    WITH ranked AS (
      SELECT customer_id, customer_name, segment, region, state, city,
             SUM(sales) AS total_sales,
             ROW_NUMBER() OVER (PARTITION BY segment ORDER BY SUM(sales) DESC) AS rn
      FROM superstore GROUP BY customer_id, customer_name, segment, region, state, city
    )
    SELECT customer_id, customer_name, segment, region, state, city, total_sales
    FROM ranked WHERE rn <= 5 ORDER BY segment, total_sales DESC

METRIC INTERPRETATION — follow these exactly:
- "total sales (SUM of sales)" → SELECT SUM(sales)
- "total profit (SUM of profit)" → SELECT SUM(profit)
- "number of orders (COUNT of rows)" → SELECT COUNT(*) as order_count
- "number of unique customers (COUNT DISTINCT customer_id)" → SELECT COUNT(DISTINCT customer_id)
- "total quantity (SUM of quantity)" → SELECT SUM(quantity)
- "average discount (AVG of discount)" → SELECT AVG(discount)
- NEVER use COUNT(*) when the user asks about a dollar amount like sales or profit

CUSTOMER QUERIES:
- When the question is about customers, ALWAYS include these columns in SELECT: customer_id, customer_name, segment, region, state, city
- Example (top 10 overall by sales — NO LIMIT, use ROW_NUMBER):
  WITH ranked AS (
    SELECT customer_id, customer_name, segment, region, state, city,
           SUM(sales) AS total_sales,
           ROW_NUMBER() OVER (ORDER BY SUM(sales) DESC) AS rn
    FROM superstore GROUP BY customer_id, customer_name, segment, region, state, city
  )
  SELECT customer_id, customer_name, segment, region, state, city, total_sales FROM ranked WHERE rn <= 10
- Never return just customer_id alone — always show the full customer profile

TOP N BY GROUP — CRITICAL RULE:
- Whenever the user says "by segment", "by category", "by region", "by state", or any grouping dimension alongside "top/bottom N", they want the top N WITHIN EACH GROUP.
- ALWAYS use ROW_NUMBER() OVER (PARTITION BY <group_col> ORDER BY <metric> DESC) for these queries.
- NEVER use LIMIT anywhere — not even at the end of a window function query.
- Example — "top 5 customers by segment by profit":
  WITH ranked AS (
    SELECT customer_id, customer_name, segment, region, state, city,
           SUM(profit) AS total_profit,
           ROW_NUMBER() OVER (PARTITION BY segment ORDER BY SUM(profit) DESC) AS rn
    FROM superstore
    GROUP BY customer_id, customer_name, segment, region, state, city
  )
  SELECT customer_id, customer_name, segment, region, state, city, total_profit
  FROM ranked WHERE rn <= 5 ORDER BY segment, total_profit DESC
- Use rn <= N where N is the number requested (default 1 if the user doesn't specify a number).

PRODUCT QUERIES:
- When the question is about products, ALWAYS include: product_name, category, sub_category alongside the metric

DATE FUNCTIONS — SQLite only, never use PostgreSQL/MySQL syntax:
- NEVER use EXTRACT(), DATE_PART(), DATE_TRUNC(), or YEAR()/MONTH() — these are NOT supported in SQLite
- ALWAYS use strftime() for all date operations:
  - Year  → strftime('%Y', order_date)
  - Month → strftime('%Y-%m', order_date)
  - Day   → strftime('%Y-%m-%d', order_date)
  - Quarter → CASE WHEN strftime('%m', order_date) IN ('01','02','03') THEN 'Q1' ... END
- Example: SELECT strftime('%Y', order_date) AS year, SUM(sales) FROM superstore GROUP BY year

DISCOUNT COLUMN — CRITICAL:
- The discount column is stored as a DECIMAL between 0.0 and 1.0, NOT as a percentage
- "20% discount" or "discount > 20%" → WHERE discount > 0.20
- "30% discount" or "discount >= 30%" → WHERE discount >= 0.30
- "less than 10% discount" → WHERE discount < 0.10
- NEVER write discount > 20 or discount > 30 — those values don't exist in the data and return 0 rows
- Always divide the user's percentage by 100 before using it in the query

RAW RECORD QUERIES — CRITICAL:
- When the user says "show me orders", "list orders", "find orders", "give me orders", "orders where", "orders with", "orders greater than", "orders above", "orders below" — they want INDIVIDUAL ROWS, not aggregates
- Use SELECT with these columns: order_id, order_date, customer_name, region, segment, category, sub_category, product_name, sales, profit, discount, quantity
- Do NOT use SUM(), AVG(), COUNT() or GROUP BY for raw listing queries
- NEVER aggregate when the user is asking to see/list/show orders — just filter and return rows
- Example: "show me orders with discount greater than 20% for central region" →
  SELECT order_id, order_date, customer_name, region, segment, category, product_name, sales, profit, discount, quantity
  FROM superstore
  WHERE discount > 0.20 AND LOWER(region) = LOWER('Central')

FILTERING:
- Always use LOWER() for string comparisons: WHERE LOWER(state) = LOWER('California')
- NEVER guess or autocomplete partial location names

User question: "{normalized_input}"
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw_sql = response.choices[0].message.content.strip()
    raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()

    # Format SQL for display
    display_sql = format_sql(raw_sql)

    df, error = run_query(raw_sql)
    if error:
        return None, display_sql, f"Query failed: {error}", None

    # Clean up column names for readability
    df = clean_column_names(df)

    chart = smart_chart(df, question=user_input)
    return df, display_sql, None, chart


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw SQL aggregate column names to human-readable labels."""
    rename_map = {}
    for col in df.columns:
        c = col.lower().strip()
        if c in ("sum(sales)", "total_sales", "totalsales"):
            rename_map[col] = "Total Sales ($)"
        elif c in ("sum(profit)", "total_profit", "totalprofit"):
            rename_map[col] = "Total Profit ($)"
        elif c in ("sum(quantity)", "total_quantity", "totalquantity", "total_units"):
            rename_map[col] = "Total Units Sold"
        elif c in ("count(*)", "order_count", "num_orders", "count(*)"):
            rename_map[col] = "Order Count"
        elif c in ("avg(discount)", "avg_discount", "avgdiscount"):
            rename_map[col] = "Avg Discount (%)"
        elif c in ("avg(sales)", "avg_sales"):
            rename_map[col] = "Avg Sales ($)"
        elif c in ("count(distinct customer_id)", "customer_count"):
            rename_map[col] = "Unique Customers"
        else:
            # General prettify: replace underscores, title case
            rename_map[col] = col.replace("_", " ").title()
    return df.rename(columns=rename_map)
