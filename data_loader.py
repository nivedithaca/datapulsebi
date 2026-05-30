import pandas as pd
import sqlite3
import os
import glob

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "superstore.db")

def load_csv_to_sqlite():
    # Only load if DB doesn't exist yet — skip on every rerun
    if os.path.exists(DB_PATH):
        return True

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("No CSV file found in data/ folder.")
        return False

    csv_path = csv_files[0]
    print(f"Loading: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="latin1")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS superstore")
    conn.commit()
    df.to_sql("superstore", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Loaded {len(df)} rows into SQLite database.")
    print(f"Columns: {list(df.columns)}")
    return True

def get_schema():
    if not os.path.exists(DB_PATH):
        return "Database not found. Run data_loader first."

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(superstore)")
    columns = cursor.fetchall()

    schema = "Table: superstore\nColumns:\n"
    for col in columns:
        schema += f"  - {col[1]} ({col[2]})\n"

    # Add sample values for key categorical columns so AI uses exact matches
    categorical_cols = ["region", "state", "category", "sub_category", "segment", "ship_mode"]
    schema += "\nSample values for categorical columns (use these exact values in WHERE clauses):\n"
    for col in categorical_cols:
        try:
            cursor.execute(f"SELECT DISTINCT {col} FROM superstore ORDER BY {col} LIMIT 20")
            vals = [str(row[0]) for row in cursor.fetchall()]
            schema += f"  - {col}: {', '.join(vals)}\n"
        except Exception:
            pass

    conn.close()
    return schema

def run_query(sql):
    if not os.path.exists(DB_PATH):
        return None, "Database not found."
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

if __name__ == "__main__":
    success = load_csv_to_sqlite()
    if success:
        print("\nSchema:")
        print(get_schema())
