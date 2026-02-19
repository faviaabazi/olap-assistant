"""
ETL: data/sales.csv  →  olap.duckdb (star schema)

Run from the project root:
    python db/etl.py
"""

import pathlib
import duckdb
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).parent.parent
CSV_PATH  = ROOT / "data" / "sales.csv"
DB_PATH   = ROOT / "olap.duckdb"
DDL_PATH  = pathlib.Path(__file__).parent / "schema.sql"

# ── Load CSV ───────────────────────────────────────────────────────────────────
print(f"Reading {CSV_PATH} …")
df = pd.read_csv(CSV_PATH, parse_dates=["order_date"])

# ── Build dimension DataFrames ─────────────────────────────────────────────────

# dim_date — one row per unique calendar date
dim_date = (
    df[["order_date", "year", "quarter", "month", "month_name"]]
    .drop_duplicates()
    .copy()
)
dim_date["date_key"] = dim_date["order_date"].dt.strftime("%Y%m%d").astype(int)
dim_date = dim_date[["date_key", "order_date", "year", "quarter", "month", "month_name"]]

# dim_geography — unique (region, country) pairs
dim_geography = (
    df[["region", "country"]]
    .drop_duplicates()
    .sort_values(["region", "country"])
    .reset_index(drop=True)
    .copy()
)
dim_geography.insert(0, "geography_key", dim_geography.index + 1)

# dim_product — unique (category, subcategory) pairs
dim_product = (
    df[["category", "subcategory"]]
    .drop_duplicates()
    .sort_values(["category", "subcategory"])
    .reset_index(drop=True)
    .copy()
)
dim_product.insert(0, "product_key", dim_product.index + 1)

# dim_customer — unique customer segments
dim_customer = (
    df[["customer_segment"]]
    .drop_duplicates()
    .sort_values("customer_segment")
    .reset_index(drop=True)
    .copy()
)
dim_customer.insert(0, "customer_key", dim_customer.index + 1)

# ── Build fact table ───────────────────────────────────────────────────────────
fact = df.copy()

# Resolve surrogate keys via merge
fact["date_key"] = fact["order_date"].dt.strftime("%Y%m%d").astype(int)

fact = fact.merge(dim_geography, on=["region", "country"])
fact = fact.merge(dim_product,   on=["category", "subcategory"])
fact = fact.merge(dim_customer,  on="customer_segment")

fact_sales = fact[[
    "order_id", "date_key", "geography_key", "product_key", "customer_key",
    "quantity", "unit_price", "revenue", "cost", "profit", "profit_margin",
]]

# ── Write to DuckDB ────────────────────────────────────────────────────────────
if DB_PATH.exists():
    DB_PATH.unlink()        # start clean on every ETL run

print(f"Creating {DB_PATH} …")
con = duckdb.connect(str(DB_PATH))

# Apply DDL — execute each statement separately (DuckDB has no executescript)
ddl = DDL_PATH.read_text()
for stmt in ddl.split(";"):
    # Strip comment lines then check whether any SQL remains
    sql_lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
    sql = "\n".join(sql_lines).strip()
    if sql:
        con.execute(sql)

# Load dimensions first (referenced by FK), then fact
loads = [
    ("dim_date",      dim_date),
    ("dim_geography", dim_geography),
    ("dim_product",   dim_product),
    ("dim_customer",  dim_customer),
    ("fact_sales",    fact_sales),
]

for table_name, frame in loads:
    con.execute(f"INSERT INTO {table_name} SELECT * FROM frame")

# ── Verify ─────────────────────────────────────────────────────────────────────
print("\nRow counts:")
for table_name, _ in loads:
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"  {table_name:<20} {count:>6,}")

con.close()
print("\nDone.")
