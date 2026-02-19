import pandas as pd
import numpy as np
from datetime import date

SEED = 42
N = 10_000

rng = np.random.default_rng(SEED)

# ── Geography ──────────────────────────────────────────────────────────────────
REGION_COUNTRIES = {
    "North America": ["United States", "Canada", "Mexico"],
    "Europe":        ["Germany", "United Kingdom", "France", "Italy", "Spain", "Netherlands"],
    "Asia Pacific":  ["China", "Japan", "Australia", "India", "South Korea", "Singapore"],
    "Latin America": ["Brazil", "Argentina", "Colombia", "Chile", "Peru"],
}

REGIONS = list(REGION_COUNTRIES.keys())
REGION_WEIGHTS = [0.35, 0.30, 0.25, 0.10]   # North America heaviest

# ── Product taxonomy ───────────────────────────────────────────────────────────
# Each entry: (subcategory, price_min, price_max)
CATEGORY_SUBS = {
    "Electronics": [
        ("Smartphones",   299, 1199),
        ("Laptops",       499, 2499),
        ("Accessories",    15,  199),
    ],
    "Furniture": [
        ("Chairs",         89,  799),
        ("Desks",         149, 1299),
        ("Storage",        49,  499),
    ],
    "Office Supplies": [
        ("Paper & Pens",    5,   49),
        ("Binders",         8,   79),
        ("Technology",     25,  299),
    ],
    "Clothing": [
        ("Tops",           20,  149),
        ("Bottoms",        25,  179),
        ("Outerwear",      59,  399),
    ],
}

CATEGORIES   = list(CATEGORY_SUBS.keys())
CAT_WEIGHTS  = [0.30, 0.20, 0.30, 0.20]

SEGMENTS        = ["Consumer", "Corporate", "Home Office"]
SEGMENT_WEIGHTS = [0.50, 0.35, 0.15]

# ── Date range ─────────────────────────────────────────────────────────────────
start = date(2022, 1, 1)
end   = date(2024, 12, 31)
total_days = (end - start).days + 1

day_offsets  = rng.integers(0, total_days, size=N)
order_dates  = pd.to_datetime(start) + pd.to_timedelta(day_offsets, unit="D")

# ── Sample regions & countries ─────────────────────────────────────────────────
regions = rng.choice(REGIONS, size=N, p=REGION_WEIGHTS)
countries = np.array([
    rng.choice(REGION_COUNTRIES[r]) for r in regions
])

# ── Sample categories & subcategories ──────────────────────────────────────────
categories = rng.choice(CATEGORIES, size=N, p=CAT_WEIGHTS)

subcategories = np.empty(N, dtype=object)
unit_prices   = np.empty(N, dtype=float)

for cat in CATEGORIES:
    cat_mask = categories == cat
    subs     = CATEGORY_SUBS[cat]
    # assign a random subcategory index to every row in this category
    chosen = np.zeros(N, dtype=int)
    chosen[cat_mask] = rng.integers(0, len(subs), size=cat_mask.sum())
    for i, (sub_name, lo, hi) in enumerate(subs):
        sub_mask = cat_mask & (chosen == i)
        subcategories[sub_mask] = sub_name
        unit_prices[sub_mask]   = rng.uniform(lo, hi, size=sub_mask.sum()).round(2)

# ── Other columns ──────────────────────────────────────────────────────────────
segments  = rng.choice(SEGMENTS, size=N, p=SEGMENT_WEIGHTS)
quantities = rng.integers(1, 11, size=N)          # 1–10 inclusive

revenue = (quantities * unit_prices).round(2)

# Cost = 70–85 % of revenue (randomised per row)
cost_pct = rng.uniform(0.70, 0.85, size=N)
cost     = (revenue * cost_pct).round(2)
profit   = (revenue - cost).round(2)

# Profit margin as percentage, 4 dp
profit_margin = ((profit / revenue) * 100).round(4)

# ── Build DataFrame ────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "order_id":        [f"ORD-{i+1:06d}" for i in range(N)],
    "order_date":      order_dates.strftime("%Y-%m-%d"),
    "year":            order_dates.year,
    "quarter":         order_dates.quarter,
    "month":           order_dates.month,
    "month_name":      order_dates.strftime("%B"),
    "region":          regions,
    "country":         countries,
    "category":        categories,
    "subcategory":     subcategories,
    "customer_segment": segments,
    "quantity":        quantities,
    "unit_price":      unit_prices,
    "revenue":         revenue,
    "cost":            cost,
    "profit":          profit,
    "profit_margin":   profit_margin,
})

# Sort by date for a natural ordering
df = df.sort_values("order_date").reset_index(drop=True)
df["order_id"] = [f"ORD-{i+1:06d}" for i in range(N)]   # re-sequence after sort

out_path = "data/sales.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df):,} rows to {out_path}")
print(df.dtypes)
print(df.head())
