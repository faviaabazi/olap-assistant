-- ─────────────────────────────────────────────────────────────────────────────
-- Star schema for retail sales OLAP database (DuckDB)
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Dimension tables ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_date (
    date_key    INTEGER     PRIMARY KEY,   -- surrogate: YYYYMMDD integer
    order_date  DATE        NOT NULL,
    year        SMALLINT    NOT NULL,
    quarter     TINYINT     NOT NULL,
    month       TINYINT     NOT NULL,
    month_name  VARCHAR(9)  NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_geography (
    geography_key   INTEGER     PRIMARY KEY,
    region          VARCHAR(32) NOT NULL,
    country         VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_key     INTEGER     PRIMARY KEY,
    category        VARCHAR(32) NOT NULL,
    subcategory     VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_customer (
    customer_key        INTEGER     PRIMARY KEY,
    customer_segment    VARCHAR(16) NOT NULL
);

-- ── Fact table ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_sales (
    order_id        VARCHAR(12)     NOT NULL,
    date_key        INTEGER         NOT NULL REFERENCES dim_date(date_key),
    geography_key   INTEGER         NOT NULL REFERENCES dim_geography(geography_key),
    product_key     INTEGER         NOT NULL REFERENCES dim_product(product_key),
    customer_key    INTEGER         NOT NULL REFERENCES dim_customer(customer_key),
    quantity        TINYINT         NOT NULL,
    unit_price      DECIMAL(10, 2)  NOT NULL,
    revenue         DECIMAL(12, 2)  NOT NULL,
    cost            DECIMAL(12, 2)  NOT NULL,
    profit          DECIMAL(12, 2)  NOT NULL,
    profit_margin   DECIMAL(7, 4)   NOT NULL,
    PRIMARY KEY (order_id)
);
