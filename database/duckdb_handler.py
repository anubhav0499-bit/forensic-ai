"""
DuckDB Handler - Analytics engine for financial trend analysis
DuckDB excels at column-oriented aggregations across multi-year data
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import duckdb
import pandas as pd
from loguru import logger

from config import DB_CONFIG


class DuckDBHandler:
    """DuckDB-powered analytics over financial time series."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_CONFIG.duckdb_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with duckdb.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_trends (
                    company_name VARCHAR,
                    ticker VARCHAR,
                    fiscal_year VARCHAR,
                    metric VARCHAR,
                    value DOUBLE,
                    unit VARCHAR,
                    yoy_change DOUBLE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fraud_signals (
                    company_name VARCHAR,
                    ticker VARCHAR,
                    fiscal_year VARCHAR,
                    signal_type VARCHAR,
                    signal_name VARCHAR,
                    signal_value DOUBLE,
                    threshold DOUBLE,
                    triggered BOOLEAN,
                    severity VARCHAR
                )
            """)
        logger.info(f"DuckDB initialized: {self.db_path}")

    def load_financial_dataframe(self, company_name: str, data: list[dict]) -> None:
        """Load a list of annual financial records into DuckDB."""
        if not data:
            return
        df = pd.DataFrame(data)
        df["company_name"] = company_name
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM financial_trends WHERE company_name = ?",
                [company_name],
            )
            # Melt to long format
            id_cols = [c for c in ["company_name", "fiscal_year"] if c in df.columns]
            value_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
            melted = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric", value_name="value")
            melted["unit"] = "absolute"
            melted["yoy_change"] = None
            conn.execute("INSERT INTO financial_trends SELECT * FROM melted")

    def get_trend_analysis(self, company_name: str, metrics: list[str]) -> pd.DataFrame:
        """Get multi-year trend for specified metrics."""
        placeholders = ", ".join(["?" for _ in metrics])
        with duckdb.connect(self.db_path) as conn:
            df = conn.execute(
                f"""SELECT fiscal_year, metric, value
                    FROM financial_trends
                    WHERE company_name = ? AND metric IN ({placeholders})
                    ORDER BY fiscal_year ASC""",
                [company_name] + metrics,
            ).df()
        return df.pivot(index="fiscal_year", columns="metric", values="value")

    def calculate_cagr(self, company_name: str, metric: str) -> Optional[float]:
        """Calculate CAGR for a metric over available history."""
        with duckdb.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT fiscal_year, value FROM financial_trends
                   WHERE company_name=? AND metric=? AND value IS NOT NULL
                   ORDER BY fiscal_year""",
                [company_name, metric],
            ).fetchall()
        if len(rows) < 2:
            return None
        start_val = rows[0][1]
        end_val = rows[-1][1]
        n_years = len(rows) - 1
        if start_val <= 0 or n_years == 0:
            return None
        return ((end_val / start_val) ** (1 / n_years) - 1) * 100

    def detect_anomalies(self, company_name: str, metric: str, std_threshold: float = 2.0) -> pd.DataFrame:
        """Flag data points that deviate significantly from trend."""
        with duckdb.connect(self.db_path) as conn:
            df = conn.execute(
                """SELECT fiscal_year, value,
                   AVG(value) OVER () as mean_val,
                   STDDEV(value) OVER () as std_val
                   FROM financial_trends
                   WHERE company_name=? AND metric=? AND value IS NOT NULL""",
                [company_name, metric],
            ).df()
        if df.empty:
            return df
        df["z_score"] = (df["value"] - df["mean_val"]) / df["std_val"].clip(lower=0.001)
        df["anomaly"] = df["z_score"].abs() > std_threshold
        return df

    def peer_benchmarking(self, company_names: list[str], metric: str, fiscal_year: str) -> pd.DataFrame:
        """Compare a metric across companies for a given year."""
        placeholders = ", ".join(["?" for _ in company_names])
        with duckdb.connect(self.db_path) as conn:
            df = conn.execute(
                f"""SELECT company_name, ticker, value
                    FROM financial_trends
                    WHERE company_name IN ({placeholders}) AND metric=? AND fiscal_year=?
                    ORDER BY value DESC""",
                company_names + [metric, fiscal_year],
            ).df()
        return df

    def query(self, sql: str, params: list = None) -> pd.DataFrame:
        with duckdb.connect(self.db_path) as conn:
            return conn.execute(sql, params or []).df()
