"""
Portfolio optimizer for Stoncs.

This module reads market metrics and narrative trends from Snowflake, computes
volatility and average returns per asset, clusters assets by risk, and computes
recommended weights combining both risk and narrative trend signals.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import importlib

# Robust loader for the Snowflake REST client module. Try package import first
# (when installed as `stoncs`), then fall back to a top-level module import.
_sf = None
try:
    _sf = importlib.import_module("stoncs.snowflake_api_client")
except Exception:
    try:
        _sf = importlib.import_module("snowflake_api_client")
    except Exception:
        _sf = None

if _sf is not None:
    run_query = getattr(_sf, "run_query")
    authenticate = getattr(_sf, "authenticate")
else:
    def _missing(*args, **kwargs):
        raise ImportError("snowflake_api_client is not available. Set SNOWFLAKE env vars or ensure the package is importable.")

    run_query = _missing
    authenticate = _missing


def compute_asset_metrics(schema_market: str = "STONCS_MARKET") -> pd.DataFrame:
    """Compute returns, average returns, and volatility for each ticker using Snowflake SQL via REST.

    Returns a pandas DataFrame with columns: ticker, avg_return, volatility
    """
    authenticate()
    market_table = f"{schema_market}.MARKET_PRICES"
    sql = f"""
    WITH r AS (
      SELECT date, ticker, close,
             close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
      FROM {market_table}
    )
    SELECT ticker, AVG(ret) AS avg_return, STDDEV_SAMP(ret) AS volatility
    FROM r
    WHERE ret IS NOT NULL
    GROUP BY ticker
    """
    res = run_query(sql)
    data = res.get("data") or {}
    rows = []
    if isinstance(data, dict) and data.get("rowset"):
        rows = data.get("rowset")
    elif isinstance(data, list):
        rows = data

    if not rows:
        return pd.DataFrame(columns=["ticker", "avg_return", "volatility"])

    df = pd.DataFrame(rows, columns=["ticker", "avg_return", "volatility"])
    # Convert types
    df["avg_return"] = pd.to_numeric(df["avg_return"], errors="coerce").fillna(0.0)
    df["volatility"] = pd.to_numeric(df["volatility"], errors="coerce").fillna(0.0)
    return df


def cluster_risk_levels(metrics_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Cluster assets into risk buckets (low/medium/high) using volatility and returns."""
    # If no data, return early with default columns to avoid KMeans errors
    if metrics_df is None or len(metrics_df) == 0:
        out = pd.DataFrame(columns=list(metrics_df.columns) if metrics_df is not None else [])
        out = out.copy()
        out["risk_cluster"] = pd.Series(dtype=int)
        out["risk_label"] = pd.Series(dtype=object)
        return out

    features = metrics_df[["avg_return", "volatility"]].fillna(0).values
    n_samples = features.shape[0]
    # If there are fewer samples than clusters, reduce n_clusters to n_samples
    if n_samples < 1:
        # no samples — handled above, but keep guard
        metrics_df = metrics_df.copy()
        metrics_df["risk_cluster"] = pd.Series([None] * len(metrics_df))
        metrics_df["risk_label"] = pd.Series([None] * len(metrics_df))
        return metrics_df

    effective_clusters = min(n_clusters, n_samples)
    # If only one sample, assign it to cluster 0 directly
    if effective_clusters == 1:
        labels = [0] * n_samples
    else:
        kmeans = KMeans(n_clusters=effective_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

    metrics_df = metrics_df.copy()
    metrics_df["risk_cluster"] = labels

    # Map cluster ids to labels by average volatility (lowest -> low risk)
    cluster_vol = metrics_df.groupby("risk_cluster")["volatility"].mean().sort_values()
    labels_list = ["low", "medium", "high"][: len(cluster_vol)]
    mapping = {int(cid): label for cid, label in zip(cluster_vol.index, labels_list)}
    metrics_df["risk_label"] = metrics_df["risk_cluster"].map(mapping)
    return metrics_df


def combine_with_narratives(metrics_df: pd.DataFrame, schema_news: str = "STONCS_NEWS") -> pd.DataFrame:
    """Fetch company trends and combine to compute a narrative score per ticker.

    We normalize mentions and combine with risk label to produce a final score.
    """
    trends_table = f"{schema_news}.COMPANY_TRENDS"
    try:
        res = run_query(f"SELECT mentioned_token, mentions FROM {trends_table}")
        data = res.get("data") or {}
        rows = []
        if isinstance(data, dict) and data.get("rowset"):
            rows = data.get("rowset")
        elif isinstance(data, list):
            rows = data
        if not rows:
            metrics_df["narrative_score"] = 0.0
            return metrics_df

        trends = pd.DataFrame(rows, columns=["mentioned_token", "mentions"])
    except Exception:
        metrics_df["narrative_score"] = 0.0
        return metrics_df

    # Normalize mentions per ticker
    trends_grouped = trends.groupby("mentioned_token")["mentions"].sum().reset_index()
    trends_grouped = trends_grouped.rename(columns={"mentioned_token": "ticker", "mentions": "total_mentions"})

    merged = metrics_df.merge(trends_grouped, on="ticker", how="left")
    merged["total_mentions"] = merged["total_mentions"].fillna(0)
    # Narrative score proportional to log(1 + mentions)
    merged["narrative_score"] = np.log1p(merged["total_mentions"]) / (np.log1p(merged["total_mentions"]).max() + 1e-9)
    return merged


def recommend_portfolio(budget: float, risk_tolerance: float, metrics_with_narratives: pd.DataFrame) -> pd.DataFrame:
    """Compute recommended weights combining risk and narrative scores.

    - risk_tolerance: 0 (risk averse) to 1 (risk seeking)
    - We compute a base weight inverse to volatility, then boost by narrative_score
    - Finally normalize weights to sum to 1 and multiply by budget
    """
    df = metrics_with_narratives.copy()
    # Ensure required columns exist
    if "total_mentions" not in df.columns:
        df["total_mentions"] = 0
    if "narrative_score" not in df.columns:
        df["narrative_score"] = 0.0

    # If empty, return empty DataFrame with expected columns (no exceptions)
    expected_cols = ["ticker", "risk_label", "avg_return", "volatility", "total_mentions", "narrative_score", "combined_weight", "allocation_amount", "sharpe"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=expected_cols)
    # base weight inverse to volatility (add epsilon)
    eps = 1e-6
    df["inv_vol"] = 1.0 / (df["volatility"] + eps)
    df["base_weight"] = df["inv_vol"] / df["inv_vol"].sum()

    # Combine with narrative: give weight to narrative proportional to risk_tolerance
    df["combined_weight"] = (1 - risk_tolerance) * df["base_weight"] + risk_tolerance * (df["narrative_score"] + 1e-6)
    # If narrative_score all zeros, combined will be near base_weight -> normalize
    df["combined_weight"] = df["combined_weight"].clip(lower=0)
    df["combined_weight"] = df["combined_weight"] / df["combined_weight"].sum()

    df["allocation_amount"] = df["combined_weight"] * budget
    # Financial metric: simple Sharpe-like ratio (for display)
    eps = 1e-6
    df["sharpe"] = df["avg_return"] / (df["volatility"] + eps)
    return df[["ticker", "risk_label", "avg_return", "volatility", "total_mentions", "narrative_score", "combined_weight", "allocation_amount", "sharpe"]]


def persist_recommendations(df: pd.DataFrame, budget: float, risk_tolerance: float, schema: str = "STONCS_MARKET", table: str = "PORTFOLIO_RECOMMENDATIONS") -> Dict[str, Any]:
    """Persist recommended allocations into Snowflake via the REST client.

    Creates a simple table and does batched INSERTs. Returns a dict with rows_inserted.
    """
    authenticate()
    full_table = f"{schema}.{table}"
    # create table if not exists
    run_query(f"CREATE TABLE IF NOT EXISTS {full_table} (ts TIMESTAMP_LTZ, budget FLOAT, risk_tolerance FLOAT, ticker STRING, allocation_amount FLOAT)")

    from datetime import datetime
    ts = datetime.utcnow().isoformat()
    rows = []
    for _, r in df.iterrows():
        ticker = str(r.get("ticker")).replace("'", "''")
        alloc = float(r.get("allocation_amount") or 0.0)
        rows.append(f"('{ts}', {float(budget)}, {float(risk_tolerance)}, '{ticker}', {alloc})")

    inserted = 0
    batch = 200
    for i in range(0, len(rows), batch):
        vals = ",".join(rows[i:i+batch])
        sql = f"INSERT INTO {full_table} (ts, budget, risk_tolerance, ticker, allocation_amount) VALUES {vals}"
        run_query(sql)
        inserted += min(batch, len(rows) - i)

    return {"rows_inserted": inserted}


if __name__ == "__main__":
    print("Run optimizer functions from a notebook or the Streamlit app")
