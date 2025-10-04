"""
Streamlit dashboard for Stoncs micro-portfolio optimizer.

Domain: stoncs.local (for demo) - change when deploying.

This app fetches recommended allocations and visualizations live from Snowflake.
It uses the modules in this package to compute metrics, narratives, and recommendations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from .snowflake_api_client import run_query, authenticate
from .optimizer import cluster_risk_levels, recommend_portfolio, persist_recommendations
from .ingest import generate_demo_market_data, generate_demo_news
import re
import os

# Optional heavy ML imports — the demo pipeline will try to use them when available.
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.cluster import KMeans as SKLearnKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    SKLearnKMeans = None
    TfidfVectorizer = None


def compute_asset_metrics_local(market_df):
    # Compute per-ticker returns, mean return and volatility from a market DataFrame
    df = market_df.copy()
    df = df.sort_values(["ticker", "date"])  # ensure ordering
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    stats = df.dropna(subset=["ret"]).groupby("ticker")["ret"].agg(["mean", "std"]).reset_index()
    stats = stats.rename(columns={"mean": "avg_return", "std": "volatility"})
    # Fill missing vol/return with zeros for demo
    stats["avg_return"] = stats["avg_return"].fillna(0.0)
    stats["volatility"] = stats["volatility"].fillna(0.0)
    return stats[["ticker", "avg_return", "volatility"]]


def detect_narratives_local(n_clusters: int = 6):
    """Compact demo narrative detector that returns (narratives_df, trend_table).

    Uses synthetic headlines from `generate_demo_news()` and runs embeddings+
    KMeans+TF-IDF in-memory without persisting to Snowflake.
    """
    news_df = generate_demo_news()
    if news_df.empty:
        return pd.DataFrame(columns=["cluster_id", "label", "size", "top_terms"]), pd.DataFrame(columns=["cluster_id", "mentioned_token", "mentions"]) 

    def preprocess_text(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s()%$]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    news_df["preprocessed"] = news_df["headline"].apply(preprocess_text)

    # Prefer the full embedding+KMeans flow when available. If heavy ML
    # packages are missing, fall back to a simple heuristic that groups by
    # ticker/company mentions so the demo remains lightweight.
    use_full_ml = SentenceTransformer is not None and SKLearnKMeans is not None and TfidfVectorizer is not None

    if use_full_ml:
        # Embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(news_df["preprocessed"].tolist(), show_progress_bar=False)

        # Clustering
        kmeans = SKLearnKMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        news_df["cluster_id"] = labels

        # TF-IDF labels
        vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
        X = vectorizer.fit_transform(news_df["preprocessed"].fillna(""))
        terms = np.array(vectorizer.get_feature_names_out())
    else:
        # Lightweight fallback: cluster by ticker/company mention (deterministic)
        def simple_labeling(row):
            if pd.notna(row.get("ticker")):
                return row.get("ticker")
            if pd.notna(row.get("company")):
                return row.get("company")
            m = re.findall(r"\b[A-Z]{1,5}\b", row.get("headline", ""))
            return m[0] if m else "misc"

        news_df["cluster_label"] = news_df.apply(simple_labeling, axis=1)
        # Map labels to integers
        label_map = {v: i for i, v in enumerate(sorted(news_df["cluster_label"].unique()))}
        news_df["cluster_id"] = news_df["cluster_label"].map(label_map)
        terms = []

    narratives = []
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        size = len(idx)
        if size == 0:
            label = "(empty)"
            top_terms = []
        else:
            cluster_tfidf = np.asarray(X[idx].sum(axis=0)).ravel()
            top_idx = cluster_tfidf.argsort()[::-1][:10]
            top_terms = terms[top_idx].tolist()
            label = ", ".join(top_terms[:5])
        narratives.append({"cluster_id": int(cid), "label": label, "size": int(size), "top_terms": top_terms})

    narratives_df = pd.DataFrame(narratives)

    # Company / ticker mentions
    def extract_tokens(row):
        if pd.notna(row.get("ticker")):
            return [row.get("ticker")]
        if pd.notna(row.get("company")):
            return [row.get("company")]
        return re.findall(r"\b[A-Z]{1,5}\b", row.get("headline", ""))

    news_df["mentioned_tokens"] = news_df.apply(extract_tokens, axis=1)
    exploded = news_df.explode("mentioned_tokens")
    exploded = exploded[exploded["mentioned_tokens"].notna()]
    trend_table = exploded.groupby(["cluster_id", "mentioned_tokens"]).size().reset_index(name="mentions")
    trend_table = trend_table.rename(columns={"mentioned_tokens": "mentioned_token"})

    return narratives_df, trend_table


def load_data():
    # Decide demo vs live based on presence of Snowflake env vars
    sf_account = os.environ.get("SNOWFLAKE_ACCOUNT")
    sf_user = os.environ.get("SNOWFLAKE_USER")
    sf_pwd = os.environ.get("SNOWFLAKE_PASSWORD")

    if sf_account and sf_user and sf_pwd:
        # Live Snowflake-backed path
        authenticate()
        # compute_asset_metrics uses Snowflake; import here to avoid heavy deps if unused
        from .optimizer import compute_asset_metrics, combine_with_narratives

        metrics = compute_asset_metrics()
        metrics = cluster_risk_levels(metrics)
        metrics = combine_with_narratives(metrics)
        return metrics

    # Demo in-memory pipeline (compact, hackathon-friendly)
    market_df = generate_demo_market_data()
    metrics = compute_asset_metrics_local(market_df)
    metrics = cluster_risk_levels(metrics)

    narratives_df, trend_table = detect_narratives_local()
    # combine narrative counts into metrics
    # Use a simple aggregation: sum mentions per token -> per ticker
    if not trend_table.empty:
        trends_grouped = trend_table.groupby("mentioned_token")["mentions"].sum().reset_index()
        trends_grouped = trends_grouped.rename(columns={"mentioned_token": "ticker", "mentions": "total_mentions"})
        merged = metrics.merge(trends_grouped, on="ticker", how="left")
        merged["total_mentions"] = merged["total_mentions"].fillna(0)
        merged["narrative_score"] = np.log1p(merged["total_mentions"]) / (np.log1p(merged["total_mentions"]).max() + 1e-9)
    else:
        merged = metrics.copy()
        merged["total_mentions"] = 0
        merged["narrative_score"] = 0.0

    return merged


def plot_pie(allocation_df: pd.DataFrame):
    chart = alt.Chart(allocation_df).mark_arc().encode(
        theta=alt.Theta(field="allocation_amount", type="quantitative"),
        color=alt.Color(field="ticker", type="nominal"),
        tooltip=["ticker", "allocation_amount"]
    )
    return chart


def generate_wordcloud(narratives_df: pd.DataFrame):
    # narratives_df expected to have top_terms column as list
    all_terms = []
    for lst in narratives_df.get("top_terms", []):
        if isinstance(lst, str):
            try:
                parsed = json.loads(lst)
            except Exception:
                parsed = []
        else:
            parsed = lst or []
        all_terms.extend(parsed)

    text = " ".join(all_terms)
    wc = WordCloud(width=400, height=200, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def main():
    st.set_page_config(page_title="Stoncs — Narrative-aware Micro-Portfolio", layout="wide")
    st.title("Stoncs — Narrative-aware Micro-Portfolio Optimizer")

    st.markdown("Domain: stoncs.local (demo) — connect your Snowflake credentials via env vars before running")

    # Snowflake connection indicator & quick test
    sf_present = bool(os.environ.get("SNOWFLAKE_ACCOUNT") and os.environ.get("SNOWFLAKE_USER") and os.environ.get("SNOWFLAKE_PASSWORD"))
    if sf_present:
        st.sidebar.success("Connected: Snowflake REST API (credentials present)")
        if st.sidebar.button("Run test query (Snowflake)"):
            try:
                authenticate()
                r = run_query("SELECT current_version()")
                st.sidebar.json(r)
            except Exception as e:
                st.sidebar.error(f"Query failed: {e}")
    else:
        st.sidebar.info("Running demo pipeline (no Snowflake credentials detected)")

    budget = st.number_input("Budget ($)", min_value=100.0, value=10000.0, step=100.0)
    risk_tolerance = st.slider("Risk tolerance", 0.0, 1.0, 0.5)

    if st.button("Run Stoncs Recommendation"):
        with st.spinner("Computing recommendations…"):
            metrics = load_data()
            rec = recommend_portfolio(budget, risk_tolerance, metrics)

            # Pie chart
            st.header("Portfolio Allocation")
            st.altair_chart(plot_pie(rec), use_container_width=True)

            # Table
            st.header("Allocations")
            st.dataframe(rec)

            # Historical vs predicted returns (simple projection)
            st.header("Historical vs Predicted Returns")
            # For demo: historical average returns vs predicted (narrative-boosted) returns
            rec["predicted_return"] = rec["avg_return"] * (1 + rec["narrative_score"] * risk_tolerance)
            line_df = rec[["ticker", "avg_return", "predicted_return"]].melt(id_vars=["ticker"], var_name="series", value_name="return")
            chart = alt.Chart(line_df).mark_line(point=True).encode(x="ticker", y="return", color="series")
            st.altair_chart(chart, use_container_width=True)

            # Word cloud of narratives — load narratives from Snowflake via REST
            try:
                authenticate()
                res = run_query("SELECT cluster_id, label, top_terms FROM STONCS_NEWS.NARRATIVES")
                data = res.get("data") or {}
                rows = []
                if isinstance(data, dict) and data.get("rowset"):
                    rows = data.get("rowset")
                elif isinstance(data, list):
                    rows = data
                if rows:
                    narratives_df = pd.DataFrame(rows, columns=["cluster_id", "label", "top_terms"])
                else:
                    narratives_df = pd.DataFrame()
            except Exception:
                narratives_df = pd.DataFrame()

            if not narratives_df.empty:
                st.header("Trending Narratives")
                wc_fig = generate_wordcloud(narratives_df)
                st.pyplot(wc_fig)

            # Trending companies
            st.header("Trending Companies")
            try:
                res = run_query("SELECT mentioned_token, SUM(mentions) as mentions FROM STONCS_NEWS.COMPANY_TRENDS GROUP BY mentioned_token ORDER BY mentions DESC LIMIT 20")
                data = res.get("data") or {}
                rows = []
                if isinstance(data, dict) and data.get("rowset"):
                    rows = data.get("rowset")
                elif isinstance(data, list):
                    rows = data
                if rows:
                    trends_agg = pd.DataFrame(rows, columns=["mentioned_token", "mentions"]).set_index("mentioned_token")
                    st.bar_chart(trends_agg)
                else:
                    st.write("No trend data available — run narratives pipeline first.")
            except Exception:
                st.write("No trend data available — run narratives pipeline first.")

            # Optionally persist recommendations to Snowflake
            if sf_present:
                save = st.checkbox("Save these allocations to Snowflake")
                if save:
                    try:
                        res = persist_recommendations(rec, budget, risk_tolerance)
                        st.success(f"Saved {res.get('rows_inserted')} rows to Snowflake")
                    except Exception as e:
                        st.error(f"Failed to save recommendations: {e}")


if __name__ == "__main__":
    main()
