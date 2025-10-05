"""
Streamlit dashboard for Stoncs micro-portfolio optimizer.

Domain: stoncs.local (for demo) - change when deploying.

This app fetches recommended allocations and visualizations live from Snowflake.
It uses the modules in this package to compute metrics, narratives, and recommendations.
"""

# Ensure HuggingFace tokenizers parallelism is disabled before any forks/imports
# This prevents repeated "The current process just got forked" warnings when
# Streamlit (or the platform) forks processes.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# If a .env file exists in the project root, load it into the environment so
# `streamlit run app.py` picks up credentials set there. We only set variables
# that are not already present to avoid overwriting real environment secrets.
def _load_dotenv_if_present():
    try:
        from pathlib import Path
        root = Path(__file__).resolve().parent
        dotenv = root / '.env'
        if not dotenv.exists():
            return
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        # Silently ignore dotenv load errors to avoid crashing the app
        pass

_load_dotenv_if_present()
# Compatibility shim: some hosting environments (Streamlit Cloud, containerized runners)
# import this file as a package module (e.g. `stoncs.app`) which makes relative
# imports like `from .snowflake_api_client import ...` valid. Other environments
# run `streamlit run app.py` from the repo root, where top-level imports are
# needed. We try package-relative imports first and silently fall back.
try:
    # package-relative imports (when the app is imported as stoncs.app)
    from .snowflake_api_client import run_query as _run_query_pkg, authenticate as _auth_pkg
    from .optimizer import cluster_risk_levels as _cluster_pkg, recommend_portfolio as _recommend_pkg, persist_recommendations as _persist_pkg
    from .ingest import generate_demo_market_data as _gdm_pkg, generate_demo_news as _gdn_pkg
    # Bind into module globals so the rest of the file can reference these names
    globals().setdefault('run_query', _run_query_pkg)
    globals().setdefault('authenticate', _auth_pkg)
    globals().setdefault('cluster_risk_levels', _cluster_pkg)
    globals().setdefault('recommend_portfolio', _recommend_pkg)
    globals().setdefault('persist_recommendations', _persist_pkg)
    globals().setdefault('generate_demo_market_data', _gdm_pkg)
    globals().setdefault('generate_demo_news', _gdn_pkg)
except Exception:
    # Ignore — we'll use the dynamic loader below which handles both cases.
    pass
import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def _import_project_modules():
    """Import project modules robustly to handle package, relative, and top-level contexts.

    Returns a tuple (run_query, authenticate, cluster_risk_levels, recommend_portfolio, persist_recommendations, generate_demo_market_data, generate_demo_news)
    """
    # Try package imports first (when installed as `stoncs`), else fall back
    # to top-level imports (when running `streamlit run app.py` from the repo root).
    try:
        import stoncs.snowflake_api_client as _sf
        import stoncs.optimizer as _opt
        import stoncs.ingest as _ing
    except Exception:
        try:
            import snowflake_api_client as _sf
            import optimizer as _opt
            import ingest as _ing
        except Exception as e:
            raise ImportError("Could not import project modules (tried package and top-level): " + str(e))

    return (_sf.run_query, _sf.authenticate, _opt.cluster_risk_levels, _opt.recommend_portfolio, getattr(_opt, 'persist_recommendations', None), _ing.generate_demo_market_data, _ing.generate_demo_news)


# Load modules
run_query, authenticate, cluster_risk_levels, recommend_portfolio, persist_recommendations, generate_demo_market_data, generate_demo_news = _import_project_modules()
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
        # compute_asset_metrics uses Snowflake; import here dynamically to avoid
        # relative import issues when running `streamlit run app.py` from repo root.
        import importlib
        try:
            _opt = importlib.import_module("stoncs.optimizer")
        except Exception:
            _opt = importlib.import_module("optimizer")

        compute_asset_metrics = getattr(_opt, "compute_asset_metrics")
        combine_with_narratives = getattr(_opt, "combine_with_narratives")

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
    st.set_page_config(page_title="Stoncs: Trend-Aware Portfolio Optimizer", layout="wide")
    st.title("Stoncs: Trend-Aware Portfolio Optimizer")
    st.subheader("Turning market hype into smarter investments")

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

    # Short explanatory sections for judges
    st.markdown("### 1- Portfolio Optimizer\nThe optimizer balances historical performance and risk to create allocations. It uses volatility and average returns to form base weights, then adjusts for current market trends.")
    st.markdown("### 2- Optimizer Using Risk Factor\nUse the risk tolerance slider to adjust how much the optimizer prioritizes trend signals vs. volatility-based risk control. Lower risk tolerance favors low-volatility assets; higher values increase sensitivity to trend signals.")
    st.markdown("### 3- Optimizer Using News (Trend Signals)\nWe measure how often companies are mentioned in recent financial news to compute a trend score. Higher trend scores can nudge the optimizer to allocate more to actively-discussed companies.")

    if st.button("Run Stoncs Recommendation"):
        with st.spinner("Computing recommendations…"):
            metrics = load_data()
            rec = recommend_portfolio(budget, risk_tolerance, metrics)

            # compute predicted returns early so display_rec can include the column
            try:
                rec["predicted_return"] = rec["avg_return"] * (1 + rec.get("narrative_score", 0) * float(risk_tolerance))
            except Exception:
                # if columns are missing or computation fails, ensure column exists
                rec["predicted_return"] = 0.0

            # Cap displayed assets to at most 20 for readable charts (keeps full rec internally)
            sort_col = next((c for c in ['allocation_amount', 'combined_weight', 'weight', 'allocation'] if c in rec.columns), None)
            if sort_col is not None:
                display_rec = rec.sort_values(sort_col, ascending=False).head(20).reset_index(drop=True)
            else:
                display_rec = rec.head(20).reset_index(drop=True)

            # Ensure display_rec has the predicted_return column (some recommenders may
            # return slightly different schemas). Prefer values from full `rec` by ticker.
            if "predicted_return" not in display_rec.columns:
                try:
                    if "ticker" in rec.columns and "predicted_return" in rec.columns:
                        display_rec = display_rec.merge(rec[["ticker", "predicted_return"]], on="ticker", how="left")
                    else:
                        display_rec["predicted_return"] = 0.0
                except Exception:
                    display_rec["predicted_return"] = 0.0

            # --- Headline impacts: which headlines contributed to each ticker's narrative score
            def compute_headline_impacts(news_df, rec_df):
                """Return a DataFrame with headline-level impact estimates for tickers present.

                Heuristic used (robust, no heavy ML):
                - Identify tickers/company mentions in each headline (simple ticker/company token match)
                - For each mention, estimate impact = narrative_score[ticker] / count_headlines_mentioning_ticker
                - This gives a per-headline share of the ticker's narrative boost.
                """
                import re
                import pandas as _pd

                if news_df is None or news_df.empty or rec_df is None or rec_df.empty:
                    return _pd.DataFrame(columns=["headline", "ticker", "impact_score"])

                # prepare mapping of narrative_score by ticker
                score_map = dict(zip(rec_df["ticker"].astype(str), rec_df["narrative_score"].astype(float)))

                # normalize headlines and find mentions
                rows = []
                # simple token matcher: split on non-word and check for ticker / company exacts
                for _, r in news_df.iterrows():
                    headline = str(r.get("headline") or "")
                    ticker = (r.get("ticker") or "")
                    company = (r.get("company") or "")
                    mentions = set()
                    if ticker and str(ticker) in score_map:
                        mentions.add(str(ticker))
                    if company:
                        # match exact company token (case-insensitive) if present in score_map keys
                        ctok = str(company).strip()
                        if ctok in score_map:
                            mentions.add(ctok)
                    # fallback: extract ALL-CAPS tokens that look like tickers
                    caps = re.findall(r"\b[A-Z]{1,5}\b", headline)
                    for c in caps:
                        if c in score_map:
                            mentions.add(c)

                    for m in mentions:
                        rows.append({"headline": headline, "ticker": m, "impact_score": float(score_map.get(m, 0.0))})

                if not rows:
                    return _pd.DataFrame(columns=["headline", "ticker", "impact_score"])

                dfh = _pd.DataFrame(rows)
                # divide each ticker's score equally among headlines that mention it
                counts = dfh.groupby("ticker").size().to_dict()
                dfh["impact_share"] = dfh.apply(lambda r: r["impact_score"] / max(1, counts.get(r["ticker"], 1)), axis=1)
                # scale to a human-friendly percentage (0-100)
                dfh["impact_pct"] = dfh["impact_share"] * 100
                return dfh.sort_values("impact_pct", ascending=False)

            # Try to load headlines (demo vs Snowflake)
            news_df = None
            try:
                if bool(os.environ.get("SNOWFLAKE_ACCOUNT") and os.environ.get("SNOWFLAKE_USER") and os.environ.get("SNOWFLAKE_PASSWORD")):
                    # live: fetch from Snowflake (limited rows for speed)
                    authenticate()
                    resn = run_query(f"SELECT published_date, headline, company, ticker FROM STONCS_NEWS.NEWS_HEADLINES LIMIT 2000")
                    nd = resn.get("data") or {}
                    if isinstance(nd, dict) and nd.get("rowset"):
                        rows = nd.get("rowset")
                        news_df = pd.DataFrame(rows, columns=["published_date", "headline", "company", "ticker"] )
                    elif isinstance(nd, list):
                        news_df = pd.DataFrame(nd, columns=["published_date", "headline", "company", "ticker"] )
                else:
                    # demo: use generator
                    news_df = generate_demo_news()
            except Exception:
                news_df = generate_demo_news()

            impacts_df = compute_headline_impacts(news_df, rec)

            # Portfolio allocation pie (visualize only top tickers)
            st.header("1- Portfolio Optimizer Charts")
            st.header("Portfolio Allocation: ")
            st.altair_chart(plot_pie(display_rec).properties(title='Current allocation by asset (top 20)'), use_container_width=True)

            # Historical vs predicted returns (simple projection)
            st.header("Historical vs Predicted Returns")
            # For demo: historical average returns vs predicted (narrative-boosted) returns
            # compute predicted returns for display (already computed on full rec)
            line_df = display_rec[["ticker", "avg_return", "predicted_return"]].melt(id_vars=["ticker"], var_name="series", value_name="return")
            chart = alt.Chart(line_df).mark_line(point=True).encode(x="ticker", y="return", color="series")
            st.altair_chart(chart, use_container_width=True)

            st.header("2- Risk Factor Charts")
            # allocation sensitivity: interactive chart showing allocations across risk tolerance
            st.header("How risk tolerance changes portfolio composition")
            st.write("Move the slider to see how allocations shift between low-volatility and high-trend assets. The chart below shows asset weights across different risk-tolerance values.")
            try:
                # Build allocation series for a range of risk tolerance values
                rts = list(np.linspace(0.0, 1.0, 21))
                rows = []
                # restrict to top tickers for visual clarity (max 20)
                top_k = 20
                # compute baseline allocation to pick top tickers
                try:
                    baseline = recommend_portfolio(budget, 0.5, metrics)
                    top_tickers = list(baseline.sort_values('combined_weight', ascending=False)['ticker'].head(top_k))
                except Exception:
                    top_tickers = list(rec['ticker'].head(top_k))

                for rt in rts:
                    try:
                        rdf = recommend_portfolio(budget, float(rt), metrics)
                        # normalize and pick tickers
                        for _, rr in rdf.iterrows():
                            t = rr['ticker']
                            if t not in top_tickers:
                                continue
                            rows.append({'risk_tolerance': float(rt), 'ticker': t, 'weight': float(rr.get('combined_weight', 0.0))})
                    except Exception:
                        continue

                df_rt = pd.DataFrame(rows)
                if not df_rt.empty:
                    chart_rt = alt.Chart(df_rt).mark_area().encode(
                        x=alt.X('risk_tolerance:Q', title='Risk tolerance (0=conservative, 1=aggressive)'),
                        y=alt.Y('weight:Q', stack='normalize', title='Portfolio weight (share)'),
                        color=alt.Color('ticker:N', title='Ticker', scale=alt.Scale(scheme='category20')),
                        tooltip=['ticker', alt.Tooltip('weight:Q', format='.2%'), alt.Tooltip('risk_tolerance:Q', format='.2f')]
                    ).properties(height=300)
                    st.altair_chart(chart_rt.interactive(), use_container_width=True)
                else:
                    st.write('Not enough data to build risk-tolerance chart.')
            except Exception as e:
                st.write('Failed to build risk-tolerance visualization:', e)

            st.header("3- Optimizer Using News (Trend Signals) Charts")

            # Trending companies (mention counts)
            st.header("Trending Companies (mention counts)")
            try:
                res = run_query("SELECT mentioned_token, SUM(mentions) as mentions FROM STONCS_NEWS.COMPANY_TRENDS GROUP BY mentioned_token ORDER BY mentions DESC LIMIT 20")
                data = res.get("data") or {}
                rows = []
                if isinstance(data, dict) and data.get("rowset"):
                    rows = data.get("rowset")
                elif isinstance(data, list):
                    rows = data
                if rows:
                    trends_agg = pd.DataFrame(rows, columns=["mentioned_token", "mentions"]).rename(columns={"mentioned_token": "Ticker", "mentions": "Mention count"})
                    chart = alt.Chart(trends_agg.head(20)).mark_bar().encode(
                        x=alt.X('Mention count:Q', title='Mention count (trend intensity)'),
                        y=alt.Y('Ticker:N', sort='-x', title='Ticker'),
                        color=alt.Color('Ticker:N', legend=None, scale=alt.Scale(scheme='category20')),
                        tooltip=['Ticker', 'Mention count']
                    ).properties(height=400)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.write("No trend data available — run trends pipeline first.")
            except Exception:
                st.write("No trend data available — run trends pipeline first.")

            # Optionally persist recommendations to Snowflake
            if sf_present:
                save = st.checkbox("Save these allocations to Snowflake")
                if save:
                    try:
                        res = persist_recommendations(rec, budget, risk_tolerance)
                        st.success(f"Saved {res.get('rows_inserted')} rows to Snowflake")
                    except Exception as e:
                        st.error(f"Failed to save recommendations: {e}")

            st.header("Top headlines and trend influence")
            # Show top companies by trend mentions and sample headlines
            try:
                # fetch top companies from COMPANY_TRENDS
                try:
                    res_tr = run_query("SELECT mentioned_token, SUM(mentions) as mentions FROM STONCS_NEWS.COMPANY_TRENDS GROUP BY mentioned_token ORDER BY mentions DESC LIMIT 10")
                    data_tr = res_tr.get('data') or {}
                    rows_tr = []
                    if isinstance(data_tr, dict) and data_tr.get('rowset'):
                        rows_tr = data_tr.get('rowset')
                    elif isinstance(data_tr, list):
                        rows_tr = data_tr
                    top_tokens = [r[0] for r in rows_tr][:5]
                except Exception:
                    # fallback to compute from impacts_df
                    top_tokens = list(impacts_df['ticker'].value_counts().head(5).index) if impacts_df is not None and not impacts_df.empty else []

                headlines_samples = []
                for t in top_tokens:
                    try:
                        res_h = run_query(f"SELECT published_date, headline FROM STONCS_NEWS.NEWS_HEADLINES WHERE ticker = '{t}' ORDER BY published_date DESC LIMIT 3")
                        dh = res_h.get('data') or {}
                        rws = []
                        if isinstance(dh, dict) and dh.get('rowset'):
                            rws = dh.get('rowset')
                        elif isinstance(dh, list):
                            rws = dh
                        sample_texts = [rw[1] for rw in rws]
                    except Exception:
                        # demo fallback
                        sample_texts = list(news_df[news_df['ticker'] == t]['headline'].head(3)) if news_df is not None else []
                    headlines_samples.append({'ticker': t, 'sample_headlines': sample_texts})

                # Build trend score list
                try:
                    trends_df = pd.DataFrame(rows_tr, columns=['ticker','mentions']) if rows_tr else pd.DataFrame()
                except Exception:
                    trends_df = pd.DataFrame()

                # Display a small table with ticker, mention count, sample headlines
                display_rows = []
                for hs in headlines_samples:
                    t = hs['ticker']
                    mentions = int(trends_df[trends_df['ticker'] == t]['mentions'].iloc[0]) if not trends_df.empty and t in list(trends_df['ticker']) else 0
                    display_rows.append({'Ticker': t, 'Mention count': mentions, 'Sample headlines': ' | '.join(hs['sample_headlines'])})

                if display_rows:
                    st.table(pd.DataFrame(display_rows))
                else:
                    st.write('No trending headlines found.')
            except Exception as e:
                st.write('Failed to fetch top headlines:', e)

            # --- Event-study: estimate short-term realized return after each headline
            def compute_event_study(news_df, rec_df, windows=(1,5)):
                """Estimate headline-level returns for given windows (days).

                For each headline/ticker pair, finds the close price on headline date and
                close price after N trading days and computes return. Works with demo
                market data (generate_demo_market_data) or queries Snowflake for price series.
                Returns a DataFrame with columns: headline, ticker, date, ret_{N} for each window.
                """
                import pandas as _pd
                import numpy as _np
                from datetime import timedelta, date

                if news_df is None or news_df.empty or rec_df is None or rec_df.empty:
                    return _pd.DataFrame()

                # Decide whether to fetch market data from Snowflake or use demo generator
                market_prices = None
                try:
                    if bool(os.environ.get("SNOWFLAKE_ACCOUNT") and os.environ.get("SNOWFLAKE_USER") and os.environ.get("SNOWFLAKE_PASSWORD")):
                        # Fetch price series for relevant tickers and date ranges
                        tickers = sorted(list(set([str(x) for x in news_df.get('ticker', []) if x])))
                        if not tickers:
                            return _pd.DataFrame()
                        # limit number of tickers to avoid huge queries
                        tickers = tickers[:50]
                        dates = pd.to_datetime(news_df['published_date']).dt.date
                        min_date = dates.min()
                        max_date = dates.max() + _pd.Timedelta(days=max(windows)+5)
                        # Query Snowflake for prices
                        authenticate()
                        placeholder = ",".join([f"'{t}'" for t in tickers])
                        sql = f"SELECT date, ticker, close FROM STONCS_MARKET.MARKET_PRICES WHERE ticker IN ({placeholder}) AND date BETWEEN '{min_date}' AND '{max_date}' ORDER BY ticker, date"
                        res = run_query(sql)
                        data = res.get('data') or {}
                        rows = []
                        if isinstance(data, dict) and data.get('rowset'):
                            rows = data.get('rowset')
                        elif isinstance(data, list):
                            rows = data
                        if not rows:
                            return _pd.DataFrame()
                        market_prices = _pd.DataFrame(rows, columns=['date','ticker','close'])
                        market_prices['date'] = _pd.to_datetime(market_prices['date']).dt.date
                    else:
                        market_prices = generate_demo_market_data(days=365*3)
                        market_prices['date'] = pd.to_datetime(market_prices['date']).dt.date
                except Exception:
                    # fallback to demo data
                    market_prices = generate_demo_market_data(days=365*3)
                    market_prices['date'] = pd.to_datetime(market_prices['date']).dt.date

                # pivot per-ticker series into dict of date -> close for quick lookup
                price_map = {}
                for t, grp in market_prices.groupby('ticker'):
                    # ensure sorted by date
                    s = grp.sort_values('date')[['date','close']].set_index('date')['close']
                    price_map[str(t)] = s.to_dict()

                rows = []
                for _, r in news_df.iterrows():
                    hdate = r.get('published_date')
                    if pd.isna(hdate):
                        continue
                    # ensure date type
                    try:
                        hdate = pd.to_datetime(hdate).date()
                    except Exception:
                        continue
                    tkr = str(r.get('ticker') or '')
                    if not tkr:
                        continue
                    pm = price_map.get(tkr)
                    if not pm:
                        continue
                    price0 = pm.get(hdate)
                    # if exact date not present, try next few days
                    if price0 is None:
                        for dshift in range(1,4):
                            price0 = pm.get(hdate + timedelta(days=dshift))
                            if price0 is not None:
                                break
                    if price0 is None:
                        continue
                    entry = {'headline': r.get('headline'), 'ticker': tkr, 'date': hdate}
                    for w in windows:
                        # find price at hdate + w (allow next trading days)
                        target = None
                        for dshift in range(w, w+6):
                            target = pm.get(hdate + timedelta(days=dshift))
                            if target is not None:
                                break
                        if target is None:
                            entry[f'ret_{w}d'] = _np.nan
                        else:
                            entry[f'ret_{w}d'] = (target / price0) - 1.0
                    rows.append(entry)

                if not rows:
                    return _pd.DataFrame()
                out = _pd.DataFrame(rows)
                # add average absolute impact per headline across windows for sorting
                out['avg_abs_ret'] = out[[c for c in out.columns if c.startswith('ret_')]].abs().mean(axis=1)
                return out.sort_values('avg_abs_ret', ascending=False)

            st.header("Headline event-study (1-day & 5-day returns)")
            try:
                ev = compute_event_study(news_df, rec, windows=(1,5))
                if ev is None or ev.empty:
                    st.write("Not enough price/headline data to compute event-study.")
                else:
                    # show top 20 headlines by absolute average return
                    display_cols = ['headline','ticker','date','ret_1d','ret_5d']
                    st.dataframe(ev[display_cols].head(20).fillna('N/A'))
            except Exception as e:
                st.write("Event-study failed:", e)


if __name__ == "__main__":
    main()
