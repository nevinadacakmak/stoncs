# Stoncs — TL;DR

Stoncs is a compact, trend-aware portfolio optimizer that combines market data and news headline narratives to produce short, interpretable portfolio recommendations. Designed for fast demos: it runs in a demo mode (no credentials) or connects to Snowflake for a live-backed experience.

Quick start (30s):

- Run locally: `streamlit run app.py` (demo mode if you don't add Snowflake secrets)
- Use budget + risk slider to get allocations, trend visualizations, and headline event-studies
- Deploy on Streamlit Cloud and add Snowflake secrets to switch from demo → live

---

## Table of Contents

1. Quick start
2. Installation
3. Run & Demo
4. Deploy (Streamlit Cloud)
5. Snowflake & Secrets
6. Data and Ingestion
7. Technical details
8. Hackathon talking points
9. Troubleshooting
10. Future work
11. License

---

## Quick start

1. Clone and install (fast demo):

```bash
git clone https://github.com/yourusername/stoncs.git
cd stoncs
python -m venv venv && source venv/bin/activate
pip install -r requirements_demo.txt
```

````

2. Run Streamlit demo UI:

```bash
streamlit run app.py
```

3. The app runs in demo mode by default (no credentials). To enable live Snowflake-backed flows, add Snowflake credentials as described below.

---

## Installation

- For a quick demo: `pip install -r requirements_demo.txt`
- For the full ML experience (embeddings, sentence-transformers): `pip install -r requirements.txt` (note: heavy packages like `sentence-transformers`/`torch` may increase build time or fail on some cloud providers; see Deploy section.)

Recommended dev setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_demo.txt
```

---

## Run & Demo

- Start the Streamlit app:

```bash
streamlit run app.py
```

- Enter a budget and risk tolerance and click **Run Stoncs Recommendation** to compute allocations and view visualizations (allocation pie, historical vs predicted returns, word cloud, trending companies, event-study table).

---

## Deploy (Streamlit Cloud)

1. Push your repo to GitHub.
2. Create a new app on Streamlit Cloud, point it to this repo and `app.py` as the entrypoint.
3. Add Snowflake credentials in the app's Secrets UI (see next section) and restart the app.

Notes:

- If `sentence-transformers` or `torch` fail to install on Streamlit Cloud, remove them from `requirements.txt` and compute embeddings offline (locally or in CI) and persist them to Snowflake.
- The app copies `st.secrets` into `os.environ` at startup so Snowflake credentials provided via Streamlit Secrets are picked up automatically.

---

## Snowflake & Secrets

The app detects Snowflake credentials from environment variables. On Streamlit Cloud, put the following keys into the Secrets UI (do NOT commit these to your repo):

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- Optional: `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_ROLE`

Local testing (do not commit): create `.streamlit/secrets.toml`:

```toml
SNOWFLAKE_ACCOUNT = "youraccount"
SNOWFLAKE_USER = "youruser"
SNOWFLAKE_PASSWORD = "yourpassword"
SNOWFLAKE_DATABASE = "STONCS_MARKET"
SNOWFLAKE_SCHEMA = "PUBLIC"
```

The repo includes `stoncs/snowflake_api_client.py`, a compact REST-first Snowflake client that falls back to the official connector when necessary. For production consider using key-pair / OAuth auth instead of username/password.

---

## Data and Ingestion

- Market data: downloaded via `yfinance` into CSVs (scripts/ingest_sp500.py) or loaded from your own OHLCV CSVs.
- News headlines: synthetic generators are provided (`scripts/generate_fake_news.py`) for demo mode; you can plug in a real news API and upload into `STONCS_NEWS.NEWS_HEADLINES`.
- Upload helpers: `snowflake_api_client.upload_csv()` supports connector-based executemany or a REST/batched-INSERT fallback for portability.

---

## Technical details

- NLP pipeline: optional `sentence-transformers` embeddings + KMeans clustering + TF-IDF label extraction (fallback deterministic heuristics used if heavy ML packages are unavailable).
- Narrative persistence: the pipeline writes simplified cluster labels and company trend counts to Snowflake; embeddings can be persisted but may require staging/COPY for robust VARIANT handling.
- Optimizer: combines historical mean/volatility metrics with narrative boosts to compute combined weights. The Streamlit UI limits visuals to the top-20 tickers for readability.

---

## Troubleshooting

- If the app starts in demo mode on Streamlit Cloud, ensure you added SNOWFLAKE\_\* keys in the app Secrets and restarted the app.
- If `sentence-transformers` fails to install on the host, remove it from `requirements.txt` and precompute embeddings elsewhere.
- If Snowflake returns "no current database", set `SNOWFLAKE_DATABASE` in Secrets or `.env` and restart.

---

## Future work

- Real-time news ingestion and streaming updates.
- Persist embeddings robustly using staged COPY or Snowpark to store VARIANT/VECTOR columns.
- Improve allocation UI: per-sector grouping, max-N asset constraint, and backtesting charts.

---

## License

MIT License — feel free to use and remix for hackathons and demos.

```

```
````
