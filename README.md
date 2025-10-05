# Stoncs

**Stoncs** is a narrative-aware micro-portfolio optimizer that combines **financial market data** and **news narratives** to create optimized, diversified portfolios. This repo focuses on a REST-first Snowflake integration (Snowflake REST API) and includes a compact demo pipeline so you can present results without external setup.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Data Sources](#data-sources)
5. [Installation](#installation)
6. [Usage / Demo](#usage--demo)
7. [Technical Details](#technical-details)
8. [Hackathon Notes](#hackathon-notes)
9. [Future Enhancements](#future-enhancements)
10. [License](#license)

---

## Project Overview

Stoncs merges **quantitative finance** and **natural language understanding** to deliver a **portfolio recommendation engine** that is both **data-driven** and **narrative-aware**.  
Users can input a budget and risk tolerance, and Stoncs will return:

- Optimized portfolio allocation
- Trending market narratives
- Companies currently driving market trends

All Snowflake interactions in this repo use a small reusable Snowflake REST client (`stoncs/snowflake_api_client.py`). For hackathon/demo usage the app falls back to an in-memory demo pipeline so no credentials are required.

---

## Architecture

```

User Input (budget, risk)
│
▼
Snowflake (Market Data + News Headlines)
│
├── Snowpark ML: Portfolio Clustering / Risk Analysis
│
└── Snowpark ML + NLP: Narrative Detection + Company Trend Scoring
│
▼
Streamlit Dashboard
├── Pie chart: portfolio allocation
├── Line chart: historical vs predicted returns
├── Word cloud: top narratives
└── Table/bar chart: trending companies

```

---

## Features

- **Market Narrative Detection**: KMeans clustering of news headlines.
- **Company Trend Detector**: Detects which companies are trending in narratives.
- **Portfolio Optimizer**: ML-driven risk/return optimization combined with narrative boosts.
- **Snowflake Integration**: Full data pipeline inside Snowflake; preprocessing, feature engineering, ML computations.
- **Interactive Dashboard**: Real-time portfolio recommendation + visualization using Streamlit.
- **Hackathon-ready**: Compact, fun, and visually engaging for judges.

---

## Data Sources

- **Market Data**: Yahoo Finance, Alpha Vantage, or Kaggle CSVs (OHLCV format).
- **News Headlines**: Kaggle financial news datasets or Reddit r/wallstreetbets headlines.
- **Company List**: Predefined list of tickers / company names for NER matching.

All datasets are ingested into **Snowflake tables** for preprocessing, storage, and ML.

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/stoncs.git
cd stoncs
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies (fast demo install):

```bash
pip install -r requirements_demo.txt
```

If you want the full ML experience (embeddings + clustering) install the full set:

```bash
pip install -r requirements.txt
```

4. (Optional) Set up Snowflake connection (for live demo):

Set the following environment variables if you want the app to use a live Snowflake account:

```
SNOWFLAKE_ACCOUNT=<your_account>
SNOWFLAKE_USER=<your_user>
SNOWFLAKE_PASSWORD=<your_password>
```

When these env vars are present the app will run Snowflake-backed flows using the REST client.

5. (Optional) Upload demo data and detect narratives in Snowflake (only if you set Snowflake env vars):

```bash
python run_demo.py
```

6. Start Streamlit dashboard:

```bash
streamlit run app.py
```

### Snowflake REST API (demo)

This repo includes `stoncs/snowflake_api_client.py` — a small reusable wrapper for Snowflake's REST endpoints (authenticate, run_query, upload_csv, manage_warehouse). For demonstration you can:

1. Set Snowflake env vars (see above).
2. In the app sidebar click **Run test query (Snowflake)** to run a live `SELECT current_version()` via the REST API and view the JSON response.

If you prefer curl, the legacy login-request + query-request pattern looks like this (replace placeholders):

```bash
# Authenticate (legacy login-request)
curl -X POST \
  "https://<account>.snowflakecomputing.com/session/v1/login-request?warehouse=&database=&schema=" \
  -H 'Content-Type: application/json' \
  -d '{"data": {"LOGIN_NAME": "<user>", "PASSWORD": "<password>"}}'

# Use token from response and call the query endpoint
curl -X POST \
  "https://<account>.snowflakecomputing.com/queries/v1/query-request" \
  -H "Authorization: Snowflake Token=<token>" \
  -H 'Content-Type: application/json' \
  -d '{"sqlText": "SELECT current_version()"}'
```

The Streamlit app also has an option to persist recommended allocations into Snowflake to demonstrate a live write via the REST API.

---

## Usage / Demo

1. Open the Streamlit app in browser.
2. Enter your **budget** and **risk tolerance**.
3. View:

   - **Optimized portfolio allocation** (pie chart)
   - **Historical vs predicted returns** (line chart)
   - **Top market narratives** (word cloud)
   - **Trending companies per narrative** (table/bar chart)

---

## Technical Details

### Snowflake + Snowpark

- **Data Storage**: Market + news datasets in Snowflake tables.
- **Vector Embeddings**: News headlines → embeddings → stored in Snowflake vector column.
- **Snowpark ML**:

  - KMeans for narrative clustering.
  - Regression/time series models for portfolio risk/return.

- **Company Trend Scoring**:

  - Extract companies via NER.
  - Compute trending scores per narrative.

### Portfolio Optimizer

- **Risk metrics**: volatility, average return, Sharpe ratio.
- **Risk Buckets**: low, medium, high via KMeans clustering.
- **Narrative Boost**: allocation weighted higher for trending companies.

---

## Hackathon Notes

- Solo-friendly, ML/data-heavy project for **Best Solo Hack + Best Use of Snowflake**.
- Emphasize **Snowflake + Snowpark integration** in your pitch:

  > "All preprocessing, clustering, and portfolio optimization runs **inside Snowflake**, making Stoncs scalable and data-driven."

- Fun branding: **Stoncs** → memorable + meme energy.

---

## Future Enhancements

- Integrate real-time news API for live trending narrative detection.
- Advanced portfolio optimization (Markowitz frontier, Reinforcement Learning).
- Sentiment analysis for narratives (positive/negative market impact).
- Interactive company drill-downs in the dashboard.

---

## License

MIT License. Free to use for hackathons, learning, or demo purposes.

---

## Deployment & Run (recommended)

Use the provided helper to load `.env` into your shell before starting Streamlit so the app (and any scripts) inherit Snowflake credentials:

```bash
# from the repo root
source scripts/load_env.sh
# Local dev (run from the project root)
streamlit run app.py
```

For hosting platforms that import your app as a package (which can cause "attempted relative import with no known parent package"), use the stable wrapper entrypoint we provide:

```bash
# Preferred entrypoint for hosting (Streamlit Cloud / containerized hosts)
streamlit run streamlit_app.py
```

Troubleshooting notes:

- If you see "attempted relative import with no known parent package", use `streamlit_app.py` as the entrypoint or make sure the host runs the app from the repository root (not installed as a package).
- If Snowflake queries complain "session does not have a current database", ensure `SNOWFLAKE_DATABASE` is set in your `.env` and you `source scripts/load_env.sh` before running the app.
- To silence HuggingFace tokenizer warnings add `TOKENIZERS_PARALLELISM=false` to your `.env`.
