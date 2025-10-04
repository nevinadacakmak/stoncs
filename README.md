# Stoncs

**Stoncs** is a narrative-aware micro-portfolio optimizer that combines **financial market data** and **news narratives** to create optimized, diversified portfolios. Powered by **Snowflake + Snowpark + ML**, Stoncs detects trending market stories and dynamically adjusts portfolio recommendations based on company trends.

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

All computations leverage **Snowflake’s data warehousing capabilities** and **Snowpark’s in-database ML** to ensure scalability and live querying.

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

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up Snowflake connection (update `config.py`):

```python
account = "<your_account>"
user = "<your_user>"
password = "<your_password>"
warehouse = "<your_warehouse>"
database = "STONCS"
schema = "MARKET_DATA"
```

5. Run Snowflake ingestion + ML scripts sequentially:

```bash
python snowflake_ingest.py
python narrative_trend_detector.py
python portfolio_optimizer.py
```

6. Start Streamlit dashboard:

```bash
streamlit run app.py
```

### Snowflake REST API demo

You can call Snowflake's REST endpoints directly. Below is a minimal demo `curl`
sequence showing how to authenticate (legacy login-request) and run a simple
query via the REST `queries/v1/query-request` endpoint. Replace placeholders
with your account + credentials. This repository's `stoncs/snowflake_api_client.py`
wraps the same calls in Python.

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

This demo uses the same REST endpoints that `stoncs/snowflake_api_client.py` calls.

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
