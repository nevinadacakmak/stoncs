"""
Data ingestion utilities for Stoncs.

This module creates demo market price and news headline datasets, then uploads
them to Snowflake into schemas/tables used by later modules. For the hackathon
demo we generate small example data in-memory so judges can run without large
dependencies.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import os

from .snowflake_api_client import upload_csv, run_query, authenticate


try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from newsapi import NewsApiClient
except Exception:
    NewsApiClient = None


def fetch_market_data_yfinance(tickers: list, period: str = "1y") -> pd.DataFrame:
    """Fetch historical close prices using yfinance for tickers.

    Returns DataFrame with columns: date, ticker, close
    """
    if yf is None:
        raise RuntimeError("yfinance missing; install yfinance to fetch live market data")

    all_rows = []
    for t in tickers:
        hist = yf.Ticker(t).history(period=period)[["Close"]].reset_index()
        hist = hist.rename(columns={"Date": "date", "Close": "close"})
        hist["ticker"] = t
        hist = hist[["date", "ticker", "close"]]
        # Convert Timestamp to date
        hist["date"] = hist["date"].dt.date
        all_rows.append(hist)

    if not all_rows:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    return pd.concat(all_rows, ignore_index=True)


def fetch_news_newsapi(query: str = "finance", from_date: Optional[str] = None, to_date: Optional[str] = None, page_size: int = 100) -> pd.DataFrame:
    """Fetch news headlines using NewsAPI (requires NEWSAPI_KEY env var).

    Fallback: if NewsApiClient not installed or key missing, returns empty DataFrame.
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if NewsApiClient is None or not api_key:
        return pd.DataFrame()

    client = NewsApiClient(api_key=api_key)
    # NewsAPI free tier restricts date ranges; for the demo we'll keep it simple
    res = client.get_everything(q=query, from_param=from_date, to=to_date, language="en", page_size=page_size)
    articles = res.get("articles", [])
    rows = []
    for a in articles:
        published = a.get("publishedAt")
        if published:
            published = pd.to_datetime(published).date()
        rows.append({"published_date": published, "headline": a.get("title"), "source": a.get("source", {}).get("name")})
    return pd.DataFrame(rows)


def generate_demo_market_data(days: int = 180, tickers=None) -> pd.DataFrame:
    """Generate simple synthetic market price data for multiple tickers."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]

    end = datetime.utcnow().date()
    dates = [end - timedelta(days=i) for i in range(days)][::-1]

    rows = []
    rng = np.random.default_rng(42)
    for t in tickers:
        price = 100 + rng.normal(0, 1)
        for d in dates:
            # Simulate returns and price
            ret = rng.normal(0.0005, 0.02)
            price = max(1.0, price * (1 + ret))
            rows.append({"date": d, "ticker": t, "close": float(round(price, 2))})

    return pd.DataFrame(rows)


def generate_demo_news(headlines_per_day: int = 5, days: int = 90) -> pd.DataFrame:
    """Generate synthetic finance news headlines mentioning tickers/companies."""
    companies = [
        ("Apple", "AAPL"),
        ("Microsoft", "MSFT"),
        ("Google", "GOOG"),
        ("Tesla", "TSLA"),
        ("Amazon", "AMZN"),
    ]

    topics = [
        "earnings beat", "supply chain", "product launch", "regulatory review",
        "partnership", "layoffs", "guidance cut", "new CEO"
    ]

    end = datetime.utcnow().date()
    rows = []
    rng = np.random.default_rng(1)
    for day_offset in range(days):
        d = end - timedelta(days=day_offset)
        for _ in range(headlines_per_day):
            comp = companies[rng.integers(0, len(companies))]
            topic = topics[rng.integers(0, len(topics))]
            sentiment = rng.choice(["positive", "neutral", "negative"], p=[0.4, 0.3, 0.3])
            headline = f"{comp[0]} ({comp[1]}) {topic} {sentiment}"
            rows.append({"published_date": d, "headline": headline, "company": comp[0], "ticker": comp[1]})

    return pd.DataFrame(rows)


def upload_demo_to_snowflake(schema_market: str = "STONCS_MARKET", schema_news: str = "STONCS_NEWS", tickers: Optional[list] = None, use_live: bool = False):
    """Create schemas/tables in Snowflake and upload demo datasets.

    This function uses Snowpark session to create the target schemas and uploads
    pandas DataFrames using to_pandas() -> write_pandas (via session.write_pandas).
    """
    # Ensure REST auth
    authenticate()

    # Create schemas via run_query
    run_query(f"CREATE SCHEMA IF NOT EXISTS {schema_market}")
    run_query(f"CREATE SCHEMA IF NOT EXISTS {schema_news}")

    # Generate or fetch data
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]

    if use_live:
        try:
            market_df = fetch_market_data_yfinance(tickers)
        except Exception:
            market_df = generate_demo_market_data()

        try:
            news_df = fetch_news_newsapi(query=" OR ".join(tickers))
            if news_df.empty:
                news_df = generate_demo_news()
        except Exception:
            news_df = generate_demo_news()
    else:
        market_df = generate_demo_market_data()
        news_df = generate_demo_news()

    # Create or replace tables
    market_table = f"{schema_market}.MARKET_PRICES"
    news_table = f"{schema_news}.NEWS_HEADLINES"

    # Drop and create tables using run_query
    run_query(f"DROP TABLE IF EXISTS {market_table}")
    run_query(f"DROP TABLE IF EXISTS {news_table}")
    run_query(f"CREATE TABLE {market_table} (date DATE, ticker STRING, close FLOAT)")
    run_query(f"CREATE TABLE {news_table} (published_date DATE, headline STRING, company STRING, ticker STRING)")

    # Write CSVs locally and upload via REST upload_csv
    market_csv = "/tmp/stoncs_market.csv"
    news_csv = "/tmp/stoncs_news.csv"
    market_df.to_csv(market_csv, index=False)
    news_df.to_csv(news_csv, index=False)

    res1 = upload_csv("MARKET_PRICES", market_csv, schema=schema_market)
    print(f"Uploaded market data via REST: {res1.get('rows_inserted')}")

    res2 = upload_csv("NEWS_HEADLINES", news_csv, schema=schema_news)
    print(f"Uploaded news data via REST: {res2.get('rows_inserted')}")

    # Confirm counts
    cnt1 = run_query(f"SELECT COUNT(*) as c FROM {market_table}")
    cnt2 = run_query(f"SELECT COUNT(*) as c FROM {news_table}")
    print(f"Market rows (confirmed): {cnt1}")
    print(f"News rows (confirmed): {cnt2}")


if __name__ == "__main__":
    print("Run as script to upload demo data into Snowflake (set env vars first)")
