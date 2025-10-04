import os
import pandas as pd
from stoncs.ingest import generate_demo_market_data, generate_demo_news


def test_generate_demo_market_data_shape():
    df = generate_demo_market_data(days=10, tickers=["AAPL", "MSFT"])
    # Expect 10 days * 2 tickers rows
    assert df.shape[0] == 20
    assert set(df.columns) == {"date", "ticker", "close"}


def test_generate_demo_news_fields():
    df = generate_demo_news(headlines_per_day=2, days=5)
    assert not df.empty
    assert all(col in df.columns for col in ["published_date", "headline", "company", "ticker"]) 
