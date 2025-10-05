"""
Generate realistic-looking synthetic news headlines for tickers present in STONCS_MARKET.MARKET_PRICES
and upload them to Snowflake into STONCS_NEWS.NEWS_HEADLINES.

This script:
- queries Snowflake for distinct tickers (falls back to 'tickers_sp500.csv' if Snowflake not available)
- generates headlines across topics, events, and sentiments over a date range
- writes a CSV and uploads it using the project's snowflake_api_client.upload_csv

Usage:
  source scripts/load_env.sh
  python3 scripts/generate_fake_news.py --days 180 --headlines-per-day 20
"""
import random
import argparse
from datetime import date, timedelta
from pathlib import Path
import csv
import sys
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    import snowflake_api_client as sfc
except Exception:
    sfc = None

def fetch_tickers_from_snowflake():
    if sfc is None:
        return []
    try:
        sfc.authenticate()
        res = sfc.run_query("SELECT DISTINCT ticker FROM STONCS_MARKET.MARKET_PRICES ORDER BY ticker")
        data = res.get('data') or {}
        rows = []
        if isinstance(data, dict) and data.get('rowset'):
            rows = [r[0] for r in data.get('rowset')]
        elif isinstance(data, list):
            rows = [r[0] for r in data]
        return [str(x) for x in rows if x]
    except Exception:
        return []

def fetch_tickers_from_csv(csv_path='tickers_sp500.csv'):
    p = Path(csv_path)
    if not p.exists():
        return []
    import pandas as pd
    df = pd.read_csv(p)
    # try common column names
    if 'symbol' in df.columns:
        return df['symbol'].astype(str).str.strip().tolist()
    if 'Symbol' in df.columns:
        return df['Symbol'].astype(str).str.strip().tolist()
    # fallback: first column
    return df.iloc[:,0].astype(str).str.strip().tolist()

def generate_headline(company, ticker, topic, sentiment):
    templates = [
        "{company} ({ticker}) {topic} {sentiment}",
        "{ticker}: {company} reports {topic}, market reacts {sentiment}",
        "{company} {topic} — analysts say {sentiment}",
        "Breaking: {company} ({ticker}) {topic}; investors {sentiment}",
        "{company} ({ticker}) {topic} — {sentiment} outlook"
    ]
    t = random.choice(templates)
    return t.format(company=company, ticker=ticker, topic=topic, sentiment=sentiment)

def pick_company_name(ticker):
    # Heuristic company name mapping: if tickers_sp500.csv includes names, prefer that
    # Otherwise use a friendly name
    csv = Path('tickers_sp500.csv')
    if csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            # try to detect name column
            if 'Name' in df.columns and 'Symbol' in df.columns:
                mapping = dict(zip(df['Symbol'].astype(str).str.strip(), df['Name'].astype(str).str.strip()))
                if ticker in mapping:
                    return mapping[ticker]
        except Exception:
            pass
    # fallback: beautify ticker
    return f"{ticker} Corp"

def build_dataset(tickers, days=90, headlines_per_day=5):
    topics = [
        'reports quarterly results', 'announces partnership', 'faces regulatory inquiry',
        'launches new product', 'issues guidance cut', 'announces layoffs', 'beats earnings estimates',
        'receives buyout interest', 'CEO steps down', 'announces share buyback', 'expands into Europe',
        'sees supply chain disruption', 'secures government contract', 'files patent application'
    ]
    sentiments = ['positive', 'neutral', 'negative']

    end = date.today()
    rows = []
    for d_offset in range(days):
        d = end - timedelta(days=d_offset)
        for _ in range(headlines_per_day):
            t = random.choice(tickers)
            company = pick_company_name(t)
            topic = random.choice(topics)
            # bias sentiment randomly
            sentiment = random.choices(['positive', 'neutral', 'negative'], weights=[0.35,0.4,0.25])[0]
            headline = generate_headline(company, t, topic, sentiment)
            rows.append({'published_date': d.isoformat(), 'headline': headline, 'company': company, 'ticker': t})
    return rows

def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['published_date','headline','company','ticker'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def upload_to_snowflake(csv_path: Path):
    if sfc is None:
        print('snowflake_api_client not available; skipping upload')
        return None
    try:
        sfc.authenticate()
        # Ensure schema/table exists
        sfc.run_query('CREATE SCHEMA IF NOT EXISTS STONCS_NEWS')
        sfc.run_query('CREATE TABLE IF NOT EXISTS STONCS_NEWS.NEWS_HEADLINES (published_date DATE, headline STRING, company STRING, ticker STRING)')
        res = sfc.upload_csv('NEWS_HEADLINES', str(csv_path), schema='STONCS_NEWS')
        return res
    except Exception as e:
        print('Upload to Snowflake failed:', e)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--headlines-per-day', type=int, default=20)
    parser.add_argument('--out', type=str, default='data/fake_news.csv')
    args = parser.parse_args()

    # Try get tickers from Snowflake; fallback to CSV
    tickers = fetch_tickers_from_snowflake()
    if not tickers:
        print('No tickers returned from Snowflake; falling back to tickers_sp500.csv')
        tickers = fetch_tickers_from_csv()
    if not tickers:
        print('No tickers available to generate news. Exiting.')
        return

    print(f'Generating synthetic headlines for {len(tickers)} tickers for {args.days} days at {args.headlines_per_day} headlines/day')
    rows = build_dataset(tickers, days=args.days, headlines_per_day=args.headlines_per_day)
    out_path = Path(args.out)
    write_csv(rows, out_path)
    print('Wrote synthetic news to', out_path)

    print('Uploading to Snowflake...')
    res = upload_to_snowflake(out_path)
    print('Upload result:', res)
    # Print sample
    print('Sample rows:')
    for r in rows[:10]:
        print(r)

if __name__ == '__main__':
    main()
