"""
Download OHLCV for a large ticker list (e.g., S&P500) using yfinance, chunk it,
and write CSVs ready for upload to Snowflake.

Usage:
    python3 scripts/ingest_sp500.py --tickers tickers_sp500.csv --start 2020-01-01 --end 2025-10-01

Notes:
- Install dependencies: `pip install yfinance pandas` (or use requirements files)
- The script writes CSVs to `data/market_csvs/` by default.
- Tune `chunk_size` to avoid rate limits.
"""
import argparse
from pathlib import Path
import pandas as pd
import time

try:
    import yfinance as yf
except Exception:
    yf = None


def download_tickers(tickers, start, end, out_dir: Path, chunk_size=50, threads=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(tickers)
    for i in range(0, total, chunk_size):
        group = tickers[i:i+chunk_size]
        print(f"Downloading tickers {i}..{min(i+chunk_size, total)-1} ({len(group)})")
        if yf is None:
            raise RuntimeError("yfinance not installed. Install with `pip install yfinance`")
        df = yf.download(group, start=start, end=end, group_by='ticker', threads=threads, progress=False)
        rows = []
        for t in group:
            try:
                sub = df[t].dropna()
            except Exception:
                # yfinance returns different shapes depending on success/failure
                continue
            if sub.empty:
                continue
            sub = sub.reset_index()
            sub['ticker'] = t
            out = sub[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
                'Date': 'date', 'Open':'open', 'High':'high','Low':'low','Close':'close','Volume':'volume'
            })
            rows.append(out)
        if not rows:
            print('No data for this batch, skipping')
            time.sleep(1)
            continue
        combined = pd.concat(rows, ignore_index=True)
        fname = out_dir / f"prices_{i}_{min(i+chunk_size-1, total-1)}.csv"
        combined.to_csv(fname, index=False)
        print('Wrote', fname)
        # polite backoff
        time.sleep(1)


def load_tickers_from_csv(path: Path):
    df = pd.read_csv(path)
    if 'symbol' in df.columns:
        return [str(x).strip() for x in df['symbol'].tolist() if pd.notna(x)]
    # assume single-column CSV of tickers
    return [str(x).strip() for x in df.iloc[:,0].tolist() if pd.notna(x)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', type=str, default='tickers_sp500.csv', help='CSV file with tickers (column symbol or single column)')
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default='2025-10-01')
    parser.add_argument('--out', type=str, default='data/market_csvs')
    parser.add_argument('--chunk', type=int, default=50)
    args = parser.parse_args()

    tickers_path = Path(args.tickers)
    if not tickers_path.exists():
        print('Tickers CSV not found:', tickers_path)
        return
    tickers = load_tickers_from_csv(tickers_path)
    print('Loaded', len(tickers), 'tickers')
    download_tickers(tickers, args.start, args.end, Path(args.out), chunk_size=args.chunk)


if __name__ == '__main__':
    main()
