"""
Upload CSV files produced by `ingest_sp500.py` into Snowflake.

This script supports two modes:
- write_pandas (connector) — convenient for medium datasets
- PUT + COPY INTO (recommended for large/bulk loads) — requires SnowSQL or connector support

Usage:
    source scripts/load_env.sh
    python3 scripts/upload_market_to_snowflake.py --dir data/market_csvs --mode write_pandas

Requirements:
    pip install snowflake-connector-python pandas
"""
from pathlib import Path
import os
import pandas as pd

try:
    import snowflake.connector as sfconn
    from snowflake.connector.pandas_tools import write_pandas
except Exception:
    sfconn = None
    write_pandas = None


def upload_with_write_pandas(conn, csv_path: Path, table='MARKET_PRICES', schema='STONCS_MARKET'):
    df = pd.read_csv(csv_path)
    # ensure column names lowercase
    df.columns = [c.lower() for c in df.columns]
    success, nchunks, nrows, _ = write_pandas(conn, df, table_name=table, schema=schema)
    print(f'Wrote {nrows} rows from {csv_path} to {schema}.{table} (success={success})')


def upload_all(dir_path: Path, mode='write_pandas'):
    if mode == 'write_pandas' and write_pandas is None:
        raise RuntimeError('write_pandas not available; install snowflake-connector-python')
    files = sorted([p for p in dir_path.glob('*.csv')])
    if not files:
        print('No CSVs found in', dir_path)
        return
    user = os.environ.get('SNOWFLAKE_USER')
    pwd = os.environ.get('SNOWFLAKE_PASSWORD')
    account = os.environ.get('SNOWFLAKE_ACCOUNT')
    db = os.environ.get('SNOWFLAKE_DATABASE')
    wh = os.environ.get('SNOWFLAKE_WAREHOUSE')
    if not (user and pwd and account and db):
        raise RuntimeError('SNOWFLAKE_USER/SNOWFLAKE_PASSWORD/SNOWFLAKE_ACCOUNT/SNOWFLAKE_DATABASE must be set in env')

    conn = sfconn.connect(user=user, password=pwd, account=account, warehouse=wh, database=db)
    try:
        for f in files:
            if mode == 'write_pandas':
                upload_with_write_pandas(conn, f)
            else:
                # TODO: implement PUT + COPY approach (requires SnowSQL or connector PUT helper)
                print('Mode', mode, 'not implemented; try write_pandas for now')
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/market_csvs')
    parser.add_argument('--mode', type=str, default='write_pandas')
    args = parser.parse_args()
    upload_all(Path(args.dir), mode=args.mode)


if __name__ == '__main__':
    main()
