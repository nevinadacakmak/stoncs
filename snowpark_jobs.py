"""
Example Snowpark job that computes per-ticker stats inside Snowflake using Snowpark.

This file contains a simple function `compute_ticker_stats()` that connects to Snowflake via
Snowpark, computes average daily returns and volatility per ticker from `MARKET_PRICES`,
and writes results to `TICKER_STATS` table in the `STONCS_MARKET` schema.

Usage (from local machine with Snowpark available):
    source scripts/load_env.sh
    python3 snowpark_jobs.py

Requirements:
    pip install snowflake-snowpark-python
    and the Snowflake account permissions to create tables and run Snowpark.
"""
import os

try:
    from snowflake.snowpark import Session
    from snowflake.snowpark.functions import col
except Exception:
    Session = None


def create_session_from_env():
    if Session is None:
        raise RuntimeError('snowflake-snowpark-python is not installed')
    params = {
        'account': os.environ.get('SNOWFLAKE_ACCOUNT'),
        'user': os.environ.get('SNOWFLAKE_USER'),
        'password': os.environ.get('SNOWFLAKE_PASSWORD'),
        'database': os.environ.get('SNOWFLAKE_DATABASE'),
        'schema': 'STONCS_MARKET',
        'warehouse': os.environ.get('SNOWFLAKE_WAREHOUSE')
    }
    missing = [k for k,v in params.items() if not v]
    if missing:
        raise RuntimeError('Missing Snowflake connection params: ' + ','.join(missing))
    return Session.builder.configs(params).create()


def compute_ticker_stats():
    sess = create_session_from_env()
    df = sess.table('MARKET_PRICES')
    # compute simple daily return and aggregate
    df2 = df.with_column('prev_close', col('close').lag(1).over(partition_by=col('ticker'), order_by=col('date')))
    df2 = df2.with_column('ret', (col('close') - col('prev_close')) / col('prev_close'))
    stats = df2.dropna(subset=['ret']).group_by('ticker').agg(
        df2['ret'].avg().as_('avg_return'),
        df2['ret'].stddev_pop().as_('volatility')
    )
    # write results to table
    stats.write.save_as_table('TICKER_STATS', mode='overwrite')
    print('Wrote TICKER_STATS')


if __name__ == '__main__':
    compute_ticker_stats()
