"""Create demo database and schemas for Stoncs and update .env.

- Creates DATABASE STONCS_DEMO if missing
- Creates SCHEMA STONCS_MARKET and STONCS_NEWS inside it
- Creates minimal tables MARKET_PRICES and NEWS_HEADLINES and NARRATIVES and COMPANY_TRENDS
- Appends SNOWFLAKE_DATABASE=STONCS_DEMO to .env if not already present

This script uses the project's snowflake_api_client (imports by path).
"""
from pathlib import Path
import importlib.util
import os

ROOT = Path(__file__).resolve().parent.parent
CLIENT_PATH = ROOT / 'snowflake_api_client.py'

# Load .env into the process so the client can pick up SNOWFLAKE_* credentials
dotenv = ROOT / '.env'
if dotenv.exists():
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip(); v = v.strip()
        if k and k not in os.environ:
            os.environ[k] = v

spec = importlib.util.spec_from_file_location('snowflake_api_client', str(CLIENT_PATH))
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)

def rq(sql):
    print('> ', sql)
    res = sf.run_query(sql)
    print('-> OK')
    return res

DB = 'STONCS_DEMO'
SCHEMA_MARKET = 'STONCS_MARKET'
SCHEMA_NEWS = 'STONCS_NEWS'

# Create database
rq(f"CREATE DATABASE IF NOT EXISTS {DB}")

# Create schemas within the database (qualified names)
rq(f"CREATE SCHEMA IF NOT EXISTS {DB}.{SCHEMA_MARKET}")
rq(f"CREATE SCHEMA IF NOT EXISTS {DB}.{SCHEMA_NEWS}")

# Create minimal tables (fully qualified)
rq(f"CREATE TABLE IF NOT EXISTS {DB}.{SCHEMA_MARKET}.MARKET_PRICES (date DATE, ticker STRING, close FLOAT)")
rq(f"CREATE TABLE IF NOT EXISTS {DB}.{SCHEMA_NEWS}.NEWS_HEADLINES (published_date DATE, headline STRING, company STRING, ticker STRING)")

# Create narratives/trends placeholders
rq(f"CREATE TABLE IF NOT EXISTS {DB}.{SCHEMA_NEWS}.NARRATIVES (cluster_id INT, label STRING, size INT, top_terms VARIANT)")
rq(f"CREATE TABLE IF NOT EXISTS {DB}.{SCHEMA_NEWS}.COMPANY_TRENDS (cluster_id INT, mentioned_token STRING, mentions INT)")

# Append to .env if missing
env_path = ROOT / '.env'
env_text = env_path.read_text() if env_path.exists() else ''
if 'SNOWFLAKE_DATABASE' not in env_text:
    with env_path.open('a') as fh:
        fh.write(f"\nSNOWFLAKE_DATABASE={DB}\n")
    print(f'Appended SNOWFLAKE_DATABASE={DB} to .env')
else:
    print('.env already contains SNOWFLAKE_DATABASE')

print('Done')
