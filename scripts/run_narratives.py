"""Run the narratives pipeline (robust).

- Loads .env
- Imports local snowflake_api_client and narratives modules by path
- Tries to run narratives.detect_narratives()
- If that raises (missing ML deps), runs a safe fallback that aggregates tokens from NEWS_HEADLINES
  and writes COMPANY_TRENDS and a minimal NARRATIVES table.
"""
from pathlib import Path
import importlib.util
import os
import re
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
# Load .env
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

# Import snowflake_api_client by path and make importable
spec_sf = importlib.util.spec_from_file_location('snowflake_api_client', str(ROOT / 'snowflake_api_client.py'))
sf = importlib.util.module_from_spec(spec_sf)
spec_sf.loader.exec_module(sf)
import sys
sys.modules['snowflake_api_client'] = sf
sys.modules['stoncs.snowflake_api_client'] = sf

# Import narratives by path
spec = importlib.util.spec_from_file_location('narratives', str(ROOT / 'narratives.py'))
nar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nar)

print('Running detect_narratives (may use heavy ML libs)')
try:
    narratives_df, trend_table = nar.detect_narratives()
    print('detect_narratives completed; narratives rows:', len(narratives_df), 'trend rows:', len(trend_table))
except Exception as e:
    print('detect_narratives failed:', e)
    print('Running fallback aggregation over NEWS_HEADLINES')
    # Read headlines from Snowflake
    res = sf.run_query(f"SELECT published_date, headline, company, ticker FROM {os.environ.get('SNOWFLAKE_DATABASE')}.STONCS_NEWS.NEWS_HEADLINES")
    data = res.get('data') or {}
    rows = []
    if isinstance(data, dict) and data.get('rowset'):
        rows = data.get('rowset')
    elif isinstance(data, list):
        rows = data

    headlines = []
    # Normalize rows into dicts
    for r in rows:
        # row may be tuple/list or dict; normalize safely
        if isinstance(r, (list, tuple)):
            published_date = r[0] if len(r) > 0 else None
            headline = r[1] if len(r) > 1 else None
            company = r[2] if len(r) > 2 else None
            ticker = r[3] if len(r) > 3 else None
        elif isinstance(r, dict):
            headline = r.get('headline')
            company = r.get('company')
            ticker = r.get('ticker')
        else:
            # unknown shape
            continue
        headlines.append({'headline': (headline or ''), 'company': (company or ''), 'ticker': (ticker or '')})

    # Extract tokens: ticker and uppercase words 1-5 letters
    counter = Counter()
    cluster_map = {}
    for h in headlines:
        tokens = []
        if h['ticker']:
            tokens.append(h['ticker'])
        if h['company']:
            tokens.append(h['company'])
        tokens += re.findall(r"\b[A-Z]{1,5}\b", h['headline'])
        for t in tokens:
            counter[t] += 1

    # Build simple narratives: top N tokens as clusters
    top_tokens = [t for t,_ in counter.most_common(10)]
    narratives = []
    for i, tok in enumerate(top_tokens):
        narratives.append({'cluster_id': i, 'label': tok, 'size': counter[tok], 'top_terms': [tok]})

    # Write narratives and trends to Snowflake (truncate and insert)
    narratives_table = f"{os.environ.get('SNOWFLAKE_DATABASE')}.STONCS_NEWS.NARRATIVES"
    trends_table = f"{os.environ.get('SNOWFLAKE_DATABASE')}.STONCS_NEWS.COMPANY_TRENDS"

    # Truncate and insert narratives (store top_terms as JSON via PARSE_JSON)
    # Prefer connector executemany with parameter binding to avoid SQL expr issues
    try:
        connector = getattr(sf, '_sf_connector', None)
    except Exception:
        connector = None

    sf.run_query(f"TRUNCATE TABLE {narratives_table}")
    if narratives:
        import json
        rows_params = []
        for n in narratives:
            top_json = json.dumps(n['top_terms'])
            rows_params.append((int(n['cluster_id']), str(n['label']), int(n['size']), top_json))

        if connector is not None:
            user = os.environ.get('SNOWFLAKE_USER')
            pwd = os.environ.get('SNOWFLAKE_PASSWORD')
            account = os.environ.get('SNOWFLAKE_ACCOUNT')
            conn = connector.connect(user=user, password=pwd, account=account)
            try:
                cur = conn.cursor()
                # ensure session DB/SCHEMA
                db = os.environ.get('SNOWFLAKE_DATABASE')
                schema = os.environ.get('SNOWFLAKE_SCHEMA')
                wh = os.environ.get('SNOWFLAKE_WAREHOUSE')
                if db:
                    try:
                        cur.execute(f"USE DATABASE {db}")
                    except Exception:
                        cur.execute(f'USE DATABASE "{db}"')
                if schema:
                    try:
                        cur.execute(f"USE SCHEMA {schema}")
                    except Exception:
                        cur.execute(f'USE SCHEMA "{schema}"')
                if wh:
                    try:
                        cur.execute(f"USE WAREHOUSE {wh}")
                    except Exception:
                        cur.execute(f'USE WAREHOUSE "{wh}"')

                # Insert rows one-by-one in batches to avoid executemany PARSE_JSON issues
                insert_sql = f"INSERT INTO {narratives_table} (cluster_id, label, size, top_terms) SELECT %s, %s, %s, parse_json(%s)"
                batch = []
                for params in rows_params:
                    batch.append(params)
                    if len(batch) >= 100:
                        for p in batch:
                            cur.execute(insert_sql, p)
                        conn.commit()
                        batch = []
                if batch:
                    for p in batch:
                        cur.execute(insert_sql, p)
                    conn.commit()
                cur.close()
                conn.close()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        else:
            # Fallback to building SQL (escaped) if connector not available
            rows_sql = []
            for n in narratives:
                top_json = json.dumps(n['top_terms'])
                top_json_escaped = top_json.replace("'", "''")
                label_escaped = str(n['label']).replace("'", "''")
                rows_sql.append(f"({int(n['cluster_id'])}, '{label_escaped}', {int(n['size'])}, parse_json('{top_json_escaped}'))")
            for i in range(0, len(rows_sql), 200):
                vals = ",".join(rows_sql[i:i+200])
                sf.run_query(f"INSERT INTO {narratives_table} (cluster_id, label, size, top_terms) VALUES {vals}")

    # trends: group token counts
    sf.run_query(f"TRUNCATE TABLE {trends_table}")
    trend_rows_params = [(0, str(tok), int(cnt)) for tok, cnt in counter.items()]
    if trend_rows_params:
        if connector is not None:
            user = os.environ.get('SNOWFLAKE_USER')
            pwd = os.environ.get('SNOWFLAKE_PASSWORD')
            account = os.environ.get('SNOWFLAKE_ACCOUNT')
            conn = connector.connect(user=user, password=pwd, account=account)
            try:
                cur = conn.cursor()
                # ensure session DB/SCHEMA
                db = os.environ.get('SNOWFLAKE_DATABASE')
                schema = os.environ.get('SNOWFLAKE_SCHEMA')
                wh = os.environ.get('SNOWFLAKE_WAREHOUSE')
                if db:
                    try:
                        cur.execute(f"USE DATABASE {db}")
                    except Exception:
                        cur.execute(f'USE DATABASE "{db}"')
                if schema:
                    try:
                        cur.execute(f"USE SCHEMA {schema}")
                    except Exception:
                        cur.execute(f'USE SCHEMA "{schema}"')
                if wh:
                    try:
                        cur.execute(f"USE WAREHOUSE {wh}")
                    except Exception:
                        cur.execute(f'USE WAREHOUSE "{wh}"')

                insert_sql = f"INSERT INTO {trends_table} (cluster_id, mentioned_token, mentions) SELECT %s, %s, %s"
                batch = []
                for p in trend_rows_params:
                    batch.append(p)
                    if len(batch) >= 200:
                        for pp in batch:
                            cur.execute(insert_sql, pp)
                        conn.commit()
                        batch = []
                if batch:
                    for pp in batch:
                        cur.execute(insert_sql, pp)
                    conn.commit()
                cur.close()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        else:
            trend_rows = []
            for tok, cnt in counter.items():
                tok_esc = str(tok).replace("'", "''")
                trend_rows.append(f"(0, '{tok_esc}', {int(cnt)})")
            for i in range(0, len(trend_rows), 200):
                vals = ",".join(trend_rows[i:i+200])
                sf.run_query(f"INSERT INTO {trends_table} (cluster_id, mentioned_token, mentions) VALUES {vals}")
