# Safe Snowflake connection test for Stoncs
# - loads .env (no printing of secrets)
# - calls authenticate() and runs a lightweight query
# - prints only a concise success/failure message (no secret leakage)

import os
from pathlib import Path

# Load .env file if present
env_path = Path(__file__).resolve().parent.parent / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        v = v.strip()
        # Only set if not already present in environment
        if k and k not in os.environ:
            os.environ[k] = v

# Now run the client
# Import the local snowflake_api_client.py by path to avoid sys.path issues
import importlib.util
client_path = Path(__file__).resolve().parent.parent / 'snowflake_api_client.py'
if not client_path.exists():
    print(f"ERROR: snowflake_api_client.py not found at {client_path}")
    raise SystemExit(1)

spec = importlib.util.spec_from_file_location("snowflake_api_client", str(client_path))
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)

try:
    sf.authenticate()
except Exception as e:
    print("ERROR: authentication failed:", str(e))
    raise SystemExit(2)

try:
    res = sf.run_query("SELECT current_version()")
    # Extract a small safe summary
    data = res.get('data') if isinstance(res, dict) else None
    if isinstance(data, dict) and data.get('rowset'):
        rv = data.get('rowset')[0][0] if data.get('rowset') and len(data.get('rowset')[0])>0 else None
        print(f"OK: Snowflake responded, current_version={rv}")
    elif isinstance(res, list):
        print(f"OK: Snowflake query returned list of {len(res)} rows")
    else:
        print("OK: Snowflake query executed (response shape unexpected, but no error)")
except Exception as e:
    print("ERROR: run_query failed:", str(e))
    raise SystemExit(3)
