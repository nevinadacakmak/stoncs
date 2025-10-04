# Snowflake REST auth diagnostic (safe)
# - loads .env
# - posts to the same login-request endpoint used by the client
# - prints only status code and top-level JSON keys and any error messages (truncated)
# - does NOT print credentials or tokens

import os
from pathlib import Path
import requests
import json

# Load .env
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
        if k and k not in os.environ:
            os.environ[k] = v

acc = os.environ.get('SNOWFLAKE_ACCOUNT')
user = os.environ.get('SNOWFLAKE_USER')
pwd = os.environ.get('SNOWFLAKE_PASSWORD')

if not acc or not user or not pwd:
    print('MISSING env: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, or SNOWFLAKE_PASSWORD')
    raise SystemExit(1)

base = f"https://{acc}.snowflakecomputing.com"
url = base + "/session/v1/login-request?warehouse=&database=&schema="
body = {"data": {"LOGIN_NAME": user, "PASSWORD": pwd}}

print('POST', url)
try:
    r = requests.post(url, json=body, timeout=20)
    print('status_code:', r.status_code)
    try:
        j = r.json()
    except Exception:
        print('no-json-response, text:', (r.text or '')[:500])
        raise SystemExit(2)
    # Show top-level keys and sizes
    if isinstance(j, dict):
        print('top-level keys:', list(j.keys()))
        # If there's a data object, show keys inside it (no values)
        if 'data' in j and isinstance(j['data'], dict):
            print("data keys:", list(j['data'].keys()))
        # Show any error fields
        if 'message' in j:
            print('message:', (j['message'] or '')[:300])
        if 'errors' in j:
            print('errors:', str(j['errors'])[:300])
    else:
        print('response (non-dict):', str(j)[:500])
    if r.status_code != 200:
        print('Server returned non-200 status; check account identifier and credentials, and ensure the account supports REST login-request endpoint')
except Exception as e:
    print('REQUEST ERROR:', str(e))
    raise SystemExit(3)
