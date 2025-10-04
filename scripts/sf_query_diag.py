# Diagnose queries endpoint using the token obtained from snowflake_api_client.py
# - loads .env
# - imports the local client by path
# - calls authenticate(), checks for token presence (masked), and issues a POST to /queries/v1/query-request
# - prints status and top-level keys (no tokens printed)

import os
from pathlib import Path
import importlib.util
import requests

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

# import client by path
client_path = Path(__file__).resolve().parent.parent / 'snowflake_api_client.py'
spec = importlib.util.spec_from_file_location('snowflake_api_client', str(client_path))
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)

try:
    res = sf.authenticate()
except Exception as e:
    print('AUTH ERROR:', str(e))
    raise SystemExit(1)

# Check token presence
token = getattr(sf, '_SESSION_TOKEN', None)
if not token:
    print('No session token present after authenticate()')
else:
    print('Session token acquired (masked):', f"{len(token)}-chars")

# Build query request similar to run_query
acc = os.environ.get('SNOWFLAKE_ACCOUNT')
if not acc:
    print('Missing SNOWFLAKE_ACCOUNT')
    raise SystemExit(2)
base = f"https://{acc}.snowflakecomputing.com"
url = base + '/queries/v1/query-request'
headers = {'Authorization': f"Snowflake Token={token}", 'Content-Type': 'application/json'}
body = {'sqlText': 'SELECT current_version()'}

try:
    r = requests.post(url, headers=headers, json=body, timeout=20)
    print('queries endpoint status:', r.status_code)
    try:
        j = r.json()
        if isinstance(j, dict):
            print('top-level keys:', list(j.keys()))
            if 'message' in j:
                print('message:', (j.get('message') or '')[:300])
            if 'headers' in j:
                print('server-headers (truncated):', str(j.get('headers'))[:800])
            if 'data' in j and isinstance(j['data'], dict):
                print('response data keys:', list(j['data'].keys()))
    except Exception:
        print('non-json response (truncated):', (r.text or '')[:500])
    if r.status_code != 200:
        print('queries endpoint returned non-200; verify token format and endpoint access for your account')
except Exception as e:
    print('REQUEST ERROR:', str(e))
    raise SystemExit(3)

def try_payload(hdrs, bdy, label):
    try:
        r = requests.post(url, headers=hdrs, json=bdy, timeout=20)
        print('\n---', label, 'status:', r.status_code)
        try:
            j = r.json()
            if isinstance(j, dict):
                print('top-level keys:', list(j.keys()))
                if 'message' in j:
                    print('message:', (j.get('message') or '')[:300])
        except Exception:
            print('non-json response (truncated):', (r.text or '')[:500])
    except Exception as e:
        print(label, 'REQUEST ERROR:', str(e))


# 1) Header only
try_payload({'Authorization': f"Snowflake Token={token}", 'Content-Type': 'application/json'}, {'sqlText': 'SELECT current_version()'}, 'header-only')

# 2) Body-only with TOKEN
try_payload({'Content-Type': 'application/json'}, {'sqlText': 'SELECT current_version()', 'data': {'TOKEN': token}}, 'body-TOKEN')

# 3) Body-only with SESSION_TOKEN
try_payload({'Content-Type': 'application/json'}, {'sqlText': 'SELECT current_version()', 'data': {'SESSION_TOKEN': token}}, 'body-SESSION_TOKEN')

# 4) Header + body token
try_payload({'Authorization': f"Snowflake Token={token}", 'Content-Type': 'application/json'}, {'sqlText': 'SELECT current_version()', 'data': {'TOKEN': token}}, 'header+body-TOKEN')
