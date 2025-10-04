"""
Lightweight Snowflake REST API client for Stoncs.

This module provides a minimal wrapper around Snowflake's REST endpoints to
authenticate and execute SQL statements without using the Snowflake Python
connector. For demo and hackathon purposes this implements username/password
login and uses the queries endpoint to run SQL. In production you'd want to
replace this with OAuth/JWT key-pair authentication and robust error handling.

NOTE: Some Snowflake REST flows (staging via presigned PUT URLs) are
non-trivial; to keep the demo portable the `upload_csv` function falls back to
batch INSERT statements when direct staging via REST is not available.
"""
import os
import requests
import json
from typing import Optional, Dict, Any

# Module-level token cache
_SESSION_TOKEN: Optional[str] = None
_BASE_URL: Optional[str] = None


def _get_base_url() -> str:
    global _BASE_URL
    if _BASE_URL:
        return _BASE_URL
    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    if not account:
        raise RuntimeError("SNOWFLAKE_ACCOUNT environment variable not set")
    # Snowflake host format
    _BASE_URL = f"https://{account}.snowflakecomputing.com"
    return _BASE_URL


def authenticate(username: Optional[str] = None, password: Optional[str] = None, warehouse: Optional[str] = None, role: Optional[str] = None, database: Optional[str] = None, schema: Optional[str] = None) -> Dict[str, Any]:
    """Authenticate with Snowflake using username/password and return session token.

    This is a simple implementation using the legacy login-request endpoint.
    For production use, prefer OAuth or key-pair JWT auth.
    """
    global _SESSION_TOKEN
    if _SESSION_TOKEN:
        return {"token": _SESSION_TOKEN}

    user = username or os.environ.get("SNOWFLAKE_USER")
    pwd = password or os.environ.get("SNOWFLAKE_PASSWORD")
    if not user or not pwd:
        raise RuntimeError("SNOWFLAKE_USER and SNOWFLAKE_PASSWORD must be set for REST auth")

    url = _get_base_url() + "/session/v1/login-request?warehouse=&database=&schema="
    # Build body following Snowflake login-request expectations (simplified)
    body = {
        "data": {
            "LOGIN_NAME": user,
            "PASSWORD": pwd,
        }
    }

    resp = requests.post(url, json=body)
    resp.raise_for_status()
    data = resp.json()
    # Extract token
    token = data.get("data", {}).get("TOKEN") or data.get("data", {}).get("sessionToken")
    if not token:
        # Try to find in response
        token = data.get("data", {}).get("masterToken")

    if not token:
        raise RuntimeError(f"Could not authenticate to Snowflake: {data}")

    _SESSION_TOKEN = token
    return {"token": _SESSION_TOKEN}


def _headers() -> Dict[str, str]:
    token = _SESSION_TOKEN or authenticate().get("token")
    return {"Authorization": f"Snowflake Token={token}", "Content-Type": "application/json"}


def run_query(sql: str, timeout: int = 120) -> Dict[str, Any]:
    """Execute a SQL statement via Snowflake REST queries endpoint and return JSON.

    This implementation posts to /queries/v1/query-request and returns the
    parsed JSON response. For long-running queries you may need to poll.
    """
    base = _get_base_url()
    url = base + "/queries/v1/query-request"
    body = {"sqlText": sql}
    headers = _headers()
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def upload_csv(table_name: str, file_path: str, schema: Optional[str] = None, batch_size: int = 500) -> Dict[str, Any]:
    """Upload a CSV file into a Snowflake table.

    For portability this function reads the CSV locally and issues batched
    INSERT INTO statements via `run_query`. If your account supports staging
    and direct REST PUTs, you can extend this function to request a
    presigned upload URL from Snowflake and then issue a `COPY INTO`.
    """
    import csv

    schema_prefix = f"{schema}." if schema else ""
    # Read CSV and build batched INSERTs
    with open(file_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        rows = []
        count = 0
        results = []
        for row in reader:
            # Escape single quotes
            escaped = [r.replace("'", "''") if isinstance(r, str) else r for r in row]
            values = [f"'{c}'" if c is not None else "NULL" for c in escaped]
            rows.append(f"({', '.join(values)})")
            if len(rows) >= batch_size:
                sql = f"INSERT INTO {schema_prefix}{table_name} ({', '.join(headers)}) VALUES " + ",".join(rows)
                results.append(run_query(sql))
                count += len(rows)
                rows = []
        if rows:
            sql = f"INSERT INTO {schema_prefix}{table_name} ({', '.join(headers)}) VALUES " + ",".join(rows)
            results.append(run_query(sql))
            count += len(rows)

    return {"rows_inserted": count, "results": results}


def manage_warehouse(action: str, warehouse_name: str) -> Dict[str, Any]:
    """Start or stop a Snowflake warehouse via SQL executed through REST.

    action: 'start' | 'stop'
    """
    action = action.lower()
    if action == "start":
        sql = f"ALTER WAREHOUSE {warehouse_name} RESUME"
    elif action == "stop":
        sql = f"ALTER WAREHOUSE {warehouse_name} SUSPEND"
    else:
        raise ValueError("action must be 'start' or 'stop'")

    return run_query(sql)
