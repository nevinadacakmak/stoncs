"""
Deprecated Snowpark connection helper.

This project has moved to the Snowflake REST API for all interactions. The
old Snowpark session helper remains for reference but should not be used by
other modules. Use `stoncs.snowflake_api_client` instead.
"""
import warnings


def create_snowpark_session():
    raise RuntimeError("create_snowpark_session is deprecated. Use stoncs.snowflake_api_client instead.")
