import pandas as pd
import json
import stoncs.narratives as nar_mod


class FakeConn:
    def __init__(self):
        self.written = {}


def test_detect_narratives_monkeypatch(monkeypatch):
    # Prepare a small headlines DataFrame
    df = pd.DataFrame({
        "published_date": [pd.Timestamp.today().date(), pd.Timestamp.today().date()],
        "headline": ["Apple (AAPL) earnings beat", "Tesla (TSLA) product launch"],
        "company": ["Apple", "Tesla"],
        "ticker": ["AAPL", "TSLA"],
    })

    # Prepare fake run_query to return the headlines
    def fake_auth():
        return {"token": "fake"}

    def fake_run_query(sql):
        # Return a Snowflake-like rowset for the SELECT
        return {"data": {"rowset": [
            [str(pd.Timestamp.today().date()), "Apple (AAPL) earnings beat", "Apple", "AAPL"],
            [str(pd.Timestamp.today().date()), "Tesla (TSLA) product launch", "Tesla", "TSLA"]
        ]}}

    monkeypatch.setattr(nar_mod, "authenticate", fake_auth)
    monkeypatch.setattr(nar_mod, "run_query", fake_run_query)

    narratives_df, trends = nar_mod.detect_narratives(n_clusters=2)
    assert "cluster_id" in narratives_df.columns
    assert isinstance(trends, pd.DataFrame)
