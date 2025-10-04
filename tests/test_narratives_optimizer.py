import pandas as pd
import numpy as np
from stoncs.narratives import preprocess_text
from stoncs.optimizer import compute_asset_metrics, cluster_risk_levels, combine_with_narratives


def test_preprocess_text():
    s = "Apple (AAPL) beats earnings!"
    p = preprocess_text(s)
    assert "apple" in p
    assert "aapl" in p


class FakeSession:
    def __init__(self, df):
        self._df = df

    def table(self, name):
        class T:
            def __init__(self, df):
                self._df = df

            def to_pandas(self):
                return self._df

        return T(self._df)

    def sql(self, q):
        class R:
            def collect(self):
                return []

        return R()

    def close(self):
        pass


def test_compute_metrics_and_cluster(monkeypatch):
    # synthetic market data
    rows = []
    dates = pd.date_range(end=pd.Timestamp.today(), periods=10).date
    for t in ["A", "B"]:
        price = 100.0
        for d in dates:
            price *= 1 + (0.001 if t == "A" else 0.002)
            rows.append({"date": d, "ticker": t, "close": price})

    market_df = pd.DataFrame(rows)

    # monkeypatch the create_snowpark_session to return a fake session
    # monkeypatch run_query to return metrics rows for compute_asset_metrics
    import stoncs.narratives as nar_mod

    def fake_auth():
        return {"token": "fake"}

    def fake_run_query(sql):
        # Return sample rows matching the SQL SELECT in compute_asset_metrics
        return {"data": {"rowset": [["A", "0.001", "0.01"], ["B", "0.002", "0.02"]]}}

    monkeypatch.setattr(nar_mod, "authenticate", fake_auth)
    monkeypatch.setattr(nar_mod, "run_query", fake_run_query)

    metrics = compute_asset_metrics()
    assert "ticker" in metrics.columns
    clustered = cluster_risk_levels(metrics)
    assert "risk_label" in clustered.columns


def test_combine_with_narratives_no_trends(monkeypatch):
    # Create metrics df
    metrics_df = pd.DataFrame({"ticker": ["A", "B"], "avg_return": [0.001, 0.002], "volatility": [0.01, 0.02]})

    # monkeypatch create_snowpark_session to a FakeSession with no COMPANY_TRENDS table
    import stoncs.optimizer as opt_mod

    def fake_auth2():
        return {"token": "fake"}

    def fake_run_query2(sql):
        # Return empty data to simulate missing trends
        return {"data": {"rowset": []}}

    monkeypatch.setattr(opt_mod, "authenticate", fake_auth2)
    monkeypatch.setattr(opt_mod, "run_query", fake_run_query2)

    combined = combine_with_narratives(metrics_df)
    assert "narrative_score" in combined.columns
