from __future__ import annotations

import pandas as pd

from ..base import FactorBase


class AnalystRevision(FactorBase):
    """
    Analyst EPS estimate revisions: up minus down over trailing 30 days.
    """

    def __init__(self, name: str | None = None):
        self.name = name or "analyst_revision_eps_30d"

    def compute_raw_factor(self, data_loader) -> pd.DataFrame:
        df = data_loader.load_long(dataset="fundamentals_earnings_estimates")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        up = pd.to_numeric(df["eps_estimate_revision_up_trailing_30_days"], errors="coerce")
        down = pd.to_numeric(df["eps_estimate_revision_down_trailing_30_days"], errors="coerce")
        df["revision"] = up - down
        wide = df.pivot(index="date", columns="ticker", values="revision").sort_index()
        return wide

    def post_process(self, raw_factor: pd.DataFrame) -> pd.DataFrame:
        return raw_factor.shift(1)
