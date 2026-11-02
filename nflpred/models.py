"""Models. Everything predicts the HOME team's point margin; win probability is derived
from that margin so the spread and the win% can never disagree.

    win_prob = P(actual margin > 0) = Phi(predicted_margin / sigma)

where sigma is the standard deviation of the model's errors on its training data
(about 13 points in the NFL). The logistic regression is fitted separately only
as a cross-check that the margin model's probabilities are sensible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import ELO_HOME_ADV, ELO_POINTS_PER_POINT, TRAIN_HALF_LIFE_SEASONS, RIDGE_ALPHA
from .elo import win_prob as elo_win_prob


def season_weights(df: pd.DataFrame) -> np.ndarray:
    """Recent seasons count more: weight = 0.5 ** (seasons_ago / half_life)."""
    return (0.5 ** ((df["season"].max() - df["season"]) / TRAIN_HALF_LIFE_SEASONS)).values


class MarginModel:
    """Linear (ridge) regression for margin, Normal errors for win probability."""

    def __init__(self, features: list[str], alpha: float = RIDGE_ALPHA, kind: str = "ridge"):
        self.features, self.kind = features, kind
        if kind == "ridge":
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        else:  # gradient boosting, kept small/regularized for a small dataset
            self.model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300,
                                                       min_samples_leaf=80, l2_regularization=1.0)

    def fit(self, df: pd.DataFrame) -> "MarginModel":
        X, y, w = df[self.features].values, df["margin"].values, season_weights(df)
        if self.kind == "ridge":
            self.model.fit(X, y, ridge__sample_weight=w)
        else:
            self.model.fit(X, y, sample_weight=w)
        resid = y - self.model.predict(X)
        self.sigma = float(np.sqrt(np.average(resid ** 2, weights=w)))
        return self

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features].values)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return norm.cdf(self.predict_margin(df) / self.sigma)

    def coefficients(self) -> pd.Series | None:
        if self.kind != "ridge":
            return None
        scaler, ridge = self.model.named_steps["standardscaler"], self.model.named_steps["ridge"]
        # Points of home margin per 1 standard deviation of each feature
        return pd.Series(ridge.coef_, index=self.features)


class LogitModel:
    """Logistic regression on home win (cross-check for the margin model)."""

    def __init__(self, features: list[str], C: float = 0.1):
        self.features = features
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, df: pd.DataFrame) -> "LogitModel":
        d = df[df.home_win != 0.5]
        self.model.fit(d[self.features].values, d["home_win"].values.astype(int),
                       logisticregression__sample_weight=season_weights(d))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features].values)[:, 1]


class EloModel:
    """Pure Elo: no training. Margin = Elo difference / 25."""

    def fit(self, df):
        return self

    def _diff(self, df):
        return df["elo_diff"].values + np.where(df["home"].values > 0, ELO_HOME_ADV, 0)

    def predict_margin(self, df):
        return self._diff(df) / ELO_POINTS_PER_POINT

    def predict_proba(self, df):
        return elo_win_prob(self._diff(df))
