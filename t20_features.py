
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SNAKE = {"Overs Played": "overs_played", "Wickets Lost": "wickets_lost", "Run Rate": "run_rate",
         "Home/Away": "home_away", "Opponent Strength": "opponent_strength",
         "Pitch Condition": "pitch_condition", "Weather": "weather"}
BASE_NUM = ["overs_played", "wickets_lost", "run_rate", "opponent_strength"]
BASE_CAT = ["home_away", "pitch_condition", "weather"]
BASIC_NUM = ["current_est_runs", "wickets_remaining", "overs_remaining", "wicket_pressure", "remaining_innings_pct"]
BASIC_CAT = ["scoring_phase", "run_rate_category"]
FULL_NUM = ["wickets_x_overs", "rr_x_wickets_remaining", "overs_remaining_x_rr"]
FEATURE_SETS = ("raw", "basic", "full")


def feature_columns(feature_set):
    num, cat = list(BASE_NUM), list(BASE_CAT)
    if feature_set in ("basic", "full"):
        num += BASIC_NUM; cat += BASIC_CAT
    if feature_set == "full":
        num += FULL_NUM
    return num, cat


class CricketFeatureEngineer(BaseEstimator, TransformerMixin):
    # Turns the 7 raw input columns into the model's feature table. Stateless, so it cannot leak.
    def __init__(self, feature_set="full"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CricketFeatureEngineer expects a pandas DataFrame with the raw column names.")
        missing = [c for c in SNAKE if c not in X.columns]
        if missing:
            raise ValueError("Missing input columns: %s" % missing)
        d = X[list(SNAKE)].rename(columns=SNAKE).copy()
        for c in BASE_NUM:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
        if self.feature_set in ("basic", "full"):
            o, w, rr = d["overs_played"], d["wickets_lost"], d["run_rate"]
            d["current_est_runs"] = o * rr
            d["wickets_remaining"] = 10 - w
            d["overs_remaining"] = 20 - o
            d["wicket_pressure"] = w / o.clip(lower=1)
            d["remaining_innings_pct"] = (20 - o) / 20 * 100
            phase = pd.Series(np.select([o <= 6, o <= 15], ["powerplay", "middle"], default="death"), index=d.index, dtype=object)
            phase[o.isna()] = np.nan
            d["scoring_phase"] = phase
            d["run_rate_category"] = pd.cut(rr, [-np.inf, 6, 9, 12, np.inf], labels=["low", "moderate", "high", "very_high"]).astype(object)
        if self.feature_set == "full":
            d["wickets_x_overs"] = d["wickets_lost"] * d["overs_played"]
            d["rr_x_wickets_remaining"] = d["run_rate"] * d["wickets_remaining"]
            d["overs_remaining_x_rr"] = d["overs_remaining"] * d["run_rate"]
        num, cat = feature_columns(self.feature_set)
        return d[num + cat]


def build_preprocessor(feature_set="full", scale=False):
    num, cat = feature_columns(feature_set)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    ct = ColumnTransformer(
        [("num", Pipeline(num_steps), num),
         ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)],
        verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def build_pipeline(model, feature_set="full", scale=False):
    return Pipeline([("fe", CricketFeatureEngineer(feature_set)),
                     ("prep", build_preprocessor(feature_set, scale)),
                     ("model", model)])
