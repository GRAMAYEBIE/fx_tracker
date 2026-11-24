"""
Daily directional forecast for USD/GHS.

1. Statistical model (logistic regression, walk-forward validated) on 1 year of daily data:
   features = today's moves in DXY, EUR/USD, gold, Brent, cocoa, US 10Y + recent USD/GHS momentum.
   target   = does USD/GHS close higher the next day (cedi weaker)?
2. News adjustment: the weighted news index shifts the model's probability.
   This step is a transparent heuristic (not yet trained) — your news_log.csv will
   let you learn its real weight after a few weeks of data.

Honest by design: the app always shows the model's measured accuracy vs a naive baseline.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "USD/GHS (market)"
DRIVER_COLS = ["DXY (Dollar Index)", "EUR/USD", "Gold ($/oz)", "Brent ($/bbl)",
               "Cocoa ($/t)", "US 10Y yield (%)"]
NEWS_WEIGHT = 0.6     # logit shift per news-index point (heuristic; calibrate later)
MIN_MOVE = 1e-5       # ignore zero-change days (stale quotes)


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    rets = daily[DRIVER_COLS + [TARGET]].pct_change()
    X = rets[DRIVER_COLS].add_suffix(" 1d")
    X["USD/GHS 1d"] = rets[TARGET]
    X["USD/GHS 5d"] = daily[TARGET].pct_change(5)
    X["DXY 5d"] = daily["DXY (Dollar Index)"].pct_change(5)
    X["Gold 5d"] = daily["Gold ($/oz)"].pct_change(5)
    return X


def fit_and_predict(daily: pd.DataFrame) -> dict:
    X = build_features(daily)
    fwd = daily[TARGET].pct_change().shift(-1)
    data = X.assign(fwd=fwd).replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data["fwd"].abs() > MIN_MOVE]
    if len(data) < 80:
        return {"ok": False, "reason": f"not enough clean history ({len(data)} days)"}

    y = (data["fwd"] > 0).astype(int)
    Xd = data.drop(columns="fwd")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))

    # walk-forward validation: always train on the past, test on the future
    hits, base_hits, n = 0, 0, 0
    for tr, te in TimeSeriesSplit(n_splits=5).split(Xd):
        if y.iloc[tr].nunique() < 2:
            continue
        model.fit(Xd.iloc[tr], y.iloc[tr])
        pred = model.predict(Xd.iloc[te])
        hits += (pred == y.iloc[te].values).sum()
        majority = int(y.iloc[tr].mean() >= 0.5)
        base_hits += (y.iloc[te].values == majority).sum()
        n += len(te)

    model.fit(Xd, y)
    latest = X.replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]]
    p_up = float(model.predict_proba(latest)[0, 1])
    coefs = pd.Series(model[-1].coef_[0], index=Xd.columns).sort_values(key=abs, ascending=False)
    return {"ok": True, "p_up": p_up, "accuracy": hits / n if n else None,
            "baseline": base_hits / n if n else None, "n_test": n, "n_train": len(Xd),
            "drivers": coefs}


def combine(p_model: float, news_idx: float) -> float:
    logit = np.log(p_model / (1 - p_model)) + NEWS_WEIGHT * news_idx
    return float(1 / (1 + np.exp(-logit)))


def label(p_up: float) -> tuple[str, str]:
    if p_up >= 0.60:
        return "🔴 Cedi under pressure (USD/GHS likely ↑)", "importers: consider covering USD needs early"
    if p_up <= 0.40:
        return "🟢 Cedi supported (USD/GHS likely ↓)", "importers can wait; exporters may sell USD now"
    return "🟡 Neutral / range-bound", "no strong edge — execute client needs as they come"


def confidence(accuracy: float | None, baseline: float | None) -> str:
    """How much the model beats a naive guess on unseen data."""
    if accuracy is None or baseline is None:
        return "unknown"
    edge = accuracy - baseline
    return "low" if edge < 0.02 else "moderate" if edge < 0.06 else "good"
