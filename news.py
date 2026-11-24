"""
News feed for the Ghana FX desk — Google News RSS (free, no key), local + global.
Each headline gets an impact score on USD/GHS (+ = cedi weaker) from the AI model (Ollama),
or from a keyword fallback when Ollama is unreachable.
"""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd

import llm

GN = "https://news.google.com/rss/search?q={q}&hl=en-GH&gl=GH&ceid=GH:en"

# (label, query, locality weight)  — local news counts more for the cedi
FEEDS = [
    ("🇬🇭 Cedi & BoG", 'cedi OR "Bank of Ghana" OR "BoG" when:2d', 1.0),
    ("🇬🇭 Ghana economy", 'Ghana (inflation OR budget OR "T-bill" OR cocoa OR gold OR IMF) when:2d', 0.9),
    ("🌍 Fed & dollar", '("Federal Reserve" OR Fed OR "dollar index") (rates OR hike OR inflation) when:1d', 0.6),
    ("🌍 Commodities", '(gold OR "Brent crude" OR cocoa) prices when:1d', 0.6),
    ("🌍 Emerging markets", '("emerging market" OR "frontier market") currencies when:2d', 0.5),
]

NEWS_LOG = Path("data/news_log.csv")

# keyword fallback: + = cedi weaker
_UP = ["depreciat", "weaken", "falls against", "slump", "pressure on cedi", "dollar strength",
       "stronger dollar", "rate hike", "hawkish", "shortage", "deficit", "downgrade", "gold falls",
       "gold drops", "cocoa falls", "oil rises", "oil jumps", "brent rises", "outflow", "undersubscri"]
_DOWN = ["appreciat", "strengthen", "gains against", "rebound", "inflow", "reserves rise",
         "upgrade", "surplus", "dovish", "rate cut", "weaker dollar", "dollar falls", "gold rises",
         "gold jumps", "cocoa rises", "oil falls", "imf approves", "disburse", "oversubscri"]
_LOCAL = ["cedi", "bank of ghana", "bog", "ghana"]


def _keyword_score(title: str) -> dict:
    t = title.lower()
    impact = sum(k in t for k in _UP) - sum(k in t for k in _DOWN)
    relevance = 3 if any(k in t for k in _LOCAL[:3]) else 2 if "ghana" in t else 1
    return {"impact": max(-2, min(2, impact)), "relevance": relevance if impact else min(relevance, 1)}


def fetch_news(max_per_feed: int = 8) -> pd.DataFrame:
    rows, seen = [], set()
    for label, q, weight in FEEDS:
        feed = feedparser.parse(GN.format(q=quote_plus(q)))
        for e in feed.entries[:max_per_feed]:
            title = e.get("title", "").strip()
            key = title.lower()[:90]
            if not title or key in seen:
                continue
            seen.add(key)
            ts = e.get("published_parsed")
            published = datetime(*ts[:6], tzinfo=timezone.utc) if ts else None
            rows.append({"feed": label, "title": title, "link": e.get("link"),
                         "source": (e.get("source") or {}).get("title", ""),
                         "published": published, "locality": weight})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("published", ascending=False, na_position="last").reset_index(drop=True)
    return df


def score_news(df: pd.DataFrame, use_llm: bool, model: str = llm.DEFAULT_MODEL) -> tuple[pd.DataFrame, str]:
    if df.empty:
        return df, "none"
    method = "keywords"
    scores = None
    if use_llm:
        try:
            scores = llm.score_headlines(df["title"].tolist(), model=model)
            method = f"AI ({model})"
        except Exception as e:
            scores = None
            method = f"keywords (AI failed: {str(e)[:120]})"
    if scores is None:
        scores = [_keyword_score(t) for t in df["title"]]
    df = df.copy()
    df["impact"] = [s["impact"] for s in scores]
    df["relevance"] = [s["relevance"] for s in scores]
    now = datetime.now(timezone.utc)
    age_h = df["published"].apply(lambda p: (now - p).total_seconds() / 3600 if pd.notna(p) else 24)
    df["recency"] = (0.5 ** (age_h / 12)).clip(0.05, 1.0)          # half-life 12 hours
    df["weight"] = df["relevance"] / 3 * df["locality"] * df["recency"]
    return df, method


def news_index(df: pd.DataFrame) -> float:
    """Weighted average impact in [-2, +2]; + = pressure for cedi weakness."""
    if df.empty or df["weight"].sum() == 0:
        return 0.0
    return float((df["impact"] * df["weight"]).sum() / df["weight"].sum())


def log_news(df: pd.DataFrame) -> None:
    """Keep every scored headline — becomes training data for a news-aware model later."""
    if df.empty:
        return
    NEWS_LOG.parent.mkdir(exist_ok=True)
    cols = ["published", "feed", "source", "title", "impact", "relevance", "weight", "link"]
    new = df[cols].copy()
    new["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    if NEWS_LOG.exists():
        try:
            old_titles = set(pd.read_csv(NEWS_LOG, usecols=["title"])["title"])
            new = new[~new["title"].isin(old_titles)]
        except Exception:
            pass
    if len(new):
        new.to_csv(NEWS_LOG, mode="a", header=not NEWS_LOG.exists(), index=False)
