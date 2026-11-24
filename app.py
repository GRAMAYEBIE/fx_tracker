"""
FX Market Tracker — Ghana desk edition (v2: 100% API, no manual input)

Run:   pip install -r requirements.txt
       streamlit run app.py

Sources (all public):
  • Global markets ........ Yahoo Finance (yfinance)          — refresh every 5 min (some feeds ~15 min delayed)
  • BoG interbank rates ... AllRatesToday API (Bank of Ghana)   — BoG publishes once per business day
  • Ghana T-bills ......... treasury-bills API (BoG T-bill page) — BoG auctions once per week

  • News ................. Google News RSS (Ghana + global), scored by AI (Ollama local or cloud) — every 15 min
  • Forecast ............. logistic regression on 1y daily data + news adjustment

Optional: set an AllRatesToday key in .streamlit/secrets.toml →  ALLRATES_API_KEY = "..."
Never load Spectra / client data into this tool.
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

import forecast
import llm
import news

st.set_page_config(page_title="FX Market Tracker", page_icon="💱", layout="wide")

# ================================================================ config
ASSETS = {
    "USD/GHS (market)": "GHS=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "DXY (Dollar Index)": "DX-Y.NYB",
    "Gold ($/oz)": "GC=F",
    "Brent ($/bbl)": "BZ=F",
    "Cocoa ($/t)": "CC=F",
    "US 10Y yield (%)": "^TNX",
}
# Top world currencies + regional peers. "direct" = quoted XXX/USD, "inverse" = quoted USD/XXX
WORLD = {
    "EUR": ("Euro", "EURUSD=X", "direct"),
    "GBP": ("British pound", "GBPUSD=X", "direct"),
    "JPY": ("Japanese yen", "JPY=X", "inverse"),
    "CNY": ("Chinese yuan", "CNY=X", "inverse"),
    "CHF": ("Swiss franc", "CHF=X", "inverse"),
    "CAD": ("Canadian dollar", "CAD=X", "inverse"),
    "AUD": ("Australian dollar", "AUDUSD=X", "direct"),
    "ZAR": ("South African rand", "ZAR=X", "inverse"),
    "NGN": ("Nigerian naira", "NGN=X", "inverse"),
    "XOF": ("CFA franc (BCEAO)", "XOF=X", "inverse"),
}
TICKERS = sorted(set(ASSETS.values()) | {t for _, t, _ in WORLD.values()})

# Ghana view: everything also expressed in Ghana cedis (GHS)
GHS_DERIVED = {
    "EUR/GHS": ("EUR/USD", "mul"),
    "GBP/GHS": ("GBP/USD", "mul"),
    "Gold (GHS/oz)": ("Gold ($/oz)", "mul"),
    "Brent (GHS/bbl)": ("Brent ($/bbl)", "mul"),
    "Cocoa (GHS/t)": ("Cocoa ($/t)", "mul"),
}
ACCRA = ZoneInfo("Africa/Accra")
now = lambda: datetime.now(ACCRA)
DRIVERS = ["DXY (Dollar Index)", "Gold ($/oz)", "Brent ($/bbl)", "Cocoa ($/t)", "US 10Y yield (%)"]
BOG_PAIRS = ["USD", "EUR", "GBP"]
TBILL_TENORS = [91, 182, 364]

ALLRATES_OPEN = "https://allratestoday.com/api/open/central-bank/bog"
ALLRATES_V1 = "https://allratestoday.com/api/v1/central-bank/bog/latest"
TBILL_API = "https://treasury-bills.onrender.com/api/get-tbill"

LOG_FILE = Path("data/market_log.csv")
REFRESH_SECONDS = 300
HTTP_TIMEOUT = 60  # the T-bill API sleeps when idle; first call can be slow


def _api_key() -> str | None:
    try:
        return st.secrets.get("ALLRATES_API_KEY")
    except Exception:
        return os.getenv("ALLRATES_API_KEY")


# ================================================================ JSON helpers
RATE_KEYS = ("rate", "mid", "middle", "interest_rate", "interestrate", "discount_rate",
             "yield", "value", "buying")
DATE_KEYS = ("rate_date", "date", "issue_date", "auction_date", "tender_date", "updated_at")


def _walk_dicts(obj):
    """Yield every dict inside a JSON structure (depth-first)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_dicts(v)


def _to_float(x):
    try:
        return float(str(x).replace("%", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def extract_rate(payload, prefer_keys=RATE_KEYS) -> tuple[float | None, str | None]:
    """Find the first numeric rate (and its date) in an unknown JSON shape."""
    for d in _walk_dicts(payload):
        lower = {str(k).lower(): v for k, v in d.items()}
        for key in prefer_keys:
            if key in lower and _to_float(lower[key]) is not None:
                when = next((str(lower[k]) for k in DATE_KEYS if k in lower and lower[k]), None)
                return _to_float(lower[key]), when
    return None, None


# ================================================================ data sources
def add_ghs(df: pd.DataFrame) -> pd.DataFrame:
    """Convert USD prices into Ghana cedis using the USD/GHS market series."""
    df = df.rename(columns={t: f"_{c}" for c, (_, t, _) in WORLD.items() if t not in ASSETS.values()})
    inv = {v: k for k, v in ASSETS.items()}
    fx = df["USD/GHS (market)"]
    for name, (src, _) in GHS_DERIVED.items():
        df[name] = df[src] * fx
    for ccy, (_, tick, style) in WORLD.items():
        col = inv.get(tick, f"_{ccy}")
        usd_per_unit = df[col] if style == "direct" else 1 / df[col]
        df[f"{ccy} in USD"] = usd_per_unit
        df[f"{ccy} in GHS"] = usd_per_unit * fx
    return df


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_intraday() -> pd.DataFrame:
    df = yf.download(TICKERS, period="5d", interval="15m",
                     progress=False, auto_adjust=False)["Close"]
    return add_ghs(df.rename(columns={v: k for k, v in ASSETS.items()}).ffill())


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily() -> pd.DataFrame:
    df = yf.download(TICKERS, period="1y", interval="1d",
                     progress=False, auto_adjust=False)["Close"]
    return add_ghs(df.rename(columns={v: k for k, v in ASSETS.items()}).ffill())


@st.cache_data(ttl=3600, show_spinner=False)
def load_bog_rates() -> dict:
    """Official BoG interbank rates for USD, EUR, GBP → GHS."""
    key = _api_key()
    out, raw = {}, {}
    for ccy in BOG_PAIRS:
        params = {"source": ccy, "target": "GHS"}
        try:
            if key:
                r = requests.get(ALLRATES_V1, params=params, timeout=HTTP_TIMEOUT,
                                 headers={"Authorization": f"Bearer {key}"})
            else:
                r = requests.get(ALLRATES_OPEN, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            raw[ccy] = payload
            rate, when = extract_rate(payload, ("rate", "mid", "middle", "buying"))
            out[ccy] = {"rate": rate, "date": when}
        except Exception as e:
            out[ccy] = {"rate": None, "date": None, "error": str(e)}
    return {"rates": out, "raw": raw}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_tbills() -> dict:
    """Latest BoG T-bill rates (91 / 182 / 364 days)."""
    out, raw = {}, {}
    for days in TBILL_TENORS:
        try:
            r = requests.get(TBILL_API, params={"days": days}, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            raw[days] = payload
            rate, when = extract_rate(payload, ("interest_rate", "interestrate", "rate",
                                                "discount_rate", "yield", "value"))
            out[days] = {"rate": rate, "date": when}
        except Exception as e:
            out[days] = {"rate": None, "date": None, "error": str(e)}
    return {"rates": out, "raw": raw}


@st.cache_data(ttl=900, show_spinner=False)
def load_news(use_llm: bool, model: str):
    raw = news.fetch_news()
    scored, method = news.score_news(raw, use_llm=use_llm, model=model)
    return scored, method


@st.cache_data(ttl=3600, show_spinner=False)
def load_forecast(daily: pd.DataFrame) -> dict:
    return forecast.fit_and_predict(daily)


@st.cache_data(ttl=900, show_spinner=False)
def llm_desk_view(facts_json: str, headlines: tuple, model: str) -> str:
    import json
    return llm.desk_view(json.loads(facts_json), list(headlines), model=model)


# ================================================================ analytics
def _pct(snap, name, col="1D %") -> float:
    v = snap.loc[name, col] if name in snap.index else None
    return 0.0 if v is None or pd.isna(v) else float(v)


def direction_word(snap, name, daily: pd.DataFrame | None = None) -> str:
    """'flat' / 'slightly higher' / 'higher' / 'sharply lower'… — size judged vs that market's usual daily move."""
    v = _pct(snap, name)
    if abs(v) < 0.05:
        return "flat"
    word = "higher" if v > 0 else "lower"
    vol = None
    if daily is not None and name in daily:
        vol = daily[name].pct_change().tail(60).std() * 100
    if vol is None or pd.isna(vol) or vol == 0:
        vol = 0.5
    z = abs(v) / vol
    if z < 0.5:
        return f"slightly {word}"
    if z >= 1.5:
        return f"sharply {word}"
    return word


def cedi_word(snap, daily) -> str:
    """Translate the USD/GHS move into plain words about the cedi."""
    w = direction_word(snap, "USD/GHS (market)", daily)
    return "flat" if w == "flat" else w.replace("higher", "weaker").replace("lower", "stronger")


def snapshot(daily: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in list(ASSETS) + list(GHS_DERIVED):
        s = intraday[name].dropna() if name in intraday else pd.Series(dtype=float)
        last = s.iloc[-1] if len(s) else None
        d = daily[name].dropna() if name in daily else pd.Series(dtype=float)
        ref = lambda n: d.iloc[-n] if len(d) >= n else None
        pct = lambda r: (last / r - 1) * 100 if last is not None and r else None
        rows.append({"Asset": name, "Last": last, "1D %": pct(ref(2)),
                     "1W %": pct(ref(6)), "1M %": pct(ref(22))})
    return pd.DataFrame(rows).set_index("Asset")


def world_table(daily: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ccy, (label, _, _) in WORLD.items():
        r = {"Code": ccy, "Currency": label}
        for unit in ("USD", "GHS"):
            s = intraday[f"{ccy} in {unit}"].dropna()
            d = daily[f"{ccy} in {unit}"].dropna()
            last = s.iloc[-1] if len(s) else None
            r[f"1 unit in {unit}"] = last
            if unit == "GHS":
                r["GHS 1 buys"] = 1 / last if last else None
                for lab, n in (("1D %", 2), ("1W %", 6), ("1M %", 22)):
                    ref = d.iloc[-n] if len(d) >= n else None
                    r[f"vs GHS {lab}"] = (last / ref - 1) * 100 if last and ref else None
        rows.append(r)
    return pd.DataFrame(rows).set_index("Code")


def _mv(snap, name) -> str:
    v = snap.loc[name, "1D %"]
    if v is None or pd.isna(v) or abs(v) < 0.05:
        return "flat"
    return f"up {v:.2f}%" if v > 0 else f"down {abs(v):.2f}%"


def auto_desk_view(snap: pd.DataFrame) -> str:
    """Rule-based starting point for the desk view — always edit before sending."""
    ghs = _pct(snap, "USD/GHS (market)")
    dxy = _pct(snap, "DXY (Dollar Index)")
    gold = _pct(snap, "Gold ($/oz)")
    oil = _pct(snap, "Brent ($/bbl)")
    pressure = sum([dxy > 0.05, gold < -0.3, oil > 0.3])
    relief = sum([dxy < -0.05, gold > 0.3, oil < -0.3])
    # the cedi's own move comes first — global drivers must not contradict it
    if ghs >= 0.5:
        return (("The cedi is weaker this morning despite a softer global dollar. " if relief >= 2 else
                 "The cedi is under pressure this morning, weaker against the dollar and major crosses. ") +
                "Importers with near-term USD payments should consider covering early; "
                "exporters can take advantage by selling USD now.")
    if ghs <= -0.5:
        return (("The cedi is firmer this morning despite a stronger global dollar. " if pressure >= 2 else
                 "The cedi is firmer this morning against the dollar. ") +
                "Importers can use the strength to buy USD at better levels; exporters may prefer to hold USD for now.")
    if pressure >= 2:
        return ("Firm dollar and weaker Ghana FX inflows/higher import bill point to cedi pressure. "
                "Importers with USD payments due may want to cover early or lock forwards.")
    if relief >= 2:
        return ("Softer dollar and supportive commodities ease pressure on the cedi. "
                "A window for importers to buy USD; exporters may hold.")
    return ("Mixed signals for the cedi. Good moment to review clients' upcoming USD needs "
            "and offer forward cover where exposure is large.")


def briefing(snap, bog, tbills, desk_view, world=None, bias=None) -> str:
    usd = bog["rates"].get("USD", {})
    eur = bog["rates"].get("EUR", {})
    gbp = bog["rates"].get("GBP", {})
    fmt = lambda x, n=4: f"{x:.{n}f}" if x else "n/a"
    ghs = lambda x: f"{x:,.0f}" if x else "n/a"
    tb = lambda d: tbills["rates"].get(d, {}).get("rate")
    tb_date = next((v.get("date") for v in tbills["rates"].values() if v.get("date")), None)
    lines = [
        f"📊 FX MORNING BRIEF — {now():%A %d %b %Y, %H:%M} GMT (Accra)",
        "",
        "🇬🇭 CEDI (GHS)",
        f"• BoG interbank: USD/GHS {fmt(usd.get('rate'))} | EUR/GHS {fmt(eur.get('rate'))} | "
        f"GBP/GHS {fmt(gbp.get('rate'))}" + (f" (BoG date {usd.get('date')})" if usd.get("date") else ""),
        f"• Market: USD/GHS {fmt(snap.loc['USD/GHS (market)', 'Last'])} ({_mv(snap, 'USD/GHS (market)')}) | "
        f"EUR/GHS {fmt(snap.loc['EUR/GHS', 'Last'])} ({_mv(snap, 'EUR/GHS')}) | "
        f"GBP/GHS {fmt(snap.loc['GBP/GHS', 'Last'])} ({_mv(snap, 'GBP/GHS')})",
        f"• Ref: GHS 1,000 = US$ {1000 / snap.loc['USD/GHS (market)', 'Last']:,.2f}"
        if snap.loc['USD/GHS (market)', 'Last'] else "",
        "",
        "🌍 TOP CURRENCIES vs GHS (1 unit = GHS · day change)",
        *([" | ".join(f"{c} {world.loc[c, '1 unit in GHS']:,.4f} ({world.loc[c, 'vs GHS 1D %']:+.2f}%)"
                      if pd.notna(world.loc[c, '1 unit in GHS']) and pd.notna(world.loc[c, 'vs GHS 1D %'])
                      else f"{c} n/a" for c in grp)
           for grp in (["EUR", "GBP", "JPY", "CNY", "CHF"], ["CAD", "AUD", "ZAR", "NGN", "XOF"])]
          if world is not None else []),
        "",
        "💵 GLOBAL DOLLAR",
        f"• DXY {fmt(snap.loc['DXY (Dollar Index)', 'Last'], 2)} ({_mv(snap, 'DXY (Dollar Index)')}); "
        f"EUR/USD {fmt(snap.loc['EUR/USD', 'Last'])} ({_mv(snap, 'EUR/USD')}); "
        f"US 10Y {fmt(snap.loc['US 10Y yield (%)', 'Last'], 2)}%.",
        "",
        "🪙 COMMODITIES — Ghana's USD earners & import bill (in GHS)",
        f"• Gold GHS {ghs(snap.loc['Gold (GHS/oz)', 'Last'])}/oz (US${fmt(snap.loc['Gold ($/oz)', 'Last'], 0)}, {_mv(snap, 'Gold ($/oz)')})",
        f"• Cocoa GHS {ghs(snap.loc['Cocoa (GHS/t)', 'Last'])}/t (US${fmt(snap.loc['Cocoa ($/t)', 'Last'], 0)}, {_mv(snap, 'Cocoa ($/t)')})",
        f"• Brent GHS {ghs(snap.loc['Brent (GHS/bbl)', 'Last'])}/bbl (US${fmt(snap.loc['Brent ($/bbl)', 'Last'], 2)}, {_mv(snap, 'Brent ($/bbl)')})",
        "",
        "🏦 GHANA T-BILLS (latest BoG auction" + (f", {tb_date}" if tb_date else "") + ")",
        f"• 91-day {fmt(tb(91), 2)}% | 182-day {fmt(tb(182), 2)}% | 364-day {fmt(tb(364), 2)}%",
        "",
        *(["🔮 TODAY'S BIAS (model + news)",
           f"• {bias['label']} — P(USD/GHS ↑) {bias['p']:.0%} · model hit-rate {bias['acc']:.0%} "
           f"vs {bias['base']:.0%} naive ({bias['conf']} confidence) · {bias['n_news']} headlines scanned",
           ""] if bias else []),
        "🎯 DESK VIEW",
        f"• {desk_view.strip()}",
    ]
    return "\n".join(lines)


def auto_log(snap, bog, tbills, extra: dict | None = None) -> bool:
    """Save one row per day automatically (builds your own history)."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    today = now().date().isoformat()
    if LOG_FILE.exists():
        try:
            if today in set(pd.read_csv(LOG_FILE, usecols=["date"])["date"].astype(str)):
                return False
        except Exception:
            pass
    row = {"date": today, "time": now().strftime("%H:%M")}
    for c in BOG_PAIRS:
        row[f"BoG {c}/GHS"] = bog["rates"].get(c, {}).get("rate")
    for d in TBILL_TENORS:
        row[f"T-bill {d}d"] = tbills["rates"].get(d, {}).get("rate")
    row.update(extra or {})
    row.update(snap["Last"].to_dict())
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False)
    return True


# ================================================================ UI
st.title("💱 FX Market Tracker — Ghana desk")
st.caption("Ghana edition · all times GMT (Accra) · prices in GHS with USD reference · top world currencies · markets every 5 min · BoG rates daily · T-bills weekly")

with st.sidebar:
    st.header("Settings")
    corr_window = st.slider("Correlation window (days)", 20, 250, 60)
    st.divider()
    where = "Ollama Cloud" if llm.IS_CLOUD else "local Ollama"
    st.subheader(f"🤖 AI ({where})")
    ollama_ok = llm.ollama_available()
    use_llm = st.toggle("Use AI for desk view & news scoring", value=ollama_ok, disabled=not ollama_ok)
    qwen_model = st.text_input("Ollama model", llm.DEFAULT_MODEL)
    st.caption(f"✅ {where} reachable" if ollama_ok else f"⚠ {where} unreachable — rule-based fallback in use")
    if st.button("🔄 Refresh all data now"):
        st.cache_data.clear()
    st.caption("API key (optional): .streamlit/secrets.toml → ALLRATES_API_KEY")


@st.fragment(run_every=REFRESH_SECONDS)
def dashboard():
    with st.spinner("Loading markets, BoG rates and T-bills…"):
        try:
            intraday, daily = load_intraday(), load_daily()
        except Exception as e:
            st.error(f"Market data download failed: {e}")
            return
        bog, tbills = load_bog_rates(), load_tbills()

    snap = snapshot(daily, intraday)
    news_df, news_method = load_news(use_llm, qwen_model)
    n_idx = news.news_index(news_df)
    fc = load_forecast(daily)
    bias = None
    if fc.get("ok"):
        p_adj = forecast.combine(fc["p_up"], n_idx)
        lab, action = forecast.label(p_adj)
        bias = {"label": lab, "action": action, "p": p_adj, "p_model": fc["p_up"],
                "acc": fc["accuracy"] or 0, "base": fc["baseline"] or 0, "n_news": len(news_df),
                "conf": forecast.confidence(fc["accuracy"], fc["baseline"])}
    news.log_news(news_df)

    # ---- source status
    usd = bog["rates"].get("USD", {})
    s1, s2, s3 = st.columns(3)
    s1.success(f"Markets ✓ · {now():%H:%M:%S} GMT")
    (s2.success if usd.get("rate") else s2.warning)(
        f"BoG rates {'✓ · ' + str(usd.get('date')) if usd.get('rate') else '⚠ unavailable'}")
    tb_ok = any(v.get("rate") for v in tbills["rates"].values())
    (s3.success if tb_ok else s3.warning)(f"T-bills {'✓' if tb_ok else '⚠ unavailable'}")

    # ---- KPI cards (Ghana view first, in GHS)
    def card(col, label, value, chg=None, digits=4):
        col.metric(label, "—" if value is None else f"{value:,.{digits}f}",
                   f"{chg:+.2f}%" if chg is not None and not pd.isna(chg) else None)

    st.subheader("🇬🇭 Cedi (GHS)")
    c = st.columns(5)
    card(c[0], "BoG USD/GHS (official)", usd.get("rate"))
    card(c[1], "BoG EUR/GHS (official)", bog["rates"].get("EUR", {}).get("rate"))
    for col, name in zip(c[2:], ["USD/GHS (market)", "EUR/GHS", "GBP/GHS"]):
        card(col, name, snap.loc[name, "Last"], snap.loc[name, "1D %"])
    ghs_usd = snap.loc["USD/GHS (market)", "Last"]
    if ghs_usd:
        st.caption(f"Ref: 1 GHS = US$ {1 / ghs_usd:.4f} · GHS 1,000 = US$ {1000 / ghs_usd:,.2f}")

    st.subheader("🪙 Commodities in GHS · 💵 Global dollar")
    c = st.columns(5)
    for col, g, u in zip(c[:3], ["Gold (GHS/oz)", "Cocoa (GHS/t)", "Brent (GHS/bbl)"],
                         ["Gold ($/oz)", "Cocoa ($/t)", "Brent ($/bbl)"]):
        card(col, g, snap.loc[g, "Last"], snap.loc[g, "1D %"], digits=0)
        usd_px = snap.loc[u, "Last"]
        col.caption(f"Ref: US$ {usd_px:,.2f}" if usd_px else "Ref: US$ —")
    card(c[3], "DXY (Dollar Index)", snap.loc["DXY (Dollar Index)", "Last"],
         snap.loc["DXY (Dollar Index)", "1D %"], digits=2)
    card(c[4], "EUR/USD", snap.loc["EUR/USD", "Last"], snap.loc["EUR/USD", "1D %"])

    mkt = snap.loc["USD/GHS (market)", "Last"]
    if usd.get("rate") and mkt:
        gap = mkt - usd["rate"]
        st.info(f"Market vs BoG gap: **{gap:+.4f} GHS** ({gap / usd['rate'] * 100:+.2f}%)")

    ghs_move = _pct(snap, "USD/GHS (market)")
    big_move_msg = None
    if abs(ghs_move) >= 2:
        big_move_msg = (f"⚠ USD/GHS moved {ghs_move:+.2f}% vs the previous close — unusually large for one day. "
                        "Yahoo's GHS quote can be stale or jump; check against another source (BoG, your dealer screen) "
                        "before sending the brief.")
        st.warning(big_move_msg)

    if auto_log(snap, bog, tbills, {"news_index": round(n_idx, 3),
                                    "p_up_model": fc.get("p_up"), "p_up_adjusted": bias and bias["p"]}):
        st.toast(f"Today's snapshot saved to {LOG_FILE}")

    world = world_table(daily, intraday)
    tab1, tab_n, tab_w, tab2, tab3, tab4, tab5 = st.tabs(
        ["📝 Morning brief", "📰 News & forecast", "🌍 World currencies", "📋 Snapshot",
         "📈 Intraday", "🔗 What drives the cedi?", "🛠 API status"])

    with tab_n:
        st.subheader("🔮 Today's bias for USD/GHS")
        if bias:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("P(USD/GHS ↑) — model", f"{bias['p_model']:.0%}")
            b2.metric("P(USD/GHS ↑) — model + news", f"{bias['p']:.0%}",
                      f"{(bias['p'] - bias['p_model']) * 100:+.0f} pts from news")
            b3.metric("Model hit-rate (walk-forward)", f"{bias['acc']:.0%}",
                      f"{(bias['acc'] - bias['base']) * 100:+.0f} pts vs naive")
            b4.metric("News pressure index", f"{n_idx:+.2f}", help="-2 = cedi-supportive … +2 = cedi-negative")
            st.markdown(f"### {bias['label']}\n**Client angle:** {bias['action']}  \n"
                        f"**Confidence:** {bias['conf']}")
            if bias["conf"] == "low":
                st.warning("The model does not beat a naive guess on past data yet — treat the bias as "
                           "a news-driven indication only. Accuracy improves as your daily logs grow.")
            st.caption(f"Model trained on {fc['n_train']} days, tested on {fc['n_test']} unseen days. "
                       "Probabilities, not certainties — FX is hard to forecast; use as one input among others.")
            with st.expander("What the model is looking at (standardised coefficients)"):
                st.bar_chart(fc["drivers"])
                st.caption("+ = pushes USD/GHS up (cedi weaker) the next day.")
        else:
            st.warning(f"Forecast unavailable: {fc.get('reason')}")

        st.subheader(f"📰 Live news — scored by {news_method}")
        if news_df.empty:
            st.info("No headlines fetched.")
        else:
            show = news_df.assign(
                time=news_df["published"].dt.tz_convert(ACCRA).dt.strftime("%d %b %H:%M"),
                effect=news_df["impact"].map({2: "🔴🔴 cedi −−", 1: "🔴 cedi −", 0: "⚪ neutral",
                                              -1: "🟢 cedi +", -2: "🟢🟢 cedi ++"}))
            show = show.sort_values(["weight", "published"], ascending=False)
            st.dataframe(show[["time", "feed", "effect", "relevance", "title", "source", "link"]],
                         width="stretch", hide_index=True,
                         column_config={"link": st.column_config.LinkColumn("open", display_text="↗")})

    with tab_w:
        st.write("Top world currencies & regional peers — value of 1 unit in US dollars and in Ghana cedis")
        fmt_cols = {c: "{:,.4f}" for c in ("1 unit in USD", "1 unit in GHS", "GHS 1 buys")}
        fmt_cols.update({c: "{:+.2f}" for c in world.columns if c.startswith("vs GHS")})
        st.dataframe(world.style.format(fmt_cols, na_rep="—"), width="stretch")
        st.caption("vs GHS % > 0 → that currency gained against the cedi (cedi weaker). "
                   "Crosses computed via USD from market quotes; the BoG rate is the official reference.")

    with tab1:
        suggestion, src = auto_desk_view(snap), "rules"
        if use_llm:
            import json
            facts = {"CEDI TODAY": cedi_word(snap, daily)}
            facts.update({n: direction_word(snap, n, daily) for n in
                          ["USD/GHS (market)", "DXY (Dollar Index)", "EUR/USD", "Gold ($/oz)",
                           "Brent ($/bbl)", "Cocoa ($/t)", "US 10Y yield (%)"]})
            facts["news pressure on cedi"] = ("negative" if n_idx > 0.3 else
                                              "supportive" if n_idx < -0.3 else "neutral")
            if bias:
                facts["model bias"] = bias["label"]
            top = tuple(news_df.sort_values("weight", ascending=False)["title"].head(8)) if len(news_df) else ()
            try:
                suggestion = llm_desk_view(json.dumps(facts), top, qwen_model)
                src = f"AI ({qwen_model})"
            except Exception as e:
                st.warning(f"AI desk view failed ({str(e)[:200]}) — rule-based view used instead.")
        if big_move_msg:
            st.warning(big_move_msg)
        desk = st.text_area(f"🎯 Desk view — drafted by {src} (edit before sending)",
                            value=suggestion, height=110)
        st.code(briefing(snap, bog, tbills, desk, world, bias), language=None)
        st.caption("Copy → WhatsApp / email. Always read the numbers before sending.")

    with tab2:
        st.dataframe(snap.style.format({"Last": "{:,.4f}", "1D %": "{:+.2f}",
                                        "1W %": "{:+.2f}", "1M %": "{:+.2f}"}, na_rep="—"),
                     width="stretch")
        tb_df = pd.DataFrame({f"{d}-day": [v.get("rate")] for d, v in tbills["rates"].items()},
                             index=["Rate %"])
        st.write("Ghana T-bills (latest auction)")
        st.dataframe(tb_df, width="stretch")
        if LOG_FILE.exists():
            st.write("📚 My history (auto-saved daily)")
            st.dataframe(pd.read_csv(LOG_FILE).tail(30), width="stretch")

    with tab3:
        pick = st.selectbox("Asset", list(ASSETS) + list(GHS_DERIVED) + [f"{c} in {u}" for c in WORLD for u in ("USD", "GHS")])
        st.line_chart(intraday[pick].dropna())

    with tab4:
        rets = daily.pct_change().tail(corr_window)
        corr = rets[DRIVERS].corrwith(rets["USD/GHS (market)"]).sort_values(key=abs, ascending=False)
        st.write(f"Correlation of daily USD/GHS changes with each driver (last {corr_window} days)")
        st.bar_chart(corr)
        st.caption("Positive = moves with USD/GHS. Correlation ≠ causation. "
                   "Your own daily BoG log will make this more reliable over time.")

    with tab5:
        st.write("Raw API responses — use this if a value shows 'n/a' to adjust the parser.")
        st.json({"BoG": bog, "T-bills": tbills}, expanded=False)


dashboard()
