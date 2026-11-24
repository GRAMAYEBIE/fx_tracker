"""
LLM helpers via the Ollama API — local Ollama or Ollama Cloud.

Local (default, nothing to configure):
    1. Install Ollama: https://ollama.com
    2. ollama pull qwen2.5:7b        (or qwen2.5:3b on a smaller laptop)

Online (Streamlit Cloud) — set these in .streamlit/secrets.toml or the app's Secrets:
    OLLAMA_HOST    = "https://ollama.com"
    OLLAMA_API_KEY = "<key from ollama.com → Settings → Keys>"
    OLLAMA_MODEL   = "gpt-oss:20b"     # any model listed at ollama.com/search?c=cloud
Environment variables with the same names also work.

The app falls back to rule-based text automatically if Ollama is unreachable.
"""
import json
import os
import re

import requests


def _setting(name: str, default: str = "") -> str:
    """Read a setting from env vars first, then Streamlit secrets (if any)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:          # no secrets file / not running under Streamlit
        return default


OLLAMA_HOST = _setting("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_API_KEY = _setting("OLLAMA_API_KEY")
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"
DEFAULT_MODEL = _setting("OLLAMA_MODEL", "qwen2.5:7b")
IS_CLOUD = "localhost" not in OLLAMA_HOST and "127.0.0.1" not in OLLAMA_HOST
HEADERS = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}


def ollama_available(timeout: float = 5.0) -> bool:
    try:
        return requests.get(f"{OLLAMA_HOST}/api/tags", headers=HEADERS, timeout=timeout).ok
    except Exception:
        return False


def chat(system: str, user: str, model: str = DEFAULT_MODEL, json_mode: bool = False,
         temperature: float = 0.3, timeout: int = 120) -> str:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        body["format"] = "json"
    r = requests.post(OLLAMA_URL, json=body, headers=HEADERS, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
    content = (r.json().get("message") or {}).get("content", "").strip()
    if not content:
        raise RuntimeError("model returned an empty answer")
    return content


# ---------------------------------------------------------------- desk view
DESK_SYSTEM = """You are a senior FX & fixed-income sales trader on a corporate broking desk in Accra, Ghana.
Write the DESK VIEW for this morning's client brief, in professional, confident English.
Rules:
- 2 to 3 short sentences, max 70 words.
- DO NOT write any digits or numbers at all (not even "ten-year" written as digits, dates or tenors). The figures are already shown above your text; refer to moves qualitatively (e.g. "firmer dollar", "gold under pressure").
- How to read the facts: "USD/GHS higher" means the CEDI IS WEAKER; "USD/GHS lower" means the CEDI IS STRONGER.
  Magnitude words: "slightly" = small move, "sharply" = unusually large move for that market.
- "CEDI TODAY" is the headline and must lead your first sentence. Global drivers (dollar index, gold, oil, cocoa, yields) are context only:
  if they point the other way, say so explicitly (e.g. "despite a softer global dollar") — never contradict the cedi's actual move.
- Focus on what it means for the cedi and for clients: importers (buy USD), exporters (sell USD), investors (T-bills).
- End with one clear, actionable suggestion for clients that is consistent with the cedi's move.
- No disclaimers, no headings, no bullet points."""


def desk_view(facts: dict, headlines: list[str], model: str = DEFAULT_MODEL) -> str:
    user = ("MARKET FACTS (direction words only):\n" + json.dumps(facts, indent=1) +
            "\n\nTOP HEADLINES:\n" + "\n".join(f"- {h}" for h in headlines[:8]))
    text = chat(DESK_SYSTEM, user, model=model)
    text = re.sub(r"\s+", " ", text).strip().strip('"')
    # guard: the LLM must never introduce figures — drop any sentence containing a digit
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = [s for s in sentences if s and not re.search(r"\d", s)]
    if not clean:
        raise ValueError("LLM output contained numbers — rejected")
    return " ".join(clean)


# ---------------------------------------------------------------- news scoring
NEWS_SYSTEM = """You are an FX analyst covering the Ghana cedi (GHS).
For each headline, estimate its likely impact on USD/GHS today.
impact: integer from -2 to +2
   +2 = strongly pushes USD/GHS UP (cedi weaker), -2 = strongly pushes USD/GHS DOWN (cedi stronger), 0 = no effect.
relevance: integer 0 to 3 (3 = directly about the cedi / Bank of Ghana / Ghana FX; 0 = irrelevant).
Return JSON only: {"scores": [{"i": <index>, "impact": <int>, "relevance": <int>}, ...]}"""


def score_headlines(headlines: list[str], model: str = DEFAULT_MODEL) -> list[dict]:
    user = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    raw = chat(NEWS_SYSTEM, user, model=model, json_mode=True, temperature=0.0)
    m = re.search(r"\{.*\}", raw, re.S)          # tolerate ```json fences or extra text
    data = json.loads(m.group(0) if m else raw)
    out = {}
    for s in data.get("scores", []):
        try:
            i = int(s["i"])
            out[i] = {"impact": max(-2, min(2, int(s["impact"]))),
                      "relevance": max(0, min(3, int(s["relevance"])))}
        except (KeyError, ValueError, TypeError):
            continue
    return [out.get(i, {"impact": 0, "relevance": 0}) for i in range(len(headlines))]
