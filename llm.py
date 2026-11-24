"""
LLM helpers — two backends, chosen by settings:

1. OpenAI-compatible API (Groq, OpenRouter, Gemini, …) — used when LLM_BASE_URL is set.
   Streamlit Cloud → Settings → Secrets (or .streamlit/secrets.toml):
       LLM_BASE_URL = "https://api.groq.com/openai/v1"
       LLM_API_KEY  = "gsk_..."                 # console.groq.com → API Keys
       LLM_MODEL    = "openai/gpt-oss-20b"      # any id from GET {LLM_BASE_URL}/models

2. Ollama (default when LLM_BASE_URL is not set):
   Local, nothing to configure:  install https://ollama.com, then `ollama pull qwen2.5:7b`
   Ollama Cloud (paid):          OLLAMA_HOST = "https://ollama.com", OLLAMA_API_KEY, OLLAMA_MODEL

Environment variables with the same names also work.
The app falls back to rule-based text automatically if the AI is unreachable.
"""
import json
import os
import re

import requests


LOCAL_HOST = "http://localhost:11434"
LOCAL_MODEL = "qwen2.5:7b"
OPENAI_DEFAULT_MODEL = "openai/gpt-oss-20b"


def _setting(name: str, default: str = "") -> str:
    """Read a setting from env vars first, then Streamlit secrets (top level or inside any [section]).
    Read on every call, so edited secrets apply without restarting the app."""
    val = os.environ.get(name)
    if not val:
        try:
            import streamlit as st
            sec = st.secrets
            val = sec.get(name)
            if val is None:            # tolerate keys placed under a [section] header
                val = next((v[name] for v in sec.values() if hasattr(v, "get") and v.get(name)), None)
        except Exception:              # no secrets file / not running under Streamlit
            val = None
    val = str(val).strip().strip('"').strip("'").strip() if val else ""
    return val or default


def openai_base() -> str:
    """Base URL of an OpenAI-compatible API, or '' to use Ollama."""
    return _setting("LLM_BASE_URL").rstrip("/")


def is_openai() -> bool:
    return bool(openai_base())


def host() -> str:
    return _setting("OLLAMA_HOST", LOCAL_HOST).rstrip("/")


def api_key() -> str:
    key = _setting("LLM_API_KEY") if is_openai() else _setting("OLLAMA_API_KEY")
    return key[7:].strip() if key.lower().startswith("bearer ") else key


def default_model() -> str:
    if is_openai():
        return _setting("LLM_MODEL", OPENAI_DEFAULT_MODEL)
    return _setting("OLLAMA_MODEL", LOCAL_MODEL)


def is_cloud() -> bool:
    if is_openai():
        return True
    h = host()
    return "localhost" not in h and "127.0.0.1" not in h


def provider_name() -> str:
    if is_openai():
        base = openai_base().lower()
        for k, name in (("groq", "Groq"), ("openrouter", "OpenRouter"), ("googleapis", "Gemini"),
                        ("openai.com", "OpenAI"), ("ollama.com", "Ollama Cloud")):
            if k in base:
                return name
        return "OpenAI-compatible API"
    return "Ollama Cloud" if is_cloud() else "local Ollama"


def _headers() -> dict:
    key = api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def available(timeout: float = 5.0) -> bool:
    """True if the AI backend answers (for OpenAI-compatible APIs this also validates the key)."""
    if is_cloud() and not api_key():
        return False
    url = f"{openai_base()}/models" if is_openai() else f"{host()}/api/tags"
    try:
        return requests.get(url, headers=_headers(), timeout=timeout).ok
    except Exception:
        return False


ollama_available = available     # backward-compatible name


def _raise_for(r: requests.Response) -> None:
    who = provider_name()
    if r.status_code in (401, 403):
        var = "LLM_API_KEY" if is_openai() else "OLLAMA_API_KEY"
        raise RuntimeError(f"{who} HTTP {r.status_code}: API key rejected — check {var} in Secrets "
                           "(no spaces, no quotes inside, key still active)")
    if r.status_code == 429:
        raise RuntimeError(f"{who} HTTP 429: free-tier rate limit reached — try again in a minute")
    if not r.ok:
        raise RuntimeError(f"{who} HTTP {r.status_code}: {r.text[:200]}")


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()   # reasoning models
    if not text:
        raise RuntimeError("model returned an empty answer")
    return text


def chat(system: str, user: str, model: str | None = None, json_mode: bool = False,
         temperature: float = 0.3, timeout: int = 120) -> str:
    model = model or default_model()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if is_openai():
        body = {"model": model, "messages": messages, "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        url = f"{openai_base()}/chat/completions"
        r = requests.post(url, json=body, headers=_headers(), timeout=timeout)
        if json_mode and r.status_code == 400:      # model without JSON mode → plain request
            body.pop("response_format")
            r = requests.post(url, json=body, headers=_headers(), timeout=timeout)
        _raise_for(r)
        choices = r.json().get("choices") or [{}]
        return _clean((choices[0].get("message") or {}).get("content", ""))

    body = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
    if json_mode:
        body["format"] = "json"
    r = requests.post(f"{host()}/api/chat", json=body, headers=_headers(), timeout=timeout)
    _raise_for(r)
    return _clean((r.json().get("message") or {}).get("content", ""))


# ---------------------------------------------------------------- desk view
DESK_SYSTEM = """You are a senior FX & fixed-income sales trader on a corporate broking desk in Accra, Ghana.
Write the DESK VIEW for this morning's client brief, in professional, confident English.
Rules:
- 2 to 3 short sentences, max 70 words.
- DO NOT write any digits or numbers at all (not even "ten-year" written as digits, dates or tenors). The figures are already shown above your text; refer to moves qualitatively (e.g. "firmer dollar", "gold under pressure").
- The facts are already interpreted for you — trust them, do not re-derive cause and effect.
  Magnitude words: "slightly" = small move, "sharply" = unusually large move for that market.
- Open with the cedi's move this morning, in natural words (e.g. "The cedi weakened sharply against the dollar this morning").
- Explain it only with the "drivers that explain the cedi move". Drivers listed as going AGAINST the move must be
  introduced with "despite" or "although" — never say the move is "in line with" or "driven by" them.
- Write client-ready prose: NEVER copy the fact keys, labels, colons, arrows or tickers (no "CEDI TODAY:", no "USD/GHS", no "DXY").
- Focus on what it means for the cedi and for clients: importers (buy USD), exporters (sell USD), investors (T-bills).
- End with one clear, actionable suggestion for clients that is consistent with the cedi's move.
- No disclaimers, no headings, no bullet points."""


def desk_view(facts: dict, headlines: list[str], model: str | None = None) -> str:
    user = ("MARKET FACTS (already interpreted for the cedi):\n" + json.dumps(facts, indent=1, ensure_ascii=False) +
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
Return json only: {"scores": [{"i": <index>, "impact": <int>, "relevance": <int>}, ...]}"""


def score_headlines(headlines: list[str], model: str | None = None) -> list[dict]:
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
