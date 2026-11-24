# FX Market Tracker — Ghana desk edition

Live cedi dashboard + automatic morning brief, news scanner and daily USD/GHS bias.

## Setup (once, on your own laptop)
```bash
pip install -r requirements.txt
# optional but recommended — local AI for desk view & news scoring
#   1. install Ollama: https://ollama.com
#   2. ollama pull qwen2.5:7b        (use qwen2.5:3b on a small laptop)
```

## Run
```bash
ollama serve            # only if Ollama is not already running
streamlit run app.py
```

## Deploy online (Streamlit Community Cloud + Groq, free tier)
Local Ollama is not available on Streamlit Cloud, so online the app uses any
OpenAI-compatible API — Groq by default. Create a key at https://console.groq.com/keys,
then in the app's **Settings → Secrets** add:
```toml
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY  = "gsk_..."
LLM_MODEL    = "openai/gpt-oss-20b"   # any id from https://api.groq.com/openai/v1/models
```
Without `LLM_BASE_URL` the app uses local Ollama (`localhost:11434`), or Ollama Cloud if
`OLLAMA_HOST` / `OLLAMA_API_KEY` / `OLLAMA_MODEL` are set (paid). It falls back to
rule-based text if no AI is reachable.

## Files
| File | Role |
|---|---|
| `app.py` | Streamlit dashboard, morning brief, auto-logging |
| `news.py` | Google News RSS (Ghana + global), headline scoring, news log |
| `forecast.py` | Walk-forward logistic model + news adjustment |
| `llm.py` | Qwen via Ollama — desk view (no numbers) & headline scoring |
| `data/market_log.csv` | one row per day (auto) — your own history |
| `data/news_log.csv` | every scored headline (auto) — future training data |

## Rules of the tool
- Figures always come from code/APIs, never from the LLM (the LLM's text is rejected if it contains digits).
- Forecasts are probabilities with their measured accuracy — not certainties.
- Public data only. Never load Spectra or client data.
