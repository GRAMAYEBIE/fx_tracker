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
