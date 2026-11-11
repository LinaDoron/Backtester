# Kabal — Backtesting SMC Hypotheses in Natural Language

A Telegram bot that transforms a trading hypothesis, written in ordinary text, into Python code and runs it across the history of EURUSD, GBPUSD, and USDCHF.

```
Test SMT on H1 between days, if there is an MSS on H1 — enter by FVG on H1, risk 1%, take 2
```

In response, the bot sends an equity chart and a report: starting and final balance, net profit, maximum drawdown, win rate, and number of trades. The log of all trades is saved to CSV.

## How it works

1. **Telegram** (`bot/`) receives the hypothesis from the administrator.
2. **LLM** (`brain/`) translates the text into a `run_strategy` function in Python. The model is not permitted to write indicators from scratch: it assembles the strategy only from the ready-made functions of the library.
3. **Engine** (`muscle/`) validates the generated code, runs it in an isolated process on M5 quotations, and computes the statistics.

```
bot/handlers.py        hypothesis intake and report to Telegram
brain/prompts.py       system prompt: rules and library API for the model
brain/llm_client.py    request to an OpenAI-compatible API and response parsing
muscle/smc_alphabet.py library of SMC patterns and trade simulation
muscle/time_theory.py  Quarterly Theory markup: weeks, true day, 6-hour and 90-minute quarters
muscle/engine.py       code validation, execution, statistics and equity chart
muscle/sandbox*.py     isolation of the strategy code
download_data.py       download of M5 quotations from Yahoo Finance
```

## What the library can do

Time is everywhere New York time. Timeframes from M5 to D1 are assembled from M5. H4 and D1 open at 17:00, as with forex brokers, and do not shift upon the transition to winter time.

**Time structure (Quarterly Theory):** week (H4), true day 18:00–18:00 (H1), 6-hour quarters (M15), 90-minute quarters (M5).

**SMT** — divergence between EURUSD, GBPUSD, and USDCHF upon the taking of the high or low of the previous quarter. The franc is accounted for inverted. There are level linkages, for example the SMT of the day, then the SMT of the 6-hour quarter in the same direction. A sweep is counted from 1 pip.

**Entry models:**
- MSS (market structure shift) and the FVG of that break
- Order Block, Breaker Block, Rejection Block, Propulsion Block
- iFVG and BPR

**Filters:**
- PSP — divergence of candle colors
- TPD — decorrelation across three candles
- DFR — premium and discount by Defining Range

**Risk management:** risk in percent of the deposit, take in R, forced closure of the trade at the Mean Threshold or CE of the original zone.

## Security

The bot executes code written by an LLM, therefore this code is constrained in several layers:

1. **Validation before execution.** Parsing of the code against a whitelist: the only permitted import is the pattern library. `open`, `eval`, `exec`, `getattr`, names and attributes containing `__`, pandas methods for reading and writing files and executing strings are forbidden. Nested loops are forbidden.
2. **Separate process.** Its environment holds no tokens, it has a curtailed set of built-in functions, a 120-second timeout, and a 2 GB memory limit (Job Object on Windows, `setrlimit` on Linux). The result is returned in JSON format and validated.
3. **bubblewrap bubble on Linux.** The process has no network, and it sees only Python, `venv`, and the library code, all read-only. `.env` and the rest of the project files do not exist inside.

Furthermore, the bot responds only to a single Telegram ID from `.env`.

## Launch

Python 3.12 or newer is required.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # Windows: copy .env.example .env
# fill in .env: bot token, your Telegram ID, LLM API key

python download_data.py           # M5 quotations for the last 60 days
python main.py
```

On a Linux server, additionally:

```bash
sudo apt install bubblewrap
python -m muscle.sandbox_check    # every item must be [OK]
```

## Limitations

- Yahoo Finance provides M5 only for the last 60 days, therefore the history is short. H4 and D1 require longer data.
- Yahoo quotations differ from broker ones by 1–2 pips. On thin sweeps and narrow FVGs this yields a discrepancy with the chart.
- This is a tool for testing hypotheses, not a trading recommendation. 🍷📉
