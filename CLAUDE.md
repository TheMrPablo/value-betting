# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# One-time DB initialization (run once at project start)
python scripts/init_db.py

# Load historical data (CSVs must already be in data/historical/)
python scripts/load_historical.py
```

Environment variables are loaded from `.env` (not committed): `ODDS_API_KEY`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.

## Running models

```bash
# Fit Dixon-Coles and predict upcoming fixtures (next 24h)
python -m src.models.dixon_coles --league BL2 --seasons 2223 2324 2425

# Load from saved model (faster, skips fitting)
python -m src.models.dixon_coles --league BL2 --model models/dc_params_BL2.pkl

# Print model summary (attack/defense params per team)
python -m src.models.dixon_coles --league BL2 --model models/dc_params_BL2.pkl --summary

# Skip writing predictions to DB
python -m src.models.dixon_coles --league BL2 --no-persist
```

## Architecture

### Data flow
```
data/historical/*.csv  →  scripts/load_historical.py  →  fixtures table
                                                           odds table (bet365_historical)

External APIs (future)  →  [scrapers/fetchers]  →  fixtures, odds, injuries, source_events

fixtures + odds  →  match_features  →  predictions (dixon_coles / xgboost)
                                     →  signals (value bets with edge%)
                                     →  bets (user decision)
                                     →  results + pnl
```

### SQLite schema (`data/betting.db`)
Core tables and their roles:
- `fixtures` — all matches (historical + upcoming), `status` = `finished` | `scheduled`
- `odds` — bookmaker odds per fixture/market/selection; `bookmaker='bet365_historical'` for CSV data, `'betplay'` for live
- `match_features` — pre-computed XGBoost features (rolling 5-match averages, H2H, rest days, injuries, standings position)
- `predictions` — model outputs; `model` = `'dixon_coles'` or `'xgboost'`; `market` = `'ou25'` | `'1x2'` | `'btts'`
- `signals` — value bets where `edge_pct` exceeds threshold; `profile` = `'conservative'` | `'aggressive'`
- `bets` — user-confirmed bets with actual stake and odds taken
- `results` — settled outcomes with P&L in COP and closing line value (CLV)
- `performance_snapshots` — weekly ROI/Brier score snapshots per league

### Season codes
Historical CSVs from football-data.co.uk use format `{league_code}_{season}.csv`:
- League codes: `D2` (BL2 = 2. Bundesliga), `N1` (NL1 = Eredivisie)
- Season codes: `"2122"`, `"2223"`, `"2324"`, `"2425"` → stored as `"2021-22"` etc. in `fixtures.season`

### Models

**Dixon-Coles** (`src/models/dixon_coles.py`):
- Bivariate Poisson with tau correction for low-scoring results (0-0, 1-0, 0-1, 1-1)
- Temporal decay weight: `exp(-0.0018 * delta_days)` ≈ half-life ~385 days
- Parameters per team: `alpha` (attack), `beta` (defense) + global `gamma` (home advantage), `rho` (tau correction)
- Identifiability constraint: `sum(log_alpha) = 0`
- Serializes to `.pkl` via `model.save()` / `DixonColesModel.load()`
- Entry point for the automated pipeline: `run_predictions()`

**XGBoost** (planned): uses `match_features` table; ensemble weight 40% (vs 60% Dixon-Coles per `config/thresholds.yaml`)

### Betting logic
- **Edge**: `(model_prob - implied_prob) / implied_prob * 100`
- **Stake**: fractional Kelly with multiplier from profile (`conservative`: 0.25×, `aggressive`: 0.125×), capped by `max_stake_pct_bankroll`
- **Ensemble**: `0.6 * dixon_coles_prob + 0.4 * xgboost_prob`
- Bankroll tracked in COP in `config/bankroll.yaml`
- Filters applied before signaling: max overround 8%, min 1h before kickoff, skip first/last 2 matchweeks

### Config files
- `config/leagues.yaml` — active leagues with football-data.co.uk codes and odds-api keys
- `config/thresholds.yaml` — two betting profiles (conservative/aggressive), portfolio split (90/10), ensemble weights
- `config/bankroll.yaml` — current bankroll in COP, monthly profit target, stop-loss rules
