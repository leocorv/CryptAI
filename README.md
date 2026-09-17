# CryptAI

> Experimental cryptocurrency market research and machine-learning project.

## Project status

**Paused experimental project.**

I have not worked on CryptAI for a while. The repository is public as a record of the architecture, experiments and tooling I built around crypto market data, model training and simulated strategy evaluation.

The project is **not production-ready**, is **not a live trading system**, and should not be used with real funds without substantial review, testing and hardening.

The core scripts have been cleaned up so they can run from the repository directory instead of depending on the original server path, but the complete pipeline has **not recently been validated end-to-end**.

## What the project explores

CryptAI was designed around several ideas:

- collecting cryptocurrency market data;
- building ML-ready features from OHLCV data;
- training classification models for market direction;
- comparing multiple model families;
- evaluating candidate strategies in a simulated arena;
- incorporating fees, slippage and drawdown constraints;
- keeping generated data, checkpoints and runtime artifacts outside Git.

## Current components

The repository currently includes:

- Binance market-data collection through `ccxt`;
- additional Hyperliquid-oriented collection code;
- market-context collection experiments;
- feature engineering utilities;
- LSTM, Transformer and MLP model definitions;
- a PyTorch training runner;
- an arena/backtesting layer;
- configuration files for symbols, fees and thresholds.

## Architecture

```text
Market data sources
        |
        +--> Binance / Hyperliquid collectors
        |
        +--> Local datasets
                 |
                 +--> Feature engineering
                           |
                           +--> ML models
                           |     ├── LSTM
                           |     ├── Transformer
                           |     └── MLP
                           |
                           +--> Training
                                   |
                                   +--> Candidate model
                                            |
                                            +--> Arena / simulation
                                                   |
                                                   +--> fees
                                                   +--> slippage
                                                   +--> drawdown
                                                   +--> performance metrics
```

## Quick start

### Requirements

- Python 3.10+
- PyTorch
- pandas
- NumPy
- ccxt

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

By default, scripts use the repository root as their working data directory.

To use another location, define:

```bash
export CRYPTAI_HOME=/path/to/CryptAI
```

On PowerShell:

```powershell
$env:CRYPTAI_HOME = "C:\path\to\CryptAI"
```

## Environment variables

A local `.env` can be created from `.env.example`.

```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret_key
```

The Binance collector only uses public OHLCV endpoints, so these credentials are optional for the current collection workflow.

The real `.env` file is ignored by Git. Do not commit API keys, exchange secrets, wallet keys or webhooks.

## Data collection

Run the Binance collector:

```bash
python scripts/collector.py
```

Run the Hyperliquid collector:

```bash
python scripts/hyperliquid_collector.py
```

The collectors verify the timeframes advertised by the exchange through CCXT. Unsupported experimental timeframes are skipped rather than crashing the whole collection run.

Generated datasets are stored under `data/` and intentionally excluded from Git.

## Training

Run the default LSTM experiment with:

```bash
python scripts/train.py
```

The current training pipeline:

- builds engineered features from the latest local datasets;
- aligns labels to the post-window feature index;
- uses chronological train/validation data per market;
- fits normalization parameters using training data only;
- selects CUDA automatically when available;
- stores generated checkpoints under `arena/champions/`.

The direction classification uses three classes:

```text
SELL / HOLD / BUY
```

This remains experimental research code, not a validated predictive system.

## Arena / simulation

The arena module was built to evaluate candidate strategies with more realistic constraints than raw prediction accuracy.

It can account for:

- trading fees;
- entry and exit slippage;
- position sizing;
- maximum drawdown;
- win rate;
- profit factor;
- Sharpe-style metrics;
- mark-to-market equity evolution.

The thresholds committed in `config/` are experimental values, not investment recommendations or validated risk settings.

## Repository layout

```text
config/     Experimental configuration
lib/        Reusable features, models and arena code
scripts/    Main executable scripts
src/        Legacy compatibility entrypoints / earlier layout
data/       Generated datasets, ignored by Git
arena/      Generated candidate artifacts, ignored by Git
```

## Known limitations

The current repository should be considered a snapshot of an experiment rather than a polished application.

Known limitations include:

- the full workflow has not recently been validated end-to-end;
- model quality and profitability are not established;
- some experimental data sources or exchange timeframes may no longer be available;
- no production trading execution layer is provided;
- test coverage and reproducibility are incomplete;
- several research modules still need consolidation and cleanup.

## Generated files excluded from Git

The repository intentionally ignores runtime and training artifacts such as:

- `.env`
- datasets
- news/cache output
- checkpoints
- generated champions
- CSV files
- PyTorch model files
- logs

## Why this repository is public

CryptAI is a personal research project around the intersection of **cryptocurrency markets, data engineering, machine learning and strategy simulation**.

It is published to document the experiments and architecture, not as financial software or a turnkey trading bot.

## Disclaimer

This project is for software and research purposes only. It is not financial advice and is not intended as a recommendation to buy, sell or trade any asset.

## Author

**Léo Corvaisier-Palluy (Nerix)**  
GitHub: [leocorv](https://github.com/leocorv)
