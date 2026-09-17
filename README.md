# CryptAI

> Experimental cryptocurrency market research and machine-learning project.

## Project status

**Paused experimental project.**

I have not worked on CryptAI for a while. The repository is public as a record of the architecture, experiments and tooling I built around crypto market data, model training and simulated strategy evaluation.

The project is **not production-ready**, is **not a live trading system**, and should not be used with real funds without substantial review, testing and hardening.

Some scripts also still reflect the original development environment and may require path/configuration fixes before they run elsewhere.

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
- news collection experiments;
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

## Data collection

The main collector uses `ccxt` and stores OHLCV datasets locally.

Configured symbols currently include examples such as:

- BTC/USDT
- ETH/USDT
- SOL/USDT
- BNB/USDT
- XRP/USDT

Several timeframes are explored, including resampling experiments.

Generated datasets are intentionally excluded from Git.

## Machine-learning experiments

The codebase contains model experiments based on:

- LSTM networks;
- Transformer-style sequence models;
- MLP baselines.

The current training code treats market direction as a classification problem with three classes:

```text
SELL / HOLD / BUY
```

This is experimental research code, not a validated predictive system.

## Arena / simulation

The arena module was built to evaluate candidate strategies with more realistic constraints than raw prediction accuracy.

It can account for:

- trading fees;
- slippage;
- position sizing;
- maximum drawdown;
- win rate;
- profit factor;
- Sharpe-style metrics;
- equity evolution.

The thresholds committed in `config/` are experimental values, not investment recommendations or validated risk settings.

## Environment variables

Create a local `.env` file from `.env.example`.

```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret_key
GITHUB_TOKEN=your_github_token
DISCORD_WEBHOOK=your_discord_webhook_url
```

The real `.env` file is ignored by Git.

Do not commit API keys, exchange secrets, wallet keys or webhooks.

## Generated files excluded from Git

The repository intentionally ignores runtime and training artifacts such as:

- `.env`
- datasets
- checkpoints
- generated champions
- CSV files
- PyTorch model files
- logs

## Known limitations

The current repository should be considered a snapshot of an experiment rather than a polished application.

Known limitations include:

- development paths are still hard-coded in parts of the codebase;
- installation is not packaged or automated;
- training scripts have not been recently validated end-to-end;
- some code may require fixes before execution;
- no production trading execution layer is provided;
- no guarantee is made regarding profitability or predictive quality;
- test coverage and reproducibility are incomplete.

## Development environment

The project was primarily written in Python and uses libraries around:

- PyTorch
- pandas
- NumPy
- ccxt

GPU acceleration is used when CUDA is available in the training code.

## Why this repository is public

CryptAI is a personal research project around the intersection of **cryptocurrency markets, data engineering, machine learning and strategy simulation**.

It is published to document the experiments and architecture, not as financial software or a turnkey trading bot.

## Disclaimer

This project is for software and research purposes only. It is not financial advice and is not intended as a recommendation to buy, sell or trade any asset.

## Author

**Léo Corvaisier-Palluy (Nerix)**  
GitHub: [leocorv](https://github.com/leocorv)
