#!/usr/bin/env python3
"""CryptAI feature engineering utilities."""

import numpy as np
import pandas as pd


def z_score(series, window=20):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-8)


def lunacy_gauge(signals):
    return signals.apply(lambda column: z_score(column)).mean(axis=1)


def bollinger_bands(price, window=20, n_std=2.0):
    mean = price.rolling(window).mean()
    std = price.rolling(window).std()
    return mean + n_std * std, mean, mean - n_std * std


def regression_channels(price, window=50, n_std=2.5):
    x = np.arange(window)
    upper, mid, lower = [], [], []

    for i in range(window, len(price) + 1):
        y = price.iloc[i - window:i].values
        matrix = np.vstack([x, np.ones(window)]).T
        slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
        trend = slope * x + intercept
        std_error = np.std(y - trend)
        upper.append(trend[-1] + n_std * std_error)
        mid.append(trend[-1])
        lower.append(trend[-1] - n_std * std_error)

    idx = price.index[window - 1:]
    return (
        pd.Series(upper, index=idx),
        pd.Series(mid, index=idx),
        pd.Series(lower, index=idx),
    )


def onchain_metrics(price, volume):
    """Experimental volume-based proxies for unavailable on-chain data.

    The original implementation required 4320 observations before one proxy became
    valid. Collectors generally retrieve far fewer rows, which made the complete
    feature matrix empty after ``dropna``. ``min_periods`` keeps the intended long
    rolling windows while allowing experiments on smaller snapshots.
    """
    features = pd.DataFrame(index=price.index)

    short_volume = volume.rolling(144, min_periods=20).mean()
    long_volume = volume.rolling(30 * 144, min_periods=30).mean()

    features["issuance_approx"] = short_volume
    features["tradable_supply_ratio"] = volume / (long_volume + 1e-8)
    features["rcap_pow_approx"] = np.sqrt((price * short_volume).clip(lower=0))
    return features


def macro_features(price):
    """Experimental regime proxies derived from price only."""
    features = pd.DataFrame(index=price.index)
    features["btc_beta"] = price.pct_change().rolling(30).std() * np.sqrt(365)
    features["risk_regime"] = z_score(price.pct_change(20), 60)
    features["momentum_regime"] = (price.rolling(50).mean() / price.rolling(200).mean()) - 1
    return features


def forecast_residuals(price, window=50):
    """Linear-trend residuals used as experimental regime features."""
    x = np.arange(window)
    residuals = []

    for i in range(window, len(price) + 1):
        y = price.iloc[i - window:i].values
        matrix = np.vstack([x, np.ones(window)]).T
        slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
        prediction = slope * (window - 1) + intercept
        residuals.append(y[-1] - prediction)

    residual_series = pd.Series(residuals, index=price.index[window - 1:])
    residual_z = (
        residual_series - residual_series.rolling(100).mean()
    ) / (residual_series.rolling(100).std() + 1e-8)
    return residual_series, residual_z


def cross_correlation_features(price_dict):
    """Return average rolling pairwise correlation for multiple symbols."""
    frame = pd.DataFrame(price_dict)
    returns = frame.pct_change()
    corr = returns.rolling(90).corr()

    n_columns = len(frame.columns)
    pairs = [(i, j) for i in range(n_columns) for j in range(i + 1, n_columns)]
    values = []
    indices = []

    for idx in returns.index:
        try:
            matrix = corr.loc[idx].values
        except KeyError:
            continue

        pair_values = [
            matrix[i, j]
            for i, j in pairs
            if i < matrix.shape[0] and j < matrix.shape[1] and np.isfinite(matrix[i, j])
        ]
        values.append(float(np.mean(pair_values)) if pair_values else np.nan)
        indices.append(idx)

    return pd.Series(values, index=indices, name="avg_corr")


def build_features(df, price_dict=None):
    """Build the experimental master feature matrix."""
    required = {"close", "high", "low", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    features = pd.DataFrame(index=df.index)

    # Core price features
    features["close"] = close
    features["returns_1"] = close.pct_change(1)
    features["returns_5"] = close.pct_change(5)
    features["returns_20"] = close.pct_change(20)

    # Trading signals
    features["z_score_20"] = z_score(close, 20)
    features["z_score_50"] = z_score(close, 50)
    features["z_score_100"] = z_score(close, 100)
    gains = close.diff().clip(lower=0).rolling(14).mean()
    losses = -close.diff().clip(upper=0).rolling(14).mean()
    features["rsi_14"] = 100 - (100 / (1 + gains / (losses + 1e-8)))
    features["bb_upper"], features["bb_mid"], features["bb_lower"] = bollinger_bands(close, 20)
    features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / (features["bb_mid"] + 1e-8)
    features["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-8)
    features["high_low_pct"] = (high - low) / (close + 1e-8)

    # Experimental on-chain proxies
    onchain = onchain_metrics(close, volume)
    for column in onchain.columns:
        features[column] = onchain[column]

    # Macro/regime proxies
    macro = macro_features(close)
    for column in macro.columns:
        features[column] = macro[column]

    # Forecast residuals
    residual, residual_z = forecast_residuals(close, 50)
    features["forecast_residual"] = residual
    features["forecast_z"] = residual_z

    if price_dict and len(price_dict) > 1:
        features["market_corr"] = cross_correlation_features(price_dict)

    # Trend features
    features["sma_50"] = close.rolling(50).mean()
    features["sma_200"] = close.rolling(200).mean()
    features["trend_strength"] = (features["sma_50"] - features["sma_200"]) / (close + 1e-8)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna()


def normalize(features, fit=None):
    if fit is None:
        fit = {
            column: {"m": features[column].mean(), "s": features[column].std()}
            for column in features.columns
        }

    normalized = features.copy()
    for column in features.columns:
        normalized[column] = (
            features[column] - fit[column]["m"]
        ) / (fit[column]["s"] + 1e-8)
    return normalized, fit
