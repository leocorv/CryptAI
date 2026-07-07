#!/usr/bin/env python3
"""CryptAI Feature Engineering — ALL 5 skills intégrées comme features"""
import numpy as np
import pandas as pd

# ─── Skill 1: Trading Signals (Z-score, Régression) ───
def z_score(series, window=20):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / (s + 1e-8)

def lunacy_gauge(signals):
    return signals.apply(lambda c: z_score(c)).mean(axis=1)

def bollinger_bands(price, window=20, n_std=2.0):
    ma = price.rolling(window).mean()
    std = price.rolling(window).std()
    return ma + n_std * std, ma, ma - n_std * std

def regression_channels(price, window=50, n_std=2.5):
    x = np.arange(window)
    upper, mid, lower = [], [], []
    for i in range(window, len(price)+1):
        y = price.iloc[i-window:i].values
        A = np.vstack([x, np.ones(window)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        trend = slope * x + intercept
        se = np.std(y - trend)
        upper.append(trend[-1] + n_std * se)
        mid.append(trend[-1])
        lower.append(trend[-1] - n_std * se)
    idx = price.index[window-1:]
    return pd.Series(upper, index=idx), pd.Series(mid, index=idx), pd.Series(lower, index=idx)

# ─── Skill 2: On-Chain Metrics (Bitcoin) ───
def onchain_metrics(price, volume):
    """Approximation on-chain. Données Glassnode idéales, sinon approximate avec volume"""
    f = pd.DataFrame(index=price.index)
    f["issuance_approx"] = volume.rolling(min(144, len(volume))).mean()  # proxy issuance
    f["tradable_supply_ratio"] = volume / volume.rolling(min(144, len(volume))).mean()
    f["rcap_pow_approx"] = np.sqrt((price * volume.rolling(min(144, len(volume))).mean()))  # geometric mean proxy
    return f

# ─── Skill 3: Macro Liquidity ───
def macro_features(price):
    """features de régime macro. Idéal: M2, taux réels, DXY. Ici: proxy via prix"""
    f = pd.DataFrame(index=price.index)
    f["btc_beta"] = price.pct_change().rolling(30).std() * np.sqrt(365)
    f["risk_regime"] = z_score(price.pct_change(20), 60)
    f["momentum_regime"] = (price.rolling(50).mean() / price.rolling(200).mean()) - 1
    return f

# ─── Skill 4: Forecasting (ARIMA/ETS residuals) ───
def forecast_residuals(price, window=50):
    """Résidus de prévision comme features de détection de régime"""
    x = np.arange(window)
    resid, resid_z = [], []
    for i in range(window, len(price)+1):
        y = price.iloc[i-window:i].values
        A = np.vstack([x, np.ones(window)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = slope * (window-1) + intercept
        r = y[-1] - pred
        resid.append(r)
    r_series = pd.Series(resid, index=price.index[window-1:])
    z = (r_series - r_series.rolling(100).mean()) / (r_series.rolling(100).std() + 1e-8)
    return r_series, z

# ─── Skill 5: Market Correlations ───
def cross_correlation_features(price_dict):
    """Matrice de corrélation glissante entre symbols. price_dict = {sym: close_series}"""
    df = pd.DataFrame(price_dict)
    returns = df.pct_change()
    corr = returns.rolling(90).corr()
    # Feature: average pairwise correlation
    n = len(df.columns)
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    # Pour chaque pas de temps, moyenne des corrélations
    avg_corr = []
    for idx in corr.index.levels[0]:
        m = corr.loc[idx].values
        vals = [m[i,j] for i,j in pairs if i < m.shape[0] and j < m.shape[1]]
        avg_corr.append(np.mean(vals) if vals else 0)
    return pd.Series(avg_corr, index=corr.index.levels[0], name="avg_corr")

# ─── Build Master Feature Matrix ───
def build_features(df, price_dict=None):
    """ALL 5 skills → feature matrix"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    f = pd.DataFrame(index=df.index)
    
    # Core price features
    f["close"] = close
    f["returns_1"] = close.pct_change(1)
    f["returns_5"] = close.pct_change(5)
    f["returns_20"] = close.pct_change(20)
    
    # Skill 1: Trading Signals
    f["z_score_20"] = z_score(close, 20)
    f["z_score_50"] = z_score(close, 50)
    f["z_score_100"] = z_score(close, 100)
    f["rsi_14"] = 100 - (100 / (1 + close.diff().clip(lower=0).rolling(14).mean() /
                                (-close.diff().clip(upper=0).rolling(14).mean() + 1e-8)))
    f["bb_upper"], f["bb_mid"], f["bb_lower"] = bollinger_bands(close, 20)
    f["bb_width"] = (f["bb_upper"] - f["bb_lower"]) / f["bb_mid"]
    f["volume_ratio"] = volume / volume.rolling(20).mean()
    f["high_low_pct"] = (high - low) / close
    
    # Skill 2: On-Chain (approximated)
    oc = onchain_metrics(close, volume)
    for col in oc.columns:
        f[col] = oc[col]
    
    # Skill 3: Macro regime
    macro = macro_features(close)
    for col in macro.columns:
        f[col] = macro[col]
    
    # Skill 4: Forecast residuals
    res, res_z = forecast_residuals(close, 50)
    f["forecast_residual"] = res
    f["forecast_z"] = res_z
    
    # Skill 5: Cross-correlations (if multi-symbol)
    if price_dict and len(price_dict) > 1:
        cc = cross_correlation_features(price_dict)
        f["market_corr"] = cc
    
    # Trend features
    f["sma_50"] = close.rolling(50).mean()
    f["sma_200"] = close.rolling(200).mean()
    f["trend_strength"] = (f["sma_50"] - f["sma_200"]) / close
    
    return f.dropna()

def normalize(features, fit=None):
    if fit is None:
        fit = {c: {"m": features[c].mean(), "s": features[c].std()} for c in features.columns}
    n = features.copy()
    for c in features.columns:
        n[c] = (features[c] - fit[c]["m"]) / (fit[c]["s"] + 1e-8)
    return n, fit
def fear_greed_feature(df, fng_value=27.0):
    """Add Fear & Greed Index as a feature column"""
    f = pd.DataFrame(index=df.index)
    f["fear_greed_index"] = fng_value
    f["fear_greed_regime"] = 0  # 0=Fear, 1=Neutral, 2=Greed
    if fng_value < 25:
        f["fear_greed_regime"] = 0  # Extreme Fear
    elif fng_value < 45:
        f["fear_greed_regime"] = 1  # Fear
    elif fng_value < 55:
        f["fear_greed_regime"] = 2  # Neutral
    elif fng_value < 75:
        f["fear_greed_regime"] = 3  # Greed
    else:
        f["fear_greed_regime"] = 4  # Extreme Greed
    return f
