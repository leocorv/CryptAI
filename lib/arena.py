#!/usr/bin/env python3
"""CryptAI arena: lightweight strategy simulation with fees and slippage."""

import numpy as np


class ArenaTrade:
    def __init__(self, symbol, entry_price, direction, size, fee_rate=0.001, slippage=0.0005):
        self.symbol = symbol
        self.entry_price = entry_price
        self.direction = direction  # 1=long, -1=short
        self.size = size
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.exit_price = None
        self.pnl = 0.0

    def close(self, exit_price):
        # Long exits suffer negative slippage; short covers suffer positive slippage.
        slip = exit_price * self.slippage * self.direction
        self.exit_price = exit_price - slip
        gross_pnl = (self.exit_price - self.entry_price) * self.direction * self.size
        fees = (self.entry_price * self.size * self.fee_rate) + (self.exit_price * self.size * self.fee_rate)
        self.pnl = gross_pnl - fees
        return self.pnl


class Arena:
    def __init__(self, initial_balance=1000.0, fee_rate=0.001, slippage=0.0005,
                 max_position_pct=0.1, max_drawdown=0.20):
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if not 0 < max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        if not 0 <= max_drawdown < 1:
            raise ValueError("max_drawdown must be in [0, 1)")

        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.equity = float(initial_balance)
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage)
        self.max_position_pct = float(max_position_pct)
        self.max_drawdown = float(max_drawdown)
        self.peak_balance = float(initial_balance)
        self.trades = []
        self.equity_curve = [float(initial_balance)]
        self.position = None

    def can_trade(self):
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        return drawdown < self.max_drawdown

    def open_trade(self, symbol, price, direction, confidence=1.0):
        if not self.can_trade() or self.position is not None:
            return False
        if price <= 0 or direction not in {-1, 1}:
            return False

        confidence = float(np.clip(confidence, 0.0, 1.0))
        if confidence == 0:
            return False

        # Apply adverse entry slippage as well as exit slippage.
        execution_price = price * (1 + self.slippage * direction)
        size = (self.balance * self.max_position_pct * confidence) / execution_price
        self.position = ArenaTrade(
            symbol,
            execution_price,
            direction,
            size,
            self.fee_rate,
            self.slippage,
        )
        self.trades.append(self.position)
        return True

    def close_trade(self, price):
        if self.position is None:
            return 0.0

        pnl = self.position.close(price)
        self.balance += pnl
        self.position = None
        self.peak_balance = max(self.peak_balance, self.balance)
        self.equity = self.balance
        return pnl

    def mark_to_market(self, price):
        if self.position is None:
            self.equity = self.balance
        else:
            unrealized = (
                (price - self.position.entry_price)
                * self.position.direction
                * self.position.size
            )
            self.equity = self.balance + unrealized

        self.equity_curve.append(float(self.equity))
        return self.equity

    def get_metrics(self):
        if not self.trades:
            return {
                "total_return": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe": 0,
                "win_rate": 0,
                "trade_count": 0,
                "pnl_by_symbol": {},
            }

        pnls = [trade.pnl for trade in self.trades]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl <= 0]
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        profit_factor = sum(wins) / (abs(sum(losses)) + 1e-8)

        equity = np.asarray(self.equity_curve, dtype=np.float64)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / np.maximum(peaks, 1e-8)
        max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0

        returns = np.diff(equity) / np.maximum(equity[:-1], 1e-8)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))
        else:
            sharpe = 0.0

        by_symbol = {}
        for trade in self.trades:
            by_symbol.setdefault(trade.symbol, []).append(trade.pnl)

        return {
            "total_return": round(total_return, 4),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe": round(sharpe, 2),
            "win_rate": round(len(wins) / len(pnls), 3),
            "trade_count": len(self.trades),
            "pnl_by_symbol": {
                symbol: round(sum(symbol_pnls), 2)
                for symbol, symbol_pnls in by_symbol.items()
            },
        }


def run_arena(champion_func, data, arena_config=None):
    """Run a strategy callback through a chronological market-data frame."""
    if arena_config is None:
        arena_config = {"initial_balance": 1000, "duration_steps": 1000}

    if data is None or data.empty:
        raise ValueError("data must contain at least one row")
    if "close" not in data.columns:
        raise ValueError("data must contain a 'close' column")

    arena = Arena(
        initial_balance=arena_config.get("initial_balance", 1000),
        fee_rate=arena_config.get("fee_rate", 0.001),
        slippage=arena_config.get("slippage", 0.0005),
        max_position_pct=arena_config.get("max_position_pct", 0.1),
        max_drawdown=arena_config.get("max_drawdown", 0.20),
    )

    last_price = None
    for step, (_, row) in enumerate(data.iterrows()):
        if step >= arena_config.get("duration_steps", 1000):
            break

        price = float(row["close"])
        if not np.isfinite(price) or price <= 0:
            continue

        last_price = price
        signal = champion_func(row)

        if signal == "buy" and arena.position is None:
            arena.open_trade("ARENA", price, 1)
        elif signal == "sell" and arena.position is not None and arena.position.direction == 1:
            arena.close_trade(price)

        arena.mark_to_market(price)

    if last_price is None:
        raise ValueError("data did not contain a usable positive close price")

    if arena.position is not None:
        arena.close_trade(last_price)
        arena.mark_to_market(last_price)

    return arena.get_metrics()


if __name__ == "__main__":
    print("[arena] CryptAI Arena module loaded")
