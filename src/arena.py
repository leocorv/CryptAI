#!/usr/bin/env python3
"""CryptAI Arena — CPU simulation with realistic fees, slippage, drawdown"""
import os, sys, json, time, random
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

BASE = "/mnt/hive_storage/CryptAI"
sys.path.insert(0, f"{BASE}/lib")

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
        slip = exit_price * self.slippage * self.direction
        self.exit_price = exit_price - slip
        gross_pnl = (self.exit_price - self.entry_price) * self.direction * self.size
        fees = (self.entry_price * self.size * self.fee_rate) + (self.exit_price * self.size * self.fee_rate)
        self.pnl = gross_pnl - fees
        return self.pnl

class Arena:
    def __init__(self, initial_balance=1000.0, fee_rate=0.001, slippage=0.0005,
                 max_position_pct=0.1, max_drawdown=0.20):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        self.max_drawdown = max_drawdown
        self.peak_balance = initial_balance
        self.trades = []
        self.equity_curve = [initial_balance]
        self.position = None
    
    def can_trade(self):
        dd = (self.peak_balance - self.balance) / self.peak_balance
        return dd < self.max_drawdown
    
    def open_trade(self, symbol, price, direction, confidence=1.0):
        if not self.can_trade():
            return False
        if self.position is not None:
            return False
        size = (self.balance * self.max_position_pct * confidence) / price
        self.position = ArenaTrade(symbol, price, direction, size, self.fee_rate, self.slippage)
        self.trades.append(self.position)
        return True
    
    def close_trade(self, price):
        if self.position is None:
            return 0.0
        pnl = self.position.close(price)
        self.balance += pnl
        self.position = None
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self.equity = self.balance
        self.equity_curve.append(self.equity)
        return pnl
    
    def get_metrics(self):
        if not self.trades:
            return {"total_return": 0, "profit_factor": 0, "max_drawdown": 0,
                    "sharpe": 0, "win_rate": 0, "trade_count": 0, "pnl_by_symbol": {}}
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        profit_factor = sum(wins) / (abs(sum(losses)) + 1e-8)
        
        eq_arr = np.array(self.equity_curve)
        dd = (np.maximum.accumulate(eq_arr) - eq_arr) / np.maximum.accumulate(eq_arr)
        max_dd = dd.max() if len(dd) > 0 else 0
        
        returns = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252) if len(returns) > 0 else 0
        
        by_sym = {}
        for t in self.trades:
            by_sym.setdefault(t.symbol, []).append(t.pnl)
        
        return {
            "total_return": round(total_return, 4),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_dd, 4),
            "sharpe": round(sharpe, 2),
            "win_rate": round(len(wins) / len(pnls), 3),
            "trade_count": len(self.trades),
            "pnl_by_symbol": {s: round(sum(p), 2) for s, p in by_sym.items()}
        }

def run_arena(champion_func, data, arena_config=None):
    """Run a champion through the arena for evaluation"""
    if arena_config is None:
        arena_config = {"initial_balance": 1000, "duration_steps": 1000}
    arena = Arena(
        initial_balance=arena_config.get("initial_balance", 1000),
        fee_rate=arena_config.get("fee_rate", 0.001),
        slippage=arena_config.get("slippage", 0.0005),
        max_position_pct=arena_config.get("max_position_pct", 0.1),
        max_drawdown=arena_config.get("max_drawdown", 0.20)
    )
    
    for step, (idx, row) in enumerate(data.iterrows()):
        if step >= arena_config.get("duration_steps", 1000):
            break
        
        signal = champion_func(row)
        
        if signal == "buy" and arena.position is None:
            arena.open_trade("ARENA", row["close"], 1)
        elif signal == "sell":
            if arena.position is not None and arena.position.direction == 1:
                arena.close_trade(row["close"])
        
        # Update equity for open positions
        if arena.position is not None:
            unrealized = (row["close"] - arena.position.entry_price) * arena.position.direction * arena.position.size
            arena.equity = arena.balance + unrealized
            arena.equity_curve[-1] = arena.equity
    
    # Close any remaining position
    if arena.position is not None:
        arena.close_trade(data.iloc[-1]["close"])
    
    return arena.get_metrics()

if __name__ == "__main__":
    print("[arena] CryptAI Arena module loaded")