"""Independent, paper-only futures research; no live-broker integration."""

from .engine import Bar, Contract, FuturesError, MovingAverage, RidgeForecast, backtest, read_bars

__all__ = ["Bar", "Contract", "FuturesError", "MovingAverage", "RidgeForecast", "backtest", "read_bars"]
