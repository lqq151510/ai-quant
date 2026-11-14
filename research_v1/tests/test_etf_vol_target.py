import numpy as np
import pandas as pd
import pytest

from quant_research.etf_vol_target import TARGET_VOL, build_schedules
from quant_research.protocol import ETF_SYMBOLS


def test_first_signal_uses_known_complete_history_and_executes_next_session():
    calendar = pd.bdate_range("2022-01-03", periods=125)
    days = calendar[59:]
    returns = pd.DataFrame({symbol: np.tile([-0.02, 0.02], 63)[:len(calendar)] for symbol in ETF_SYMBOLS},
                           index=calendar)
    schedules, evidence = build_schedules(returns, days)
    future_changed = returns.copy()
    future_changed.loc[calendar[60]:] *= 3
    changed, _ = build_schedules(future_changed, days)
    first_open = calendar[60]

    assert schedules["vol_target"][first_open] == changed["vol_target"][first_open]
    assert evidence[0]["signal_date"] == calendar[59].date().isoformat()
    assert list(schedules["vol_target"]) == list(schedules["equal_full"]) == list(schedules["equal_60"])
    assert sum(schedules["vol_target"][first_open].values()) < 1
    assert evidence[0]["projected_annual_volatility"] == pytest.approx(TARGET_VOL)

    returns.loc[calendar[59], ETF_SYMBOLS[0]] = np.nan
    with pytest.raises(ValueError, match="cannot estimate basket volatility"):
        build_schedules(returns, days)
