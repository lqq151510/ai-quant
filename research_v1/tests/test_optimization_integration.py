import json

import pandas as pd

from quant_research.baseline import _simulate_optimized
from quant_research.cli import main
from quant_research.ledger import Costs
from quant_research import snapshot


def test_close_callback_retries_a_missed_fill_and_only_then_skips():
    days = pd.bdate_range("2024-01-02", periods=5)
    bars = pd.DataFrame({
        "date": days, "symbol": "510300", "kind": "etf",
        "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
        "volume": [100000, 0, 100000, 100000, 100000],
        "amount": [1000000, 0, 1000000, 1000000, 1000000],
        "tradable": [True, False, True, True, True],
        "upper_limit": 11.0, "lower_limit": 9.0,
    })
    decisions = {day: {"allocation_hrp": {"510300": 0.1}} for day in days}
    result, diagnostics = _simulate_optimized(
        bars, days, None, decisions, pd.Series(1.0, index=days), "etf",
        100000, Costs(0, 0, 0), use_band=True, use_breadth=False,
    )

    assert list(pd.to_datetime(result.orders["date"])) == [days[1], days[2]]
    assert list(result.orders["filled_qty"]) == [0, 1000]
    assert diagnostics == {
        "scheduled_count": 4, "skipped_count": 2, "defensive_count": 0, "unknown_breadth_count": 0,
    }


def test_stock_optimization_audits_its_etf_breadth_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(snapshot, "read_snapshot", lambda _: ({}, {}))
    calls = []

    def audit(manifest, frames, period, kind):
        calls.append(kind)
        return {"kind": kind, "status": "ready" if kind == "stock" else "data_insufficient"}

    monkeypatch.setattr(snapshot, "audit_snapshot", audit)
    out = tmp_path / "result"
    assert main([
        "baseline", str(tmp_path / "snapshot"), "--kind", "stock", "--end", "2024-12-31",
        "--strategy-optimization", "--out", str(out),
    ]) == 2
    assert calls == ["etf", "stock"]
    assert json.loads(capsys.readouterr().out)[0] == {"kind": "etf", "status": "data_insufficient"}
    assert not out.exists()
