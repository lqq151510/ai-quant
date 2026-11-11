from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_research.futures.engine import Bar, Contract, FuturesError, MovingAverage, RidgeForecast, backtest, read_bars
from quant_research.futures.output import sha256
from quant_research.futures.paper import apply_event, initial_state, run_session
from quant_research.futures.probe import capture_sina
from quant_research.futures.tqfeed import append_settlement, completed_bar, quote_event, run_feed


SYMBOL = "SHFE.ag2701"


def spec(**changes):
    values = {"symbol": SYMBOL, "multiplier": 10, "tick_size": 1, "margin_rate": "0.10",
              "max_margin_fraction": "0.5", "fee_open": 2, "fee_close": 3,
              "slippage_ticks": 1, "max_lots": 1, "exit_by": "2026-01-20"}
    return Contract.from_mapping(values | changes)


def daily(contract, day, opening, closing, settlement=None, **changes):
    raw = {"trading_day": day, "symbol": contract.symbol, "open": opening,
           "high": max(opening, closing) + 1, "low": min(opening, closing) - 1,
           "close": closing, "settlement": settlement if settlement is not None else closing,
           "upper_limit": 300, "lower_limit": 1} | changes
    return Bar.from_mapping(raw, contract)


class Script:
    def __init__(self, targets):
        self.targets = targets

    def target(self, history, position):
        return self.targets.get(history[-1].trading_day.isoformat())


def event(kind, minute, **values):
    base = {"type": kind, "time": f"2026-01-02T09:{minute:02d}:00+08:00",
            "symbol": SYMBOL, "trading_day": "2026-01-02"}
    if kind == "quote":
        base.update(last=100, bid=99, ask=101, bid_size=10, ask_size=10, lower_limit=50, upper_limit=200)
    elif kind == "bar":
        base.update(bar_end=base["time"], close=100)
    elif kind == "settlement":
        base["price"] = 100
    return base | values


def feed_to_signal(state, contract, strategy):
    for minute, close in enumerate((100, 101, 102)):
        apply_event(state, event("quote", minute, last=close, bid=close - 1, ask=close + 1), contract, strategy)
        apply_event(state, event("bar", minute, close=close), contract, strategy)


def test_daily_long_short_gap_settlement_and_fees_reconcile():
    contract = spec()
    bars = [daily(contract, "2026-01-02", 100, 100), daily(contract, "2026-01-05", 110, 115),
            daily(contract, "2026-01-06", 105, 100)]
    result = backtest(bars, contract, Script({"2026-01-02": 1, "2026-01-05": -1}), capital=10000)
    assert [x["equity"] for x in result.equity] == ["10000.00", "10038.00", "9963.00"]
    assert [x["position"] for x in result.equity] == [0, 1, -1]
    assert (result.equity[2]["gap_pnl"], result.equity[2]["intraday_pnl"]) == ("-100.00", "50.00")
    assert (result.orders[1]["closed_lots"], result.orders[1]["opened_lots"]) == (1, 1)
    assert (result.orders[1]["fee"], result.orders[1]["slippage"]) == ("5.00", "20.00")


def test_future_price_cannot_affect_previous_signals_or_executions():
    contract = spec(fee_open=0, fee_close=0, slippage_ticks=0)
    bars = [daily(contract, f"2026-01-{day:02d}", price, price) for day, price in
            ((2, 100), (3, 101), (4, 102), (5, 103), (6, 99))]
    first = backtest(bars, contract, MovingAverage(2, 3), capital=10000)
    changed = backtest(bars[:-1] + [daily(contract, "2026-01-06", 80, 80)], contract,
                       MovingAverage(2, 3), capital=10000)
    assert first.equity[:-1] == changed.equity[:-1]
    assert first.decisions[:-1] == changed.decisions[:-1]
    assert first.orders[0]["decision_day"] == "2026-01-04"
    assert first.orders[0]["trading_day"] == "2026-01-05"


def test_margin_blocks_opening_and_reversal_may_only_close():
    contract = spec(margin_rate="0.5", max_margin_fraction="1", fee_open=0, fee_close=0, slippage_ticks=0)
    bars = [daily(contract, "2026-01-02", 100, 100), daily(contract, "2026-01-03", 100, 100)]
    denied = backtest(bars, contract, Script({"2026-01-02": 1}), capital=100)
    assert denied.orders[0]["reason"] == "insufficient_margin_or_risk_limit"
    assert denied.equity[-1]["position"] == 0

    contract = spec(margin_rate="0.5", max_margin_fraction="0.5", fee_open=0, fee_close=3, slippage_ticks=0)
    bars.append(daily(contract, "2026-01-04", 90, 90))
    partial = backtest(bars, contract, Script({"2026-01-02": 1, "2026-01-03": -1}), capital=1000)
    assert (partial.orders[-1]["status"], partial.orders[-1]["filled_delta"], partial.equity[-1]["position"]) == ("partial", -1, 0)


def test_expiry_and_price_limits_do_not_invent_a_fill():
    contract = spec(exit_by="2026-01-04", slippage_ticks=0)
    bars = [daily(contract, "2026-01-02", 100, 100), daily(contract, "2026-01-03", 100, 100),
            daily(contract, "2026-01-04", 80, 80, low=80, lower_limit=80)]
    result = backtest(bars, contract, Script({"2026-01-02": 1}), capital=10000)
    assert result.status == "exit_unresolved"
    assert result.orders[-1]["trigger"] == "exit_by"
    assert result.orders[-1]["reason"] == "price_limit_or_slippage"
    assert result.equity[-1]["position"] == 1


def test_bad_contract_and_missing_settlement_are_rejected(tmp_path):
    with pytest.raises(FuturesError, match="dated contract"):
        spec(symbol="SHFE.ag9999")
    data = tmp_path / "bars.csv"
    data.write_text("trading_day,symbol,open,high,low,close,upper_limit,lower_limit\n"
                    f"2026-01-02,{SYMBOL},100,101,99,100,200,50\n")
    with pytest.raises(FuturesError, match="settlement"):
        read_bars(data, spec())


def test_paper_next_quote_and_settlement_reconcile():
    contract, strategy = spec(), MovingAverage(2, 3)
    state = initial_state(contract, strategy, 10000, "hash", Path("events.jsonl"))
    feed_to_signal(state, contract, strategy)
    assert state["position"] == 0 and state["pending"]["target"] == 1
    apply_event(state, event("quote", 3, last=102, bid=101, ask=103), contract, strategy)
    assert state["position"] == 1
    assert state["basis"] == "104"
    assert state["cash"] == "9998.00"
    assert state["equity"][-1]["equity"] == "9978.00"
    apply_event(state, event("settlement", 4, price=102), contract, strategy)
    assert state["cash"] == "9978.00"
    assert state["basis"] == "102"


def test_paper_cross_day_requires_settlement_and_no_stale_fill():
    contract, strategy = spec(), MovingAverage(2, 3)
    state = initial_state(contract, strategy, 10000, "hash", Path("events.jsonl"))
    feed_to_signal(state, contract, strategy)
    apply_event(state, event("quote", 3), contract, strategy)
    with pytest.raises(FuturesError, match="settlement event"):
        apply_event(state, event("quote", 4, time="2026-01-03T09:04:00+08:00", trading_day="2026-01-03"), contract, strategy)

    stale = initial_state(contract, strategy, 10000, "hash", Path("events.jsonl"))
    feed_to_signal(stale, contract, strategy)
    apply_event(stale, event("quote", 5), contract, strategy)
    assert stale["position"] == 0
    assert stale["orders"][-1]["reason"] == "signal_expired_60s"


def test_resume_is_idempotent_and_processed_history_is_immutable(tmp_path):
    contract, strategy = spec(), MovingAverage(2, 3)
    config = tmp_path / "contract.json"
    config.write_text("{}", encoding="utf-8")
    events = tmp_path / "events.jsonl"
    sequence = []
    for minute, close in enumerate((100, 101, 102)):
        sequence.extend([event("quote", minute, last=close, bid=close-1, ask=close+1), event("bar", minute, close=close)])
    sequence.append(event("quote", 3, last=102, bid=101, ask=103))
    events.write_text("".join(json.dumps(item) + "\n" for item in sequence), encoding="utf-8")
    session = tmp_path / "session"
    run_session(session, events, config, contract, strategy, capital=10000)
    first = json.loads((session / "state.json").read_text())
    assert first["position"] == 1 and len(first["orders"]) == 1
    run_session(session, events, config, contract, strategy, capital=10000)
    second = json.loads((session / "state.json").read_text())
    assert first == second
    second["account_code_sha256"]["engine.py"] = "different-code"
    (session / "state.json").write_text(json.dumps(second))
    with pytest.raises(FuturesError, match="account_code_sha256"):
        run_session(session, events, config, contract, strategy, capital=10000)
    (session / "state.json").write_text(json.dumps(first))
    with events.open("a") as handle:
        handle.write(json.dumps(event("quote", 4)) + "\n")
    run_session(session, events, config, contract, strategy, capital=10000)
    third = json.loads((session / "state.json").read_text())
    assert len(third["orders"]) == 1 and third["offset"] > first["offset"]
    events.write_bytes(events.read_bytes().replace(b'"last": 100', b'"last": 101', 1))
    with pytest.raises(FuturesError, match="previously processed"):
        run_session(session, events, config, contract, strategy, capital=10000)
    assert sha256(config) == third["config_sha256"]


def test_incomplete_bar_and_out_of_order_quote_rejected():
    contract, strategy = spec(), MovingAverage(2, 3)
    state = initial_state(contract, strategy, 10000, "hash", Path("events.jsonl"))
    apply_event(state, event("quote", 0), contract, strategy)
    with pytest.raises(FuturesError, match="completed"):
        apply_event(state, event("bar", 0, bar_end="2026-01-02T09:01:00+08:00"), contract, strategy)
    with pytest.raises(FuturesError, match="strictly increasing"):
        apply_event(state, event("quote", 0), contract, strategy)


def test_external_model_signal_requires_available_features_and_next_quote():
    contract, strategy = spec(), MovingAverage(2, 3)
    state = initial_state(contract, strategy, 10000, "hash", Path("events.jsonl"), "external")
    apply_event(state, event("quote", 0), contract, strategy)
    signal = event("signal", 0, model_id="example-model-v1", target=1,
                   feature_cutoff="2026-01-02T09:00:01+08:00")
    with pytest.raises(FuturesError, match="feature_cutoff"):
        apply_event(state, signal, contract, strategy)
    signal["feature_cutoff"] = "2026-01-02T08:59:59+08:00"
    apply_event(state, signal, contract, strategy)
    assert state["position"] == 0
    apply_event(state, event("quote", 1), contract, strategy)
    assert state["position"] == 1


def test_quote_adapter_rejects_stale_or_wrong_contract_settings():
    contract = spec()
    now = datetime(2026, 1, 2, 9, 0, tzinfo=timezone(timedelta(hours=8)))
    quote = SimpleNamespace(datetime="2026-01-02 08:59:59.000000", last_price=100, bid_price1=99,
                            ask_price1=101, bid_volume1=1, ask_volume1=2, upper_limit=200, lower_limit=50,
                            price_tick=1, volume_multiple=10)
    result = quote_event(quote, contract, now.date(), now, 30)
    assert result["time"] == now.isoformat() and result["source"] == "tqsdk"
    assert quote_event(quote, contract, now.date(), now + timedelta(minutes=2), 30) is None
    quote.volume_multiple = 20
    with pytest.raises(FuturesError, match="does not match"):
        quote_event(quote, contract, now.date(), now, 30)


def test_completed_bar_adapter_never_uses_unfinished_or_old_bar():
    contract = spec()
    now = datetime(2026, 1, 2, 1, 0, tzinfo=timezone.utc)
    start = now - timedelta(seconds=60)
    row = {"datetime": int(start.timestamp() * 1_000_000_000), "close": 100}
    completed = completed_bar(row, contract, now.date(), now, 60, None, 30)
    assert completed["bar_end"] == now.isoformat()
    assert completed_bar(row, contract, now.date(), now - timedelta(seconds=1), 60, None, 30) is None
    assert completed_bar(row, contract, now.date(), now + timedelta(minutes=2), 60, None, 30) is None


def test_optional_feed_uses_local_sim_only_and_never_calls_trade_api(tmp_path, monkeypatch):
    import pandas as pd
    from quant_research.futures import tqfeed

    now = datetime.now(tqfeed.SHANGHAI)
    contract = spec(exit_by=(now.date() + timedelta(days=1)).isoformat())
    captured = {}

    class FakeSim:
        def __init__(self, init_balance):
            captured["sim"] = self
            captured["balance"] = init_balance

    class FakeApi:
        def __init__(self, account, auth):
            assert account is captured["sim"]
            captured["auth"] = auth

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_quote(self, symbol):
            assert symbol == SYMBOL
            return SimpleNamespace(datetime=now.strftime("%Y-%m-%d %H:%M:%S.%f"), last_price=100,
                                   bid_price1=99, ask_price1=101, bid_volume1=1, ask_volume1=2,
                                   upper_limit=200, lower_limit=50, price_tick=1, volume_multiple=10)

        def get_kline_serial(self, symbol, duration_seconds, data_length):
            start = now - timedelta(seconds=60)
            return pd.DataFrame([{"datetime": int(start.timestamp() * 1e9), "close": 100},
                                 {"datetime": int(now.timestamp() * 1e9), "close": 101}])

        def wait_update(self, deadline):
            return True

    counter = iter((0, 0.1, 0.2, 0.3, 1.5, 1.6))
    monkeypatch.setattr(tqfeed.time, "monotonic", lambda: next(counter, 2))
    monkeypatch.setenv("TQ_ACCOUNT", "test-account")
    monkeypatch.setenv("TQ_PASSWORD", "test-secret")
    fake_sdk = SimpleNamespace(TqSim=FakeSim, TqApi=FakeApi, TqAuth=lambda user, password: object())
    events = tmp_path / "quotes.jsonl"
    run_feed(events, contract, trading_day=now.date().isoformat(), seconds=1, sdk=fake_sdk)
    text = events.read_text()
    kinds = [json.loads(line)["type"] for line in text.splitlines()]
    assert kinds == ["quote", "bar"]
    assert captured["balance"] == 1
    assert "test-secret" not in text and "test-account" not in text


def test_append_settlement_requires_current_unsettled_day(tmp_path):
    contract = spec()
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(event("quote", 0)) + "\n")
    now = datetime.fromisoformat("2026-01-02T15:30:00+08:00")
    append_settlement(events, contract, day="2026-01-02", price="99.5", observed=now)
    result = [json.loads(line) for line in events.read_text().splitlines()]
    assert result[-1]["type"] == "settlement" and result[-1]["price"] == "99.5"
    with pytest.raises(FuturesError, match="unsettled"):
        append_settlement(events, contract, day="2026-01-02", price=100, observed=now)


def test_no_new_position_on_exit_day():
    contract, strategy = spec(exit_by="2026-01-03"), MovingAverage(2, 3)
    state = initial_state(contract, strategy, 10000, "hash", Path("events.jsonl"))
    feed_to_signal(state, contract, strategy)
    apply_event(state, event("settlement", 3), contract, strategy)
    state["pending"] = {"target": 1, "time": "2026-01-02T09:03:00+08:00", "reason": "test"}
    next_day = event("quote", 4, time="2026-01-03T09:04:00+08:00", trading_day="2026-01-03")
    apply_event(state, next_day, contract, strategy)
    assert state["position"] == 0 and state["pending"] is None


def test_market_probe_preserves_failure_without_fabricated_data(tmp_path, monkeypatch):
    from quant_research.futures import probe

    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout='{"status":"provider_error","error_type":"Timeout"}', returncode=0))
    output = tmp_path / "probe"
    result = capture_sina("AG2609", output, timeout=1, attempts=2)
    assert result["qualification"] == "data_insufficient" and len(result["attempts"]) == 2
    assert not (output / "raw.csv").exists()
    assert json.loads((output / "audit.json").read_text())["attempts"][0]["status"] == "provider_error"
    with pytest.raises(FuturesError, match="dated"):
        capture_sina("AG9999", tmp_path / "bad")


def test_ridge_baseline_only_trains_on_completed_past_labels():
    contract = spec(fee_open=0, fee_close=0, slippage_ticks=0)
    bars = [daily(contract, f"2026-01-{day:02d}", 100, 101, 102) for day in range(2, 10)]
    model = RidgeForecast(lookback=2, min_train=3, train_window=3, alpha=1, threshold=0.001)
    assert model.target(bars[:5], 0) is None
    assert model.target(bars[:6], 0) == 1
    first = backtest(bars, contract, model, capital=10000)
    changed = backtest(bars[:-1] + [daily(contract, "2026-01-09", 100, 80, 80)], contract, model, capital=10000)
    assert first.decisions[:-1] == changed.decisions[:-1]
    assert first.equity[:-1] == changed.equity[:-1]
    assert first.orders[0]["decision_day"] == "2026-01-07"
    assert first.orders[0]["trading_day"] == "2026-01-08"


def test_ridge_rejects_invalid_training_configuration():
    with pytest.raises(FuturesError, match="ridge strategy"):
        RidgeForecast(lookback=2, min_train=5, train_window=4)
    with pytest.raises(FuturesError, match="ridge strategy"):
        RidgeForecast(alpha=float("nan"))


def test_backtest_archives_exact_inputs_and_marks_synthetic(tmp_path):
    from quant_research.futures.__main__ import fixture, main

    config, bars, _ = fixture(tmp_path / "fixture")
    output = tmp_path / "run"
    result = main(["backtest", "--config", str(config), "--bars", str(bars), "--source", "synthetic test",
                   "--capital", "100000", "--strategy", "ridge", "--lookback", "2", "--min-train", "3",
                   "--train-window", "5", "--synthetic", "--out", str(output)])
    assert result == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["origin"] == "synthetic_fixture" and manifest["strategy"]["name"] == "ridge"
    assert (output / "inputs" / "bars.csv").read_bytes() == bars.read_bytes()
    assert (output / "inputs" / "contract.json").read_bytes() == config.read_bytes()
    assert manifest["inputs"]["bars_sha256"] == sha256(output / "inputs" / "bars.csv")
    assert manifest["code_sha256"]["engine.py"] == sha256(output / "source" / "engine.py")
    assert "合成数据" in (output / "report.html").read_text()
