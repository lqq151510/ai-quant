from datetime import date
import json

import pandas as pd
import pytest

from quant_research.etf_bulk import prepare_bulk_etf_snapshot
from quant_research.protocol import ETF_SYMBOLS, ResearchPeriod
from quant_research.snapshot import audit_snapshot, file_sha256, read_snapshot


_PERIOD = ResearchPeriod(date(2015, 4, 10), date(2015, 4, 15))


def _capture(root, *, range_conflict=False):
    (root / "raw").mkdir(parents=True)
    items = []

    def save(name, rows, columns=None):
        path = root / "raw" / name
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        items.append({"path": f"raw/{name}", "status": "success", "sha256": file_sha256(path), "rows": len(rows),
                      "source_function": "fixture", "params": {}, "observed_at_utc": "2026-11-08T00:00:00Z"})

    days = pd.date_range(_PERIOD.start, _PERIOD.end)
    save("trading_calendar_baostock_2015.csv", [
        {"calendar_date": day.date().isoformat(), "is_trading_day": int(day.weekday() < 5)} for day in days
    ])
    for symbol in ETF_SYMBOLS:
        prices = []
        for day in days[days.day_of_week < 5]:
            halt = symbol == "510500" and day.day in (13, 14)
            resume = symbol == "510500" and day.day == 15
            close = "8.001" if resume else "2.243"
            high = "3.000" if range_conflict and symbol == "510300" and day.day == 15 else close
            prices.append({"date": day.date().isoformat(), "open": close, "high": high,
                           "low": close, "close": close, "volume": 0 if halt else 100,
                           "amount": 0 if halt else 224.3, "preclose": close, "tradestatus": int(not halt),
                           "code": f"sh.{symbol}", "adjustflag": "3"})
        save(f"etf_bars_baostock_{symbol}.csv", prices)
        save(f"etf_bars_tx_{symbol}.csv", [row for row in prices if row["tradestatus"]])
        save(f"etf_actions_f10_{symbol}_splits.csv", [], ["拆分折算日"])
        save(f"etf_actions_f10_{symbol}_dividends.csv", [], ["除息日", "权益登记日", "分红发放日", "每10份分红"])
    (root / "manifest.json").write_text(json.dumps({"complete": True, "items": items}))


def test_bulk_snapshot_retains_halt_prices_and_separate_accounting_mark(tmp_path):
    capture = tmp_path / "capture"
    _capture(capture)
    output = tmp_path / "snapshot"
    manifest = prepare_bulk_etf_snapshot(capture, output, _PERIOD)
    _, frames = read_snapshot(output)
    bars = frames["bars"].loc[lambda frame: frame["symbol"].eq("510500")].set_index("date")
    halted = bars.loc["2015-04-14"]
    resumed = bars.loc["2015-04-15"]
    assert not halted.tradable and halted.close == 2.243 and halted.valuation_close == 8.001
    assert pd.isna(halted.upper_limit) and halted.tradability_source == "issuer_announcement"
    assert resumed.tradable and resumed.preclose == 8.001 and resumed.upper_limit == 8.801
    assert resumed.limit_source == "rule_derived_candidate"
    assert frames["actions"].iloc[0].share_multiplier == 0.28032483
    assert frames["actions"].iloc[0].split_rounding == "floor"
    assert manifest["sources"][0]["by_symbol"]["510500"]["verified_halts"] == 2
    assert audit_snapshot(manifest, frames, _PERIOD, "etf")["status"] == "data_insufficient"


def test_bulk_snapshot_rejects_modified_source_bytes(tmp_path):
    capture = tmp_path / "capture"
    _capture(capture)
    source = capture / "raw" / "etf_bars_tx_510500.csv"
    source.write_bytes(source.read_bytes() + b"\n")
    output = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="source CSV hash mismatch"):
        prepare_bulk_etf_snapshot(capture, output, _PERIOD)
    assert not output.exists()


def test_bulk_snapshot_leaves_conflicting_candidate_limits_unresolved(tmp_path):
    capture = tmp_path / "capture"
    _capture(capture, range_conflict=True)
    output = tmp_path / "snapshot"
    manifest = prepare_bulk_etf_snapshot(capture, output, _PERIOD)
    _, frames = read_snapshot(output)
    bad = frames["bars"].loc[lambda frame: frame["symbol"].eq("510300")].iloc[-1]
    assert bad.limit_source == "unresolved" and pd.isna(bad.upper_limit)
    assert manifest["sources"][0]["by_symbol"]["510300"]["issues"][0]["check"] == "candidate_limit_range_conflict"
