"""Regression checks for the paired entry-timing replay."""

import csv
import hashlib
import json
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from studies.entry_timing.replay import (
    DELAYED,
    HERE,
    NORMAL,
    OPENING,
    Costs,
    Opportunity,
    execution_price,
    group_summary,
    load_opportunities,
    replay_arm,
    run,
)


D = Decimal
HEADER = "symbol,signal_date,entry_date,exit_date,previous_close,overnight_open,delayed_open,exit_open\n"
ETF = "510300,2026-10-14,2026-10-15,2026-10-16,4.130,4.125,4.132,4.180\n"
SESSIONS = [date(2026, 10, day) for day in (14, 15, 16)]


class EntryReplayTests(unittest.TestCase):
    def test_buy_rounds_up_and_sell_rounds_down_after_slippage(self):
        cost = Costs()
        self.assertEqual(execution_price(D("4.125"), D(".001"), cost, "buy"), D("4.128"))
        self.assertEqual(execution_price(D("4.125"), D(".001"), cost, "sell"), D("4.122"))
        self.assertEqual(execution_price(D("6.96"), D(".01"), cost, "buy"), D("6.97"))
        self.assertEqual(execution_price(D("6.96"), D(".01"), cost, "sell"), D("6.95"))

    def test_limit_equality_fills_and_other_arm_misses_with_zero_fees(self):
        opportunity = Opportunity(
            "510300", date(2026, 10, 14), date(2026, 10, 15), date(2026, 10, 16),
            D("4.128"), D("4.125"), D("4.132"), D("4.180"),
        )
        opening = replay_arm(opportunity, OPENING, NORMAL, 100, "etf", D(".001"), Costs())
        delayed = replay_arm(opportunity, DELAYED, NORMAL, 100, "etf", D(".001"), Costs())
        self.assertEqual(opening.entry_price, opening.limit_price)
        self.assertEqual(opening.exit_price, D("4.177"))
        self.assertEqual(opening.fees, D("10"))
        self.assertEqual(opening.net_pnl, D("-5.10"))
        self.assertEqual(delayed.limit_price, opening.limit_price)
        self.assertEqual(delayed.reason, "projected_price_above_limit")
        self.assertEqual(delayed.filled_quantity, 0)
        self.assertEqual(delayed.fees, D("0"))
        self.assertEqual(delayed.net_pnl, D("0"))
        self.assertIsNone(delayed.exit_price)
        summary = group_summary([opening, delayed])
        self.assertEqual(summary[DELAYED]["opportunities"], 1)
        self.assertEqual(summary["paired"]["wait_minus_overnight_net"], D("5.10"))
        self.assertEqual(summary["paired"]["by_fill_case"]["overnight_only"]["count"], 1)

    def test_rejects_same_day_exit_duplicate_symbol_date_and_invalid_price(self):
        bad_rows = (
            (ETF.replace("2026-10-16", "2026-10-15"), "must be consecutive"),
            (ETF.replace("2026-10-16", "2026-10-19"), "must be consecutive"),
            (ETF + ETF, "duplicate opportunity"),
            (ETF.replace("4.130", "NaN"), "positive and finite"),
            (ETF.replace("510300", "300706"), "unsupported symbol"),
            ("", "Incomplete symbol and session opportunities"),
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "opportunities.csv"
            for rows, error in bad_rows:
                with self.subTest(error=error):
                    path.write_text(HEADER + rows, encoding="utf-8")
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(ValueError, error):
                        load_opportunities(path, {"510300"}, SESSIONS, digest)

    def test_run_keeps_all_scenario_arm_rows_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "opportunities.csv"
            protocol_path = Path(temp) / "protocol.json"
            out = Path(temp) / "results"
            source.write_text(HEADER + ETF, encoding="utf-8")
            protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
            protocol["last_complete_session"] = SESSIONS[-1].isoformat()
            protocol["symbols"].pop("601988")
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            manifest = {
                "sessions": [day.isoformat() for day in SESSIONS],
                "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "opportunities_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            source.with_name("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            run(protocol_path, source, out)
            with (out / "trades.csv").open(encoding="utf-8", newline="") as handle:
                trades = list(csv.DictReader(handle))
            self.assertEqual(len(trades), 6)
            self.assertEqual(sum(row["arm"] == DELAYED for row in trades), 3)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["opportunities"], 1)
            self.assertEqual(summary["scenarios"][NORMAL]["overall"][DELAYED]["unfilled"], 1)
            self.assertFalse(summary["screen"]["passed"])
            with self.assertRaises(FileExistsError):
                run(protocol_path, source, out)
            tampered = Path(temp) / "tampered"
            source.write_text(HEADER + ETF.replace("4.130", "4.120"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Opportunity CSV SHA256"):
                run(protocol_path, source, tampered)
            protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Protocol SHA256"):
                run(protocol_path, source, tampered)
            self.assertFalse(tampered.exists())


if __name__ == "__main__":
    unittest.main()
