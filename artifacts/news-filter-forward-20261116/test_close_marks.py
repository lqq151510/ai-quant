"""Use the archived Friday responses to verify close data acceptance."""

import json
import sys
import tempfile
import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import collect_close_marks as collector


BARS = Path(__file__).resolve().parent / "monday-preparation/bars"
FRIDAY = date(2026, 11, 13)
THURSDAY = date(2026, 11, 12)
OBSERVED = datetime(2026, 11, 13, 18, 52, tzinfo=collector.CST)
PREVIOUS_CLOSE = Decimal("7.04")
EXPECTED_CLOSE = Decimal("7.18")


class CloseMarksTest(unittest.TestCase):
    def archived_bar(self, filename):
        try:
            return (BARS / filename).read_bytes()
        except FileNotFoundError:
            self.skipTest(f"local archive missing: {BARS.name}/{filename}")

    def test_exact_unadjusted_close_agrees_with_both_sources(self):
        close, quote_at = collector.parse_tencent(
            self.archived_bar("601988.SH-0.json"), "sh601988", PREVIOUS_CLOSE, OBSERVED,
            day=FRIDAY, prior_day=THURSDAY,
        )
        self.assertEqual(close, collector.parse_sina(self.archived_bar("601988.SH-sina.txt"), day=FRIDAY))
        self.assertEqual(close, EXPECTED_CLOSE)
        self.assertGreaterEqual(quote_at.time(), collector.CLOSE)

    def test_friday_quote_cannot_mark_monday(self):
        with self.assertRaisesRegex(ValueError, "exact_session_close_missing"):
            collector.parse_tencent(self.archived_bar("601988.SH-0.json"), "sh601988", PREVIOUS_CLOSE, OBSERVED)

    def test_intraday_quote_cannot_mark_even_with_same_day_kline(self):
        data = json.loads(self.archived_bar("601988.SH-0.json"))
        data["data"]["sh601988"]["qt"]["sh601988"][30] = "20261113145959"
        with self.assertRaisesRegex(ValueError, "post_close_quote_missing"):
            collector.parse_tencent(
                json.dumps(data).encode(), "sh601988", PREVIOUS_CLOSE, OBSERVED,
                day=FRIDAY, prior_day=THURSDAY,
            )

    def test_prior_close_mismatch_rejects_unresolved_corporate_action(self):
        with self.assertRaisesRegex(ValueError, "corporate_action_unresolved"):
            collector.parse_tencent(
                self.archived_bar("601988.SH-0.json"), "sh601988", Decimal("7.00"), OBSERVED,
                day=FRIDAY, prior_day=THURSDAY,
            )

    def test_partial_feed_outage_keeps_primary_mark_and_raw_evidence(self):
        raw = b"synthetic Tencent response"
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(collector, "download", side_effect=[(raw, OBSERVED), OSError("offline")]),
            patch.object(collector, "parse_tencent", return_value=(EXPECTED_CLOSE, OBSERVED)),
        ):
            root = Path(tmp)
            _, mark, error, _ = collector.collect(root, "601988.SH", PREVIOUS_CLOSE)
            self.assertIsNone(error)
            self.assertEqual(mark["secondary_check"]["status"], "unavailable")
            self.assertEqual((root / mark["raw_file"]).read_bytes(), raw)

    def test_conflicting_sources_fail_closed(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(collector, "download", side_effect=[(b"synthetic Tencent response", OBSERVED),
                                                            (b"synthetic Sina response", OBSERVED)]),
            patch.object(collector, "parse_tencent", return_value=(EXPECTED_CLOSE, OBSERVED)),
            patch.object(collector, "parse_sina", return_value=Decimal("7.19")),
        ):
            root = Path(tmp)
            _, mark, error, attempts = collector.collect(root, "601988.SH", PREVIOUS_CLOSE)
            self.assertIsNone(mark)
            self.assertEqual(error, "tencent_sina_close_mismatch")
            self.assertTrue((root / attempts["sina"]["raw_file"]).exists())

    def test_only_real_forward_held_symbols_are_requested(self):
        candidates = {"candidates": [
            {"symbol": "601988.SH", "previous_close": "7.04"},
            {"symbol": "513100.SH", "previous_close": "2.23"},
        ]}
        opening = {"date": collector.DAY.isoformat(), "mode": "forward",
                   "baseline": {"positions": {"601988.SH": 100}}, "filtered": {"positions": {}}}
        self.assertEqual(collector.held_symbols(opening, candidates), {"601988.SH": PREVIOUS_CLOSE})
        with self.assertRaisesRegex(ValueError, "forward opening evidence"):
            collector.held_symbols({**opening, "mode": "synthetic"}, candidates)

    def test_later_day_carries_unknown_prior_close_as_unknown(self):
        tuesday = date(2026, 11, 17)
        opening = {"date": tuesday.isoformat(), "mode": "forward",
                   "baseline": {"positions": {"601988.SH": 100}},
                   "filtered": {"positions": {"513100.SH": 200}},
                   "prior_closes": {"513100.SH": "2.23", "601988.SH": None}}
        self.assertEqual(collector.prior_session(tuesday), collector.DAY)
        self.assertEqual(collector.held_symbols(opening, day=tuesday),
                         {"601988.SH": None, "513100.SH": Decimal("2.23")})

    def test_unknown_prior_close_requires_both_sources_and_current_quote(self):
        data = json.loads(self.archived_bar("601988.SH-0.json"))
        security = data["data"]["sh601988"]
        security["day"] = security["day"][-2:]
        security["day"][0][0] = collector.PRIOR_DAY.isoformat()
        security["day"][1][0] = collector.DAY.isoformat()
        security["qt"]["sh601988"][30] = "20261116161448"
        payload = json.dumps(data).encode()
        secondary = b'var _a=([{"day":"2026-11-13","close":"7.04","volume":"10"},'
        secondary += b'{"day":"2026-11-16","close":"7.18","volume":"10"}]);'
        observed = datetime(2026, 11, 16, 18, tzinfo=collector.CST)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(collector, "download", side_effect=[(payload, observed), (secondary, observed)]):
                _, mark, error, _ = collector.collect(root, "601988.SH", None)
                self.assertIsNone(error)
                self.assertEqual(mark["previous_close_basis"], "dual_source_recovery")
                self.assertEqual(mark["previous_close"], "7.040")
            with patch.object(collector, "download", side_effect=[(payload, observed), OSError("offline")]):
                _, mark, error, _ = collector.collect(root, "601988.SH", None)
                self.assertIsNone(mark)
                self.assertEqual(error, "prior_close_evidence_missing")
            conflicting = secondary.replace(b'7.04', b'7.05')
            with patch.object(collector, "download", side_effect=[(payload, observed), (conflicting, observed)]):
                _, mark, error, _ = collector.collect(root, "601988.SH", None)
                self.assertIsNone(mark)
                self.assertEqual(error, "previous_close_mismatch_corporate_action_unresolved")

    def test_monday_close_restores_saved_open_before_collecting_held_marks(self):
        now = datetime(2026, 11, 16, 18, tzinfo=collector.CST)
        candidates = {"candidates": [{"symbol": "601988.SH", "previous_close": "7.04"}]}
        opening = {"date": str(collector.DAY), "mode": "forward",
                   "baseline": {"positions": {"601988.SH": 100}}, "filtered": {"positions": {}}}
        for capture_path in ("books/books.json", "books-retry-2/capture-progress.json"):
            with self.subTest(capture_path=capture_path), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "artifacts" / "study"
                out = root / str(collector.DAY)
                capture = out / capture_path
                capture.parent.mkdir(parents=True)
                capture.write_text("{}")

                def replay(command, **kwargs):
                    self.assertEqual(command[1:], [str(root / "scheduled_open.py"), "--date", str(collector.DAY)])
                    self.assertEqual(kwargs["env"]["PYTHONPATH"], str(Path(tmp) / "research_v1"))
                    (out / "opening.json").write_text(json.dumps(opening))

                with (
                    patch.object(collector, "BASE", root),
                    patch.object(collector, "datetime") as clock,
                    patch.object(collector, "load_frozen_inputs", return_value=candidates),
                    patch.object(collector.subprocess, "run", side_effect=replay) as dispatch,
                    patch.object(collector, "collect", return_value=("601988.SH", None, "offline", {})) as marks,
                    patch.object(sys, "argv", ["collect_close_marks.py"]),
                    patch("builtins.print"),
                ):
                    clock.now.return_value = now
                    collector.main()
                    dispatch.assert_called_once()
                    marks.assert_called_once_with(out, "601988.SH", PREVIOUS_CLOSE, day=collector.DAY)

    def test_monday_close_does_not_replay_without_capture_or_overwrite_invalid_opening(self):
        now = datetime(2026, 11, 16, 18, tzinfo=collector.CST)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / str(collector.DAY)
            out.mkdir()
            with (
                patch.object(collector, "BASE", root),
                patch.object(collector, "datetime") as clock,
                patch.object(collector, "load_frozen_inputs"),
                patch.object(collector.subprocess, "run") as dispatch,
                patch.object(collector, "collect") as marks,
                patch.object(sys, "argv", ["collect_close_marks.py"]),
                patch("sys.stderr") as stderr,
            ):
                clock.now.return_value = now
                with self.assertRaises(SystemExit) as raised:
                    collector.main()
                self.assertEqual(raised.exception.code, 1)
                stderr.write.assert_called_once_with("周一开盘账本缺失，未找到可用于恢复的已保存盘口。\n")
                opening = out / "opening.json"
                opening.write_text("{invalid")
                (out / "books").mkdir()
                (out / "books" / "books.json").write_text("{}")
                with self.assertRaises(json.JSONDecodeError):
                    collector.main()
                dispatch.assert_not_called()
                marks.assert_not_called()
                self.assertEqual(opening.read_text(), "{invalid")


if __name__ == "__main__":
    unittest.main()
