from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant_research import providers


class ProviderEvidenceTests(unittest.TestCase):
    def test_requests_cover_frozen_universe_and_dated_samples(self):
        plan = providers._request_plan(date(2015, 1, 1), date(2026, 11, 6))
        keys = [request.key for request in plan]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len([key for key in keys if key.startswith("etf_bars_")]), 4)
        self.assertEqual(len([key for key in keys if key.startswith("csi300_membership_")]), 2)
        self.assertTrue(all("push2" not in request.endpoint for request in plan))

    def test_calendar_requests_partition_the_range_into_natural_years(self):
        start, end = date(2015, 1, 1), date(2026, 11, 6)
        calendars = [request for request in providers._request_plan(start, end)
                     if request.endpoint == "query_trade_dates"]
        self.assertEqual(len(calendars), 12)
        self.assertEqual(calendars[0].arguments["start_date"], start.isoformat())
        self.assertEqual(calendars[-1].arguments["end_date"], end.isoformat())
        for request in calendars:
            first = date.fromisoformat(request.arguments["start_date"])
            last = date.fromisoformat(request.arguments["end_date"])
            self.assertEqual(first.year, last.year)
            self.assertLessEqual((last - first).days + 1, 366)

    def test_capture_preserves_rows_and_hashes_without_normalizing_source_fields(self):
        request = providers._Request("fixture", "akshare_sina", "fixture", {}, "date", ("date", "open"))
        raw = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "open": [1.25, 1.5]})
        with tempfile.TemporaryDirectory() as directory, patch.object(
            providers, "_fetch", return_value=({"": (raw, "date")}, "test")
        ):
            result = providers._capture_tables(request, Path(directory))
            evidence = result["files"][0]
            file = Path(directory) / evidence["name"]
            self.assertEqual(result["status"], "captured")
            self.assertEqual(evidence["rows"], 2)
            self.assertEqual(evidence["first_date"], "2024-01-02")
            self.assertEqual(evidence["sha256"], hashlib.sha256(file.read_bytes()).hexdigest())
            pd.testing.assert_frame_equal(pd.read_csv(file), raw)

    def test_invalid_schema_remains_in_evidence(self):
        request = providers._Request("fixture", "akshare_sina", "fixture", {}, "date", ("date", "close"))
        raw = pd.DataFrame({"date": ["invalid"], "open": [1.0]})
        with tempfile.TemporaryDirectory() as directory, patch.object(
            providers, "_fetch", return_value=({"": (raw, "date")}, "test")
        ):
            result = providers._capture_tables(request, Path(directory))
            self.assertEqual(result["status"], "invalid_schema")
            self.assertEqual(result["files"][0]["missing_columns"], ["close"])
            self.assertEqual(result["files"][0]["invalid_date_rows"], 1)
            self.assertTrue((Path(directory) / "fixture.csv").exists())

    def test_timeout_is_failure_and_cleans_partial_worker_output(self):
        request = providers._Request("fixture", "akshare_sina", "fixture", {})

        def timed_out(command, **kwargs):
            (Path(command[-1]) / "partial.csv").write_text("unfinished", encoding="utf-8")
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        with tempfile.TemporaryDirectory() as directory, patch.object(providers.subprocess, "run", timed_out):
            result = providers._collect(request, Path(directory))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["type"], "TimeoutExpired")
            self.assertEqual(result["files"], [])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_calendar_alignment_counts_missing_non_session_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            pd.DataFrame({
                "calendar_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "is_trading_day": ["0", "1", "1"],
            }).to_csv(output / "calendar.csv", index=False)
            pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-02"]}).to_csv(
                output / "bars.csv", index=False
            )
            records = [
                {"request": {"key": "trading_calendar"}, "status": "captured", "files": [{"path": "calendar.csv"}]},
                {"request": {"key": "etf_bars_510050"}, "status": "captured", "files": [{"path": "bars.csv"}]},
            ]
            result = providers._session_alignment(output, records, date(2024, 1, 1), date(2024, 1, 3))
            comparison = result["etfs"]["510050"]
            self.assertEqual(comparison["missing_sessions"], 1)
            self.assertEqual(comparison["non_session_dates"], 1)
            self.assertEqual(comparison["duplicate_rows"], 1)

    def test_probe_failure_is_saved_and_never_promoted_to_ready(self):
        def failed(request, raw):
            return {"request": providers.asdict(request), "status": "failed", "error": {"type": "TimeoutError"}, "files": []}

        with tempfile.TemporaryDirectory() as directory, patch.object(providers, "_collect", failed):
            output = Path(directory) / "probe"
            result = providers.run_probe(output, start=date(2015, 1, 1), end=date(2026, 11, 6))
            saved = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(result, saved)
            self.assertEqual(result["acquisition_status"], "failed")
            self.assertEqual(result["status"], "data_insufficient")
            self.assertEqual(result["captured_requests"], 0)
            with self.assertRaises(FileExistsError):
                providers.run_probe(output, start=date(2015, 1, 1), end=date(2026, 11, 6))

    def test_baostock_requests_use_the_callers_serial_session_lane(self):
        owner = threading.get_ident()
        baostock_threads = []

        def failed(request, raw):
            if request.provider == "baostock":
                baostock_threads.append(threading.get_ident())
            return {"request": providers.asdict(request), "status": "captured", "files": []}

        samples = [request for request in providers._request_plan(date(2015, 1, 1), date(2026, 11, 6))
                   if request.endpoint == "query_hs300_stocks"]
        with tempfile.TemporaryDirectory() as directory, patch.object(providers, "_collect", failed), patch.object(
            providers, "_request_plan", return_value=samples
        ):
            providers.run_probe(Path(directory) / "probe", start=date(2015, 1, 1), end=date(2026, 11, 6))
        self.assertEqual(len(baostock_threads), 2)
        self.assertEqual(set(baostock_threads), {owner})


if __name__ == "__main__":
    unittest.main()
