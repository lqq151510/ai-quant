"""Verify legacy research output and its upstream provenance."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import duckdb
import pandas as pd

from src.application import trading_plan_generator as plans
from src.controller import run_pipeline
from src.infrastructure import market_data
from src.infrastructure.storage import QuantDuckDBStorage


class TradingPlanGateTests(unittest.TestCase):
    def test_legacy_generator_cannot_render_trade_instructions(self):
        bars = pd.DataFrame({
            "symbol": ["510300", "510300", "510300", "002594", "601988"],
            "trade_date": pd.date_range("2024-01-01", periods=5),
            "data_origin": ["real", "real", "synthetic", "pit", "real"],
            "data_source": ["akshare_sina", "akshare_sina", "synthetic_panel", "akshare_sina", None],
        })
        with TemporaryDirectory() as directory:
            output = Path(directory) / "latest_trading_plan.md"
            output.write_text("旧实盘指令", encoding="utf-8")
            with patch.object(plans, "infer_latest_live_alpha", side_effect=AssertionError("inference must not run")):
                report = plans.generate_semi_auto_trading_plan(
                    bars, object(), {"510300": 100}, pd.DataFrame(), 1.0, output_path=str(output),
                )
            self.assertEqual(output.read_text(encoding="utf-8"), report)
        self.assertIn("研究专用（未通过验收）", report)
        self.assertIn("510300 | real | akshare_sina | 2024-01-01 | 2024-01-02 | 2", report)
        self.assertIn("510300 | synthetic | synthetic_panel", report)
        self.assertIn("002594 | unknown | unknown", report)
        self.assertIn("601988 | unknown | unknown", report)
        for term in ("旧实盘指令", "操作指令", "买入区间", "止损位", "止盈目标", "目标仓位"):
            self.assertNotIn(term, report)

    def test_pipeline_replaces_old_report_before_fetch_or_training_failure(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "latest_trading_plan.md"
            output.write_text("旧实盘指令", encoding="utf-8")
            with (
                patch.object(run_pipeline, "write_prototype_research_report",
                             side_effect=lambda: plans.write_prototype_research_report(output_path=str(output))),
                patch.object(run_pipeline, "QuantDuckDBStorage", side_effect=RuntimeError("storage unavailable")),
                self.assertRaisesRegex(RuntimeError, "storage unavailable"),
            ):
                run_pipeline.main()
            self.assertIn("研究专用（未通过验收）", output.read_text(encoding="utf-8"))
            self.assertNotIn("旧实盘指令", output.read_text(encoding="utf-8"))

    def test_report_only_reads_existing_metadata_without_schema_migration_or_training(self):
        with TemporaryDirectory() as directory:
            db = Path(directory) / "legacy.duckdb"
            output = Path(directory) / "report.md"
            with duckdb.connect(str(db)) as conn:
                conn.execute("CREATE TABLE daily_bars (symbol VARCHAR, trade_date DATE)")
                conn.execute("INSERT INTO daily_bars VALUES ('510300', '2024-01-02')")
            before = db.read_bytes()
            with (
                patch.object(run_pipeline, "DEFAULT_DATABASE", str(db)),
                patch.object(run_pipeline, "DEFAULT_REPORT", str(output)),
                patch.object(run_pipeline, "main", side_effect=AssertionError("training must not run")),
            ):
                run_pipeline.cli(["--report-only"])
            self.assertEqual(db.read_bytes(), before)
            self.assertIn("510300 | unknown | unknown | 2024-01-02 | 2024-01-02", output.read_text(encoding="utf-8"))

    def test_source_survives_storage_roundtrip_and_legacy_rows_stay_unknown(self):
        with TemporaryDirectory() as directory:
            db = Path(directory) / "test.duckdb"
            with duckdb.connect(str(db)) as conn:
                conn.execute("""
                    CREATE TABLE daily_bars (
                        symbol VARCHAR, name VARCHAR, sector VARCHAR, trade_date DATE,
                        open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, amount DOUBLE,
                        PRIMARY KEY (symbol, trade_date)
                    )
                """)
                conn.execute("""
                    INSERT INTO daily_bars VALUES ('510300', 'ETF', 'ETF', '2024-01-02', 4, 4, 4, 4, 1, 4)
                """)
            storage = QuantDuckDBStorage(str(db))
            original = storage.load_daily_bars()
            self.assertEqual(original["symbol"].tolist(), ["510300"])
            self.assertTrue(original["data_origin"].isna().all())
            self.assertTrue(original["data_source"].isna().all())
            with patch("src.infrastructure.market_data._fetch_from_akshare", side_effect=OSError):
                fallback = market_data.fetch_or_generate_market_data(n_days=1)
            self.assertEqual(set(fallback["data_origin"]), {"synthetic"})
            self.assertEqual(set(fallback["data_source"]), {"synthetic_panel"})
            storage.replace_all_daily_bars(fallback)
            loaded = storage.load_daily_bars()
            self.assertEqual(len(loaded), len(fallback))
            self.assertEqual(set(loaded["data_origin"]), {"synthetic"})
            self.assertEqual(set(loaded["data_source"]), {"synthetic_panel"})
            broken = fallback.copy()
            broken["open"] = "not a price"
            with self.assertRaises(duckdb.ConversionException):
                storage.replace_all_daily_bars(broken)
            self.assertTrue(storage.load_daily_bars().equals(loaded))

    def test_online_source_labels_follow_the_actual_provider_branch(self):
        day = pd.Timestamp.today().normalize()
        sina = pd.DataFrame({
            "date": [day], "open": [4.0], "high": [4.0], "low": [4.0], "close": [4.0],
            "volume": [100.0], "amount": [400.0],
        })
        eastmoney = sina.rename(columns={
            "date": "日期", "open": "开盘", "high": "最高", "low": "最低", "close": "收盘",
            "volume": "成交量", "amount": "成交额",
        })
        provider = SimpleNamespace(
            fund_etf_hist_sina=lambda **kwargs: sina,
            stock_zh_a_daily=lambda **kwargs: None,
            stock_zh_a_hist=lambda **kwargs: eastmoney,
        )
        assets = [
            market_data.AssetMeta("510300", "ETF", "sector", kind="etf"),
            market_data.AssetMeta("002594", "Stock", "sector"),
        ]
        with patch.dict("sys.modules", {"akshare": provider}), patch.object(market_data, "CORE_UNIVERSE", assets):
            bars = market_data.fetch_or_generate_market_data(n_days=10).set_index("symbol")
        self.assertEqual(set(bars["data_origin"]), {"real"})
        self.assertEqual(bars.loc["510300", "data_source"], "akshare_sina")
        self.assertEqual(bars.loc["002594", "data_source"], "akshare_eastmoney")


if __name__ == "__main__":
    unittest.main()
