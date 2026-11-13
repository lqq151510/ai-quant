import pandas as pd

from quant_research.cli import main


def test_cli_refuses_sealed_interval_before_reading_nonexistent_input(tmp_path, capsys):
    assert main(["baseline", str(tmp_path / "does-not-exist"), "--kind", "etf", "--end", "2025-01-01",
                 "--out", str(tmp_path / "results")]) == 2
    assert "sealed test interval" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()
    assert main(["audit", str(tmp_path / "does-not-exist"), "--end", "2025-01-01"]) == 2
    assert "audit command does not open the sealed test interval" in capsys.readouterr().err


def test_etf_line_runs_without_stock_membership_data(tmp_path, monkeypatch):
    from quant_research import baseline
    from quant_research.demo import fixture

    frames = fixture()
    frames.pop("membership")
    frames["bars"] = frames["bars"][frames["bars"]["kind"].eq("etf")]
    frames["calendar"] = frames["calendar"].iloc[:4]
    monkeypatch.setattr(baseline, "CAPITALS", (10000,))
    monkeypatch.setattr(baseline, "HORIZONS", (5,))
    result = baseline.run_baselines(frames, tmp_path, start=pd.Timestamp("2024-01-02"),
                                    end=pd.Timestamp("2024-01-05"), origin="synthetic_fixture", kinds=("etf",))
    assert len(result) == 8
    assert all(row["line"] == "etf-5d" for row in result)
    assert all(row["ledger_reconciliation_error"] < 1e-6 and row["min_cash"] >= 0 for row in result)
