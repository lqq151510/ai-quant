import json

import pandas as pd
import pytest

from quant_research.fund_flow import SOURCE_FIELDS, fund_flow_asof, read_fund_flow
from quant_research.snapshot import file_sha256


def test_flow_capture_is_hash_checked_and_never_backdated(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    files = []
    values = dict(zip(SOURCE_FIELDS, [3.5, -1.0, 0.0], strict=True))
    for key, endpoint, arguments, rows in (
        ("stock_flow_600000", "stock_individual_fund_flow", {"stock": "600000"},
         [{"日期": "2024-01-02", **values}, {"日期": "2024-01-03", **values}]),
        ("etf_flow_spot", "fund_etf_spot_em", {}, [
            {"代码": "510300", "数据日期": "2024-01-02", "更新时间": "2024-01-02T15:04:00+08:00", **values},
            {"代码": "510500", "数据日期": "2024-01-03", "更新时间": "2024-01-03T12:00:00+08:00", **values},
        ]),
    ):
        path = raw / f"{key}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        files.append({"request": {"key": key, "endpoint": endpoint, "arguments": arguments},
                      "status": "captured", "observed_at_utc": "2024-01-03T05:00:00Z",
                      "files": [{"sha256": file_sha256(path)}]})
    meta = {"schema_version": 1, "records": files, "stocks": ["600000"], "etfs": ["510300", "510500"]}
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(meta))
    observations = read_fund_flow(tmp_path)
    assert list(observations["symbol"]) == ["600000", "510300"]
    assert observations["main_flow_fraction"].tolist() == [0.035, 0.035]
    decisions = observations[["date", "symbol", "kind"]].assign(as_of=pd.Timestamp("2024-01-02T20:00:00+08:00"))
    assert fund_flow_asof(observations, decisions)["main_flow_fraction"].isna().all()
    files[0].update(started_at_utc="2024-01-03T06:59:59Z", observed_at_utc="2024-01-03T07:00:01Z")
    capture.write_text(json.dumps(meta))
    assert len(read_fund_flow(tmp_path).query("kind == 'stock'")) == 1
    files[0].update(started_at_utc="2024-01-03T07:00:00Z")
    capture.write_text(json.dumps(meta))
    assert len(read_fund_flow(tmp_path).query("kind == 'stock'")) == 2
    with (raw / "stock_flow_600000.csv").open("a") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="hash"):
        read_fund_flow(tmp_path)


def test_future_fund_flow_revision_cannot_change_previous_signal():
    when = pd.Timestamp("2024-01-02")
    past = {"date": when, "symbol": "510300", "kind": "etf", "known_at": pd.Timestamp("2024-01-02T08:00Z"),
            "main_flow_fraction": 0.0, "extra_large_flow_fraction": -0.1, "large_flow_fraction": 0.1}
    future = dict(past, known_at=pd.Timestamp("2024-01-03T00:00Z"), main_flow_fraction=0.2)
    decision = pd.DataFrame([{"date": when, "symbol": "510300", "kind": "etf", "as_of": pd.Timestamp("2024-01-02T12:00Z")}])
    joined = fund_flow_asof(pd.DataFrame([past, future]), decision)
    assert joined.loc[0, "main_flow_fraction"] == 0.0
