"""Run isolated experiments and write only into a newly created output directory."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import duckdb
import numpy as np
import pandas as pd
import torch

from .data import fetch_akshare_inputs
from .experiment import run_experiment, synthetic_bars


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="仅限研究：数据及算法升级，不连接交易账户")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="使用明确标记的合成行情")
    source.add_argument("--bars-csv", type=Path, help="本地日线 CSV，symbol 必须为六位代码")
    source.add_argument("--bars-duckdb", type=Path, help="只读本地 DuckDB 的 daily_bars 表；来源仍需核验")
    parser.add_argument("--flow-csv", type=Path, help="可选：已规范化的资金流 CSV")
    parser.add_argument("--minutes-csv", type=Path, help="可选：已规范化的 5 分钟 K 线 CSV")
    parser.add_argument("--macro-bars-csv", type=Path, help="可选：用于市场状态监测的更广泛标的日线")
    parser.add_argument("--fetch-akshare", action="store_true", help="只读获取最近资金流与分钟 K 线")
    parser.add_argument("--fetch-symbol", action="append", help="限制联网标的；可重复，默认输入中的所有标的")
    parser.add_argument("--out", type=Path, required=True, help="新的输出目录；已有目录不会被覆盖")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--synthetic-days", type=int, default=360)
    args = parser.parse_args()
    if args.synthetic and (args.fetch_akshare or args.flow_csv or args.minutes_csv or args.macro_bars_csv):
        parser.error("合成行情不能与真实或外部资金流/分钟数据混用")
    if args.fetch_akshare and (args.flow_csv or args.minutes_csv):
        parser.error("联网获取与外部资金流/分钟 CSV 请分别运行")
    if args.fetch_symbol and not args.fetch_akshare:
        parser.error("--fetch-symbol 需要与 --fetch-akshare 一起使用")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)

    if args.synthetic:
        bars = synthetic_bars(args.synthetic_days)
        source_label = "synthetic"
    elif args.bars_csv:
        bars = pd.read_csv(args.bars_csv, dtype={"symbol": str})
        source_label = "user_supplied_csv_unverified"
    else:
        with duckdb.connect(str(args.bars_duckdb), read_only=True) as connection:
            bars = connection.execute("SELECT symbol, trade_date, open, high, low, close, volume FROM daily_bars").df()
        source_label = "duckdb_unverified"
    flow = pd.read_csv(args.flow_csv, dtype={"symbol": str}) if args.flow_csv else None
    minutes = pd.read_csv(args.minutes_csv, dtype={"symbol": str}) if args.minutes_csv else None
    macro_bars = pd.read_csv(args.macro_bars_csv, dtype={"symbol": str}) if args.macro_bars_csv else None
    status = {"source": "not_requested", "results": []}
    if args.fetch_akshare:
        symbols = args.fetch_symbol or sorted(bars["symbol"].astype(str).str.zfill(6).unique())
        flow, minutes, status = fetch_akshare_inputs(symbols)
    (args.out / "provider_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    bars.to_csv(args.out / "daily_bars.csv", index=False)
    if flow is not None:
        flow.to_csv(args.out / "fund_flow.csv", index=False)
    if minutes is not None:
        minutes.to_csv(args.out / "minute_bars.csv", index=False)
    if macro_bars is not None:
        macro_bars.to_csv(args.out / "macro_bars.csv", index=False)

    report, model, tree, transform, monitor = run_experiment(bars, flow, minutes, epochs=args.epochs, macro_bars=macro_bars)
    torch.save(model.state_dict(), args.out / "temporal_model.pt")
    joblib.dump(tree, args.out / "tree_model.joblib")
    joblib.dump(monitor, args.out / "regime_monitor.joblib")
    np.savez(args.out / "lowdin.npz", mean=transform.mean, matrix=transform.matrix,
             projection=transform.projection, rank=transform.rank)
    report["bars_source"] = source_label
    report["auxiliary_data_source"] = status["source"] if args.fetch_akshare else ("user_supplied_csv_unverified" if args.flow_csv or args.minutes_csv else "not_supplied")
    report["macro_data_source"] = "user_supplied_csv_unverified" if args.macro_bars_csv else source_label
    report["input_hashes"] = {p.name: _hash(p) for p in sorted(args.out.glob("*.csv"))}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
