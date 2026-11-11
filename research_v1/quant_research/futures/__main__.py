"""Run an isolated futures backtest or local paper account."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .engine import Contract, FuturesError, MovingAverage, RidgeForecast, backtest, number, read_bars
from .output import save_backtest


def load_contract(path: Path) -> Contract:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FuturesError("contract JSON must be an object")
    return Contract.from_mapping(raw)


def fixture(path: Path) -> tuple[Path, Path, Path]:
    path.mkdir(parents=True, exist_ok=False)
    spec = {"symbol": "SHFE.ag2701", "multiplier": 15, "tick_size": 1,
            "margin_rate": "0.12", "max_margin_fraction": "0.40", "fee_open": 3,
            "fee_close": 3, "slippage_ticks": 1, "max_lots": 1, "exit_by": "2026-12-31"}
    config = path / "synthetic_contract.json"
    config.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bars = path / "synthetic_bars.csv"
    closes = (100, 101, 102, 104, 107, 106, 103, 99, 95, 97, 101, 105, 102, 98, 94, 96)
    with bars.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("trading_day", "symbol", "open", "high", "low", "close", "settlement", "upper_limit", "lower_limit"))
        for idx, close in enumerate(closes):
            day = date(2026, 1, 1) + timedelta(days=idx)
            opening = 100 if idx == 0 else closes[idx - 1]
            writer.writerow((day.isoformat(), spec["symbol"], opening, max(opening, close) + 1,
                             min(opening, close) - 1, close, close, 200, 50))
    events = path / "synthetic_events.jsonl"
    with events.open("w", encoding="utf-8") as handle:
        for idx, close in enumerate(closes):
            when = (datetime(2026, 1, 2, 1, 0, tzinfo=timezone.utc) + timedelta(minutes=idx)).isoformat()
            base = {"time": when, "trading_day": "2026-01-02", "symbol": spec["symbol"]}
            quote = base | {"type": "quote", "last": close, "bid": close - 1, "ask": close + 1,
                            "bid_size": 10, "ask_size": 10, "lower_limit": 50, "upper_limit": 200}
            bar = base | {"type": "bar", "bar_end": when, "close": close}
            for event in (quote, bar):
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        settlement = {"type": "settlement", "time": "2026-01-02T07:05:00+00:00",
                      "trading_day": "2026-01-02", "symbol": spec["symbol"], "price": closes[-1]}
        handle.write(json.dumps(settlement, ensure_ascii=False) + "\n")
    return config, bars, events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="期货研究：只使用虚拟资金，不连接期货交易账户")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="合成数据的账本演示")
    demo.add_argument("--out", type=Path, required=True)
    test = sub.add_parser("backtest", help="用自己的单一真实月份合约日线进行回测")
    test.add_argument("--config", type=Path, required=True)
    test.add_argument("--bars", type=Path, required=True)
    test.add_argument("--source", required=True, help="数据来源说明；不会因此被视为已审计")
    test.add_argument("--synthetic", action="store_true", help="将输入明确标为合成数据，仅用于工程验证")
    test.add_argument("--capital", required=True, help="虚拟本金，单位元")
    test.add_argument("--fast", type=int, default=3)
    test.add_argument("--slow", type=int, default=5)
    test.add_argument("--lots", type=int, default=1)
    test.add_argument("--strategy", choices=("moving_average", "ridge"), default="moving_average")
    test.add_argument("--lookback", type=int, default=3, help="ridge 过去收益特征数量")
    test.add_argument("--min-train", type=int, default=30, help="ridge 至少需要的历史训练样本数")
    test.add_argument("--train-window", type=int, default=120, help="ridge 最多使用的历史训练样本数")
    test.add_argument("--ridge-alpha", type=float, default=1.0)
    test.add_argument("--threshold", type=float, default=0.0005)
    test.add_argument("--out", type=Path, required=True)
    paper = sub.add_parser("paper", help="持续处理追加到 JSONL 的行情事件，可重启恢复")
    paper.add_argument("--config", type=Path, required=True)
    paper.add_argument("--events", type=Path, required=True)
    paper.add_argument("--capital", required=True)
    paper.add_argument("--fast", type=int, default=3)
    paper.add_argument("--slow", type=int, default=5)
    paper.add_argument("--lots", type=int, default=1)
    paper.add_argument("--strategy", choices=("moving_average", "external"), default="moving_average",
                       help="external 接收带时间戳的外部模型信号")
    paper.add_argument("--session", type=Path, required=True)
    paper.add_argument("--watch", action="store_true", help="持续监测行情文件，Ctrl+C 停止")
    paper.add_argument("--poll-seconds", type=float, default=2)
    feed = sub.add_parser("tq-feed", help="可选：读取天勤行情并追加到本地事件文件，不向券商下单")
    feed.add_argument("--config", type=Path, required=True)
    feed.add_argument("--events", type=Path, required=True)
    feed.add_argument("--trading-day", required=True, help="明确指定交易日，夜盘不能简单使用自然日")
    feed.add_argument("--seconds", type=int, required=True)
    feed.add_argument("--period", type=int, default=60, choices=(60, 300, 900))
    feed.add_argument("--max-age", type=float, default=30)
    settlement = sub.add_parser("settle", help="追加人工核验的上一交易日结算价")
    settlement.add_argument("--config", type=Path, required=True)
    settlement.add_argument("--events", type=Path, required=True)
    settlement.add_argument("--trading-day", required=True)
    settlement.add_argument("--price", required=True)
    probe = sub.add_parser("probe-sina", help="采集原始期货日线；缺少历史限价时不会视为回测准入")
    probe.add_argument("--symbol", required=True, help="新浪真实月份合约代码，例如 AG2609")
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--timeout", type=int, default=30)
    probe.add_argument("--attempts", type=int, choices=(1, 2), default=1)
    try:
        args = parser.parse_args(argv)
        if args.command == "demo":
            config, bars, events = fixture(args.out)
            contract = load_contract(config)
            result = backtest(read_bars(bars, contract), contract, MovingAverage(2, 3), capital="100000")
            out = args.out / "backtest"
            save_backtest(out, result, contract=contract, capital="100000", bars_path=bars, config_path=config,
                          source="synthetic_fixture", fast=2, slow=3, lots=1, synthetic=True)
            from .paper import run_session

            paper_dir = args.out / "paper"
            run_session(paper_dir, events, config, contract, MovingAverage(2, 3), capital="100000")
            print(json.dumps({"origin": "synthetic_fixture", "report": str((out / "report.html").resolve()),
                              "paper_report": str((paper_dir / "report.html").resolve()),
                              "status": result.status}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "backtest":
            # Read once into a private snapshot, so results and archived input refer to identical bytes.
            with tempfile.TemporaryDirectory(prefix="futures-backtest-") as scratch:
                frozen = Path(scratch)
                frozen_config, frozen_bars = frozen / "contract.json", frozen / "bars.csv"
                frozen_config.write_bytes(args.config.read_bytes())
                frozen_bars.write_bytes(args.bars.read_bytes())
                contract = load_contract(frozen_config)
                if args.lots > contract.max_lots:
                    raise FuturesError("strategy lots exceeds contract max_lots")
                if args.strategy == "ridge":
                    strategy = RidgeForecast(args.lookback, args.min_train, args.train_window,
                                             args.ridge_alpha, args.threshold, args.lots)
                    metadata = {"name": "ridge", "lookback": args.lookback, "min_train": args.min_train,
                                "train_window": args.train_window, "alpha": args.ridge_alpha,
                                "threshold": args.threshold, "lots": args.lots}
                else:
                    strategy = MovingAverage(args.fast, args.slow, args.lots)
                    metadata = {"name": "moving_average", "fast": args.fast, "slow": args.slow, "lots": args.lots}
                capital = str(number(args.capital, "capital"))
                result = backtest(read_bars(frozen_bars, contract), contract, strategy, capital=capital)
                save_backtest(args.out, result, contract=contract, capital=capital, bars_path=frozen_bars,
                              config_path=frozen_config, source=args.source, fast=args.fast, slow=args.slow, lots=args.lots,
                              strategy_metadata=metadata, synthetic=args.synthetic)
            print(json.dumps({"origin": "synthetic_fixture" if args.synthetic else "user_supplied_unverified", "status": result.status,
                              "warnings": result.warnings, "report": str((args.out / "report.html").resolve())},
                             ensure_ascii=False, indent=2))
            return 0 if result.status == "completed" else 2
        if args.command == "tq-feed":
            from .tqfeed import run_feed

            run_feed(args.events, load_contract(args.config), trading_day=args.trading_day,
                     seconds=args.seconds, period=args.period, max_age=args.max_age)
            return 0
        if args.command == "probe-sina":
            from .probe import capture_sina

            audit = capture_sina(args.symbol, args.out, timeout=args.timeout, attempts=args.attempts)
            print(json.dumps(audit, ensure_ascii=False, indent=2))
            return 0 if "raw_sha256" in audit else 2
        if args.command == "settle":
            from .tqfeed import append_settlement

            append_settlement(args.events, load_contract(args.config), day=args.trading_day, price=args.price)
            print("已追加结算事件；价格仍须由使用者核验。")
            return 0
        from .paper import run_session

        contract = load_contract(args.config)
        strategy = MovingAverage(args.fast, args.slow, args.lots)
        if args.lots > contract.max_lots:
            raise FuturesError("strategy lots exceeds contract max_lots")
        run_session(args.session, args.events, args.config, contract, strategy, capital=args.capital,
                    watch=args.watch, poll_seconds=args.poll_seconds, signal_mode=args.strategy)
        return 0
    except (FuturesError, OSError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已停止；本地模拟会话已保留。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
