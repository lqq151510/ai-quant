"""Readable reports and immutable backtest artifacts."""

from __future__ import annotations

import csv
import hashlib
import html
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .engine import Contract, Result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_file(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def report_html(*, title: str, explanation: str, equity: list[dict[str, object]],
                orders: list[dict[str, object]], status: str, capital: str,
                symbol: str = "", pending: dict | None = None) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    values = [float(Decimal(str(row["equity"]))) for row in equity]
    points = [float(Decimal(capital))] + values
    low, high = min(points), max(points)
    spread = max(high - low, abs(high) * 0.001, 1.0)
    coordinates = " ".join(f"{20 + i * 760 / max(len(points)-1, 1):.1f},{190 - (v-low)*150/spread:.1f}"
                           for i, v in enumerate(points))
    final = values[-1] if values else float(Decimal(capital))
    recent = orders[-20:]
    translations = {"filled": "已成交", "partial": "部分成交", "rejected": "未成交", "cancelled": "已取消",
                    "price_limit_or_slippage": "涨跌停或滑点限制", "insufficient_margin_or_risk_limit": "保证金不足或超出风控",
                    "limited_quoted_liquidity": "盘口手数不足", "no_quoted_liquidity": "无可用盘口",
                    "signal_expired_60s": "信号超过 60 秒", "max_lots": "超过最大手数"}
    table = "".join("<tr>" + "".join(f"<td>{esc(translations.get(row.get(k, ''), row.get(k, '')))}</td>" for k in
                            ("trading_day", "requested_target", "filled_delta", "fill_price", "status", "reason"))
                    + "</tr>" for row in recent)
    if not table:
        table = "<tr><td colspan='6'>暂无订单</td></tr>"
    latest = equity[-1] if equity else {}
    status_text = {"paper_only": "仅虚拟资金", "completed": "计算完成", "exit_unresolved": "到期平仓未完成"}.get(status, status)
    refresh = '<meta http-equiv="refresh" content="5">' if status == "paper_only" else ""
    pending_text = f"等待下一笔报价，目标 {pending['target']} 手" if pending else "暂无待执行信号"
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title>
<style>body{{max-width:1060px;margin:36px auto;padding:0 20px;background:#f3f7f5;color:#142720;font:16px/1.6 system-ui,sans-serif}}
h1{{line-height:1.2}}.note{{background:#e0f0e8;border-left:4px solid #247455;padding:12px 16px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:white;border:1px solid #dbe7e1;border-radius:10px;padding:12px 18px;flex:1;min-width:150px}}
.value{{font-size:1.4em;font-weight:700}}svg{{width:100%;height:auto;background:white;border:1px solid #dbe7e1;border-radius:10px}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;border-bottom:1px solid #dbe7e1;text-align:left;white-space:nowrap}}
.scroll{{overflow:auto}}thead{{background:#193b32;color:white}}small{{color:#536961}}</style>
<h1>{esc(title)}</h1><p class="note">{esc(explanation)}</p><div class="cards">
<div class="card">虚拟初始资金<div class="value">{esc(capital)}</div></div>
<div class="card">最新模拟权益<div class="value">{final:.2f}</div></div>
<div class="card">运行状态<div class="value">{esc(status_text)}</div></div></div>
<div class="cards" style="margin-top:12px"><div class="card">模拟合约<div class="value">{esc(symbol or '—')}</div></div>
<div class="card">持仓手数<div class="value">{esc(latest.get('position', 0))}</div></div>
<div class="card">占用保证金<div class="value">{esc(latest.get('margin', '0.00'))}</div></div>
<div class="card">可用资金<div class="value">{esc(latest.get('available', capital))}</div></div></div>
<p><small>{esc(pending_text) if status == 'paper_only' else ''}{'；页面每 5 秒刷新' if status == 'paper_only' else ''}</small></p>
<h2>模拟权益变化</h2><svg viewBox="0 0 800 220" role="img" aria-label="模拟权益曲线">
<line x1="20" y1="190" x2="780" y2="190" stroke="#c6d9ce"/><polyline points="{coordinates}" fill="none" stroke="#16724f" stroke-width="3"/></svg>
<small>从初始权益开始；纵轴按当前范围缩放，具体数据见 CSV 或状态文件。</small>
<h2>最近订单</h2><div class="scroll"><table><thead><tr><th>交易日/时间</th><th>目标手数</th><th>成交变化</th>
<th>成交价</th><th>状态</th><th>原因</th></tr></thead><tbody>{table}</tbody></table></div></html>"""


def save_backtest(output: Path, result: Result, *, contract: Contract, capital: str,
                  bars_path: Path, config_path: Path, source: str, fast: int, slow: int, lots: int,
                  synthetic: bool = False, strategy_metadata: dict | None = None) -> None:
    # The caller owns the fresh output directory; never silently replace a prior run.
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    (inputs / "bars.csv").write_bytes(bars_path.read_bytes())
    (inputs / "contract.json").write_bytes(config_path.read_bytes())
    source_copy = output / "source"
    source_copy.mkdir()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        (source_copy / path.name).write_bytes(path.read_bytes())
    eq_columns = ("trading_day", "position", "settlement", "gap_pnl", "intraday_pnl", "fees", "slippage",
                  "equity", "margin", "available", "drawdown", "margin_call")
    order_columns = ("decision_day", "trading_day", "trigger", "requested_target", "position_before", "filled_delta",
                     "position_after", "open_price", "fill_price", "closed_lots", "opened_lots", "fee", "slippage", "status", "reason")
    csv_file(output / "equity.csv", result.equity, eq_columns)
    csv_file(output / "orders.csv", result.orders, order_columns)
    csv_file(output / "decisions.csv", result.decisions, ("trading_day", "position", "target", "execution"))
    origin = "synthetic_fixture" if synthetic else "user_supplied_unverified"
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    manifest = {"mode": "backtest", "origin": origin, "source_description": source,
                "status": result.status, "warnings": result.warnings,
                "symbol": contract.symbol, "virtual_capital": capital,
                "strategy": strategy_metadata or {"name": "moving_average", "fast": fast, "slow": slow, "lots": lots},
                "inputs": {"bars_sha256": sha256(inputs / "bars.csv"), "config_sha256": sha256(inputs / "contract.json")},
                "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "numpy_version": numpy_version,
                "code_sha256": {path.name: sha256(path) for path in sorted(source_copy.glob("*.py"))}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    label = "合成数据，仅验证工程流程，不代表真实行情或收益。" if synthetic else (
        "用户提供的数据和合约参数尚未经独立审计；回测结果不代表未来表现。")
    label += " 日线开盘撮合是假设，缺少盘口、排队、交易日历和自动换月。"
    (output / "report.html").write_text(report_html(title="期货虚拟资金回测", explanation=label,
                                                    equity=result.equity, orders=result.orders,
                                                    status=result.status, capital=capital,
                                                    symbol=contract.symbol), encoding="utf-8")
