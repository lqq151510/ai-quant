"""HTML summary and machine-readable output of the exact same run."""

from __future__ import annotations

import html
import json
from pathlib import Path


def save_report(output: Path, metrics: list[dict], *, origin: str, period: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    synthetic = origin == "synthetic_fixture"
    title = "合成夹具｜工程对账演示" if synthetic else "真实历史｜基线研究（不含封存测试）"
    interpretation = (
        "所有行情均为程序构造；本页只验证时间、账本和报告流程，不能用于判断真实收益。"
        if synthetic else "按快照审计准入运行的基线；本页不包含样本外增量证据或前向验证。"
    )
    summary = {"origin": origin, "status": "engineering_only" if synthetic else "baseline_only",
               "period": period, "interpretation": interpretation, "metrics": metrics}
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")

    def cell(value: object) -> str:
        return f"<td>{html.escape(str(value))}</td>"

    allocation_study = any("allocation_target_metrics" in row for row in metrics)
    allocation_headers = (
        "<th>成交额 / 平均净值</th><th>平均目标总仓位</th><th>最高单票目标</th><th>目标集中度 HHI</th>"
        if allocation_study else ""
    )
    allocation_note = (
        '<p>分配对照共用候选和目标总仓位。HHI 为目标股票仓位归一化后的权重平方和，数值越高越集中。'
        '查看 <a href="allocation_plan.json">模拟目标记录</a>。</p>'
        if allocation_study else ""
    )
    optimization_rows = []
    rows = []
    for row in metrics:
        ledger = "-".join(str(row[k]) for k in ("line", "capital", "cost_case", "strategy"))
        cells = [row["line"], f"{row['capital']:,}", row["cost_case"], row["strategy"],
                 f"{row['net_return']:+.2%}", f"{row['max_drawdown']:.2%}", row["filled_orders"],
                 f"{row['transaction_fees']:.2f}"]
        if allocation_study:
            allocation = row.get("allocation_target_metrics", {})
            gross = allocation.get("mean_gross_weight")
            single = allocation.get("max_single_name_weight")
            hhi = allocation.get("mean_invested_hhi")
            cells.extend([
                f"{row['traded_notional_over_average_nav']:.2%}",
                "—" if gross is None else f"{gross:.2%}",
                "—" if single is None else f"{single:.2%}",
                "—" if hhi is None else f"{hhi:.3f}",
            ])
        target = html.escape(f"ledgers/{ledger}/equity.parquet", quote=True)
        rows.append(f"<tr>{''.join(map(cell, cells))}<td><a href='{target}'>净值明细</a></td></tr>")
        if "vs_hrp" in row:
            difference = row["vs_hrp"]
            decisions = row["optimization_diagnostics"]
            comparison = [row["line"], f"{row['capital']:,}", row["cost_case"], row["strategy"]]
            for metric in ("net_return", "max_drawdown", "traded_notional_over_average_nav", "mean_realized_gross_weight"):
                comparison.append(f"{100 * difference[metric]:+.2f}")
            comparison.extend([decisions["skipped_count"], decisions["defensive_count"], decisions["unknown_breadth_count"]])
            optimization_rows.append(f"<tr>{''.join(map(cell, comparison))}<td><a href='{target}'>净值明细</a></td></tr>")
    optimization = ""
    if optimization_rows:
        optimization = f"""<h2>HRP 策略对照</h2>
<p>每组与同研究线、同本金、同费用的原始 HRP 比较。收益、回撤、换手和仓位差均为百分点，候选减原始 HRP。</p>
<div class="scroll"><table><thead><tr><th>研究线</th><th>虚拟本金</th><th>费用情景</th><th>候选</th><th>净收益差</th>
<th>最大回撤差</th><th>换手差</th><th>平均实仓差</th><th>跳过调仓</th><th>降仓决策</th><th>趋势数据缺失</th>
<th>账本</th></tr></thead><tbody>{''.join(optimization_rows)}</tbody></table></div>"""
    (output / "report.html").write_text(f"""<!DOCTYPE html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{max-width:1150px;margin:44px auto;padding:0 20px;font:16px/1.65 system-ui,sans-serif;color:#13251f;background:#f4f8f6}}
h1{{line-height:1.2}} .note{{padding:12px 16px;border-left:4px solid #1a6854;background:#e1f0e9}}
table{{border-collapse:collapse;width:100%;background:#fff;font-variant-numeric:tabular-nums}}
td,th{{border-bottom:1px solid #dce5e0;text-align:left;padding:8px 10px}}
.scroll{{overflow:auto}} thead{{background:#193b32;color:white}} tr:nth-child(even){{background:#f5f9f7}} a{{color:#13614d}}
</style><h1>{title}</h1><p>{html.escape(period)}</p><p class="note">{interpretation}</p>
<p>权益、现金、持仓、成交和分红事件分别保存。重采样情景见 <a href="metrics.json">metrics.json</a>。</p>
{optimization}
{allocation_note}
<div class="scroll"><table><thead><tr><th>研究线</th><th>虚拟本金</th><th>费用情景</th><th>策略</th><th>净收益</th>
<th>最大回撤</th><th>有成交订单</th><th>交易费用</th>{allocation_headers}<th>账本</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></html>
""", encoding="utf-8")
