# 2026 年 11 月 20 日模拟策略阶段收尾

本轮结果来自 11 月 16–18 日已核实的公开盘口和本地模拟账本。各对照以冻结协议中的 100,000 模拟元为起点。11 月 19–20 日缺少连续可对账记录，三日数据属于已观察样本。

| 三日对照 | 扣费后损益（模拟元） | 收盘最大回撤 | 主要发现 |
| --- | ---: | ---: | --- |
| 原规则 | −1,248.41 | 1.24841% | 共 37 笔模拟成交 |
| 同一批成交股数翻倍 | −2,311.91 | 2.31191% | 相同报价下增加的损失为 1,063.50 元 |
| 同盘口反向订单净额 | −1,014.03 | 1.06534% | 每日收盘持仓一致，成交缩至 18 笔，成本减少 234.38 元 |
| 开盘确认 | +51.39 | 0.04448% | 保留 6/21 笔买入，平均收盘仓位由 29.49% 降至 7.85% |
| 留现金 | 0 | 0% | 独立等额现金基准，利息记为零 |

按每日含买费的实际投入匹配后，开盘确认组相对原规则的 1,299.80 元改善可拆成 919.27 元的投入变化与 380.53 元的同金额选单差额。选单差额集中在前两天。在佣金、最低佣金和滑点加倍的固定整手复算中，确认组为 −18.61 元，相同投入的按比例对照为 −389.13 元，现金为 0。候选仍属于研究观察，现有证据不足以晋级或支持加仓。

## 可复用的模拟工具

- [仓位回放](../artifacts/paper-exposure-20261120/report.md)复核同批成交的股数、现金与费用。
- [同盘口净额对照](../artifacts/paper-order-netting-20261120/report.md)以原始双边盘口为依据，只抵消同时可判断的反向订单。
- [开盘确认](../artifacts/paper-open-confirmation-20261120/protocol.md)在原可成交买单中对比盘口中间价与未复权昨收，并记录跳过的亏损与盈利。
- [同金额选单对照](../artifacts/paper-entry-selection-20261120/protocol.md)比较同日、同金额及同笔数结果，单列已退出交易与压力成本。
- [独立前向试验](../artifacts/paper-forward-20261123/protocol.md)使用 11 月 23–27 日的新账本观察原规则和留现金。已存在的盘前快照中，63 个标的有 45 个可判定，18 个 ETF 仍有复权证据缺口。收尾时没有已结算交易日。ETF 定向补采仍在本地实验状态。

这些本地脚本复用原冻结盘口和费用规则。日结和逐笔原始快照留在各自的本地目录。仓库提供源码、测试和结果摘要。历史复算需要在本机保存对应交易日的原始候选、盘口和收盘证据，测试会明确报告缺失的本地档案。

## 验证入口

在仓库根目录使用研究线的 Python 环境运行以下离线验证。三组策略对照各用独立的发现入口。

```bash
for study in paper-order-netting-20261120 paper-open-confirmation-20261120 paper-entry-selection-20261120; do
  PYTHONDONTWRITEBYTECODE=1 research_v1/.venv/bin/python -m unittest discover -s "artifacts/$study" -p 'test_*.py' || exit
done
```

有本地原始档案时可运行以下命令逐笔复算。输出到标准输出，原账本可继续核对。

```bash
research_v1/.venv/bin/python artifacts/paper-exposure-20261120/replay.py
research_v1/.venv/bin/python artifacts/paper-order-netting-20261120/archive.py
research_v1/.venv/bin/python artifacts/paper-open-confirmation-20261120/confirm.py archive
research_v1/.venv/bin/python artifacts/paper-entry-selection-20261120/matched.py archive
research_v1/.venv/bin/python artifacts/paper-entry-selection-20261120/stress_matched.py archive
```
