# AI量化

面向 A 股与场内股票 ETF 的个人研究项目，探索时序分析、知识图谱、强化学习与新闻证据的组合。当前以独立模拟账本和可复现验证为基础，尚未形成通过验收的真实市场盈利策略。

## 项目结构

| 目录 | 内容与状态 |
|---|---|
| [`src/`](src/) | PatchTST → GAT → RL 算法原型，包含合成数据机制实验和部分历史行情滚动回放；尚未通过独立的策略有效性验收 |
| [`research_v1/`](research_v1/) | 独立的数据审计、现金与持仓账本、基准策略、时间隔离、新闻／舆论证据、带时间版本的知识图谱与区块重采样工具 |

`research_v1` 不导入原型代码，两条线路使用独立环境和产物目录。原型已发现的核算与时点问题及后续比较要求，见[独立验收记录](research_v1/PROTOTYPE_REVIEW.md)。

## 当前验证状态

截至 2026-11-14，本机针对独立研究线、原型和新闻前向模块分别通过 **104 项、72 项（另有 19 个子测试）和 138 项测试**。原型测试出现两条 NumPy 计算警告。此前完成的 64 个工程演示场景使用合成夹具，只验证现金、持仓、费用、分红、时间规则和报告流程。测试通过不代表真实行情流程或盈利能力已通过验收。

真实数据入口保留来源、哈希和失败记录，不会静默切换为合成数据。本地 `market-etf-verified-20261108` 在 2015—2024 年 ETF 审计中为 `ready`，四只 ETF 均有 2,431 个交易日；`market-wide28-verified-20261108` 的通用个股审计仍为 `data_insufficient`。历史供应商快照缺完整的历史获知时间及修订证据，文件完整不等于数据已经完全可靠。

[固定双动量研究](research_v1/MOMENTUM_STUDY.md)在已探索的 2023—2024 主窗落后于基线，已拒绝采用；该实验当时的源码快照和本地数据已用于隔离重放，逐日账本、决策和订单文件与原结果逐字节一致。[候选登记表](research_v1/REJECTED.md)保留了其他已拒绝、未晋级或停用的实验。四只 ETF 历史研究中，2025 年起的封存行情未进入这些计算；已反复查看的 2023—2024 结果也不是新的独立样本外证据。

详细的数据盘点、备份检查范围、实验版本与重放步骤见[项目状态与复现清单](PROJECT_STATUS.md)。行情、数据库和多数实验产物被 Git 忽略，只有代码提交不足以恢复这些本地资料；目前尚未核实到覆盖它们的独立备份。

较早阶段的交付记录见[2026-11-07 实施检查点](research_v1/CHECKPOINT.md)。

## 快速开始

在仓库根目录执行，需要 Python 3.11 与 [uv](https://docs.astral.sh/uv/)：

```bash
cd research_v1
uv sync --frozen --extra models
uv run --frozen --extra models python -m pytest -q

# 使用新的输出目录运行工程演示。
uv run --frozen --extra models python -m quant_research.cli demo --out artifacts/demo-001
```

演示结束后打开 `research_v1/artifacts/demo-001/report.html`。已有产物目录禁止覆盖；再次运行时选择新的目录名。

数据探测、真实快照审计、分研究线运行与前向证据接口，见[研究工具使用说明](research_v1/README.md)。

计划于 2026-11-16 至 11-20 开展的本地前向对照使用独立模拟账本。截至 2026-11-14，仓库已收录[冻结协议](artifacts/news-filter-forward-20261116/protocol.md)、[运行说明](artifacts/news-filter-forward-20261116/runbook.md)及源码；正式结果仍待实际采集。安装上面的研究环境后，在仓库根目录运行离线测试：

```bash
PYTHONPATH=research_v1 research_v1/.venv/bin/python -m unittest discover \
  -s artifacts/news-filter-forward-20261116 -p 'test_*.py'
```

新检出可运行合成情景；需要本地候选、新闻、原始行情或特征归档的回放测试会报告缺失并跳过。时点校验、盘口回放、收盘对账和周报代码都用于虚拟资金对照。

## 数据与研究边界

- 原始行情、数据库、虚拟环境和本地实验产物不在仓库中；使用者需自行依法获取并遵守各数据源的使用条件。
- 所有账户和账本均为独立模拟，不连接真实交易账户。回测和重采样结果不能直接解释为未来收益概率。
- 最大回撤 15% 是预登记研究筛选线，不是实际亏损上限保证。真实数据未通过审计前，不出具策略有效性结论。

设计参考：[Qlib](https://github.com/microsoft/qlib)、[FinRL](https://github.com/AI4Finance-Foundation/FinRL)。依赖和数据源的许可分别适用。
