# AI量化

面向 A 股与场内股票 ETF 的个人研究项目，探索时序分析、知识图谱、强化学习与新闻证据的组合。当前以独立模拟账本和可复现验证为基础，尚未形成通过验收的真实市场盈利策略。

## 项目结构

| 目录 | 内容与状态 |
|---|---|
| [`src/`](src/) | Antigravity 构建的 PatchTST → GAT → RL 算法原型，使用合成行情进行机制实验；其中收益指标不代表真实市场收益 |
| [`research_v1/`](research_v1/) | 独立的数据审计、现金与持仓账本、基准策略、时间隔离、新闻／舆论证据、带时间版本的知识图谱与区块重采样工具 |

`research_v1` 不导入原型代码，两条线路使用独立环境和产物目录。原型已发现的核算与时点问题及后续比较要求，见[独立验收记录](research_v1/PROTOTYPE_REVIEW.md)。

## 当前验证状态

截至 2026-11-07，`research_v1` 在 Python 3.11 环境中通过 **51 项测试**，并完成 **64 个工程演示场景**。这些演示使用明确标记的合成夹具，验证现金、持仓、费用、分红、时间规则和报告流程。

真实数据入口保留来源、哈希和失败记录，不会静默切换为合成数据。当前真实数据闸门仍为 `data_insufficient`：ETF 缺日、公司行动、历史限价及股票池时点等证据尚待补齐。ARIMA／GARCH／LightGBM 滚动训练、PPO 对照、模型 Agent 与长期前向观察尚未完成；依赖声明不等于功能已实现。

详细交付和后续工作见[实施检查点](research_v1/CHECKPOINT.md)。

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

## 数据与研究边界

- 原始行情、数据库、虚拟环境和本地实验产物不在仓库中；使用者需自行依法获取并遵守各数据源的使用条件。
- 所有账户和账本均为独立模拟，不连接真实交易账户。回测和重采样结果不能直接解释为未来收益概率。
- 最大回撤 15% 是预登记研究筛选线，不是实际亏损上限保证。真实数据未通过审计前，不出具策略有效性结论。

设计参考：[Qlib](https://github.com/microsoft/qlib)、[FinRL](https://github.com/AI4Finance-Foundation/FinRL)。依赖和数据源的许可分别适用。
