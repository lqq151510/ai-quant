# 泽宝的 A 股 / ETF 研究

这是获批研究方案的独立实现。上一级 `src/`、`.venv/` 和 `data/` 属于 Antigravity 算法原型，本子项目不导入或修改它们。

当前交付：真实来源探测与原始证据、不可变 Parquet 快照及数据闸门、可对账的模拟账本、四条研究线的基准、时间隔离工具、前向新闻/舆论与知识图谱、区块重采样风险报告。

尚未交付：ARIMA/GARCH/LightGBM 的滚动训练、PPO 五种子对照、真实本地模型 Agent、候选实验注册与自动复盘、封存测试一次性开启、60 日前向运行。依赖已声明不代表这些功能已实现。

## 安装与运行

```bash
cd ai-quant/research_v1
uv sync --frozen --extra models
uv run --frozen --extra models python -m pytest -q

# 明确的合成夹具，只验证工程，不能据其收益判断市场有效性。
uv run --frozen --extra models python -m quant_research.cli demo --out artifacts/demo-001

# 匿名公开数据探测；输出目录必须不存在。失败记录与实际CSV都保存。
uv run --frozen --extra models python -m quant_research.cli probe \
  --start 2015-01-01 --end 2026-11-06 --out artifacts/real-probe-001

# 经过整理的完整快照才可审计与运行真实基线。
uv run --frozen --extra models python -m quant_research.cli audit /absolute/snapshot --end 2024-12-31
uv run --frozen --extra models python -m quant_research.cli baseline /absolute/snapshot \
  --end 2024-12-31 --out artifacts/baseline-001
```

本轮已建好的独立环境也可直接运行 `.venv/bin/python -m pytest -q` 和 `.venv/bin/python -m quant_research.cli ...`。`--kind etf` 或 `--kind stock` 可分别审计、运行研究线，另一类数据不足不会阻断本类研究；省略该参数则检查两类。

`demo` 的 `report.html` 和 `metrics.json` 始终标记 `synthetic_fixture/engineering_only`。输出64份独立账本：ETF/个股 × 5/20交易日 × 1万/10万元 × 正常/双倍摩擦 × 现金/等权/动量/沪深300ETF持有。每份含每日权益、逐笔订单、持仓和现金事件。用 DuckDB 独立查询检查 `净值=现金+持仓价值+应收`。

`probe` 保存的是供应商实际返回表和部分源页面，不是完整研究数据集，也不是HTTP线上字节的全量抓包。日线采集成功不能证明历史成分、限价和公司行动覆盖合格。报告状态保持 `data_insufficient`，直到各层证据齐全。这个入口没有合成降级。

将本轮已保存的原始证据转为可审计的 Parquet 快照：

```bash
.venv/bin/python -m quant_research.cli prepare-etf \
  --probe artifacts/real-probe-20261107 \
  --calendar artifacts/real-calendar-sina-20261107/raw/trading_calendar_sina.csv \
  --start 2015-01-01 --end 2026-11-06 --out artifacts/market-etf-next
```

该命令先核对原始CSV哈希，显式解析每10份现金分红与份额比例，保存未知限价和未核实公司行动状态。它不补齐510500缺失的2015-04-13/14，不会将“已转换格式”当成“市场研究资格通过”。真实快照为 `market-etf-snapshot-20261107`；数据对照在 `real-calendar-sina-20261107/comparison.json`。

## 时间与研究约束

入场时点对照和 510300 净收益筛选的复验命令见 [2026-11-11 研究记录](STUDIES_20261111.md)。

- 预登记 ETF 为510050、510300、510500、510880。2015—2022开发，2023—2024选模，2025起封存；本版 `baseline` 命令拒绝进入封存段。
- 完整收盘信息生成信号，下一交易日开盘模拟执行。标签从 `t+1` 开盘至 `t+h+1` 开盘，h仅5或20。训练样本的标签结束时间必须严格早于下一分区。
- 个股按历史成分快照（需已知时间）与严格此前60个交易日成交额每月筛前50只，缺数据不退回当前股票池。供应商历史成分是周快照，不能据查询日期伪造调整公告的已知时间。
- 原始价格模拟成交；公司行动分别处理份额、应收、到账。累计分红序列不具备完整现金账本所需字段。510500在2015/2022发生份额变化，未核实前不能发布完整收益结论。
- 15%最大回撤是研究筛选线，不是止损保证。日线开盘价含不利滑点只是近似执行，缺盘口和开盘实际成交量。

## 交易账本

`ledger.simulate(bars, calendar, targets, actions, initial_cash=..., cost=Costs())` 接受执行日目标权重，记录实际股数、费用、现金与未成交原因。无交易时份额不变、权重随价格自然漂移。

`bars` 必需列：date、symbol、kind(stock/etf)、open、high、low、close、volume、amount、tradable、upper_limit、lower_limit。缺限价会拒单；持仓无可核收盘价会停止。可选 preclose 用于发现未解释的除权。

`actions` 必需列：symbol、ex_date、pay_date、cash_per_share、share_multiplier；可选 record_date、share_available_date。现金分红在除息时形成应收、发放后才可交易；股份变化出现小数份额时要求现金替代或准确分配证据，不默默舍入。cash_per_share按输入实入金额，个股持有期差异分红税尚须上游提供，不能把税前现金当个人税后收益。

默认佣金万三、最低5元、单边滑点5基点是研究假设。股票印花税、过户费按生效日期分段；默认股票过户费证据表只覆盖2015-08-01起，更早股票成交将停止，不能悄悄改研究起点。`Costs.stressed()`仅将佣金、最低佣金和滑点翻倍。

## 前向新闻、舆论与图谱

`evidence.load_evidence(path)`读取events、relations、coverage三组记录；`evidence_features(bundle, as_of=带时区时间, lookback=..., targets=...)`按该时点生成特征。CLI：

```bash
uv run --frozen --extra models python -m quant_research.cli evidence examples/evidence.json \
  --as-of 2026-11-07T18:00:00+08:00 --target etf:510300
```

每条新闻保存来源、发布/首次看到/特征就绪时间；每条关系保存事实有效期与版本获知时间。先选当时关系版本再遍历，撤销不会复活旧版本。三跳包括“事件→实体”的隐式第一跳。同事件的转载、多条路径只记一次；未知权重不伪造成分敞口。无可靠覆盖时计数为null，覆盖完整且无事件才为0。首版提供文件摄入与特征工具，尚未配置自动抓新闻调度或模型推理。

## 证据与回滚

原始来源、请求参数、采集时间、文件哈希与失败都保留；快照和运行目录禁止覆盖。重新运行要用新目录，重复读取相同快照会先校验哈希。移除本子项目即可撤销新增实现；保留其 artifacts 便于审计，不需要更改原型和原数据。

最终工程演示见 `artifacts/demo-final-20261107/report.html`。该目录的 `source/` 保存实际运行时源码副本，`run_manifest.json` 记录其指纹、环境与依赖锁指纹；后续修改源码不会覆盖既有运行版本。`artifacts/verification-20261107.json` 保存最后一次测试、真实闸门与账本核验结果。

以上 `artifacts/` 路径是本地验收记录，不随公开仓库分发。克隆仓库后可用 `demo` 命令重新生成工程报告；真实来源需自行采集，并重新进行数据准入检查。`examples/evidence.json` 为虚构工程样例，不是真实新闻。

区块重采样以候选/基准同步索引生成2000条250交易日路径，检查20/40/60日区块敏感性。情景亏损比例不是未来概率，也不是新的独立市场证据。

参考设计：[Qlib](https://github.com/microsoft/qlib)、[FinRL](https://github.com/AI4Finance-Foundation/FinRL)。本实现未复制两库代码或数据集；各依赖与数据源许可需分别遵守。
