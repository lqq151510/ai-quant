# 项目状态与复现清单

检查时间：2026-11-14，北京时间。数据与实验检查的代码基准：`15aed9053a5f3320d56d2f9151007f07f0fb5bc5`；本文也记录了该基准之后的新闻归档 `legacy` 覆盖修复。本文是当日的本机检查记录，不会随行情、自动化或后续代码变动自动更新。

## 2026-11-27 将收盘换留模型从 2021 年底接续至 2022 年底

- 保留 2020 年初的换留系数、20 日调仓日程和前一年收盘持仓。原定起点正常费用下，2022 年模型净收益 −11.02%，最大回撤 20.42%；一直留仓分别为 −7.54%、14.38%，前二与持仓不同时就换为 −13.43%、20.99%。正常和双倍费用下同时胜过两者、回撤不加深的起点，2017–2018、2020–2021、2022 年依次为 2/20、1/20、0/20。
- 2022 年共有 148 个可换且已到期的机会。模型选择换仓 17 次，仅 1 次换仓后净值更高；选择留仓 131 次，其中 72 次留仓更好。按基金公告修正了本次模拟中 510500 拆分的折算精度与进位方法，120 条账户路径的年末净值最大改动 6.29 元，换留决定全部一致。生产策略沿用既有规则。[研究报告、逐日净值图和独立复核](artifacts/etf-close-carry-2022-20261127/report.md)

## 2026-11-20 增量复核

11-14 的检查记录原样保留；本节只记 11-20 的增量。

- **质量门**：新增 `tools/quality_gate.py`，一次运行研究线 108 项、原型 90 项（30 个子测试）、前向 138 项测试，校验证据归档哈希；本机快照存在时追加 ETF 数据审计，并核对 README 测试数与实测。11-20 运行结果全部通过；前向窗口缺 11-19 当日记录，11-20 待收盘。
- **证据入库**：`research_v1/evidence/` 归档 `REJECTED.md` 引用的 14 份摘要文件；`MANIFEST.json` 记录原件路径、字节数与 SHA-256，新增 `tests/test_rejected_evidence.py` 校验哈希与链接，新检出无需本地 `artifacts/` 即可复核 15 条候选。
- **回测性能**：`ledger.simulate` 改为一次性构建按日行情索引，去掉每日 `groupby.get_group` 的重复扫描；第二轮把校验后的索引在 54 次模拟间复用，并改用向量化日期转换。同一 ETF 审计快照下，`capital_scale` 默认三信号 × 六档本金 × 两档费用（54 次账本）由 8.3 秒降至冷启动 1.3 秒、三次中位约 0.8 秒（约 7–10 倍）；`python3 tools/bench_capital_scale.py` 可复现计时；两轮改动后 `equity.csv` 与 `orders.csv` 均逐字节一致，`report.json` 除源码哈希外全部字段一致。
- **前向时限**：`collect_books.capture` 在读取完成后复核实际耗时，避免 macOS 上慢响应被内部重试后仍当作按时完成；`test_capture_deadline` 恢复通过。
- **复现口径**：5 日对照归档由显式信号集生成；当前默认 `equal_weight_hold` 是首次买入后持有，直接重跑 500,000 元正常费用等权为 +9.05%，不能与归档的 +9.55% 直接比较。用 `--signals momentum reverse equal_weight csi300_hold` 重跑后，四组收益与回撤均与归档一致；命令见 `research_v1/evidence/README.md`。
- **并行改动**：工作区中 `src/` 与 `tests/` 的 7 个未提交文件来自原型线的并行工作，本次没有修改或提交。

## 当前状态与边界

- 项目已有独立研究账本、数据审计、基准对照、部分模型和新闻前向模拟工具；尚无通过验收的真实市场盈利策略。
- `src/` 的原型与 `research_v1/` 研究线环境和产物分开。原型有合成实验及部分历史回放；历史报告不自动代表当前源码重新训练后的结果。
- 四只 ETF 历史研究中，2023—2024 年已反复查看，不能称为新的独立样本外结果；2025 年起的封存历史行情继续保持封存。2026-11-16 至 11-20 的新闻前向观察是另一个预先冻结的实验。
- 真实历史数据存在供应商事后修订、历史首次获知时间不全和开盘价近似成交等限制。数据审计 `ready` 不等于消除了这些限制，也不等于策略通过验收。
- 不把已拒绝候选恢复为运行策略；候选和判定见 [REJECTED.md](research_v1/REJECTED.md)。所有金额均为研究中的虚拟金额。

## 数据盘点

| 本地位置 | 当日检查结果 | 说明 |
|---|---|---|
| `research_v1/artifacts/market-etf-verified-20261108/` | 三个 Parquet 表的 SHA-256 和行数均与 `manifest.json` 一致；2015—2024 年 ETF 审计为 `ready`，四只各 2,431 个交易日 | 含完整快照文件，审计只针对所选截止日期；不能证明所有历史修订或当时可得时间 |
| `research_v1/artifacts/market-wide28-verified-20261108/` | 五个 Parquet 表哈希及行数一致，清单中 25 个来源文件路径和哈希匹配 | 通用个股审计为 `data_insufficient`，原因包括股票池时点、上市退市信息、原始价格及费用规则等证据不足；该快照的来源清单含本机绝对路径 |
| `research_v1/artifacts/` | 约 247 MB、12,588 个文件 | 包含原始来源、快照、账本、报告和部分源码快照；大小与文件数会变化 |
| 根目录 `artifacts/` | 约 20 MB、483 个文件 | 含历史实验、新闻归档及前向模拟文件；其中部分协议和源码由 Git 跟踪，其余多数产物被忽略 |
| `data/` | 约 2.5 MB、2 个文件 | 包含本地 DuckDB 数据库和历史计划文件，被 Git 忽略；不要把历史计划视为当前交易指令 |

快照标识（SHA-256 是文件内容指纹）：

```text
ETF manifest:    6227df4d2415ae88faa8bc5fca48f504972a4bfe5c5c76abd180053eccb0b169
wide28 manifest: ddec7e0c7c932cc04a7b87ab26ec9f1d43ed6090eead0d293b11101c66f33e5b
research_v1/uv.lock: 0aecbb9b1f165d10977e54a2daae8848f137504e4517cdfb38df75fb77f39081
```

ETF 清单涵盖的完整文件包含 2015—2026 年的数据行；本次市场审计和下面的历史重放限定截止到 2024-12-31。对完整文件计算哈希是检查字节完整性，并未将封存行情用于研究。

## 关键实验与可复现程度

| 实验 | 本地证据及检查 | 限制 |
|---|---|---|
| 双动量，R-01 | `research_v1/artifacts/momentum-study-20261111-a/` 中 7 份源码快照、输入清单及旧对照报告哈希匹配；按下方步骤隔离重放后，`daily.csv`、`decisions.csv`、`orders.csv` 与原文件逐字节一致 | 当前源码中 4 份文件已与当时版本不同；旧实验属于历史探索，候选已拒绝 |
| 60 日反向动量，R-05 | `research_v1/artifacts/reversal-60-exposure-20261114/` 的 8 份来源代码哈希与当前文件一致；年度拆分另见 `reversal-60-subwindows-20261114/` | 本次未重新运行；年度拆分报告没有单独记录源码哈希，候选未晋级 |
| 波动率目标，R-14 | `research_v1/artifacts/etf-vol-target-12pct-20261114/` 的入口源码哈希与报告记录一致 | 本次未重新运行，候选未晋级 |
| 岭回归退出价，R-07 | `artifacts/net-edge-study-20261111/` 记录的 15 份运行源码与快照哈希一致 | 结果为历史筛选失败；本次未重新运行 |
| IQL 五种子，R-09/R-10 | `artifacts/rl-self-review-20261111/` 的 4 项输入和 59 份源码快照哈希一致 | 本次未重新训练；历史候选未通过筛选 |
| 28 标的原型回放 | `artifacts/walk-forward-wide28-observed-labels-20261114/` 保存结果报告 | 该报告未单独保存运行源码哈希；本次未证明可以用当前代码复现，且通用个股数据审计仍未通过 |
| 新闻过滤前向观察 | [协议](artifacts/news-filter-forward-20261116/protocol.md)、运行说明和测试已纳入 Git；本地周一候选和协议哈希与排程固定值一致 | 11 月 16—20 日尚未开始；候选和新闻档案是本地忽略文件，不能把演练结果当正式结果 |

11 月 13 日新闻档案没有 `collection_status` 字段，程序按 `legacy` 读取；11 月 14 日档案为 `partial`。修复后的新快照会将 `legacy` 标为覆盖不完整；收盘摘要与周报也会从旧快照的状态字段重新识别缺口，不改写原始档案。11 月 15 日档案应按实际采集情况记录，不事后补造。

## 备份核对

- Time Machine 报告未配置备份目标；`latestbackup` 无法挂载目标，对 `/` 的本地快照查询没有返回快照。
- 检查了本机 `Documents`、`Desktop`、`Downloads`、`Backups`、可见 iCloud Drive 路径和两个已挂载卷的可见目录，检索深度最多约三层且未遍历隐藏目录；没有找到名称可识别为本项目的副本。已挂载卷和 `Backups` 的这一范围内也未找到常见格式的压缩归档。
- **结论：本次未核实到覆盖这些数据的独立备份。** 这不排除更深目录、其他设备、未挂载磁盘或云端存在备份；也没有进行恢复演练。本次没有创建或覆盖备份。
- 若后续备份，优先保留 `research_v1/artifacts/`、根目录 `artifacts/` 和 `data/`；源码另保留对应 Git 提交。先停止写入，尤其要为 DuckDB 使用一致性快照或数据库导出；备份到独立介质后逐文件校验哈希并抽样恢复。注意数据源许可、模拟账户记录和隐私，不包含 `.env`、密钥或虚拟环境。
- wide28 来源清单记录了原机器绝对路径。迁移时保留原始清单和文件，另行记录新旧路径映射与校验结果；不要直接修改旧清单并声称其原哈希仍然有效。

## 本机验证命令

从仓库根目录运行。研究线环境为 Python 3.11；原型本机环境为 Python 3.12。命令只运行测试或审计，不覆盖原实验结果。

```bash
cd research_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m quant_research.cli audit \
  artifacts/market-etf-verified-20261108 --end 2024-12-31 --kind etf

# 预期返回 data_insufficient，退出码为 2；不要把它当作成功准入。
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m quant_research.cli audit \
  artifacts/market-wide28-verified-20261108 --end 2024-12-31 --kind stock

cd ..
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_v1 research_v1/.venv/bin/python \
  -m unittest discover -s artifacts/news-filter-forward-20261116 -p 'test_*.py'

# 或一次运行全部门禁（研究线、原型、前向、证据归档与前向记录审计）。
python3 tools/quality_gate.py
```

2026-11-14 的结果：研究线 104 项通过；原型 72 项、19 个子测试通过，另有两条 NumPy 警告；新闻归档修复后前向模块 138 项通过。该验证未接入真实盘口或模拟账户，也未重新训练模型。

## 双动量旧实验的隔离重放

该实验的 `protocol.json` 记录 Python 3.11.15、NumPy 2.4.6、Pandas 3.1.1 和 DuckDB 2.0.0；本次使用相同版本。需要完整 ETF 快照、旧对照报告及实验归档源码。下列命令从仓库根目录运行，在临时目录重放，不覆盖原实验；若缺文件或版本不同，应先记录差异。

```bash
project_dir="$(pwd -P)"
run_dir="$(mktemp -d "${TMPDIR:-/tmp}/aiquant-replay.XXXXXX")"
run_dir="$(cd "$run_dir" && pwd -P)"
mkdir -p "$run_dir/research_v1"
cp -R "$project_dir/research_v1/artifacts/momentum-study-20261111-a/source/research_v1/quant_research" \
  "$run_dir/research_v1/quant_research"
touch "$run_dir/research_v1/quant_research/__init__.py"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$run_dir/research_v1" \
  "$project_dir/research_v1/.venv/bin/python" -m quant_research.momentum_study \
  --snapshot "$project_dir/research_v1/artifacts/market-etf-verified-20261108" \
  --prior-report "$project_dir/artifacts/etf-deadband-study-20261110/results/report.json" \
  --out "$run_dir/output"

comparison_status=0
for artifact_name in daily.csv decisions.csv orders.csv; do
  if cmp -s "$project_dir/research_v1/artifacts/momentum-study-20261111-a/$artifact_name" \
            "$run_dir/output/$artifact_name"; then
    printf '%s: 一致\n' "$artifact_name"
  else
    printf '%s: 不一致，需排查\n' "$artifact_name"
    comparison_status=1
  fi
done
printf '临时结果：%s\n' "$run_dir/output"
test "$comparison_status" -eq 0
```

本次三份 CSV 均一致，配置、快照清单、旧对照和七份源码的记录哈希也一致。报告中的运行时间等元数据会随新运行改变；临时目录可能被系统清理，不能当备份。直接使用当前 `quant_research` 重跑会生成新版本结果，不应声称与旧版本完全相同。

## 下一步

1. 先选择独立备份目标，备份并执行校验与抽样恢复；继续遵守数据使用许可与隐私边界。
2. 按冻结协议观察 11 月 16—20 日新闻对照，逐日保留数据覆盖、执行和估值缺口，一周结果只作为阶段观察。
3. 在不触碰封存历史行情的前提下，复用已有简单模型和同一份额账本，预先定义与现金、等权及成本压力的对照，再判断是否值得引入更复杂模型。
