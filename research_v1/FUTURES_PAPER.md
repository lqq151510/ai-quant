# 独立期货虚拟账户

这是本项目的一条独立研究线：**单个真实月份合约、虚拟资金、日线回测和本地行情事件模拟**。它不导入 A 股账本、不绑定真实期货账户，也不能向券商发送委托。内置均线策略和一个简单的机器学习基线；外部模型也可以提供带时间戳的目标手数，但项目现有 AI 模型尚未自动接入本模块。

## 1. 先运行完全离线的演示

在 `research_v1` 目录执行，输出目录必须不存在：

```bash
.venv/bin/python -m quant_research.futures demo --out artifacts/futures-demo-001
```

打开 `artifacts/futures-demo-001/backtest/report.html` 和 `artifacts/futures-demo-001/paper/report.html`。演示使用**虚构合约参数、虚构行情和 100000 元虚拟本金**，不代表真实交易成本或收益。运行目录还包含合成行情、订单、权益、策略决策和账户状态。

命令列表：

```bash
.venv/bin/python -m quant_research.futures --help
```

## 2. 回测自己的日线

先准备两份文件：

1. **合约配置 JSON**：字段为 `symbol`、`multiplier`（合约乘数）、`tick_size`（最小变动价位）、`margin_rate`（假设保证金比例）、`max_margin_fraction`（保证金最多占权益的比例）、`fee_open` / `fee_close`（每手固定开/平仓费用）、`slippage_ticks`（不利滑点的跳数）、`max_lots`、`exit_by`。
2. **日线 CSV**：表头必须包含 `trading_day,symbol,open,high,low,close,settlement,upper_limit,lower_limit`，按交易日严格递增。`settlement` 是当日结算价，`upper_limit` / `lower_limit` 是当日涨跌停价。

演示目录中的 `synthetic_contract.json` 和 `synthetic_bars.csv` 只说明格式；使用真实数据时必须按交易所、期货公司及数据源的具体历史规则重新核验，不能直接照搬示例参数。`symbol` 必须是具体月份，例如 `SHFE.ag2701`；主连、指数等不能作为交易合约。`exit_by` 是使用者核实后的**最迟允许持仓日期**，可能早于交易所的最后交易日。

```bash
.venv/bin/python -m quant_research.futures backtest \
  --config /path/to/contract.json --bars /path/to/daily.csv \
  --source '数据来源和采集说明' --capital 100000 \
  --fast 3 --slow 5 --lots 1 --out artifacts/futures-backtest-001
```

增加 `--strategy ridge --lookback 3 --min-train 30 --train-window 120 --ridge-alpha 1 --threshold 0.0005` 可以运行**岭回归**（带正则化的线性回归）基线。每个交易日仅用当时已完成的历史样本滚动训练，预测下一交易日从开盘到结算的收益方向；训练不足时不交易。标准化只使用训练窗口；模型设置及输入哈希会记录在报告清单中。该基线没有经过可信的期货数据验收或独立样本外比较，不能据此判断真实盈利能力。实时模拟当前默认均线；岭回归尚未自动接到实时事件流。

示例本金只是虚拟数字，可自行改动。输出目录必须是新的；`report.html`、`equity.csv`、`orders.csv`、`decisions.csv` 和 `manifest.json` 分别记录报告、权益、订单、收盘决策和输入/代码哈希，`inputs/` 与 `source/` 保存当次输入和代码副本。用户提供的数据默认标记 `user_supplied_unverified`，不会因为能够运行就被视为已审计；合成输入可使用 `--synthetic` 明确标记。

处理顺序：上次结算到本次开盘的盈亏归原仓；上一交易日信号在开盘近似执行；新仓从开盘持有至结算，扣除费用及不利滑点。涨跌停方向、可用资金和保证金约束可能导致拒单；反手时可以只平旧仓而开仓失败。到 `exit_by` 会尝试平仓，若被价格限制阻止则报告 `exit_unresolved`，不假装已经成交。回测要求输入方确认信号和结算数据在下一次开盘前已可用；程序不会自动验证夜盘日历和数据发布时刻。

### 采集原始历史日线

该命令使用现有研究环境中的 AkShare，读取新浪提供的具体月份合约日线；有超时、有限重试和失败记录，不会自动使用合成数据替代：

```bash
.venv/bin/python -m quant_research.futures probe-sina \
  --symbol AG2609 --out artifacts/futures-sina-probe-001 --timeout 30 --attempts 1
```

结果为 `raw.csv` 和 `audit.json`。数据可能含结算字段，但这里尚未补齐逐日涨跌停价、完整历史合约和费率、交易日历及历史可用时间，所以审计状态仍为 `data_insufficient`；`raw.csv` **不能直接冒充本模块的完整回测输入**。采集范围和数据许可由使用者核验。

## 3. 持续模拟账户和真实行情

模拟账户消费追加写入的 JSONL 行情事件，账户保存在独立的 `session` 文件夹。命令每次处理新完整行；加 `--watch` 可持续监听。重启时使用完全相同的参数、配置、账户代码与事件文件；历史文件被截断、改写或账本代码变化时会拒绝继续，防止重复成交或混用不同规则。`state.json` 原子保存本地虚拟账户，`report.html` 是只读可视化报告；权益曲线按分钟保存最后一个快照，原始事件保留在 JSONL。

真实行情需要独立的快期行情账号和 [TqSdk](https://github.com/shinnytech/tqsdk-python) 服务权限。本机已为此模块准备 `futures_runtime/.venv`；在其他机器可这样安装：

```bash
uv venv --python 3.11 futures_runtime/.venv
uv pip install --python futures_runtime/.venv/bin/python -r futures-live-requirements.txt
```

先创建新的本地文件夹和空事件文件，准备一份**已核实**的真实合约配置 `contract.json`：

```bash
mkdir -p artifacts/futures-live-001
futures_runtime/.venv/bin/python -c "from pathlib import Path; Path('artifacts/futures-live-001/events.jsonl').touch(exist_ok=False)"
```

终端 A 启动本地模拟账户（虚拟本金自行指定）：

```bash
futures_runtime/.venv/bin/python -m quant_research.futures paper \
  --config artifacts/futures-live-001/contract.json \
  --events artifacts/futures-live-001/events.jsonl \
  --capital 100000 --fast 3 --slow 5 --lots 1 \
  --session artifacts/futures-live-001/session --watch
```

终端 B 启动只读行情（将交易日替换为实际所属交易日）：

```bash
futures_runtime/.venv/bin/python -m quant_research.futures tq-feed \
  --config artifacts/futures-live-001/contract.json \
  --events artifacts/futures-live-001/events.jsonl \
  --trading-day YYYY-MM-DD --seconds 3600 --period 60
```

按终端提示输入快期行情账号和密码，密码不会回显，也不写进事件文件。也可由本地环境提供 `TQ_ACCOUNT` / `TQ_PASSWORD`，不要把凭据放到代码、命令参数或聊天中。适配器使用 TqSdk 的本地 `TqSim` 对象**仅创建行情客户端**，只调用行情读取方法；实际的模拟订单由本项目的本地账户计算。行情需满足新鲜度、买卖价、涨跌停、最小变动价位和合约乘数检查；当前未完成的 K 线不会用于策略。

停止行情进程后，从可信来源核实结算价，并在新交易日行情接入前追加结算事件：

```bash
futures_runtime/.venv/bin/python -m quant_research.futures settle \
  --config artifacts/futures-live-001/contract.json \
  --events artifacts/futures-live-001/events.jsonl \
  --trading-day YYYY-MM-DD --price 已核实的结算价
```

夜盘的所属交易日不能按自然日猜测，`--trading-day` 必须由使用者明确填写。结算价由使用者提供且标记为 `manual_unverified`；持仓跨日时缺少结算事件会停止处理。行情适配器不会自动寻找下个交易日、自动重连、自动换月或自动获取结算价；断线、权限、网络延迟和休市均可能导致无新行情。使用 Ctrl+C 可停止监听，状态留在本地。

## 4. 外部 AI 信号

将 `paper` 命令改成 `--strategy external`，由单一事件生产者按时间顺序写入 `quote` 和 `signal` 事件。信号示例：

```json
{"type":"signal","time":"2026-01-02T09:02:00+08:00","trading_day":"2026-01-02","symbol":"SHFE.ag2701","feature_cutoff":"2026-01-02T09:01:00+08:00","model_id":"example-v1","target":1}
```

`target` 是有符号整数目标手数：正数为多、负数为空、零为平仓；不得超过 `max_lots`。信号在可用时刻之后的下一笔报价尝试成交，超过 60 秒仍无报价会取消。代码只检查 `feature_cutoff <= time`，无法证明模型训练集和特征没有泄露未来数据。

当前 `tq-feed` 独占事件文件，不能让第二个进程同时写入信号；若需要行情和 AI 同时运行，应自行实现一个按时间顺序合并行情与信号的单一生产者。内置适配器、外部模型信号以及回测目前不会自动对接同花顺期货通模拟账户。

## 5. 边界与验证

- 第一版只支持单合约、固定每手费用及固定保证金比例，不支持自动换月、多合约组合、期权、手续费按成交额计费、平今特殊费率或自动强平。
- 日线开盘撮合无法还原真实盘口；本地实时模拟按下一笔买一/卖一和显示手数近似成交，也没有模拟排队、冲击成本和所有交易所风控。
- 缺少真实连续行情、真实交易所撮合、长期运行和第三方账户互通的验证。模拟盈利不能证明真实交易可盈利。
- 删除本次新增的 `quant_research/futures/`、`tests/test_futures_paper.py`、本文件和可选依赖文件即可撤销代码；独立的 `futures_runtime/.venv` 和选定的演示产物可按需另行删除。已有 A 股/ETF 代码和其未提交改动无需回滚。
