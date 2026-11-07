# 研究实验：资金流、日内特征与分布漂移

本目录是与原型主入口隔离的实验：只训练与报告，不连接真实账户、不生成交易计划、不修改现有数据库。它不会替代 `research_v1` 的数据闸门、账本和执行时点审计。

## 运行

在仓库根目录执行（输出目录必须不存在）：

```bash
# 合成行情，仅验证工程链路；不联网，也不生成虚构资金流或分钟行情
.venv/bin/python -m src.research_upgrades --synthetic --out /tmp/ai-quant-upgrade-demo

# 只读本项目现有数据库；输入来源仍待独立核验
.venv/bin/python -m src.research_upgrades --bars-duckdb data/quant_market.duckdb \
  --out /tmp/ai-quant-upgrade-local

# 本地日线 + 可选 AkShare 在线数据；先选择少量标的，避免给供应商造成压力
.venv/bin/python -m src.research_upgrades --bars-csv /path/to/daily.csv \
  --fetch-akshare --fetch-symbol 600900 --fetch-symbol 510300 --out /tmp/ai-quant-upgrade-real
```

日线 CSV 需含 `symbol,trade_date,open,high,low,close,volume`，同一天必须包含相同的至少三个标的，价格有效且键唯一。真实交易日历、停牌、复权、股票池时点和公司行动仍须独立审计。输入行情可以改用 `--flow-csv` 与 `--minutes-csv`：资金流列为 `symbol,trade_date,flow_super_ratio,flow_large_ratio`，净占比用小数（0.01 代表 1%）；分钟列为 `symbol,timestamp,open,close`，时间为北京时间。自行提供的文件应只含收盘后可用的完整数据，并保留原始来源证据。

如需使用独立的更广泛市场数据计算 Wasserstein 与状态监测，可追加 `--macro-bars-csv /path/to/market.csv`。该 CSV 至少包含 `symbol,trade_date,close`，允许每天的标的集合不同，但必须自行提供并审计历史时点的股票池和完整行情。标的缺失一天后重新出现时，该次跨缺口收益不会伪装成单日收益。此入口不会自动获得或验证全市场数据。

运行会在新目录保存输入副本和 SHA-256、供应商状态、模型、正交化参数与 JSON 报告。`provider_status.json` 在联网后立即保存，即使后续训练失败也能保留接口失败信息。默认不联网；触发东财限流后停止后续请求。AkShare 日线资金流 API 名称为 `stock_individual_fund_flow`，[官方文档](https://akshare.akfamily.xyz/data/stock/stock_flow.html)说明只提供近期约 100 个交易日；[股票](https://akshare.akfamily.xyz/data/stock/stock_quote.html)及 [ETF](https://akshare.akfamily.xyz/data/fund/fund_public.html)分钟接口也只提供近期数据。若这些数据只覆盖测试期，特征组会因训练期缺失被屏蔽，需另备足够的历史数据并审计时点。对 ETF 资金流只做实际请求尝试，能否返回以状态和覆盖率为准；不把 ETF 净申购赎回额冒充大单交易净流入。数据源提供的历史记录可能修订，抓取时间不能证明历史时点可获得，不能未经审计用作真实回测证据。

## 特征和算法

- CLV = `(2 × 收盘 − 最高 − 最低) / (最高 − 最低)`。最高等于最低时标记缺失。新特征按当天有效标的截面标准化，缺失值填零并附可用性标记，不做前向或后向填充。它们在收盘后才可能完整可用。如果训练期完全没有资金流或分钟数据，对应特征组在所有数据切分中屏蔽，报告仍保留屏蔽前的分段覆盖率，避免模型用随机初始化的权重处理仅在测试期出现的新字段。
- 5 分钟数据需覆盖正常交易时段共 48 根、时间戳唯一。第一根计算开盘到收盘的对数收益，其后计算相邻收盘收益；分别累加正负收益平方并取平滑后的对数比。缺根或不完整的交易日标记不可用。它是 5 分钟采样下的波动近似，不是逐笔成交。
- Löwdin 使用训练集的六个多周期收益及风险调整收益，基于训练样本最后时点计算均值和协方差；验证集和测试集复用同一变换。数值秩不足时对零方向使用伪逆，报告有效秩；无法保证所有原始列完全独立。
- 1D Wasserstein 比较最近 20 个样本日的截面收益分布与此前不重叠的 120 个样本日。它作为第五个观测输入 GMM 发射分布和粘性前向滤波；95% 阈值仅由训练期拟合，用作分布变化提示。仅覆盖输入标的，距离升高可能由均值、波动等多种变化导致，不等于黑天鹅概率。
- 指数权重以 120 个有效样本日为半衰期，深度模型按训练全集平均权重为 1 加权逐日损失；GBDT 将每日权重复制到各标的。缺少交易日时，样本日不等于真实交易所交易日；验证和测试不参与加权拟合。

输出 IC 仅是这次给定数据和切分上的预测相关性，无法证明投资收益。没有执行完整真实数据回测、成本核算或交易。
