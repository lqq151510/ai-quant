# 双动量 IQL 风险叠加研究

证据等级：previously inspected historical exploration; not new independent OOS。2023—2024 区间此前已查看，不是新的独立样本外证据。

固定策略：过去 126 个交易日总收益，正动量前两只，每只 35%，每 20 日复核。RL 只能把该组合总仓位乘以离散风险倍率；不能改选标的。
训练区间：2015-07-10 — 2022-12-30 (1821 sessions)；评估区间：2023-02-14 — 2024-12-24 (454 sessions)。封存区间从 2025-01-01 开始，未读取。

|费用情景|策略|净收益|最大回撤|年化收益|费用|
|---|---|---:|---:|---:|---:|
|normal|equal70_monthly|3.44%|10.92%|1.89%|45.00|
|normal|band70_weekly|3.09%|10.88%|1.70%|65.00|
|normal|dual_momentum126_monthly|-9.70%|13.26%|-5.51%|110.00|
|normal|overlay_fixed_85|-8.42%|11.41%|-4.76%|120.00|
|normal|overlay_fixed_60|-6.16%|8.17%|-3.47%|110.00|
|normal|overlay_fixed_35|-3.74%|4.74%|-2.09%|120.00|
|normal|overlay_drawdown_guard|-9.03%|12.20%|-5.12%|130.00|
|normal|overlay_volatility_target_10pct|-11.07%|14.17%|-6.30%|125.00|
|normal|overlay_momentum_breadth|-9.49%|13.06%|-5.39%|130.00|
|normal|iql_seed_42|-0.44%|1.59%|-0.24%|30.00|
|normal|iql_seed_101|0.00%|0.00%|0.00%|0.00|
|normal|iql_seed_202|0.00%|0.00%|0.00%|0.00|
|normal|iql_seed_303|-1.59%|2.08%|-0.89%|30.00|
|normal|iql_seed_404|0.00%|0.00%|0.00%|0.00|
|normal|IQL 五种子均值|-0.41%|0.74%|-0.23%|—|
|double_friction|equal70_monthly|2.94%|11.10%|1.62%|90.00|
|double_friction|band70_weekly|2.40%|11.34%|1.32%|140.00|
|double_friction|dual_momentum126_monthly|-11.13%|14.30%|-6.34%|220.00|
|double_friction|overlay_fixed_85|-9.91%|12.54%|-5.63%|240.00|
|double_friction|overlay_fixed_60|-7.46%|9.13%|-4.21%|220.00|
|double_friction|overlay_fixed_35|-5.05%|5.73%|-2.83%|240.00|
|double_friction|overlay_drawdown_guard|-11.66%|14.20%|-6.65%|290.00|
|double_friction|overlay_volatility_target_10pct|-12.64%|15.31%|-7.23%|250.00|
|double_friction|overlay_momentum_breadth|-11.14%|14.32%|-6.35%|260.00|
|double_friction|iql_seed_42|-0.76%|1.71%|-0.42%|60.00|
|double_friction|iql_seed_101|0.00%|0.00%|0.00%|0.00|
|double_friction|iql_seed_202|0.00%|0.00%|0.00%|0.00|
|double_friction|iql_seed_303|-1.76%|2.19%|-0.98%|50.00|
|double_friction|iql_seed_404|0.00%|0.00%|0.00%|0.00|
|double_friction|IQL 五种子均值|-0.50%|0.78%|-0.28%|—|

全部种子及费用情景均超过原双动量候选、且最大回撤不超过 15%：**True**。这只是预先定义的历史筛选条件，不是收益保证。

区块重采样与原候选的配对终值区间：
- normal，块长 20：[-2.89%, 13.78%]。
- normal，块长 40：[-2.32%, 14.11%]。
- normal，块长 60：[-1.01%, 12.69%]。
- double_friction，块长 20：[-2.42%, 14.81%]。
- double_friction，块长 40：[-1.86%, 15.10%]。
- double_friction，块长 60：[-0.47%, 13.74%]。

行为数据由固定仓位、回撤降仓、波动目标和动量广度规则在训练区间的完整账本轨迹构成，不是用户实盘记录。每日特征在收盘生成，订单进入次日开盘份额账本；训练奖励按 20 日区间相对双动量基线的净收益、换手及超过 15% 回撤惩罚计算。
历史价格快照缺完整的点时获知与修订记录；封存区间没有参与特征、训练或筛选。IQL 增量需与固定仓位和简单风控行为规则对照阅读。

区块重采样仅是已看历史序列的情景分析，不是未来盈利概率。策略仍需独立前向验证，不能自动替换原候选。
