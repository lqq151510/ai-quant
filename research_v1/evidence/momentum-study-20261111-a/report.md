# 固定双动量候选历史探索

状态：historical_exploration。2023—2024已被查看，本次结果不是新的独立样本外证据。

|窗口|模拟本金|成本|策略|净收益|年化收益|最大回撤|费用|成交笔数|
|---|---:|---|---|---:|---:|---:|---:|---:|
|primary_2023_2024|10000|normal|equal70_monthly|3.44%|1.89%|10.92%|45.00|9|
|primary_2023_2024|10000|normal|dual_momentum126_monthly|-9.70%|-5.51%|13.26%|110.00|22|
|primary_2023_2024|10000|normal|band70_weekly|3.09%|1.70%|10.88%|65.00|13|
|primary_2023_2024|10000|double_friction|equal70_monthly|2.94%|1.62%|11.10%|90.00|9|
|primary_2023_2024|10000|double_friction|dual_momentum126_monthly|-11.13%|-6.34%|14.30%|220.00|22|
|primary_2023_2024|10000|double_friction|band70_weekly|2.40%|1.32%|11.34%|140.00|14|
|primary_2023_2024|100000|normal|equal70_monthly|3.93%|2.16%|12.25%|315.84|63|
|primary_2023_2024|100000|normal|dual_momentum126_monthly|-9.23%|-5.24%|13.36%|305.29|39|
|primary_2023_2024|100000|normal|band70_weekly|4.42%|2.43%|11.88%|70.84|14|
|primary_2023_2024|100000|double_friction|equal70_monthly|3.69%|2.03%|12.31%|621.72|62|
|primary_2023_2024|100000|double_friction|dual_momentum126_monthly|-9.86%|-5.60%|13.77%|589.30|37|
|primary_2023_2024|100000|double_friction|band70_weekly|4.37%|2.40%|11.98%|151.72|15|
|retrospective_warmed_2015_2022|10000|normal|equal70_monthly|2.79%|0.38%|22.39%|165.00|33|
|retrospective_warmed_2015_2022|10000|normal|dual_momentum126_monthly|17.35%|2.24%|20.89%|325.00|65|
|retrospective_warmed_2015_2022|10000|normal|band70_weekly|0.82%|0.11%|22.96%|285.00|57|
|retrospective_warmed_2015_2022|10000|double_friction|equal70_monthly|0.82%|0.11%|22.44%|390.00|39|
|retrospective_warmed_2015_2022|10000|double_friction|dual_momentum126_monthly|11.17%|1.48%|20.93%|650.00|65|
|retrospective_warmed_2015_2022|10000|double_friction|band70_weekly|-0.91%|-0.13%|23.22%|450.00|45|
|retrospective_warmed_2015_2022|100000|normal|equal70_monthly|4.94%|0.67%|23.75%|1145.76|229|
|retrospective_warmed_2015_2022|100000|normal|dual_momentum126_monthly|20.57%|2.62%|21.90%|917.63|125|
|retrospective_warmed_2015_2022|100000|normal|band70_weekly|6.15%|0.83%|24.07%|215.76|43|
|retrospective_warmed_2015_2022|100000|double_friction|equal70_monthly|3.58%|0.49%|23.84%|2421.54|242|
|retrospective_warmed_2015_2022|100000|double_friction|dual_momentum126_monthly|18.66%|2.40%|21.92%|1886.12|131|
|retrospective_warmed_2015_2022|100000|double_friction|band70_weekly|5.94%|0.80%|24.10%|421.54|42|

主窗历史筛选通过：False；回顾窗历史筛选通过：False；独立前向验证：pending。

参数：126期含分红收益、正动量前2、每只35%、20交易日复核。只有1只合格则35%，无合格则现金。
复核日收盘决策，下一交易日开盘模拟；含整手约束、费用、滑点、限价及公司行动账本。
回顾窗共同暖启动需127个收盘观察，因此不覆盖2015最早阶段。现金收益假设为0。
历史供应商快照是事后采集，缺完整历史known_at证据；因果公司行动计算不保证供应商历史数据从未修订。
配对重采样是历史情景，不是未来获利概率。2025起封存行情未进入研究计算。
冻结协议见protocol.json；逐日账本、订单、决策见daily.csv、orders.csv、decisions.csv。

|窗口|模拟本金|成本|回撤≤15%|超过月度等权|超过周度容差|全部通过|
|---|---:|---|---|---|---|---|
|primary_2023_2024|10000|normal|True|False|False|False|
|primary_2023_2024|10000|double_friction|True|False|False|False|
|primary_2023_2024|100000|normal|True|False|False|False|
|primary_2023_2024|100000|double_friction|True|False|False|False|
|retrospective_warmed_2015_2022|10000|normal|False|True|True|False|
|retrospective_warmed_2015_2022|10000|double_friction|False|True|True|False|
|retrospective_warmed_2015_2022|100000|normal|False|True|True|False|
|retrospective_warmed_2015_2022|100000|double_friction|False|True|True|False|
