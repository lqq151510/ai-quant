# 盘口采集与对账

开盘前按 protocol.md 准备同一份候选排名，以及两组需要退出的持仓。取两者的代码并集。在窗口一开始从本目录运行：

    python3 collect_books.py --date YYYY-MM-DD --out YYYY-MM-DD/books 代码.SH 代码.SZ

保存的 books.json 含原始响应文件名、行情来源、供应商时间和收到响应的时间。先用新浪，缺失或过期时再取腾讯并在窗口内重试。selected 是每只标的首份通过时间检查的报价，价格和挂量仍须逐笔核对。先卖后买，挂量不足分别记录 insufficient_bid_units 或 insufficient_ask_units，零挂量单独记录。买入盘口不可用看 `unavailable`。卖出用独立核对的腾讯 `exit_selected` 和 `exit_unavailable`，只按报价时效、涨跌停价及原始证据判断盘口是否可用。513100 的盘中参考净值和 1% 溢价门槛用于买入。收盘摘要分别记录两份参考净值原文、供应商按最新成交价给出的溢价（百分数）及含滑点模拟买入溢价（比例值）。规则未计算模拟买价时，模拟溢价及 1% 比较结果留空，并按实际拒单原因记录。若进程错过窗口，记录采样失败。演练模式不生成 selected 报价。

自动化名义北京时间 09:43 启动，为调度抖动留出余量。入口按上海时间等待至 09:45，正式盘口窗口仍是 09:45–09:50。工作日之外和实验日期之外不生成正式记录。每次响应保存时间戳、原文、哈希及进度。盘口请求使用不超过 8 秒且受当前窗口余额限制的完整响应读取时限。超时或载荷截断时记录失败，窗口内继续尝试下一来源。正式决策从原始响应复核报价和参考净值证据。收盘分别按标的统计计划、收到响应行、成功解析、时效合格、双边价格和挂量可核、买入盘口可用、卖出盘口可用。双边价格齐全时，明确报出的零挂量也计入挂量证据。附带的报价年龄按接收时间减供应商报价时间计算，未被选中的报价仅用于解释缺口。原文核验失败或缺少正式选中记录时，账本按买卖方向保存具体的盘口证据错误，收盘和周报保留原因。

    PYTHONPATH=research_v1 research_v1/.venv/bin/python artifacts/news-filter-forward-20261116/scheduled_open.py

11 月 16 日的开盘计算入口如下。可以用已落盘的窗口内证据重算，重启时沿用已选的第一份合格快照。开盘账本会标记 forward。

    PYTHONPATH=research_v1 research_v1/.venv/bin/python artifacts/news-filter-forward-20261116/monday_open.py

开盘会将当时可用的新闻归档固化到 opening-news.json。回放时按每笔盘口的原始接收时间核对快照可用性。快照损坏或晚于该笔报价时，原规则组仍按盘口判断，新闻组在该笔记录新闻证据缺口。前三个自然日的 18:00 采集都应写入同级 news-forward/YYYY-MM-DD.json，包含周末。已完成定向采集但无事件时保存空 events，单列缺档、格式错误和部分或失败状态。没有采集状态的旧档按 legacy 保留已观察事件，并标记覆盖不完整；收盘和周报也会根据已保存快照的状态识别旧缺口，不改写原始证据。相同公告按事件标识、来源和最早采集时间跨日去重。归档覆盖缺口进入周报证据判定。旧归档和正式盘口窗口不倒填。

周一收盘采集若发现开盘账本缺失，会先尝试用当天已保存的盘口恢复，然后采集恢复出的持仓收盘价。恢复只使用开盘窗口内的证据。采集进度中没有任何请求记录时，不能据此生成开盘账本。收盘摘要会把缺少开盘执行证据的两组成交、费用及净值标记为未知。只有持仓的收盘证据缺失时，仍报告已核实的成交和费用。

某个标的最终只剩盘口不可用时，逐笔记录和汇总附上可核请求日志里最近一次相关的供应商及错误。合格备用报价、过期盘口和参考净值未核实等具体结果按原规则记录。

    python3 artifacts/news-filter-forward-20261116/collect_close_marks.py
    python3 artifacts/news-filter-forward-20261116/close_monday.py

11 月 17 至 20 日使用以下流程，把 DATE 换成实验交易日。使用上一日 15:00 后和当日 09:45 前采集的候选，优先按上一日已对账的持仓退出。前一日的部分账本可承接已验证的持仓和现金，缺净值时保留买入预算无法核实的原因。多份部分账本的执行记录和已知收盘价须一致，并且其中一份须包含其他版本的全部已核估值，才会自动选用该文件。冲突或无法选出单个版本时保留原始记录和错误。续跑在第一次请求盘口前保存 `opening-inputs.json`，记录所用前序账本和候选文件的路径及哈希，候选缺失时也如实记录。收盘任务可从当天已保存盘口恢复开盘账本，使用原先固定的输入。证据缺失或原文件与校验值不符时，报告不可核实。已发布的续跑账本重跑时仍核对原前序记录。

    python3 artifacts/news-filter-forward-20261116/prepare_monday.py --date DATE
    PYTHONPATH=research_v1 research_v1/.venv/bin/python artifacts/news-filter-forward-20261116/continue_open.py --date DATE
    python3 artifacts/news-filter-forward-20261116/collect_close_marks.py --date DATE
    python3 artifacts/news-filter-forward-20261116/close_monday.py --date DATE

收盘程序返回摘要和实际账本路径。若估值尚未完整，输出落在当日 close-partials 下，后续补齐证据可发布正式账本。周五从项目根目录运行：

    python3 artifacts/news-filter-forward-20261116/weekly_report.py

本地演算使用独立的假设数据：

    PYTHONPATH=research_v1 research_v1/.venv/bin/python artifacts/news-filter-forward-20261116/monday_scenario_matrix.py
    PYTHONPATH=research_v1 research_v1/.venv/bin/python artifacts/news-filter-forward-20261116/joint_scenarios.py
    PYTHONPATH=research_v1 research_v1/.venv/bin/python artifacts/news-filter-forward-20261116/price_scenarios.py
    PYTHONPATH=research_v1 research_v1/.venv/bin/python artifacts/news-filter-forward-20261116/exit_scenarios.py
    PYTHONPATH=research_v1 research_v1/.venv/bin/python -m unittest discover -s artifacts/news-filter-forward-20261116 -p 'test_*.py'

每日按冻结规则记录订单和收盘估值，保存 YYYY-MM-DD/daily.json。文件顶层字段 date、baseline、filtered。每组字段 fills、cash、positions、close_prices、nav；positions 为代码到整数股数或份额的映射，close_prices 为代码到价格字符串的映射。fills 每笔含 symbol、units、side、price、commission、stamp_tax、transfer_fee、fees、cash_delta，金额用十进制字符串。保留拒单、来源和新闻决策的其他字段。公司行动须附来源，可用 corporate_actions 记录盘后现金或份额变化；若公司行动必须先于交易处理，先按冻结方案核实并人工对账，不按盘后顺序强行记账。估值缺失时 nav 填 null，并附 valuation_gap。真实账本应与演练文件分开。

每日和周五用完整、按日期排序的账本运行：

    python3 reconcile.py YYYY-MM-DD/daily.json ... > reconciliation.json

先核对实际交易日是否齐全，再用输出汇总两组扣费收益和最大回撤。缺日时保留已核结果并明确区间不完整。周报按已卖出和周五仍持仓分列已核实的避亏、错过盈利与持平机会，另记待核原因。逐笔可归因金额和两组周五收盘净值差分别报告，未归因差额保留。有效过滤机会和周五判定按 protocol.md，从逐笔决策另行核对。
