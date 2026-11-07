"""基础设施层：A股/ETF 行情获取与知识图谱初始边构建."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssetMeta:
    """标的元数据定义."""
    symbol: str
    name: str
    sector: str
    is_leader: bool = False


# 精选 12 只具备明显产业链传导与板块轮动特征的核心标的池
CORE_UNIVERSE: List[AssetMeta] = [
    # AI 与半导体算力链
    AssetMeta("512480", "半导体ETF", "Tech_Semi", is_leader=True),
    AssetMeta("688981", "中芯国际", "Tech_Semi", is_leader=True),
    AssetMeta("002371", "北方华创", "Tech_Semi", is_leader=False),
    AssetMeta("515880", "通信设备ETF", "AI_Compute", is_leader=True),
    AssetMeta("300308", "中际旭创", "AI_Compute", is_leader=False),
    # 新能源与智能制造链
    AssetMeta("300750", "宁德时代", "New_Energy", is_leader=True),
    AssetMeta("002594", "比亚迪", "New_Energy", is_leader=False),
    AssetMeta("515790", "光伏ETF", "New_Energy", is_leader=False),
    # 价值红利与宏观对冲链
    AssetMeta("510880", "红利ETF", "Dividend_Value", is_leader=True),
    AssetMeta("600900", "长江电力", "Dividend_Value", is_leader=False),
    AssetMeta("510300", "沪深300ETF", "Broad_Index", is_leader=True),
    AssetMeta("518880", "黄金ETF", "Macro_Hedge", is_leader=True),
    # 跨市场全球科技锚点 (纳斯达克100：映射英伟达/费半/海外AI资本开支冲击)
    AssetMeta("513100", "纳指ETF", "Global_Tech_Anchor", is_leader=True),
]

# 显式知识图谱先验关系：(源节点, 目标节点, 关系类型, 先验权重)
# 包含：跨市场海外算力映射 (overseas_ai_chain)、同行业共振 (same_sector)、产业链传导 (supply_chain)、避险跷跷板 (risk_off_hedge)
PRIOR_KG_RELATIONS: List[Tuple[str, str, str, float]] = [
    ("513100", "300308", "overseas_nvda_chain", 0.95),  # 纳指/英伟达 Capex -> 光模块核心映射
    ("513100", "515880", "overseas_ai_capex", 0.90),    # 纳指/海外算力 -> 通信设备ETF
    ("513100", "512480", "overseas_sox_link", 0.78),    # 费城半导体 -> A股半导体ETF
    ("513100", "600900", "risk_off_hedge", 0.70),       # 海外科技暴跌 -> 高股息水电避险跷跷板
    ("513100", "510880", "risk_off_hedge", 0.70),       # 海外科技暴跌 -> 红利低波避险跷跷板
    ("512480", "688981", "same_sector", 0.90),
    ("688981", "002371", "supply_chain", 0.85),         # 晶圆代工龙头 -> 国产半导体设备(内需自主可控)
    ("512480", "002371", "same_sector", 0.80),
    ("515880", "300308", "same_sector", 0.90),
    ("512480", "515880", "tech_theme", 0.75),
    ("688981", "300308", "tech_theme", 0.70),
    ("300750", "002594", "supply_chain", 0.85),
    ("300750", "515790", "same_sector", 0.70),
    ("510880", "600900", "same_sector", 0.90),
    ("510300", "510880", "style_link", 0.65),
    ("510300", "300750", "index_weight", 0.60),
    ("518880", "510880", "defensive_link", 0.55),
]


def build_initial_kg_edges(trade_date: str = "2026-01-01") -> pd.DataFrame:
    """生成双向知识图谱初始边表."""
    rows = []
    for src, dst, rel, w in PRIOR_KG_RELATIONS:
        rows.append({
            "src_symbol": src,
            "dst_symbol": dst,
            "relation_type": rel,
            "weight": w,
            "updated_at": pd.to_datetime(trade_date).date(),
        })
        # 反向传导边（下游或同板块反馈，权重略微衰减以体现龙头领涨不对称性）
        rows.append({
            "src_symbol": dst,
            "dst_symbol": src,
            "relation_type": f"{rel}_rev",
            "weight": round(w * 0.8, 4),
            "updated_at": pd.to_datetime(trade_date).date(),
        })
    return pd.DataFrame(rows)


def fetch_or_generate_market_data(
    n_days: int = 750,
    try_online_akshare: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """优先尝试从 AkShare 拉取真实历史日线，若网络受限则生成高保真含产业链动量溢出效应的基准日线."""
    if try_online_akshare:
        try:
            df = _fetch_from_akshare(n_days=n_days)
            if not df.empty and df["symbol"].nunique() >= len(CORE_UNIVERSE) // 2:
                return df
        except Exception:
            pass
    return generate_realistic_ashare_panel(n_days=n_days, seed=seed)


def _apply_split_qfq_adjustment(sub: pd.DataFrame) -> pd.DataFrame:
    """修复 ETF 份额拆分/折算导致的未复权断崖跳空，保持最新盘面价格与真实收盘一致."""
    df = sub.sort_values("trade_date").reset_index(drop=True).copy()
    symbol = str(df["symbol"].iloc[0])
    gap_limit = 0.21 if symbol.startswith(("30", "68")) else 0.11

    closes = df["close"].to_numpy(dtype=np.float64)
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)

    # 从后往前累积前复权因子，使得最新一天的价格等于真实盘面价格，且历史拆分日无断崖跳空
    adj_factor = np.ones(len(df), dtype=np.float64)
    cum_factor = 1.0
    for t in range(len(df) - 1, 0, -1):
        raw_ret = closes[t] / max(closes[t - 1], 1e-8) - 1.0
        if abs(raw_ret) > gap_limit:
            # 发生份额拆分或大比例除权：用当日开盘相对前收的比例作为拆分折算因子
            split_ratio = opens[t] / max(closes[t - 1], 1e-8)
            cum_factor *= split_ratio
        adj_factor[t - 1] = cum_factor

    df["open"] = np.round(opens * adj_factor, 4)
    df["high"] = np.round(highs * adj_factor, 4)
    df["low"] = np.round(lows * adj_factor, 4)
    df["close"] = np.round(closes * adj_factor, 4)
    return df


def _fetch_from_akshare(n_days: int = 750) -> pd.DataFrame:
    """使用 AkShare (新浪/东财双通道) 拉取 12 只核心场内 ETF 与个股近 3 年真实前复权日线."""
    import akshare as ak

    end_dt = pd.Timestamp.today().strftime("%Y%m%d")
    start_dt = (pd.Timestamp.today() - pd.Timedelta(days=int(n_days * 1.65))).strftime("%Y%m%d")
    start_ts = pd.to_datetime(start_dt)
    frames = []

    for asset in CORE_UNIVERSE:
        sub = None
        is_etf = asset.symbol.startswith(("51", "15"))
        prefix = "sh" if asset.symbol.startswith(("5", "6")) else "sz"

        # 通道 1: 新浪高速历史行情接口 (对 ETF 和 A股前复权最稳定)
        try:
            if is_etf:
                raw = ak.fund_etf_hist_sina(symbol=f"{prefix}{asset.symbol}")
                if raw is not None and not raw.empty:
                    raw["date"] = pd.to_datetime(raw["date"])
                    raw = raw[raw["date"] >= start_ts]
                    sub = pd.DataFrame({
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "sector": asset.sector,
                        "trade_date": raw["date"],
                        "open": raw["open"].astype(float),
                        "high": raw["high"].astype(float),
                        "low": raw["low"].astype(float),
                        "close": raw["close"].astype(float),
                        "volume": raw["volume"].astype(float),
                        "amount": raw["amount"].astype(float),
                    })
            else:
                raw = ak.stock_zh_a_daily(
                    symbol=f"{prefix}{asset.symbol}",
                    start_date=start_dt,
                    end_date=end_dt,
                    adjust="qfq",
                )
                if raw is not None and not raw.empty:
                    sub = pd.DataFrame({
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "sector": asset.sector,
                        "trade_date": pd.to_datetime(raw["date"]),
                        "open": raw["open"].astype(float),
                        "high": raw["high"].astype(float),
                        "low": raw["low"].astype(float),
                        "close": raw["close"].astype(float),
                        "volume": raw["volume"].astype(float),
                        "amount": raw["amount"].astype(float),
                    })
        except Exception:
            sub = None

        # 通道 2: 东方财富备用接口
        if sub is None or sub.empty:
            try:
                if is_etf:
                    raw = ak.fund_etf_hist_em(
                        symbol=asset.symbol,
                        period="daily",
                        start_date=start_dt,
                        end_date=end_dt,
                        adjust="qfq",
                    )
                else:
                    raw = ak.stock_zh_a_hist(
                        symbol=asset.symbol,
                        period="daily",
                        start_date=start_dt,
                        end_date=end_dt,
                        adjust="qfq",
                    )
                if raw is not None and not raw.empty:
                    sub = pd.DataFrame({
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "sector": asset.sector,
                        "trade_date": pd.to_datetime(raw["日期"]),
                        "open": raw["开盘"].astype(float),
                        "high": raw["最高"].astype(float),
                        "low": raw["最低"].astype(float),
                        "close": raw["收盘"].astype(float),
                        "volume": raw["成交量"].astype(float),
                        "amount": raw["成交额"].astype(float),
                    })
            except Exception:
                sub = None

        if sub is not None and not sub.empty:
            frames.append(_apply_split_qfq_adjustment(sub))

    if not frames:
        return pd.DataFrame()
    full_df = pd.concat(frames, ignore_index=True)
    common_dates = (
        full_df.groupby("trade_date")["symbol"]
        .nunique()
        .loc[lambda s: s == len(frames)]
        .index.sort_values()
    )
    if len(common_dates) > n_days:
        common_dates = common_dates[-n_days:]
    return (
        full_df[full_df["trade_date"].isin(common_dates)]
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )


def fetch_latest_eastmoney_news(top_per_asset: int = 4) -> pd.DataFrame:
    """从东方财富 (ak.stock_news_em) 抓取标的池最新个股/行业新闻快讯."""
    import akshare as ak

    news_rows = []
    for asset in CORE_UNIVERSE:
        try:
            raw = ak.stock_news_em(symbol=asset.symbol)
            if raw is None or raw.empty:
                continue
            for _, row in raw.head(top_per_asset).iterrows():
                news_rows.append({
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "sector": asset.sector,
                    "pub_time": str(row.get("发布时间", "")),
                    "title": str(row.get("新闻标题", "")),
                    "content": str(row.get("新闻内容", ""))[:300],
                    "source": str(row.get("文章来源", "东方财富")),
                })
        except Exception:
            continue
    return pd.DataFrame(news_rows)


def generate_realistic_ashare_panel(n_days: int = 750, seed: int = 42) -> pd.DataFrame:
    """生成蕴含真实金融物理规律的 A 股面板数据 (用于离线训练与 GNN/RL 消融验证).

    内置三大真实市场机制：
    1. 宏观市场状态切换 (牛市/震荡/熊市波动率聚类 GARCH 效应)
    2. 个人时序动量与短期均值回归特征
    3. 知识图谱产业链动量溢出效应 (Lead-Lag Spillover)：上游/龙头标的在 t-1, t-2 的超额异动会向图谱邻居传导
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-10-30", periods=n_days)
    n_assets = len(CORE_UNIVERSE)
    sym_to_idx: Dict[str, int] = {a.symbol: i for i, a in enumerate(CORE_UNIVERSE)}

    # 构建知识图谱传导矩阵 A (dst <- src)
    adj_spillover = np.zeros((n_assets, n_assets), dtype=np.float64)
    for src, dst, _, w in PRIOR_KG_RELATIONS:
        i_src, i_dst = sym_to_idx[src], sym_to_idx[dst]
        adj_spillover[i_dst, i_src] = w * 0.18
        adj_spillover[i_src, i_dst] = w * 0.06  # 龙头向跟随者传导更强

    # 行归一化
    row_sums = adj_spillover.sum(axis=1, keepdims=True)
    adj_spillover = np.divide(adj_spillover, np.maximum(row_sums, 1.0))

    rets = np.zeros((n_days, n_assets), dtype=np.float64)
    vols = np.zeros((n_days, n_assets), dtype=np.float64)
    market_vol = 0.014

    for t in range(n_days):
        # GARCH 风格动态市场波动率与牛熊周期漂移
        regime_drift = 0.0003 * np.sin(2 * np.pi * t / 180.0)
        market_shock = rng.normal(regime_drift, market_vol)
        market_vol = np.clip(0.92 * market_vol + 0.08 * abs(market_shock) + 0.0005, 0.008, 0.032)

        # 板块轮动冲击
        sector_shocks: Dict[str, float] = {}
        for sec in {a.sector for a in CORE_UNIVERSE}:
            sector_shocks[sec] = rng.normal(0.0, 0.011)

        for i, asset in enumerate(CORE_UNIVERSE):
            idiosyncratic = rng.normal(0.0, 0.011)
            # 1. 时序自回归动量 (5日趋势延续 + 20日中期动量)
            ts_signal = 0.0
            if t >= 20:
                mom_5 = np.mean(rets[t - 5:t, i])
                mom_20 = np.mean(rets[t - 20:t, i])
                ts_signal = 0.25 * mom_5 + 0.15 * mom_20

            # 2. 知识图谱空间动量溢出 (Lead-Lag: 龙头/上游过去 1~3 日的超额异动持续传导至下游)
            kg_spill = 0.0
            if t >= 3:
                ex_1 = rets[t - 1] - np.mean(rets[t - 1])
                ex_2 = rets[t - 2] - np.mean(rets[t - 2])
                ex_3 = rets[t - 3] - np.mean(rets[t - 3])
                kg_spill = float(
                    adj_spillover[i] @ (0.40 * ex_1 + 0.30 * ex_2 + 0.20 * ex_3)
                )

            raw_ret = 0.40 * market_shock + 0.30 * sector_shocks[asset.sector] + ts_signal + kg_spill + idiosyncratic
            # A股涨跌停裁剪 (-10% ~ +10%)
            rets[t, i] = np.clip(raw_ret, -0.098, 0.098)
            vols[t, i] = np.exp(15.5 + 35.0 * abs(rets[t, i]) + rng.normal(0, 0.25))

    rows = []
    for i, asset in enumerate(CORE_UNIVERSE):
        prices = 10.0 * np.exp(np.cumsum(rets[:, i]))
        for t in range(n_days):
            close_p = float(prices[t])
            prev_c = float(prices[t - 1]) if t > 0 else close_p
            open_p = prev_c * (1.0 + float(rng.normal(0, 0.004)))
            spread = abs(rets[t, i]) * 0.7 + float(rng.uniform(0.004, 0.015))
            high_p = max(open_p, close_p) * (1.0 + spread * 0.6)
            low_p = min(open_p, close_p) * (1.0 - spread * 0.6)
            vol = float(vols[t, i])
            rows.append({
                "symbol": asset.symbol,
                "name": asset.name,
                "sector": asset.sector,
                "trade_date": dates[t],
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close_p, 4),
                "volume": round(vol, 2),
                "amount": round(vol * close_p, 2),
            })
    return pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
