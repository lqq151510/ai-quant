"""领域层 Stage 2：动态知识图谱演化规则、金融提示词定义与产业链共振评分模型."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


# 领域层提示词：严格限定输出 JSON 结构，提取新闻对标的及关联产业链边的催化强度
KG_AGENT_SYSTEM_PROMPT = """你是一位资深的 A 股量化知识图谱研究员（Graph-Agent）。
你的任务是根据输入的东方财富最新个股/ETF新闻快讯，评估各标的的短期事件催化极性（sentiment_score: -1.0 到 +1.0），
并识别哪些产业链或板块关联边（edge_updates）因共同主题或上下游传导而需要增强或衰减。
请严格返回如下 JSON 格式：
{
  "asset_sentiments": {"股票代码": 0.35},
  "edge_boosts": [
    {"src_symbol": "688981", "dst_symbol": "002371", "delta_weight": 0.12, "reason": "晶圆厂扩产带动国产半导体设备需求"}
  ]
}"""


POSITIVE_FIN_KEYWORDS: List[Tuple[str, float]] = [
    ("大涨", 0.35), ("突破", 0.25), ("净流入", 0.30), ("增持", 0.35), ("回购", 0.30),
    ("超预期", 0.40), ("中标", 0.30), ("扩产", 0.25), ("满产", 0.30), ("新高", 0.30),
    ("利好", 0.30), ("领涨", 0.35), ("算力", 0.20), ("国产替代", 0.25), ("分红", 0.20),
    ("翻倍", 0.35), ("增长", 0.20), ("获批", 0.25), ("签约", 0.20), ("景气", 0.25),
]

NEGATIVE_FIN_KEYWORDS: List[Tuple[str, float]] = [
    ("减持", -0.40), ("下跌", -0.25), ("净流出", -0.30), ("亏损", -0.35), ("立案", -0.50),
    ("警示", -0.35), ("下滑", -0.25), ("暴跌", -0.45), ("解禁", -0.30), ("低于预期", -0.35),
    ("风险", -0.15), ("退潮", -0.30), ("下调", -0.25), ("停牌", -0.40), ("溢价风险", -0.40),
    ("暂停令", -0.45), ("放缓", -0.35), ("集体回调", -0.30),
]

# 基金/自媒体常见逆势营销软文话术：在大跌日出现时属于典型「诱多噪音」，必须剥离正面权重
SOFT_MARKETING_TRAPS: List[str] = [
    "逢低布局", "凸显投资价值", "释放配置性价比", "主力重金抢筹", "底仓配置价值",
]

THEME_KEYWORDS_TO_SECTOR: Dict[str, List[str]] = {
    "Tech_Semi": ["半导体", "芯片", "晶圆", "光刻", "存储", "北方华创", "中芯国际", "国产替代", "设备"],
    "AI_Compute": ["算力", "光模块", "CPO", "通信", "AI", "大模型", "中际旭创", "智算中心", "英伟达"],
    "New_Energy": ["固态电池", "锂电", "储能", "光伏", "新能源", "宁德时代", "比亚迪", "智驾", "出海"],
    "Dividend_Value": ["红利", "高股息", "央企", "分红", "水电", "长江电力", "险资"],
    "Macro_Hedge": ["黄金", "避险", "美联储", "降息", "央行购金", "通胀"],
}


@dataclass
class DynamicKGUpdateResult:
    """动态知识图谱更新结果."""
    scored_news_df: pd.DataFrame
    updated_edges_df: pd.DataFrame
    asset_sentiment_map: Dict[str, float]
    active_catalysts: List[str]
    llm_mode_used: str


def score_financial_text_rule(text: str) -> float:
    """领域词典与反软文诱多极性规则打分 (范围 [-1.0, 1.0])."""
    cleaned_text = text
    for trap in SOFT_MARKETING_TRAPS:
        cleaned_text = cleaned_text.replace(trap, "")

    score = 0.0
    for kw, w in POSITIVE_FIN_KEYWORDS:
        if kw in cleaned_text:
            score += w
    for kw, w in NEGATIVE_FIN_KEYWORDS:
        if kw in cleaned_text:
            score += w
    return float(np.clip(score, -1.0, 1.0))


def compute_dynamic_kg_updates(
    news_df: pd.DataFrame,
    base_edges_df: pd.DataFrame,
    llm_json_result: Dict[str, Any] | None = None,
    ema_alpha: float = 0.35,
) -> DynamicKGUpdateResult:
    """结合东方财富最新新闻与 LLM/领域规则，动态更新知识图谱边权重.

    核心数学机制：
    1. 计算每条新闻的情绪催化分 s_i in [-1, 1] 及主题热度。
    2. 当源节点 u 出现强催化 (|s_u| > 0.1) 或 u, v 共现同一高热度产业主题时，
       动态增强传导边权重：w_new = clip((1 - alpha) * w_old + alpha * (w_old + boost), 0.15, 0.99)。
    """
    if news_df.empty:
        return DynamicKGUpdateResult(
            scored_news_df=news_df,
            updated_edges_df=base_edges_df,
            asset_sentiment_map={},
            active_catalysts=["当前无新增新闻，维持先验知识图谱权重"],
            llm_mode_used="none",
        )

    scored_df = news_df.copy()
    scored_df["sentiment_score"] = scored_df.apply(
        lambda r: score_financial_text_rule(f"{r['title']} {r['content']}"), axis=1
    )

    # 若本地 Qwen / DeepSeek 返回了覆盖分数，则融合大模型判断
    llm_sentiments = (llm_json_result or {}).get("asset_sentiments", {})
    asset_sent_map: Dict[str, float] = {}
    for sym, grp in scored_df.groupby("symbol"):
        rule_s = float(grp["sentiment_score"].mean())
        if sym in llm_sentiments:
            asset_sent_map[str(sym)] = float(np.clip(0.6 * float(llm_sentiments[sym]) + 0.4 * rule_s, -1.0, 1.0))
        else:
            asset_sent_map[str(sym)] = rule_s

    # 统计各赛道主题在今日新闻中的提及密度 (Topic Heat)
    all_news_text = " ".join((scored_df["title"] + " " + scored_df["content"]).tolist())
    sector_heat: Dict[str, float] = {}
    for sec, kws in THEME_KEYWORDS_TO_SECTOR.items():
        hits = sum(all_news_text.count(k) for k in kws)
        sector_heat[sec] = float(np.clip(hits * 0.04, 0.0, 0.25))

    # LLM 显式边增益映射
    llm_edge_boosts: Dict[Tuple[str, str], float] = {}
    for item in (llm_json_result or {}).get("edge_boosts", []):
        u = str(item.get("src_symbol", ""))
        v = str(item.get("dst_symbol", ""))
        llm_edge_boosts[(u, v)] = float(item.get("delta_weight", 0.0))

    # 跨市场外盘冲击传染 (Global Tech Anchor Contagion):
    # 若 513100 (纳指ETF) 出现负向风险 (如海外AI算力暴跌/溢价停牌)，沿海外映射边向光模块/通信传导负向修正，并向红利避险边注入正向增益
    nasdaq_sent = asset_sent_map.get("513100", 0.0)
    if nasdaq_sent < -0.10:
        for _, row in base_edges_df[base_edges_df["src_symbol"] == "513100"].iterrows():
            dst = str(row["dst_symbol"])
            rel = str(row["relation_type"])
            w_edge = float(row["weight"])
            if "overseas" in rel:
                asset_sent_map[dst] = float(np.clip(asset_sent_map.get(dst, 0.0) + 0.75 * nasdaq_sent * w_edge, -1.0, 1.0))
            elif "risk_off" in rel:
                asset_sent_map[dst] = float(np.clip(asset_sent_map.get(dst, 0.0) - 0.55 * nasdaq_sent * w_edge, -1.0, 1.0))

    updated_edges = base_edges_df.copy()
    catalysts: List[str] = []
    today_dt = pd.Timestamp.today().date()

    new_weights = []
    for _, row in updated_edges.iterrows():
        u = str(row["src_symbol"])
        v = str(row["dst_symbol"])
        old_w = float(row["weight"])

        # 源节点情绪动量驱动 + 目标共鸣
        s_u = asset_sent_map.get(u, 0.0)
        s_v = asset_sent_map.get(v, 0.0)
        synergy = 0.18 * abs(s_u) + (0.10 if s_u * s_v > 0 else 0.0)
        llm_delta = llm_edge_boosts.get((u, v), 0.0)

        target_w = old_w + synergy + llm_delta
        fused_w = float(np.clip((1.0 - ema_alpha) * old_w + ema_alpha * target_w, 0.15, 0.99))
        new_weights.append(round(fused_w, 4))

        if abs(fused_w - old_w) >= 0.015 and len(catalysts) < 5:
            catalysts.append(
                f"边 [{u} -> {v}] ({row['relation_type']}) 权重 {old_w:.2f} -> {fused_w:.2f} (源催化={s_u:+.2f})"
            )

    updated_edges["weight"] = new_weights
    updated_edges["updated_at"] = today_dt

    mode = "LLM + 领域规则混合" if llm_json_result else "东方财富实时新闻 + 领域语义图谱引擎"
    return DynamicKGUpdateResult(
        scored_news_df=scored_df,
        updated_edges_df=updated_edges,
        asset_sentiment_map=asset_sent_map,
        active_catalysts=catalysts or ["已完成全部知识图谱边新闻热度对齐"],
        llm_mode_used=mode,
    )
