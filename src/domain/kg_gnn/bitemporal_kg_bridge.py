"""领域层方向 C：双时点 (Bitemporal) 知识图谱 RAG 抽取、三跳产业链传导与动态时点邻接张量桥接器.

核心防穿越原理 (对齐 research_v1/quant_research/evidence.py)：
1. 事件三时点约束：published_at <= observed_at <= ready_at；在任意历史交易日 t 收盘 (15:00 CST) 截面，
   仅当 ready_at <= cutoff(t) 时该事件才进入可见视野。
2. 关系双时点演化：每条产业链/跨市场传导边具有事实有效期 [valid_from, valid_to) 与市场获知时间 known_at。
   在截面 cutoff(t) 构建图谱时，必须先按 known_at <= cutoff(t) 筛选每个 relation_id 的最新已知版本，
   再检查 withdrawn == False 且 valid_from <= cutoff(t) < valid_to，彻底杜绝未来新闻修改历史图权重的穿越偏差。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import fmean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.domain.kg_gnn.dynamic_kg_rules import (
    KG_AGENT_SYSTEM_PROMPT,
    THEME_KEYWORDS_TO_SECTOR,
    score_financial_text_rule,
)
from src.infrastructure.llm_client import DualModelLLMClient

MAX_RELATION_HOPS = 2  # 事件提及实体为第1跳，加最多2跳图关系 = 3跳传导
_VERSIONED_RELATION_COLUMNS = {"relation_id", "valid_from", "known_at", "valid_to", "withdrawn"}


def has_versioned_relation_columns(edges: pd.DataFrame) -> bool:
    """Recognize versioned input, including incomplete schemas that must be rejected."""
    return bool(_VERSIONED_RELATION_COLUMNS.intersection(edges.columns))


def _ensure_utc(dt: datetime | pd.Timestamp | str) -> datetime:
    ts = pd.Timestamp(dt)
    if pd.isna(ts):
        raise ValueError("missing evidence timestamp")
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Shanghai")
    return ts.tz_convert("UTC").to_pydatetime()


def session_cutoff_utc(session_date: pd.Timestamp | datetime | str) -> datetime:
    """将交易日映射为当日 A 股收盘决策时点 (15:00 Asia/Shanghai) 的 UTC 时间."""
    day = pd.Timestamp(session_date).normalize()
    if day.tzinfo is not None:
        day = day.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    return (day + pd.Timedelta(hours=15)).tz_localize("Asia/Shanghai").tz_convert("UTC").to_pydatetime()


@dataclass(frozen=True)
class BitemporalEvent:
    """带严格三时点 (published_at <= observed_at <= ready_at) 的财经资讯/舆论事件观测."""

    event_id: str
    source_url: str
    entity_symbols: Tuple[str, ...]
    category: str
    published_at: datetime
    observed_at: datetime
    ready_at: datetime
    confidence: float
    sentiment: float
    heat: float = 1.0

    def __post_init__(self) -> None:
        pub = _ensure_utc(self.published_at)
        obs = _ensure_utc(self.observed_at)
        rdy = _ensure_utc(self.ready_at)
        if not (pub <= obs <= rdy):
            raise ValueError(f"BitemporalEvent violated published_at <= observed_at <= ready_at: {self.event_id}")
        if not self.entity_symbols or len(set(self.entity_symbols)) != len(self.entity_symbols):
            raise ValueError(f"BitemporalEvent requires unique non-empty entity_symbols: {self.event_id}")
        if not (0.0 <= self.confidence <= 1.0) or not (-1.0 <= self.sentiment <= 1.0):
            raise ValueError(f"BitemporalEvent confidence/sentiment out of bounds: {self.event_id}")


@dataclass(frozen=True)
class BitemporalRelationVersion:
    """带事实有效期 [valid_from, valid_to) 与获知时点 known_at 的知识图谱关系版本."""

    relation_id: str
    src_symbol: str
    dst_symbol: str
    relation_type: str
    valid_from: datetime
    known_at: datetime
    weight: float
    confidence: float = 0.85
    valid_to: Optional[datetime] = None
    withdrawn: bool = False

    def __post_init__(self) -> None:
        vf = _ensure_utc(self.valid_from)
        if self.valid_to is not None and _ensure_utc(self.valid_to) <= vf:
            raise ValueError(f"Relation validity interval must be positive: {self.relation_id}")
        if not (0.0 <= self.weight <= 1.0) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Relation weight/confidence must be in [0, 1]: {self.relation_id}")


@dataclass(frozen=True)
class BitemporalKGBundle:
    """不可变双时点图谱与事件证据包."""

    events: Tuple[BitemporalEvent, ...] = field(default_factory=tuple)
    relations: Tuple[BitemporalRelationVersion, ...] = field(default_factory=tuple)


def build_asof_directed_graph(
    relations: Iterable[BitemporalRelationVersion],
    as_of: datetime | pd.Timestamp | str,
) -> nx.DiGraph:
    """在指定时点 as_of 构建零未来函数的有向知识图谱.

    严格顺序：
    1. 先过滤 known_at <= as_of，取同一 relation_id 的最新已知版本；
    2. 若最新版本标记 withdrawn=True，则该边被撤销（绝不回退到旧版本）；
    3. 检查 valid_from <= as_of < valid_to。
    """
    cutoff = _ensure_utc(as_of)
    latest_by_id: Dict[str, BitemporalRelationVersion] = {}
    seen_versions: set[Tuple[str, datetime]] = set()

    for rel in relations:
        known = _ensure_utc(rel.known_at)
        if known > cutoff:
            continue
        version_key = (rel.relation_id, known)
        if version_key in seen_versions:
            raise ValueError(f"Duplicate relation version for {rel.relation_id} at {known.isoformat()}")
        seen_versions.add(version_key)
        prev = latest_by_id.get(rel.relation_id)
        if prev is None or known > _ensure_utc(prev.known_at):
            latest_by_id[rel.relation_id] = rel

    graph = nx.DiGraph(as_of=cutoff)
    for rel_id, rel in sorted(latest_by_id.items()):
        if rel.withdrawn:
            continue
        if _ensure_utc(rel.valid_from) > cutoff:
            continue
        if rel.valid_to is not None and _ensure_utc(rel.valid_to) <= cutoff:
            continue
        u, v = str(rel.src_symbol), str(rel.dst_symbol)
        if u == v:
            continue
        if graph.has_edge(u, v):
            if rel.weight > graph[u][v]["weight"]:
                graph[u][v].update(weight=rel.weight, confidence=rel.confidence, relation_id=rel_id)
        else:
            graph.add_edge(u, v, weight=rel.weight, confidence=rel.confidence, relation_id=rel_id)
    return graph


def _find_3hop_paths_to_targets(
    graph: nx.DiGraph,
    start_symbol: str,
    target_symbols: set[str],
) -> Dict[str, List[Tuple[float, float]]]:
    """从事件提及实体出发，沿当时有效图边最多遍历 MAX_RELATION_HOPS=2 跳 (合计3跳).

    Returns:
        mapping target_symbol -> list of (path_weight_product, path_confidence_product)
    """
    hits: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    queue: List[Tuple[Tuple[str, ...], float, float]] = [((start_symbol,), 1.0, 1.0)]
    while queue:
        nodes, w_prod, c_prod = queue.pop()
        curr = nodes[-1]
        if curr in target_symbols:
            hits[curr].append((w_prod, c_prod))
        if len(nodes) - 1 >= MAX_RELATION_HOPS or curr not in graph:
            continue
        for nxt, edge_attr in graph[curr].items():
            if nxt in nodes:
                continue
            queue.append((
                (*nodes, nxt),
                w_prod * float(edge_attr["weight"]),
                c_prod * float(edge_attr["confidence"]),
            ))
    return hits


def compute_bitemporal_3hop_features_and_adjacency(
    dates: Sequence[pd.Timestamp | datetime | str],
    symbols: Sequence[str],
    bundle: BitemporalKGBundle,
    lookback_days: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """按交易日序列逐日构建零穿越动态邻接张量 A_t 与三跳传导情绪特征张量 E_t.

    Args:
        dates: 交易日序列, 长度 T
        symbols: 资产代码列表, 长度 N
        bundle: 双时点知识图谱与事件证据包
        lookback_days: 事件回溯窗口天数

    Returns:
        dynamic_adj: shape = (T, N, N), 无自环时点显式邻接矩阵序列 (dst 接收 src: adj[dst, src])
        evidence_feats: shape = (T, N, 4), 四维特征依次为:
            [0] log1p(去重事件数), [1] 加权三跳传导情绪均值, [2] 情绪分歧度, [3] 传导置信度均值
    """
    sym_list = [str(s) for s in symbols]
    sym_to_idx = {s: i for i, s in enumerate(sym_list)}
    target_set = set(sym_list)
    n_days = len(dates)
    n_assets = len(sym_list)

    dynamic_adj = np.zeros((n_days, n_assets, n_assets), dtype=np.float32)
    evidence_feats = np.zeros((n_days, n_assets, 4), dtype=np.float32)

    for t_idx, raw_day in enumerate(dates):
        cutoff = session_cutoff_utc(raw_day)
        window_start = cutoff - timedelta(days=lookback_days)
        graph = build_asof_directed_graph(bundle.relations, as_of=cutoff)

        # 1. 导出当日 cutoff 可见的显式无自环邻接矩阵 A_t[dst, src]
        for u, v, attr in graph.edges(data=True):
            if u in sym_to_idx and v in sym_to_idx and u != v:
                dynamic_adj[t_idx, sym_to_idx[v], sym_to_idx[u]] = float(attr["weight"])

        # 2. 按 event_id 去重聚合可见窗口 [window_start, cutoff] 内的事件 (防止同事件多渠道转载重复计数)
        visible_by_event: Dict[str, List[BitemporalEvent]] = defaultdict(list)
        for ev in bundle.events:
            rdy = _ensure_utc(ev.ready_at)
            if window_start <= rdy <= cutoff:
                visible_by_event[ev.event_id].append(ev)

        per_target_sentiments: Dict[str, List[float]] = defaultdict(list)
        per_target_confidences: Dict[str, List[float]] = defaultdict(list)

        for _, obs_group in sorted(visible_by_event.items()):
            ev_sentiment = fmean(item.sentiment for item in obs_group)
            ev_conf = max(item.confidence for item in obs_group)
            mentioned = {s for item in obs_group for s in item.entity_symbols}

            # 同一事件通过多条路径传导至同一目标标的时仅计一次，取最强传导路径
            target_best_path: Dict[str, Tuple[float, float]] = {}
            for entity_sym in sorted(mentioned):
                for tgt, path_list in _find_3hop_paths_to_targets(graph, entity_sym, target_set).items():
                    best_w, best_c = max(path_list, key=lambda item: item[0] * item[1])
                    prev = target_best_path.get(tgt)
                    if prev is None or best_w * best_c > prev[0] * prev[1]:
                        target_best_path[tgt] = (best_w, best_c)

            for tgt, (path_w, path_c) in target_best_path.items():
                per_target_sentiments[tgt].append(ev_sentiment * path_w)
                per_target_confidences[tgt].append(ev_conf * path_c)

        for s_idx, sym in enumerate(sym_list):
            sents = per_target_sentiments.get(sym, [])
            confs = per_target_confidences.get(sym, [])
            if sents:
                evidence_feats[t_idx, s_idx, 0] = float(np.log1p(len(sents)))
                evidence_feats[t_idx, s_idx, 1] = float(fmean(sents))
                evidence_feats[t_idx, s_idx, 2] = float(pstdev(sents)) if len(sents) > 1 else 0.0
                evidence_feats[t_idx, s_idx, 3] = float(fmean(confs))

    return dynamic_adj, evidence_feats


def _versioned_relations(edges: pd.DataFrame) -> List[BitemporalRelationVersion]:
    required = {"relation_id", "src_symbol", "dst_symbol", "weight", "valid_from", "known_at"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"Missing versioned relation fields: {', '.join(sorted(missing))}")

    relations = []
    seen = set()
    for idx, row in edges.iterrows():
        try:
            identifiers = [row[key] for key in ("relation_id", "src_symbol", "dst_symbol")]
            if any(pd.isna(value) or not str(value).strip() for value in identifiers):
                raise ValueError("relation_id and endpoints cannot be empty")
            relation_id, src, dst = (str(value).strip() for value in identifiers)
            withdrawn = row.get("withdrawn", False)
            if not isinstance(withdrawn, (bool, np.bool_)):
                raise ValueError("withdrawn must be a boolean")
            timestamps = {}
            for field in ("valid_from", "known_at", "valid_to"):
                value = row.get(field)
                if field == "valid_to" and pd.isna(value):
                    timestamps[field] = None
                else:
                    if isinstance(value, (int, float, np.number, bool)):
                        raise ValueError(f"{field} must be a calendar timestamp")
                    timestamps[field] = _ensure_utc(value)
            version = (relation_id, timestamps["known_at"])
            if version in seen:
                raise ValueError(f"duplicate relation version for {relation_id}")
            seen.add(version)
            relations.append(
                BitemporalRelationVersion(
                    relation_id=relation_id,
                    src_symbol=src,
                    dst_symbol=dst,
                    relation_type=str(row.get("relation_type", row.get("relation", "industry_chain"))),
                    weight=float(row["weight"]),
                    confidence=float(row.get("confidence", 0.85)),
                    withdrawn=bool(withdrawn),
                    **timestamps,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid versioned relation at row {idx}: {exc}") from exc
    return relations


def build_bitemporal_bundle_from_news_and_edges(
    base_edges_df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    *,
    base_known_at: datetime | str = "2014-12-31T07:00:00+00:00",
    llm_client: Optional[DualModelLLMClient] = None,
    ingestion_delay_minutes: int = 15,
) -> BitemporalKGBundle:
    """Load versioned snapshot edges or timeless legacy priors, plus dated news.

    Versioned edges change only when the supplied revisions become visible.
    Legacy priors may receive heuristic news overlays at the event's ready time.
    """
    versioned = has_versioned_relation_columns(base_edges_df)
    events: List[BitemporalEvent] = []
    if versioned:
        relations = _versioned_relations(base_edges_df)
    else:
        base_dt = _ensure_utc(base_known_at)
        relations = []
        for idx, row in base_edges_df.iterrows():
            u, v = str(row["src_symbol"]), str(row["dst_symbol"])
            rel_type = str(row.get("relation", "industry_chain"))
            w = float(np.clip(float(row["weight"]), 0.05, 1.0))
            relations.append(
                BitemporalRelationVersion(
                    relation_id=f"base_{u}_{v}_{idx}",
                    src_symbol=u,
                    dst_symbol=v,
                    relation_type=rel_type,
                    valid_from=base_dt,
                    known_at=base_dt,
                    weight=w,
                    confidence=0.85,
                )
            )

    if news_df is None or news_df.empty:
        return BitemporalKGBundle(events=tuple(events), relations=tuple(relations))

    # 可选调用 DeepSeek / 本地 Qwen3.5-9B 获取结构化情绪与边增量
    llm_sentiment_map: Dict[str, float] = {}
    if llm_client is not None and (llm_client.is_local_online() or llm_client.is_deepseek_configured()):
        digest = [
            f"[{r.get('symbol', '')}] {r.get('title', '')}"
            for _, r in news_df.head(15).iterrows()
        ]
        res = llm_client.chat_json(
            system_prompt=KG_AGENT_SYSTEM_PROMPT,
            user_prompt="请抽取以下快讯的标的事件情绪与产业链传导权重：\n" + "\n".join(digest),
            prefer_tier="local_fast",
        )
        if res and isinstance(res.get("news_sentiments"), list):
            for item in res["news_sentiments"]:
                sym = str(item.get("symbol", ""))
                if sym:
                    llm_sentiment_map[sym] = float(np.clip(float(item.get("sentiment", 0.0)), -1.0, 1.0))

    for idx, row in news_df.reset_index(drop=True).iterrows():
        sym = str(row["symbol"])
        title = str(row.get("title", ""))
        raw_pub = row.get("publish_time")
        if pd.isna(raw_pub):
            raw_pub = row.get("published_at")
        pub_dt = _ensure_utc(raw_pub)
        raw_observed = row.get("observed_at")
        obs_dt = pub_dt if pd.isna(raw_observed) else _ensure_utc(raw_observed)
        raw_ready = row.get("ready_at")
        rdy_dt = (obs_dt + timedelta(minutes=max(1, ingestion_delay_minutes))
                  if pd.isna(raw_ready) else _ensure_utc(raw_ready))

        if "sentiment" in row and pd.notna(row["sentiment"]):
            score = float(np.clip(float(row["sentiment"]), -1.0, 1.0))
        elif sym in llm_sentiment_map:
            score = llm_sentiment_map[sym]
        else:
            content = str(row.get("content", ""))
            score = score_financial_text_rule(f"{title} {content}")

        events.append(
            BitemporalEvent(
                event_id=f"ev_{sym}_{idx}_{int(pub_dt.timestamp())}",
                source_url=str(row.get("source_url", f"https://finance.eastmoney.com/news/{sym}/{idx}.html")),
                entity_symbols=(sym,),
                category=str(row.get("category", "market_news")),
                published_at=pub_dt,
                observed_at=obs_dt,
                ready_at=rdy_dt,
                confidence=0.80,
                sentiment=score,
                heat=1.0 + abs(score),
            )
        )

        # Versioned snapshots supply their own revisions. Generate heuristic overlays only for legacy priors.
        if not versioned and not base_edges_df.empty and abs(score) >= 0.15:
            delta = 0.08 * score
            matching_edges = base_edges_df[base_edges_df["src_symbol"].astype(str) == sym]
            for _, edge_row in matching_edges.iterrows():
                u = str(edge_row["src_symbol"])
                v = str(edge_row["dst_symbol"])
                old_w = float(edge_row["weight"])
                new_w = float(np.clip(old_w + delta, 0.10, 0.98))
                relations.append(
                    BitemporalRelationVersion(
                        relation_id=f"base_{u}_{v}",
                        src_symbol=u,
                        dst_symbol=v,
                        relation_type=str(edge_row.get("relation", "catalyst_transmission")),
                        valid_from=pub_dt,
                        valid_to=rdy_dt + timedelta(days=30),
                        known_at=rdy_dt,
                        weight=new_w,
                        confidence=0.85,
                    )
                )

    return BitemporalKGBundle(events=tuple(events), relations=tuple(relations))
