"""应用编排层 Stage 2：动态知识图谱演化智能体 (Graph-Agent)."""

from typing import Optional
from src.domain.kg_gnn.dynamic_kg_rules import (
    DynamicKGUpdateResult,
    KG_AGENT_SYSTEM_PROMPT,
    compute_dynamic_kg_updates,
)
from src.infrastructure.llm_client import DualModelLLMClient
from src.infrastructure.market_data import CORE_UNIVERSE, fetch_latest_eastmoney_news
from src.infrastructure.storage import QuantDuckDBStorage


class KnowledgeGraphAgent:
    """自动抓取东方财富个股/行业新闻并更新 DuckDB 知识图谱权重的智能体编排器."""

    def __init__(
        self,
        storage: QuantDuckDBStorage,
        llm_client: Optional[DualModelLLMClient] = None,
    ) -> None:
        self.storage = storage
        self.llm_client = llm_client or DualModelLLMClient()

    def run_daily_kg_update(self, top_news_per_asset: int = 3) -> DynamicKGUpdateResult:
        """执行每日新闻感知 -> LLM/领域提取 -> 动态图谱边更新 -> DuckDB 持久化闭环."""
        cached_news = self.storage.load_market_news()
        if not cached_news.empty and cached_news["symbol"].nunique() >= len(CORE_UNIVERSE):
            news_df = cached_news
        else:
            news_df = fetch_latest_eastmoney_news(top_per_asset=top_news_per_asset)
        base_edges_df = self.storage.load_kg_edges()

        llm_result = None
        if not news_df.empty and (
            self.llm_client.is_local_online() or self.llm_client.is_deepseek_configured()
        ):
            digest_lines = [
                f"[{r['symbol']} {r['name']}] {r['title']}"
                for _, r in news_df.head(18).iterrows()
            ]
            user_prompt = "以下为今日最新东方财富快讯，请输出知识图谱边权重调整 JSON：\n" + "\n".join(digest_lines)
            llm_result = self.llm_client.chat_json(
                system_prompt=KG_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                prefer_tier="local_fast",
            )

        update_res = compute_dynamic_kg_updates(
            news_df=news_df,
            base_edges_df=base_edges_df,
            llm_json_result=llm_result,
        )

        if not update_res.scored_news_df.empty:
            self.storage.save_market_news(update_res.scored_news_df)
        if not update_res.updated_edges_df.empty:
            self.storage.save_kg_edges(update_res.updated_edges_df)

        return update_res
