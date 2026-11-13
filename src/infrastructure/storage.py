"""基础设施层：基于 DuckDB 的本地高性能列式量化存储引擎."""

from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd


def read_daily_bar_metadata(db_path: str) -> pd.DataFrame:
    """Read only the metadata needed by the prototype research report."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info('daily_bars')").fetchall()}
        selected = ["symbol", "trade_date"]
        selected.extend(column for column in ("data_origin", "data_source") if column in columns)
        return conn.execute(f"SELECT {', '.join(selected)} FROM daily_bars").df()


class QuantDuckDBStorage:
    """管理本地行情日线、截面特征及知识图谱关系边的 DuckDB 存储."""

    def __init__(self, db_path: str = "data/quant_market.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_bars (
                    symbol VARCHAR,
                    name VARCHAR,
                    sector VARCHAR,
                    trade_date DATE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    amount DOUBLE,
                    PRIMARY KEY (symbol, trade_date)
                );
            """)
            # Legacy rows have no recorded source. Keep those columns NULL.
            conn.execute("ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS data_origin VARCHAR")
            conn.execute("ALTER TABLE daily_bars ADD COLUMN IF NOT EXISTS data_source VARCHAR")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_edges (
                    src_symbol VARCHAR,
                    dst_symbol VARCHAR,
                    relation_type VARCHAR,
                    weight DOUBLE,
                    updated_at DATE,
                    PRIMARY KEY (src_symbol, dst_symbol, relation_type)
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_news (
                    symbol VARCHAR,
                    pub_time VARCHAR,
                    title VARCHAR,
                    content VARCHAR,
                    source VARCHAR,
                    sentiment_score DOUBLE,
                    PRIMARY KEY (symbol, title)
                );
            """)

    def save_daily_bars(self, df: pd.DataFrame) -> int:
        """幂等写入或更新日线行情数据."""
        return self._write_daily_bars(df, replace=False)

    def _write_daily_bars(self, df: pd.DataFrame, *, replace: bool) -> int:
        if df.empty and not replace:
            return 0
        if not df.empty:
            required = ["symbol", "name", "sector", "trade_date",
                        "open", "high", "low", "close", "volume", "amount"]
            clean_df = df[required].copy()
            clean_df["trade_date"] = pd.to_datetime(clean_df["trade_date"]).dt.date
            clean_df["data_origin"] = df.get("data_origin")
            clean_df["data_source"] = df.get("data_source")
        with self._connect() as conn:
            conn.begin()
            if replace:
                conn.execute("DELETE FROM daily_bars")
            if not df.empty:
                conn.register("tmp_bars", clean_df)
                conn.execute("""
                    INSERT OR REPLACE INTO daily_bars (
                        symbol, name, sector, trade_date, open, high, low, close, volume, amount,
                        data_origin, data_source
                    )
                    SELECT * FROM tmp_bars
                """)
            conn.commit()
        return len(df)

    def load_daily_bars(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """按时间顺序加载全截面日线行情."""
        query = "SELECT * FROM daily_bars WHERE 1=1"
        params = []
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)
        query += " ORDER BY trade_date ASC, symbol ASC"
        with self._connect() as conn:
            df = conn.execute(query, params).df()
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def save_kg_edges(self, edges_df: pd.DataFrame) -> int:
        """保存知识图谱关系边 (行业同属、上下游供应链、概念主题)."""
        if edges_df.empty:
            return 0
        with self._connect() as conn:
            conn.register("tmp_edges", edges_df)
            conn.execute("""
                INSERT OR REPLACE INTO kg_edges
                SELECT src_symbol, dst_symbol, relation_type, weight, updated_at
                FROM tmp_edges
            """)
            conn.unregister("tmp_edges")
        return len(edges_df)

    def load_kg_edges(self) -> pd.DataFrame:
        """加载知识图谱静态与动态关系边."""
        with self._connect() as conn:
            return conn.execute("SELECT * FROM kg_edges").df()

    def replace_all_daily_bars(self, df: pd.DataFrame) -> int:
        """在同一事务中替换行情及其来源标签。"""
        return self._write_daily_bars(df, replace=True)

    def save_market_news(self, news_df: pd.DataFrame) -> int:
        """保存个股/ETF 最新财经新闻及情绪打分."""
        if news_df.empty:
            return 0
        cols = ["symbol", "pub_time", "title", "content", "source", "sentiment_score"]
        clean = news_df[cols].drop_duplicates(subset=["symbol", "title"]).copy()
        with self._connect() as conn:
            conn.register("tmp_news", clean)
            conn.execute("""
                INSERT OR REPLACE INTO market_news
                SELECT symbol, pub_time, title, content, source, sentiment_score
                FROM tmp_news
            """)
            conn.unregister("tmp_news")
        return len(clean)

    def load_market_news(self) -> pd.DataFrame:
        """加载本地存储的财经新闻."""
        with self._connect() as conn:
            return conn.execute("SELECT * FROM market_news ORDER BY pub_time DESC").df()
