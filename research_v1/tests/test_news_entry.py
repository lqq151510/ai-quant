import pytest

from quant_research.news_entry import compare_news_entry


@pytest.mark.parametrize(
    ("available_by", "observed_at", "expected_reason"),
    [
        pytest.param("2026-11-16T09:45:02+08:00", "2026-11-16T01:45:02Z", "news_abnormal_volatility",
                     id="snapshot-boundary-inclusive"),
        pytest.param("2026-11-16T09:45:02+08:00", "2026-11-16T09:45:03+08:00", "within_limit",
                     id="observed-between-snapshot-and-decision"),
        pytest.param("2026-11-16T09:45:02+08:00", "2026-11-13T09:45:09+08:00", "within_limit",
                     id="older-than-72-hours-at-decision"),
        pytest.param("2026-11-16T09:45:02+08:00", "2026-11-13T09:45:10+08:00", "news_abnormal_volatility",
                     id="72-hour-boundary-inclusive"),
        pytest.param("2026-11-16T09:45:12+08:00", "2026-11-16T09:45:11+08:00", "within_limit",
                     id="snapshot-cannot-extend-decision-cutoff"),
        pytest.param(None, "2026-11-16T09:45:10+08:00", "news_abnormal_volatility",
                     id="default-decision-boundary-inclusive"),
    ],
)
def test_news_availability_uses_snapshot_without_moving_decision_window(available_by, observed_at, expected_reason):
    symbol = "600001.SH"
    observation = {
        "symbol": symbol, "as_of": "2026-11-16T09:45:10+08:00", "signal": True,
        "signal_at": "2026-11-13T15:00:00+08:00", "expires_at": "2026-11-16T09:50:00+08:00",
        "limit_price": "10.00", "ordered_units": 100, "max_quote_age_seconds": 30,
        "quote": {"symbol": symbol, "at": "2026-11-16T09:45:09+08:00", "ask_price": "9.99", "ask_units": 100},
    }
    batches = [{"events": [{
        "event_id": "synthetic-event", "symbols": [symbol], "title": "股票交易异常波动公告",
        "first_observed_at": observed_at, "source_url": "https://example.invalid/synthetic-event",
    }]}]
    snapshot_arg = {} if available_by is None else {"available_by": available_by}
    result = compare_news_entry(observation, batches, **snapshot_arg)

    assert result["baseline"] == {"eligible": True, "reason": "within_limit"}
    assert result["reason"] == expected_reason
    assert result["eligible"] is (expected_reason == "within_limit")
