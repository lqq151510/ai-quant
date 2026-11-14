"""Compare the existing paper entry rule with a forward news risk filter."""

from datetime import timedelta

from .paper_entry import _timestamp, evaluate_paper_buy


def compare_news_entry(observation: dict, batches: list[dict], *, available_by: str | None = None) -> dict:
    baseline = evaluate_paper_buy(observation)
    cutoff = _timestamp(observation["as_of"], "as_of")
    start = cutoff - timedelta(hours=72)
    visible_until = cutoff if available_by is None else min(cutoff, _timestamp(available_by, "available_by"))
    symbol = observation["symbol"].upper()
    visible = {}

    for batch in batches:
        if not isinstance(batch, dict) or not isinstance(batch.get("events"), list):
            raise ValueError("news batch must contain an events list")
        for event in batch["events"]:
            if not isinstance(event, dict) or not isinstance(event.get("symbols"), list):
                raise ValueError("news event must contain a symbols list")
            if symbol not in event["symbols"]:
                continue
            observed = _timestamp(event.get("first_observed_at"), "first_observed_at").astimezone(cutoff.tzinfo)
            if observed > visible_until:
                continue
            event_id = event.get("event_id")
            source = event.get("source_url")
            if not all(isinstance(value, str) and value.strip() for value in (event_id, source)):
                if observed >= start:
                    raise ValueError("news event requires event_id and source_url")
                continue
            key = (event_id, source)
            prior = visible.get(key)
            if prior is None or observed < prior[0]:
                visible[key] = (observed, [event])
            elif observed == prior[0]:
                prior[1].append(event)

    recent = []
    for (event_id, source), (observed, events) in visible.items():
        if observed < start:
            continue
        titles = set()
        for event in events:
            title = event.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("news event requires title")
            published = event.get("published_at")
            if published is not None and _timestamp(published, "published_at") > observed:
                raise ValueError("news publication cannot be later than first observation")
            titles.add(title)
        if len(titles) != 1:
            raise ValueError("conflicting titles for the same news observation")
        recent.append((observed, titles.pop(), source, event_id))

    matched = [
        {"event_id": event_id, "title": title, "source_url": source, "first_observed_at": observed.isoformat()}
        for observed, title, source, event_id in sorted(recent)
        if "股票交易异常波动" in title
    ]
    deferred = baseline["eligible"] and bool(matched)
    return {
        **baseline,
        "eligible": baseline["eligible"] and not deferred,
        "reason": "news_abnormal_volatility" if deferred else baseline["reason"],
        "baseline": {"eligible": baseline["eligible"], "reason": baseline["reason"]},
        "news": {
            "lookback_hours": 72,
            "coverage": "partial" if recent else "unknown",
            "visible_observations": len(recent),
            "matched": matched,
        },
    }
