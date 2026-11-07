"""Regression checks for chronological news and relation evidence."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from quant_research.evidence import (
    CoverageObservation,
    EventObservation,
    EvidenceBundle,
    RelationVersion,
    build_asof_graph,
    evidence_features,
    load_evidence,
)


def stamp(day, hour=0):
    return datetime(2026, 11, day, hour, tzinfo=timezone.utc)


def event(**changes):
    values = dict(
        event_id="policy-1",
        source_url="https://example.com/policy",
        channel="news",
        category="policy",
        entity_ids=("entity:600000",),
        published_at=stamp(2),
        observed_at=stamp(2, 1),
        ready_at=stamp(2, 2),
        confidence=0.8,
        sentiment=0.4,
        heat=2.0,
    )
    return EventObservation(**(values | changes))


def relation(**changes):
    values = dict(
        relation_id="bank-index",
        relation_type="entity_index",
        source_id="entity:600000",
        target_id="index:000300",
        source_url="https://example.com/members",
        valid_from=stamp(1),
        known_at=stamp(1),
        confidence=0.9,
    )
    return RelationVersion(**(values | changes))


def feature(bundle, target="etf:510300", channel="news", as_of=None):
    return next(item for item in evidence_features(
        bundle,
        as_of=as_of or stamp(4),
        lookback=timedelta(days=3),
        targets=(target,),
    ) if item.channel == channel)


def test_unknown_coverage_and_audited_empty_interval_are_distinct():
    unknown = feature(EvidenceBundle())
    assert unknown.coverage == "unknown"
    assert unknown.event_count is None
    assert unknown.mean_sentiment is None
    covered = CoverageObservation(
        target_id="etf:510300", channel="news", from_at=stamp(1),
        to_at=stamp(4), known_at=stamp(4),
        source_url="https://example.com/coverage",
    )
    zero = feature(EvidenceBundle(coverage=(covered,)))
    assert zero.coverage == "complete"
    assert zero.covered_fraction == 1
    assert zero.event_count == 0
    assert zero.mean_sentiment is None


def test_future_news_relation_and_correction_leave_past_features_identical():
    first = relation()
    second = relation(
        relation_id="index-etf", relation_type="index_etf",
        source_id="index:000300", target_id="etf:510300",
    )
    original = EvidenceBundle(events=(event(),), relations=(first, second))
    future = event(
        event_id="policy-2", source_url="https://example.com/later",
        published_at=stamp(5), observed_at=stamp(5), ready_at=stamp(5),
    )
    revised = relation(known_at=stamp(5), withdrawn=True)
    extra = relation(
        relation_id="future-direct", relation_type="entity_etf",
        target_id="etf:510300", valid_from=stamp(1), known_at=stamp(5),
    )
    enlarged = EvidenceBundle(
        events=(*original.events, future),
        relations=(*original.relations, revised, extra),
    )
    assert feature(original).model_dump() == feature(enlarged).model_dump()
    assert feature(enlarged, as_of=stamp(5), target="index:000300").event_count is None


def test_latest_withdrawal_does_not_resurrect_old_relationship():
    bundle = EvidenceBundle(
        events=(event(),),
        relations=(relation(), relation(known_at=stamp(3), withdrawn=True)),
    )
    assert feature(bundle, target="index:000300").hits == ()
    earlier = build_asof_graph(bundle.relations, stamp(2))
    assert earlier.has_edge("entity:600000", "index:000300")


def test_revised_validity_interval_supersedes_old_version():
    records = (
        relation(),
        relation(known_at=stamp(3), valid_from=stamp(5)),
    )
    assert len(build_asof_graph(records, stamp(4)).edges) == 0
    assert len(build_asof_graph(records, stamp(2)).edges) == 1


def test_reposts_and_parallel_routes_count_once_and_keep_unknown_weights():
    routes = (
        relation(),
        relation(
            relation_id="index-etf", relation_type="index_etf",
            source_id="index:000300", target_id="etf:510300", weight=0.2,
        ),
        relation(
            relation_id="direct-etf", relation_type="entity_etf",
            target_id="etf:510300",
        ),
    )
    original = event()
    repost = event(
        source_url="https://example.com/repost",
        observed_at=stamp(3), ready_at=stamp(3),
        sentiment=-0.5, heat=20.0,
    )
    found = feature(EvidenceBundle(events=(original, repost), relations=routes))
    assert found.coverage == "partial"
    assert found.event_count == 1
    assert found.mean_sentiment == original.sentiment
    assert found.mean_heat == original.heat
    assert len(found.hits[0].paths) == 2
    assert len(found.hits[0].source_urls) == 2
    assert any(None in path.relation_weights for path in found.hits[0].paths)


def test_old_event_does_not_reenter_window_after_repost():
    old = event(published_at=stamp(1))
    repost = event(
        source_url="https://example.com/repost",
        published_at=stamp(5), observed_at=stamp(5), ready_at=stamp(5),
    )
    bundle = EvidenceBundle(events=(old, repost))
    found = feature(bundle, target="entity:600000", as_of=stamp(5))
    assert found.event_count is None


def test_feature_ready_time_controls_visibility_and_channels_remain_distinct():
    late = event(ready_at=stamp(5))
    opinion = event(
        event_id="opinion-1", source_url="https://example.com/opinion",
        channel="opinion", sentiment=-0.2,
    )
    bundle = EvidenceBundle(events=(late, opinion))
    assert feature(bundle, target="entity:600000").event_count is None
    assert feature(bundle, target="entity:600000", channel="opinion").event_count == 1


def test_relation_paths_obey_three_total_hops():
    relations = (
        relation(
            relation_id="entity-industry", relation_type="entity_industry",
            target_id="industry:bank",
        ),
        relation(
            relation_id="industry-index", relation_type="industry_index",
            source_id="industry:bank",
        ),
        relation(
            relation_id="index-etf", relation_type="index_etf",
            source_id="index:000300", target_id="etf:510300",
        ),
    )
    bundle = EvidenceBundle(events=(event(),), relations=relations)
    assert feature(bundle, target="index:000300").event_count == 1
    assert feature(bundle).event_count is None


def test_partial_overlapping_coverage_intervals_are_merged():
    def covered(begin, end):
        return CoverageObservation(
            target_id="etf:510300", channel="news",
            from_at=stamp(begin), to_at=stamp(end), known_at=stamp(4),
            source_url="https://example.com/coverage",
        )
    bundle = EvidenceBundle(coverage=(covered(1, 3), covered(2, 3)))
    found = feature(bundle)
    assert found.coverage == "partial"
    assert found.covered_fraction == pytest.approx(2 / 3)
    assert found.event_count is None


def test_local_json_roundtrip_and_validation(tmp_path):
    bundle = EvidenceBundle(events=(event(),), relations=(relation(),))
    file = tmp_path / "evidence.json"
    file.write_text(bundle.model_dump_json(), encoding="utf-8")
    assert load_evidence(file) == bundle
    invalid = json.loads(bundle.model_dump_json())
    invalid["events"][0]["confidence"] = 2
    file.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_evidence(file)
    with pytest.raises(ValidationError):
        event(observed_at=stamp(1))
    with pytest.raises(ValidationError):
        event(ready_at=datetime(2026, 11, 4))
    with pytest.raises(ValidationError):
        relation(target_id="entity:510300")
    with pytest.raises(ValueError, match="timezone"):
        feature(bundle, as_of=datetime(2026, 11, 4))


def test_duplicate_versions_and_duplicate_source_observations_are_rejected():
    with pytest.raises(ValidationError, match="duplicate observation"):
        EvidenceBundle(events=(event(), event()))
    with pytest.raises(ValidationError, match="duplicate relation"):
        EvidenceBundle(relations=(relation(), relation()))
    with pytest.raises(ValueError, match="duplicate relation"):
        build_asof_graph((relation(), relation()), stamp(4))
