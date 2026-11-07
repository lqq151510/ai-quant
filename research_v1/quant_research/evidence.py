"""Point-in-time news, opinion, and graph evidence for forward research."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Annotated, Literal

import networkx as nx
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

Channel = Literal["news", "opinion"]
Channels: tuple[Channel, ...] = ("news", "opinion")
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
NodeId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=256, pattern=r"^(entity|industry|index|etf):[^\s:]+$"),
]
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Sentiment = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]
Heat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
RelationType = Literal["entity_industry", "entity_index", "entity_etf", "industry_index", "industry_etf", "index_etf"]

# The event's mention of an entity accounts for the first hop.
MAX_RELATION_HOPS = 2


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EventObservation(EvidenceRecord):
    """One source observation. Reposts of one event share its canonical ID."""

    event_id: Identifier
    source_url: HttpUrl
    channel: Channel
    category: Identifier
    entity_ids: tuple[NodeId, ...] = Field(min_length=1)
    published_at: AwareDatetime
    observed_at: AwareDatetime
    ready_at: AwareDatetime
    confidence: Unit
    sentiment: Sentiment | None = None
    heat: Heat | None = None

    @model_validator(mode="after")
    def check_observation(self) -> EventObservation:
        if not self.published_at <= self.observed_at <= self.ready_at:
            raise ValueError("expected published_at <= observed_at <= ready_at")
        if any(not item.startswith("entity:") for item in self.entity_ids):
            raise ValueError("event mentions must use entity: identifiers")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("event contains duplicate entity identifiers")
        return self


class RelationVersion(EvidenceRecord):
    """A full revision of one fact interval, with an explicit knowledge time."""

    relation_id: Identifier
    relation_type: RelationType
    source_id: NodeId
    target_id: NodeId
    source_url: HttpUrl
    valid_from: AwareDatetime
    valid_to: AwareDatetime | None = None
    known_at: AwareDatetime
    confidence: Unit
    weight: Unit | None = None
    withdrawn: bool = False

    @model_validator(mode="after")
    def check_relation(self) -> RelationVersion:
        source_kind, target_kind = self.relation_type.split("_")
        if not self.source_id.startswith(source_kind + ":") or not self.target_id.startswith(target_kind + ":"):
            raise ValueError("relation endpoints do not match the relation type")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("relation validity interval must have positive duration")
        return self


class CoverageObservation(EvidenceRecord):
    """An explicit record of a completed observation interval for a target and channel."""

    target_id: NodeId
    channel: Channel
    from_at: AwareDatetime
    to_at: AwareDatetime
    known_at: AwareDatetime
    source_url: HttpUrl

    @model_validator(mode="after")
    def check_coverage(self) -> CoverageObservation:
        if not self.from_at < self.to_at <= self.known_at:
            raise ValueError("expected from_at < to_at <= known_at")
        return self


class EvidenceBundle(EvidenceRecord):
    events: tuple[EventObservation, ...] = ()
    relations: tuple[RelationVersion, ...] = ()
    coverage: tuple[CoverageObservation, ...] = ()

    @model_validator(mode="after")
    def check_identity(self) -> EvidenceBundle:
        event_sources = {(item.event_id, str(item.source_url), item.channel) for item in self.events}
        if len(event_sources) != len(self.events):
            raise ValueError("duplicate observation for the same event, source, and channel")
        versions = {(item.relation_id, item.known_at.astimezone(timezone.utc)) for item in self.relations}
        if len(versions) != len(self.relations):
            raise ValueError("duplicate relation version for the same logical ID and knowledge time")
        return self


class EvidencePath(EvidenceRecord):
    nodes: tuple[NodeId, ...]
    relation_ids: tuple[Identifier, ...]
    relation_sources: tuple[str, ...]
    relation_weights: tuple[Unit | None, ...]
    confidence: Unit


class EvidenceHit(EvidenceRecord):
    event_id: Identifier
    category: Identifier
    published_at: AwareDatetime
    observed_at: AwareDatetime
    ready_at: AwareDatetime
    source_urls: tuple[str, ...]
    paths: tuple[EvidencePath, ...]
    sentiment: Sentiment | None
    heat: Heat | None
    confidence: Unit


class EvidenceFeature(EvidenceRecord):
    target_id: NodeId
    channel: Channel
    as_of: AwareDatetime
    window_start: AwareDatetime
    coverage: Literal["complete", "partial", "unknown"]
    covered_fraction: Unit
    coverage_sources: tuple[str, ...]
    event_count: int | None
    mean_sentiment: Sentiment | None
    sentiment_disagreement: Unit | None
    mean_heat: Heat | None
    hits: tuple[EvidenceHit, ...]


def load_evidence(path: Path) -> EvidenceBundle:
    """Load a local JSON bundle with capture timestamps supplied in the evidence."""
    return EvidenceBundle.model_validate_json(Path(path).read_bytes())


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("as_of must include a timezone")
    return value.astimezone(timezone.utc)


def build_asof_graph(relations: Iterable[RelationVersion], as_of: datetime) -> nx.MultiDiGraph:
    """Apply knowledge time before effective time to prevent superseded facts resurfacing."""
    cutoff = _utc(as_of)
    latest: dict[str, RelationVersion] = {}
    seen: set[tuple[str, datetime]] = set()
    for relation in relations:
        known = _utc(relation.known_at)
        if known > cutoff:
            continue
        key = (relation.relation_id, known)
        if key in seen:
            raise ValueError("duplicate relation version for the same logical ID and knowledge time")
        seen.add(key)
        previous = latest.get(relation.relation_id)
        if previous is None or known > _utc(previous.known_at):
            latest[relation.relation_id] = relation

    graph = nx.MultiDiGraph(as_of=cutoff)
    for relation_id, item in sorted(latest.items()):
        if item.withdrawn or _utc(item.valid_from) > cutoff:
            continue
        if item.valid_to is not None and _utc(item.valid_to) <= cutoff:
            continue
        graph.add_edge(item.source_id, item.target_id, key=relation_id, relation=item)
    return graph


def _paths(graph: nx.MultiDiGraph, entity: str, targets: set[str]) -> dict[str, tuple[EvidencePath, ...]]:
    results: dict[str, list[EvidencePath]] = defaultdict(list)
    pending: list[tuple[tuple[str, ...], tuple[RelationVersion, ...]]] = [((entity,), ())]
    while pending:
        nodes, edges = pending.pop()
        current = nodes[-1]
        if current in targets:
            results[current].append(EvidencePath(
                nodes=nodes,
                relation_ids=tuple(item.relation_id for item in edges),
                relation_sources=tuple(str(item.source_url) for item in edges),
                relation_weights=tuple(item.weight for item in edges),
                confidence=min((item.confidence for item in edges), default=1.0),
            ))
        if len(edges) < MAX_RELATION_HOPS and current in graph:
            for _, neighbor, relation in graph.out_edges(current, data="relation"):
                if neighbor not in nodes:
                    pending.append(((*nodes, neighbor), (*edges, relation)))
    return {target: tuple(sorted(paths, key=lambda path: (path.nodes, path.relation_ids)))
            for target, paths in results.items()}


def _coverage(
    records: Iterable[CoverageObservation], target: str, channel: Channel, start: datetime, cutoff: datetime,
) -> tuple[float, tuple[str, ...]]:
    intervals = []
    urls = set()
    for record in records:
        if record.target_id != target or record.channel != channel or _utc(record.known_at) > cutoff:
            continue
        begin, end = max(start, _utc(record.from_at)), min(cutoff, _utc(record.to_at))
        if begin < end:
            intervals.append((begin, end))
            urls.add(str(record.source_url))
    covered = timedelta()
    end = start
    for left, right in sorted(intervals):
        if right > end:
            covered += right - max(left, end)
            end = right
    return covered / (cutoff - start), tuple(sorted(urls))


def _hit(
    event_id: str, published_at: datetime,
    observations: list[tuple[EventObservation, tuple[EvidencePath, ...]]],
) -> EvidenceHit:
    ordered = sorted(observations, key=lambda item: (_utc(item[0].ready_at), str(item[0].source_url)))
    first = ordered[0][0]
    paths = {(path.nodes, path.relation_ids): path for _, routes in ordered for path in routes}
    return EvidenceHit(
        event_id=event_id,
        category=first.category,
        published_at=published_at,
        observed_at=min(_utc(item.observed_at) for item, _ in ordered),
        ready_at=_utc(first.ready_at),
        source_urls=tuple(sorted({str(item.source_url) for item, _ in ordered})),
        paths=tuple(paths[key] for key in sorted(paths)),
        sentiment=next((item.sentiment for item, _ in ordered if item.sentiment is not None), None),
        heat=next((item.heat for item, _ in ordered if item.heat is not None), None),
        confidence=max(min(item.confidence, path.confidence) for item, routes in ordered for path in routes),
    )


def evidence_features(
    bundle: EvidenceBundle, *, as_of: datetime, lookback: timedelta, targets: Iterable[str],
) -> list[EvidenceFeature]:
    """Aggregate distinct events per target/channel in a publication-time window.

    A repost retains the event's earliest visible publication time. Scores use
    the first ready observation with a value, then average across distinct
    events. Relation weights are preserved as evidence, including unknowns.
    """
    cutoff = _utc(as_of)
    if lookback <= timedelta():
        raise ValueError("lookback must have positive duration")
    start = cutoff - lookback
    requested = set(targets)
    if not requested:
        return []
    graph = build_asof_graph(bundle.relations, cutoff)
    ready = [item for item in bundle.events if _utc(item.ready_at) <= cutoff]
    first_publication: dict[str, datetime] = {}
    for item in ready:
        stamp = _utc(item.published_at)
        first_publication[item.event_id] = min(stamp, first_publication.get(item.event_id, stamp))
    associations = defaultdict(lambda: defaultdict(list))
    for item in ready:
        if first_publication[item.event_id] < start:
            continue
        per_target: dict[str, list[EvidencePath]] = defaultdict(list)
        for entity in item.entity_ids:
            for target, paths in _paths(graph, entity, requested).items():
                per_target[target].extend(paths)
        for target, paths in per_target.items():
            associations[target, item.channel][item.event_id].append((item, tuple(paths)))

    features = []
    for target in sorted(requested):
        for channel in Channels:
            hits = tuple(
                _hit(event_id, first_publication[event_id], items)
                for event_id, items in sorted(associations[target, channel].items())
            )
            fraction, coverage_sources = _coverage(bundle.coverage, target, channel, start, cutoff)
            status = "complete" if fraction == 1 else "partial" if fraction > 0 or hits else "unknown"
            sentiment = [hit.sentiment for hit in hits if hit.sentiment is not None]
            heat = [hit.heat for hit in hits if hit.heat is not None]
            features.append(EvidenceFeature(
                target_id=target,
                channel=channel,
                as_of=cutoff,
                window_start=start,
                coverage=status,
                covered_fraction=fraction,
                coverage_sources=coverage_sources,
                event_count=len(hits) if hits or status == "complete" else None,
                mean_sentiment=fmean(sentiment) if sentiment else None,
                sentiment_disagreement=pstdev(sentiment) if len(sentiment) > 1 else None,
                mean_heat=fmean(heat) if heat else None,
                hits=hits,
            ))
    return features
