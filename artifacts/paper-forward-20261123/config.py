"""Fixed calendar and candidate evidence checks for the November paper pilot."""

import hashlib
import json
import sys
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'news-filter-forward-20261116'
REPO = ROOT.parents[1]
sys.path[:0] = [str(SOURCE), str(REPO / 'research_v1')]

from prepare_monday import ETF_SH, ETF_SZ, SH, SZ

CST = ZoneInfo('Asia/Shanghai')
DAYS = tuple(date(2026, 11, day) for day in (23, 24, 25, 26, 27))
CALENDAR = tuple(date(2026, 11, day) for day in (13, 16, 17, 18, 19, 20)) + DAYS
UNIVERSE = tuple(
    (f'{code}.{exchange}', kind)
    for codes, exchange, kind in (
        (SH, 'SH', 'stock'), (SZ, 'SZ', 'stock'), (ETF_SH, 'SH', 'etf'), (ETF_SZ, 'SZ', 'etf'),
    )
    for code in codes.split()
)
COVERAGE = 'bounded_sample_not_full_market'


def candidate_sessions(day):
    if day not in DAYS:
        raise ValueError(f'{day} outside the recorded pilot calendar')
    index = CALENDAR.index(day)
    return [item.isoformat() for item in CALENDAR[index - 6:index]]


def previous_session(day):
    return date.fromisoformat(candidate_sessions(day)[-1])


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(reviewed):
    return sorted(
        (row for row in reviewed if row['status'] == 'eligible'),
        key=lambda row: (-Decimal(row['five_day_return']), row['symbol']),
    )[:10]


def candidate_path(day):
    folder = ROOT / day.isoformat()
    first = folder / 'candidates.json'
    paths = ([first] if first.is_file() else []) + sorted(folder.glob('candidates-*.json'))
    if not paths:
        raise FileNotFoundError(first)

    def preference(path):
        snapshot = load_candidates(day, path)
        known = sum(row['status'] not in {'data_gap', 'turnover_threshold_uncertain'}
                    for row in snapshot['reviewed'])
        return -known, datetime.fromisoformat(snapshot['captured_at']), path.name

    return min(paths, key=preference)


def load_candidates(day, snapshot_path=None):
    folder = ROOT / day.isoformat()
    snapshot_path = snapshot_path or candidate_path(day)
    if snapshot_path.parent != folder or snapshot_path.resolve() != snapshot_path:
        raise ValueError(f'{day}: candidate snapshot is outside the pilot session directory')
    snapshot = json.loads(snapshot_path.read_bytes())
    captured = datetime.fromisoformat(snapshot['captured_at'])
    if (
        snapshot['date'] != day.isoformat() or snapshot['sessions'] != candidate_sessions(day)
        or snapshot['coverage'] != COVERAGE or captured.utcoffset() is None
        or not datetime.combine(previous_session(day), time(15), CST)
        <= captured < datetime.combine(day, time(9, 45), CST)
    ):
        raise ValueError(f'{day}: candidate snapshot date, coverage or capture window is invalid')
    reviewed = snapshot['reviewed']
    if tuple((row['symbol'], row['kind']) for row in reviewed) != UNIVERSE:
        raise ValueError(f'{day}: candidate universe differs from the pilot rules')
    if snapshot['candidates'] != rank(reviewed):
        raise ValueError(f'{day}: candidate ranking differs from the reviewed universe')
    raw = folder / snapshot['raw_directory']
    if raw.parent != folder or not raw.name.startswith('bars-') or raw.resolve() != raw:
        raise ValueError(f'{day}: candidate raw directory is not canonical')
    hashes = snapshot['raw_sha256']
    if not hashes or set(hashes) != {path.name for path in raw.iterdir()}:
        raise ValueError(f'{day}: candidate raw manifest is incomplete')
    for name, expected in hashes.items():
        path = raw / name
        if path.parent != raw or path.resolve() != path or sha256_file(path) != expected:
            raise ValueError(f'{day}: candidate raw evidence mismatch for {name}')
    return snapshot
