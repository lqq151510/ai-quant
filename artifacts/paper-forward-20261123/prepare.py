"""Collect and publish a pre-open candidate snapshot for the new paper pilot."""

import argparse
import json
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time
from pathlib import Path

from config import (
    COVERAGE, CST, DAYS, ROOT, UNIVERSE, candidate_path, candidate_sessions, load_candidates,
    previous_session, rank, sha256_file,
)
from monday_open import publish_once
from prepare_monday import inspect


def prepare(day, *, refresh=False):
    folder = ROOT / day.isoformat()
    destination = folder / 'candidates.json'
    if not refresh and (destination.is_file() or any(folder.glob('candidates-*.json'))):
        return load_candidates(day)
    start = datetime.combine(previous_session(day), time(15), CST)
    cutoff = datetime.combine(day, time(9, 45), CST)
    if not start <= datetime.now(CST) < cutoff:
        raise ValueError(f'{day}: candidate collection requires prior close and must finish before 09:45')
    if (folder / 'opening-inputs.json').exists():
        raise ValueError(f'{day}: opening inputs have already been pinned')
    sessions = candidate_sessions(day)
    folder.mkdir(exist_ok=True)
    raw = Path(tempfile.mkdtemp(prefix='bars-', dir=folder))
    with ThreadPoolExecutor(max_workers=4) as pool:
        reviewed = list(pool.map(
            lambda item: inspect(item, raw, sessions, prefer_recent_bars=True), UNIVERSE,
        ))
    captured_at = datetime.now(CST)
    snapshot = {
        'date': day.isoformat(), 'captured_at': captured_at.isoformat(),
        'coverage': COVERAGE, 'sessions': sessions, 'raw_directory': raw.name,
        'raw_sha256': {path.name: sha256_file(path) for path in sorted(raw.iterdir())},
        'reviewed': reviewed, 'candidates': rank(reviewed),
    }
    if captured_at >= cutoff or not any(row['status'] in ('eligible', 'signal_not_triggered') for row in reviewed):
        publish_once(folder / f'{raw.name}-unavailable.json', snapshot)
        raise ValueError(f'{day}: collection finished late or found no verifiable signal data')
    if refresh:
        destination = folder / f'candidates-{captured_at:%Y%m%dT%H%M%S%f}.json'
    publish_once(destination, snapshot, read_existing=lambda: load_candidates(day, destination))
    return load_candidates(day)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=date.fromisoformat, choices=DAYS, required=True)
    parser.add_argument('--refresh', action='store_true', help='record another independent pre-open data snapshot')
    args = parser.parse_args()
    try:
        snapshot = prepare(args.date, refresh=args.refresh)
        path = candidate_path(args.date)
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        parser.exit(2, f'Candidate preparation failed: {exc}\n')
    print(json.dumps({
        'date': snapshot['date'], 'captured_at': snapshot['captured_at'],
        'snapshot': path.name,
        'status_counts': dict(Counter(row['status'] for row in snapshot['reviewed'])),
        'ranked': [row['symbol'] for row in snapshot['candidates']],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
