"""Dispatch the paper experiment at the frozen China-market opening window."""
import argparse
import re
import subprocess
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo


BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
CST = ZoneInfo('Asia/Shanghai')
MONDAY = date(2026, 11, 16)
FRIDAY = date(2026, 11, 20)
EARLY_TOLERANCE = timedelta(minutes=8)


def capture_directory(directory):
    """Reuse a single journal and use the canonical directory for new captures."""
    canonical = directory / 'books'
    retry_dirs = [path for path in directory.glob('books-retry-*')
                  if re.fullmatch(r'books-retry-[1-9][0-9]*', path.name)]
    saved = [path for path in (canonical, *retry_dirs)
             if any((path / name).is_file() for name in ('books.json', 'capture-progress.json'))]
    if len(saved) > 1:
        raise ValueError('multiple saved capture directories; first eligible quotes cannot be established')
    return saved[0] if saved else canonical


def has_saved_capture(directory, *, legacy_retry=False):
    """Recognize recovery inputs. The entry point validates their content."""
    location = capture_directory(directory) if legacy_retry else directory / 'books'
    return any((location / name).is_file() for name in ('books.json', 'capture-progress.json'))


def run_scheduled(day, *, clock, wait, launch, base=BASE):
    if not MONDAY <= day <= FRIDAY:
        return 0
    start = datetime.combine(day, time(9, 45), CST)
    end = datetime.combine(day, time(9, 50), CST)
    while True:
        now = clock()
        if now.utcoffset() is None:
            raise ValueError('scheduler clock must be timezone-aware')
        now = now.astimezone(CST)
        if now.date() != day:
            raise ValueError('scheduler date does not match the requested China trading date')
        if now < start - EARLY_TOLERANCE:
            raise ValueError('scheduler started more than eight minutes before the capture window')
        if now >= start:
            break
        wait(min(30, (start - now).total_seconds()))
    if now > end and not has_saved_capture(base / str(day), legacy_retry=day == MONDAY):
        raise ValueError('opening window missed and no saved capture is available')
    command = [sys.executable]
    if day == MONDAY:
        command.append(str(base / 'monday_open.py'))
    else:
        command.extend([str(base / 'continue_open.py'), '--date', str(day)])
    return launch(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=date.fromisoformat,
                        help='China trading date, defaults to the current date in Asia/Shanghai')
    args = parser.parse_args()
    clock = lambda: datetime.now(CST)
    try:
        return run_scheduled(args.date or clock().date(), clock=clock, wait=sleep,
                             launch=lambda command: subprocess.run(command, cwd=PROJECT).returncode)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
