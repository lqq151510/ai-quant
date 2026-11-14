"""Verify the scheduler wrapper independently of the operating-system clock."""
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scheduled_open import CST, MONDAY, capture_directory, run_scheduled


class Runtime:
    def __init__(self, now, result=0):
        self.now = now
        self.result = result
        self.waits = []
        self.commands = []

    def clock(self):
        return self.now

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += timedelta(seconds=seconds)

    def launch(self, command):
        self.commands.append(command)
        return self.result

    def run(self, day, base):
        return run_scheduled(day, clock=self.clock, wait=self.wait, launch=self.launch, base=base)


class ScheduledOpenTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)

    def test_earliest_jitter_waits_in_short_chunks_to_open(self):
        runtime = Runtime(datetime(2026, 11, 16, 1, 37, tzinfo=timezone.utc), result=7)
        self.assertEqual(runtime.run(MONDAY, self.base), 7)
        self.assertEqual(sum(runtime.waits), 480)
        self.assertLessEqual(max(runtime.waits), 30)
        self.assertEqual(runtime.now.astimezone(CST).strftime('%H:%M:%S'), '09:45:00')
        self.assertEqual(runtime.commands, [[sys.executable, str(self.base / 'monday_open.py')]])

    def test_weekday_continuation_starts_immediately_in_window(self):
        tuesday = MONDAY + timedelta(days=1)
        runtime = Runtime(datetime(2026, 11, 17, 9, 48, tzinfo=CST))
        self.assertEqual(runtime.run(tuesday, self.base), 0)
        self.assertEqual(runtime.waits, [])
        self.assertEqual(runtime.commands, [
            [sys.executable, str(self.base / 'continue_open.py'), '--date', '2026-11-17']
        ])

    def test_recovery_after_window_requires_saved_capture(self):
        runtime = Runtime(datetime(2026, 11, 16, 18, 0, tzinfo=CST))
        with self.assertRaisesRegex(ValueError, 'opening window missed'):
            runtime.run(MONDAY, self.base)
        self.assertEqual(runtime.commands, [])
        retry = self.base / str(MONDAY) / 'books-retry-3'
        retry.mkdir(parents=True)
        (retry / 'capture-progress.json').touch()
        self.assertEqual(runtime.run(MONDAY, self.base), 0)
        self.assertEqual(runtime.commands, [[sys.executable, str(self.base / 'monday_open.py')]])

    def test_fresh_capture_reuses_canonical_directory_before_first_checkpoint(self):
        folder = self.base / str(MONDAY)
        canonical = folder / 'books'
        canonical.mkdir(parents=True)
        (canonical / '.capture.lock').touch()
        (folder / 'books-retry-1').mkdir()
        self.assertEqual(capture_directory(folder), canonical)

    def test_legacy_journal_recovered_without_canonical_or_consecutive_folders(self):
        folder = self.base / str(MONDAY)
        retry = folder / 'books-retry-3'
        retry.mkdir(parents=True)
        (retry / 'books.json').touch()
        self.assertEqual(capture_directory(folder), retry)
        canonical = folder / 'books'
        canonical.mkdir()
        (canonical / 'capture-progress.json').touch()
        with self.assertRaisesRegex(ValueError, 'multiple saved capture directories'):
            capture_directory(folder)

    def test_invalid_clock_or_day_never_launches(self):
        cases = (
            (datetime(2026, 11, 16, 9, 36, 59, tzinfo=CST), 'eight minutes'),
            (datetime(2026, 11, 17, 9, 45, tzinfo=CST), 'scheduler date does not match'),
            (datetime(2026, 11, 16, 9, 45), 'timezone-aware'),
        )
        for timestamp, error in cases:
            with self.subTest(timestamp=timestamp):
                runtime = Runtime(timestamp)
                with self.assertRaisesRegex(ValueError, error):
                    runtime.run(MONDAY, self.base)
                self.assertEqual(runtime.commands, [])

    def test_outside_experiment_is_quiet(self):
        saturday = MONDAY + timedelta(days=5)
        runtime = Runtime(datetime(2026, 11, 21, 9, 43, tzinfo=CST))
        self.assertEqual(runtime.run(saturday, self.base), 0)
        self.assertEqual(runtime.waits, [])
        self.assertEqual(runtime.commands, [])


if __name__ == '__main__':
    unittest.main()
