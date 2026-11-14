"""Exercise dated news evidence without touching the real archive or ledgers."""

import json
import unittest
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import monday_open
from close_monday import close_report
from continue_open import run_day
from monday_scenario_matrix import synthetic_books
from test_weekly_report import SYMBOLS, with_sale
from weekly_report import weekly_report


FRIDAY, SATURDAY, SUNDAY = '2026-11-13', '2026-11-14', '2026-11-15'
MONDAY = '2026-11-16'


class NewsSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.base = root / 'study'
        self.base.mkdir()
        self.archive = root / 'news-forward'
        self.archive.mkdir()
        self.output = self.base / 'opening-news.json'
        base_patch = patch('monday_open.BASE', self.base)
        base_patch.start()
        self.addCleanup(base_patch.stop)

    def write_batch(self, filename_date, *, collection_status=None, **values):
        batch = {'date': filename_date, 'events': [], **values}
        if collection_status is not None:
            batch['collection_status'] = collection_status
        (self.archive / f'{filename_date}.json').write_text(json.dumps(batch), encoding='utf-8')

    def snapshot(self, day=MONDAY, at='09:45:02'):
        with patch('monday_open.datetime', wraps=datetime) as clock:
            clock.now.return_value = datetime.fromisoformat(f'{day}T{at}+08:00')
            return monday_open.news_snapshot(self.output, date.fromisoformat(day))

    def test_missing_weekend_is_a_coverage_gap_and_empty_batches_fill_it(self):
        self.write_batch(FRIDAY, events=[{'id': 'sample'}])
        friday_only = self.snapshot()
        info = friday_only['news_archive']
        self.assertEqual(info['expected_dates'], [FRIDAY, SATURDAY, SUNDAY])
        self.assertEqual(info['loaded_dates'], [FRIDAY])
        self.assertEqual(info['missing_dates'], [SATURDAY, SUNDAY])
        self.assertEqual(info['captured_at'], f'{MONDAY}T09:45:02+08:00')
        self.assertFalse(friday_only['errors'])

        for day in (SATURDAY, SUNDAY):
            self.write_batch(day, collection_status='complete')
        self.assertEqual(self.snapshot(), friday_only)
        self.output.unlink()
        complete = self.snapshot()['news_archive']
        self.assertEqual(complete['loaded_dates'], [FRIDAY, SATURDAY, SUNDAY])
        self.assertEqual(complete['empty_dates'], [SATURDAY, SUNDAY])
        self.assertFalse(complete['missing_dates'])

    def test_concurrent_retry_uses_first_saved_snapshot(self):
        existing = {'batches': [], 'errors': ['first writer'], 'sources': []}
        self.write_batch(FRIDAY)
        link = monday_open.os.link

        def commit_first(source, destination):
            destination.write_text(json.dumps(existing), encoding='utf-8')
            link(source, destination)

        with patch('monday_open.os.link', side_effect=commit_first):
            self.assertEqual(self.snapshot(), existing)
        self.assertEqual(json.loads(self.output.read_text()), existing)

    def test_wrong_date_and_invalid_events_are_reported_as_invalid(self):
        self.write_batch(FRIDAY)
        self.write_batch(SATURDAY, date=SUNDAY)
        self.write_batch(SUNDAY, events={})
        snapshot = self.snapshot()
        self.assertEqual(snapshot['news_archive']['invalid_dates'], [SATURDAY, SUNDAY])
        self.assertFalse(snapshot['news_archive']['missing_dates'])
        self.assertEqual(len(snapshot['errors']), 2)
        self.assertEqual([batch['date'] for batch in snapshot['batches']], [FRIDAY])

    def test_older_archives_are_loaded_for_dedup_but_corruption_is_coverage_only(self):
        for day in (FRIDAY, SATURDAY, SUNDAY, MONDAY):
            self.write_batch(day)
        snapshot = self.snapshot('2026-11-17')
        self.assertEqual(snapshot['news_archive']['expected_dates'], [SATURDAY, SUNDAY, MONDAY])
        self.assertEqual([batch['date'] for batch in snapshot['batches']], [FRIDAY, SATURDAY, SUNDAY, MONDAY])
        self.output.unlink()
        self.write_batch(FRIDAY, date=SATURDAY)
        snapshot = self.snapshot('2026-11-17')
        self.assertEqual(snapshot['news_archive']['invalid_dates'], [FRIDAY])
        self.assertFalse(snapshot['errors'])

    def test_partial_keeps_observed_events_and_failed_empty_is_not_usable(self):
        self.write_batch(FRIDAY, collection_status='partial', events=[{'id': 'partial'}])
        self.write_batch(SATURDAY, collection_status='failed')
        self.write_batch(SUNDAY, collection_status='complete')
        snapshot = self.snapshot()
        info = snapshot['news_archive']
        self.assertEqual(info['incomplete_dates'], [FRIDAY, SATURDAY])
        self.assertEqual(info['empty_dates'], [SUNDAY])
        self.assertEqual([batch['date'] for batch in snapshot['batches']], [FRIDAY, SUNDAY])
        self.assertFalse(snapshot['errors'])

    def test_all_missing_or_failed_archives_keep_unavailable_semantics(self):
        for status in (None, 'failed'):
            with self.subTest(collection_status=status):
                if status:
                    self.output.unlink()
                    self.write_batch(FRIDAY, collection_status=status)
                snapshot = self.snapshot()
                self.assertEqual(snapshot['batches'], [])
                self.assertEqual(snapshot['errors'], ['no news batches available'])

    def test_late_recovery_does_not_load_new_archives(self):
        self.write_batch(FRIDAY)
        snapshot = self.snapshot(at='09:50:01')
        self.assertFalse(snapshot['batches'])
        self.assertTrue(snapshot['errors'])
        info = snapshot['news_archive']
        self.assertIsNone(info['captured_at'])
        self.assertFalse(info['missing_dates'])
        self.assertEqual(info['checked_at'], f'{MONDAY}T09:50:01+08:00')

    def test_bad_saved_snapshot_remains_untouched_even_if_archive_is_usable(self):
        self.write_batch(FRIDAY)
        valid = self.snapshot()
        cases = (
            (b'{', 'Expecting property name'),
            (b'{}', 'expected batches, errors and sources lists'),
            (json.dumps({**valid, 'batches': [], 'sources': []}).encode(),
             'snapshot has no news batches or recorded errors'),
            (json.dumps({k: valid[k] for k in ('batches', 'errors', 'sources')}).encode(),
             'snapshot capture timestamp is missing'),
            (json.dumps({**valid, 'news_archive': {'captured_at': f'{SUNDAY}T09:45:00+08:00'}}).encode(),
             'snapshot capture timestamp is outside the opening session'),
            (json.dumps({**valid, 'news_archive': {'captured_at': f'{MONDAY}T09:50:01+08:00'}}).encode(),
             'snapshot capture timestamp is outside the opening session'),
        )
        for contents, error in cases:
            with self.subTest(error=error):
                self.output.write_bytes(contents)
                snapshot = self.snapshot()
                self.assertEqual(snapshot['batches'], [])
                self.assertIn(error, snapshot['errors'][0])
                self.assertEqual(self.output.read_bytes(), contents)

    def test_concurrent_retry_preserves_a_damaged_first_write(self):
        self.write_batch(FRIDAY)
        link = monday_open.os.link

        def commit_first(source, destination):
            destination.write_bytes(b'{')
            link(source, destination)

        with patch('monday_open.os.link', side_effect=commit_first):
            snapshot = self.snapshot()
        self.assertEqual(snapshot['batches'], [])
        self.assertIn('Expecting property name', snapshot['errors'][0])
        self.assertEqual(self.output.read_bytes(), b'{')

    def test_archive_reads_crossing_cutoff_do_not_supply_news(self):
        self.write_batch(FRIDAY)
        times = [datetime.fromisoformat(f'{MONDAY}T09:49:59+08:00'),
                 datetime.fromisoformat(f'{MONDAY}T09:50:01+08:00')]
        with patch('monday_open.datetime', wraps=datetime) as clock:
            clock.now.side_effect = times
            snapshot = monday_open.news_snapshot(self.output)
        self.assertIsNone(snapshot['news_archive']['captured_at'])
        self.assertEqual(snapshot['news_archive']['checked_at'], times[1].isoformat())
        self.assertEqual(snapshot['batches'], [])
        self.assertEqual(snapshot['errors'], ['archive reads did not finish during the opening window'])
        self.assertEqual(self.snapshot(), snapshot)

    def test_news_snapshot_time_is_checked_per_quote_in_both_opening_paths(self):
        rows = [{'symbol': symbol, 'previous_close': '10.00', 'tick': '.01', 'kind': 'stock',
                 'cross_border': False} for symbol in SYMBOLS[:3]]
        prepared = {'coverage': 'synthetic', 'candidates': rows}
        symbols = [row['symbol'] for row in rows]
        self.write_batch(FRIDAY, events=[
            {'symbols': symbols, 'event_id': 'example', 'source_url': 'https://example.test/a',
             'title': '股票交易异常波动公告', 'first_observed_at': f'{FRIDAY}T18:00:00+08:00'}
        ])
        self.write_batch(SATURDAY, events=[
            {'symbols': symbols, 'event_id': 'example', 'source_url': 'https://example.test/b',
             'title': '股票交易异常波动公告', 'first_observed_at': f'{SATURDAY}T18:00:00+08:00'}
        ])
        due = '09:46:00'
        for day in (MONDAY, '2026-11-17'):
            with self.subTest(day=day):
                if self.output.exists():
                    self.output.unlink()
                snapshot = self.snapshot(day=day, at=due)
                captured = synthetic_books(rows, 'eligible')
                for symbol, second in zip(symbols, ('09:45:59', '09:46:00', '09:46:01')):
                    book = captured['selected'][symbol]
                    book.update(at=f'{day}T{second}+08:00', received_at=f'{day}T{second}+08:00')
                if day == MONDAY:
                    opening = monday_open.run(prepared, captured, snapshot['batches'],
                                               news_errors=snapshot['errors'],
                                               news_captured_at=snapshot['news_archive']['captured_at'])
                else:
                    empty = {'cash': '100000', 'positions': {}, 'nav': '100000'}
                    opening = run_day(date.fromisoformat(day), dict.fromkeys(('baseline', 'filtered'), empty),
                                      prepared, captured, snapshot)
                self.assertEqual([fill['symbol'] for fill in opening['baseline']['fills']], symbols)
                self.assertEqual([row['reason'] for row in opening['filtered']['decisions']],
                                 ['news_data_unavailable', 'news_abnormal_volatility',
                                  'news_abnormal_volatility'])
                error = opening['filtered']['decisions'][0]['evaluation']['news_error']
                self.assertEqual(error, 'saved news snapshot was captured after the quote was received')

    def test_partial_archive_propagates_to_summary_without_blocking_fills(self):
        self.write_batch(FRIDAY)
        snapshot = self.snapshot()
        row = {'symbol': SYMBOLS[0], 'previous_close': '10.00', 'tick': '.01', 'kind': 'stock',
               'cross_border': False}
        prepared = {'coverage': 'synthetic', 'candidates': [row]}
        empty = {'cash': '100000', 'positions': {}, 'nav': '100000'}
        day = date(2026, 11, 17)
        captured = synthetic_books(prepared['candidates'], 'eligible')
        for book in captured['selected'].values():
            for key in ('at', 'received_at'):
                book[key] = book[key].replace(MONDAY, str(day))
        opening = run_day(day, dict.fromkeys(('baseline', 'filtered'), empty), prepared, captured, snapshot)
        self.assertEqual([len(opening[group]['fills']) for group in ('baseline', 'filtered')], [1, 1])
        self.assertEqual(opening['news_archive'], snapshot['news_archive'])
        prior = {'mode': 'forward', 'date': MONDAY,
                 **dict.fromkeys(('baseline', 'filtered'), {**empty, 'fills': [], 'close_prices': {}})}
        daily, summary = close_report(opening, day=str(day), prior=[prior])
        self.assertEqual(daily['news_archive'], snapshot['news_archive'])
        self.assertEqual(summary['news_evidence']['archive'], snapshot['news_archive'])
        self.assertTrue(summary['news_evidence']['degraded'])

    def test_archive_gap_blocks_verdict_even_with_five_verified_losses(self):
        records = with_sale(dict.fromkeys(SYMBOLS, '9.90'))
        self.assertEqual(weekly_report(records)['verdict'], 'both_improved')
        self.write_batch(FRIDAY)
        records[0]['news_archive'] = self.snapshot()['news_archive']
        report = weekly_report(records)
        self.assertEqual(report['news_archive_gaps'], [{'date': MONDAY, 'missing_dates': [SATURDAY, SUNDAY]}])
        self.assertEqual(report['insufficient_evidence_reasons'], ['incomplete_news_archive_coverage'])
        self.assertEqual(report['verdict'], 'insufficient_evidence')
        self.assertEqual(report['distinct_verified_filtered_symbols'], list(SYMBOLS))
        self.assertIsNotNone(report['comparison'])
