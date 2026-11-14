"""Exercise a hypothetical full week and a stopped quote capture in isolation."""
import hashlib
import json
import shutil
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import collect_books
import continue_open
import monday_open
from book_evidence import verify_capture
from test_book_evidence import save_attempt, select, tencent_row
from close_monday import close_report, save_report
from continue_open import run_day
from monday_open import run
from monday_scenario_matrix import frozen_inputs, synthetic_books
from weekly_report import DATES, GROUPS, weekly_report


def sina_book():
    fields = ['0'] * 32
    fields[0] = 'fixture'
    fields[2:4] = ['7.18', '7.19']
    fields[8] = '1000'
    fields[10:12] = ['1000', '7.18']
    fields[20:22] = ['1000', '7.19']
    fields[30:32] = [DATES[0], '09:45:09']
    return ('var hq_str_sh601988="' + ','.join(fields) + '";').encode()


def synthetic_close(opening, references, prior=()):
    day = opening['date']
    held = {symbol for group in GROUPS for symbol in opening[group]['positions']}
    marks = {
        symbol: {
            'date': day, 'close': references[symbol], 'price_type': 'unadjusted_close',
            'source_url': 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get',
        }
        for symbol in held if symbol in references
    }
    return close_report(opening, {'date': day, 'marks': marks}, day=day, prior=prior)


class ForwardStudyIntegration(unittest.TestCase):
    def archived_inputs(self):
        try:
            return frozen_inputs()
        except FileNotFoundError as exc:
            self.skipTest(f'local archive missing: {exc.filename}')

    def test_monday_fills_next_day_exits_and_five_day_report(self):
        prepared, news = self.archived_inputs()
        rows = prepared['candidates']
        references = {row['symbol']: row['previous_close'] for row in rows}
        snapshot = {'batches': news, 'errors': [], 'sources': []}
        opening = {'mode': 'forward', **run(prepared, synthetic_books(rows, 'eligible'), news)}
        monday, summary = synthetic_close(opening, references)
        self.assertEqual(
            [(summary['groups'][g]['buy_count'], summary['groups'][g]['fees']) for g in GROUPS],
            [(10, '50.25'), (9, '45.20')],
        )
        records = [monday]
        selected = {**prepared, 'candidates': [rows[0]]}
        for day in DATES[1:]:
            books = {
                symbol: {
                    **book,
                    'at': day + 'T09:45:20+08:00',
                    'received_at': day + 'T09:45:21+08:00',
                    'bid': book['previous'],
                    'bid_units': 100000,
                    'lower_limit': '0.001',
                    'upper_limit': '9999',
                    'source_url': f'https://qt.gtimg.cn/q={symbol[-2:].lower()}{symbol[:6]}',
                }
                for symbol, book in synthetic_books(rows, 'eligible')['selected'].items()
            }
            opening = run_day(date.fromisoformat(day), records[-1], selected,
                              {'selected': books, 'exit_selected': books, 'unavailable': {}}, snapshot)
            current, summary = synthetic_close(opening, references, prior=records)
            if day == DATES[1]:
                self.assertEqual([summary['groups'][g]['sell_count'] for g in GROUPS], [10, 9])
            records.append(current)
        report = weekly_report(records)
        self.assertEqual(report['reconciled_dates'], list(DATES))
        self.assertEqual(report['execution_evidence_gaps'], [])
        self.assertEqual(report['insufficient_evidence_reasons'],
                         ['fewer_than_five_distinct_verified_filtered_symbols'])
        self.assertEqual(report['verdict'], 'insufficient_evidence')
        self.assertEqual([report['groups'][g]['ending_nav'] for g in GROUPS],
                         ['99734.320', '99756.710'])
        [opportunity] = report['effective_opportunities']
        self.assertEqual((opportunity['symbol'], opportunity['net_pnl'], opportunity['exit_date']),
                         ('603135.SH', '-22.39', DATES[1]))

    def test_tuesday_recovers_archived_quotes_with_the_original_inputs_after_close(self):
        monday, tuesday = DATES[:2]
        stock, etf = '601988.SH', '513100.SH'
        rows = [
            {'symbol': stock, 'kind': 'stock', 'tick': '.01', 'previous_close': '7.18', 'cross_border': False},
            {'symbol': etf, 'kind': 'etf', 'tick': '.001', 'previous_close': '2.00', 'cross_border': True},
        ]
        prepared = {'candidates': rows, 'coverage': 'synthetic'}
        open_monday = {'mode': 'forward', **run(prepared, synthetic_books(rows, 'eligible'), [])}
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            shutil.copyfile(monday_open.BASE / 'protocol.md', base / 'protocol.md')
            for references in ({}, {stock: '2.00'}):
                daily, summary = synthetic_close(open_monday, references)
                partial = Path(save_report(base / monday, daily, summary, True)['artifacts']['daily'])
            self.assertIsNone(daily['baseline']['nav'])
            session = base / tuesday
            books = session / 'books'
            pin = session / 'opening-inputs.json'
            candidate = base / f'{tuesday}-preparation' / 'candidates.json'
            candidate.parent.mkdir()
            candidate.write_text(json.dumps({'date': tuesday, 'captured_at': tuesday + 'T09:30:00+08:00',
                                             'coverage': 'synthetic',
                                             'candidates': [{**rows[0], 'previous_close': '2.00'}]}))
            original_candidate = candidate.read_bytes()

            class Clock(datetime):
                current = tuesday + 'T09:45:08+08:00'

                @classmethod
                def now(cls, tz):
                    return cls.fromisoformat(cls.current).astimezone(tz)

            def interrupt_after_capture(*args, **kwargs):
                self.assertTrue(pin.is_file())
                books.mkdir()
                entry = save_attempt(books, 'tencent.txt', tencent_row(stock, 10, latest='2.000', ask='2.000',
                                                                     ask_lots='100').replace('20261116', '20261117'),
                                     '09:45:11', ['sh601988'])
                entry['retrieved_at'] = tuesday + 'T09:45:11+08:00'
                selected = {stock: select(entry, stock)}
                capture = {'date': tuesday, 'mode': 'forward', 'attempts': [entry], 'selected': selected,
                           'exit_selected': selected, 'unavailable': {}}
                (books / 'books.json').write_text(json.dumps(capture))
                raise InterruptedError('after saving the capture')

            with (
                patch.object(continue_open, 'BASE', base),
                patch.object(sys, 'argv', ['continue_open.py', '--date', tuesday]),
                patch.object(continue_open, 'datetime', Clock),
                patch.object(continue_open, 'news_snapshot', return_value={'batches': [], 'errors': [], 'sources': []}),
                patch.object(continue_open.subprocess, 'run', side_effect=interrupt_after_capture) as collector,
            ):
                with self.assertRaisesRegex(InterruptedError, 'after saving'):
                    continue_open.main()
                published = session / 'opening.json'
                self.assertFalse(published.exists())
                pinned_bytes = pin.read_bytes()
                pinned = json.loads(pinned_bytes)
                self.assertEqual(pinned['prepared']['sha256'], hashlib.sha256(original_candidate).hexdigest())

                # A subsequent valuation can be reconciled, but would newly allow Tuesday buys.
                complete, close = synthetic_close(open_monday, {stock: '2.00', etf: '2.00'})
                save_report(base / monday, complete, close, True)
                Clock.current = tuesday + 'T18:00:00+08:00'
                pin.unlink()
                with self.assertRaisesRegex(ValueError, 'not pinned'):
                    continue_open.main()
                pin.write_bytes(pinned_bytes)

                candidate.write_bytes(original_candidate + b' ')
                with self.assertRaisesRegex(ValueError, 'pinned candidate hash'):
                    continue_open.main()
                candidate.write_bytes(original_candidate)

                original_partial = partial.read_bytes()
                partial.write_bytes(original_partial + b' ')
                with self.assertRaisesRegex(ValueError, 'pinned prior ledger hash'):
                    continue_open.main()
                partial.write_bytes(original_partial)

                pin.write_text(json.dumps({**pinned, 'pinned_at': tuesday + 'T09:45:12+08:00'}))
                with self.assertRaisesRegex(ValueError, 'received before the inputs were pinned'):
                    continue_open.main()
                pin.write_bytes(pinned_bytes)
                continue_open.main()
                original = published.read_bytes()
                opened = json.loads(original)
                self.assertEqual(opened['starting_from']['path'], str(partial.relative_to(base)))
                self.assertEqual(opened['starting_from']['sha256'], hashlib.sha256(partial.read_bytes()).hexdigest())
                self.assertEqual(opened['baseline']['positions'], {etf: daily['baseline']['positions'][etf]})
                self.assertEqual([f['symbol'] for f in opened['baseline']['fills']], [stock])
                self.assertIn('previous_nav_unverified', {d['reason'] for d in opened['baseline']['decisions']})
                continue_open.main()
                self.assertEqual(published.read_bytes(), original)
                self.assertEqual(pin.read_bytes(), pinned_bytes)
                partial.write_bytes(partial.read_bytes() + b' ')
                with self.assertRaisesRegex(ValueError, 'prior ledger.*hash'):
                    continue_open.main()
                collector.assert_called_once()

    def test_partial_capture_finalizes_without_network_after_cutoff(self):
        class Clock(datetime):
            current = DATES[0] + 'T09:45:10+08:00'

            @classmethod
            def now(cls, tz):
                return cls.fromisoformat(cls.current).astimezone(tz)

        def interrupt_after_sina(url, timeout):
            if url.startswith(collect_books.SOURCES['tencent']):
                raise KeyboardInterrupt('before 513100 receives a quote')
            return Clock.current, sina_book()

        with TemporaryDirectory() as directory:
            output = Path(directory) / 'capture'
            argv = ['collect_books.py', '--date', DATES[0], '--out', str(output),
                    '--require-513100-iopv', '601988.SH', '513100.SH']
            with patch.object(collect_books, 'datetime', Clock), patch.object(sys, 'argv', argv):
                with patch.object(collect_books, 'capture', interrupt_after_sina):
                    with self.assertRaises(KeyboardInterrupt):
                        collect_books.main()
                checkpoint = json.loads((output / 'capture-progress.json').read_text())
                self.assertEqual(list(checkpoint['selected']), ['601988.SH'])
                Clock.current = DATES[0] + 'T18:00:00+08:00'
                with patch.object(collect_books, 'capture') as request:
                    collect_books.main()
                    request.assert_not_called()
            final = json.loads((output / 'books.json').read_text())
            self.assertEqual(final['selected'], checkpoint['selected'])
            self.assertEqual(final['unavailable'], {'513100.SH': 'quote_unavailable'})
            raw_file = final['selected']['601988.SH']['raw_file']
            self.assertEqual((output / raw_file).read_bytes(), sina_book())


    def test_evening_replay_requires_a_recorded_capture_attempt(self):
        class Clock(datetime):
            current = DATES[0] + 'T09:45:10+08:00'

            @classmethod
            def now(cls, tz):
                return cls.fromisoformat(cls.current).astimezone(tz)

        def launch_collector(command, *, check):
            with patch.object(sys, 'argv', command[1:]):
                collect_books.main()

        prepared, _ = self.archived_inputs()
        symbols = [row['symbol'] for row in prepared['candidates']]
        for attempted in (False, True):
            with self.subTest(attempted=attempted), TemporaryDirectory() as directory:
                base = Path(directory)
                (base / 'monday-preparation').mkdir()
                for name in ('protocol.md', 'monday-preparation/candidates.json'):
                    shutil.copyfile(monday_open.BASE / name, base / name)
                session = base / DATES[0]
                books = session / 'books'
                args = ['collect_books.py', '--date', DATES[0], '--out', str(books),
                        '--require-513100-iopv', *symbols]
                Clock.current = DATES[0] + 'T09:45:10+08:00'
                failure = KeyboardInterrupt('before the first response')
                capture_results = [OSError('vendor unavailable'), failure] if attempted else [failure]
                with patch.object(collect_books, 'datetime', Clock), patch.object(monday_open, 'datetime', Clock), \
                     patch.object(monday_open, 'BASE', base):
                    with patch.object(sys, 'argv', args), \
                         patch.object(collect_books, 'capture', side_effect=capture_results):
                        with self.assertRaises(KeyboardInterrupt):
                            collect_books.main()
                    Clock.current = DATES[0] + 'T18:00:00+08:00'
                    opening = session / 'opening.json'
                    with patch.object(sys, 'argv', ['monday_open.py']), \
                         patch.object(monday_open.subprocess, 'run', side_effect=launch_collector), \
                         patch.object(collect_books, 'capture') as request:
                        if attempted:
                            monday_open.main()
                        else:
                            with self.assertRaisesRegex(ValueError, 'no verifiable quote attempts'):
                                monday_open.main()
                        request.assert_not_called()
                    self.assertEqual(opening.is_file(), attempted)
                    _, summary = close_report(json.loads(opening.read_text()) if attempted else None)
                    self.assertEqual(summary['status'], 'reconciled' if attempted else 'opening_unverified')
                    for group in GROUPS:
                        metrics = summary['groups'][group]
                        values = tuple(metrics[k] for k in ('buy_count', 'sell_count', 'fees', 'close_nav'))
                        self.assertEqual(values, (0, 0, '0', '100000') if attempted else (None,) * 4)

    def test_first_response_survives_interruption_without_duplicate_attempts(self):
        original_receipt = DATES[0] + 'T09:45:10+08:00'

        class Clock(datetime):
            current = original_receipt

            @classmethod
            def now(cls, tz):
                return cls.fromisoformat(cls.current).astimezone(tz)

        payload = sina_book()
        write_json = collect_books.write_json

        for stage in ('parse', 'commit', 'final'):
            with self.subTest(stage=stage), TemporaryDirectory() as directory:
                output = Path(directory) / 'capture'
                argv = ['collect_books.py', '--date', DATES[0], '--out', str(output), '601988.SH']
                checkpoint_path = output / 'capture-progress.json'
                Clock.current = original_receipt

                def interrupted_write(path, value):
                    if stage == 'commit' and value.get('attempts') and path == checkpoint_path:
                        raise KeyboardInterrupt('after raw write, before checkpoint commit')
                    if stage == 'final' and path.name == 'books.json':
                        raise KeyboardInterrupt('after checkpoint commit, before final publication')
                    write_json(path, value)

                def capture_response(url, timeout):
                    initial = json.loads(checkpoint_path.read_text())
                    self.assertEqual(initial['attempts'], [])
                    self.assertNotIn('pending_response', initial)
                    return original_receipt, payload

                with patch.object(collect_books, 'datetime', Clock), patch.object(sys, 'argv', argv):
                    parse = collect_books.parse_books
                    if stage == 'parse':
                        parse = unittest.mock.Mock(side_effect=KeyboardInterrupt('before response parsing'))
                    with patch.object(collect_books, 'capture', capture_response), \
                            patch.object(collect_books, 'write_json', interrupted_write), \
                            patch.object(collect_books, 'parse_books', parse):
                        with self.assertRaises(KeyboardInterrupt):
                            collect_books.main()
                    saved = json.loads(checkpoint_path.read_text())
                    self.assertEqual('pending_response' in saved, stage != 'final')
                    self.assertEqual(len(saved['attempts']), 1 if stage == 'final' else 0)
                    Clock.current = DATES[0] + 'T18:00:00+08:00'
                    with patch.object(collect_books, 'capture') as request:
                        collect_books.main()
                        request.assert_not_called()

                final = json.loads((output / 'books.json').read_text())
                attempt, = final['attempts']
                self.assertEqual(attempt['raw_sha256'], hashlib.sha256(payload).hexdigest())
                self.assertEqual(attempt['retrieved_at'], original_receipt)
                self.assertEqual(list(output.glob('*.txt')), [output / attempt['raw_file']])
                self.assertEqual((output / attempt['raw_file']).read_bytes(), payload)
                self.assertNotIn('pending_response', json.loads(checkpoint_path.read_text()))
                audited = verify_capture(final, output, date.fromisoformat(DATES[0]), ['601988.SH'])
                self.assertEqual(audited['selected']['601988.SH']['received_at'], original_receipt)


    def test_pending_second_etf_response_keeps_the_original_iopv_pair(self):
        symbol = collect_books.IOPV_SYMBOL
        receipts = [DATES[0] + f'T09:45:{second}+08:00' for second in (11, 21)]

        class Clock(datetime):
            current = receipts[0]

            @classmethod
            def now(cls, tz):
                return cls.fromisoformat(cls.current).astimezone(tz)

        payloads = [
            tencent_row(symbol, 10).encode(),
            tencent_row(symbol, 20, nav='2.010', premium='0.50').encode(),
        ]
        write_json = collect_books.write_json

        def interrupt_after_pending(path, value):
            write_json(path, value)
            if value.get('pending_response', {}).get('index') == 2:
                raise KeyboardInterrupt('second response saved before IOPV confirmation')

        with TemporaryDirectory() as directory:
            output = Path(directory)
            argv = ['collect_books.py', '--date', DATES[0], '--out', str(output),
                    '--require-513100-iopv', symbol]
            with patch.object(collect_books, 'datetime', Clock), patch.object(sys, 'argv', argv):
                with patch.object(collect_books, 'capture', side_effect=zip(receipts, payloads)), \
                        patch.object(collect_books, 'write_json', interrupt_after_pending), \
                        patch.object(collect_books.time, 'sleep'):
                    with self.assertRaises(KeyboardInterrupt):
                        collect_books.main()
                saved = json.loads((output / 'capture-progress.json').read_text())
                self.assertEqual(len(saved['attempts']), 1)
                self.assertEqual(saved['pending_response']['index'], 2)
                self.assertEqual(saved['selected'], {})
                Clock.current = DATES[0] + 'T18:00:00+08:00'
                with patch.object(collect_books, 'capture') as request:
                    collect_books.main()
                    request.assert_not_called()
            final = json.loads((output / 'books.json').read_text())
            self.assertEqual([item['retrieved_at'] for item in final['attempts']], receipts)
            audited = verify_capture(final, output, date.fromisoformat(DATES[0]), [symbol])
            book = audited['selected'][symbol]
            self.assertEqual(book['received_at'], receipts[1])
            self.assertEqual(book['iopv_update']['previous_received_at'], receipts[0])
            self.assertEqual(book['iopv_update']['previous_nav'], '2.000')
            self.assertEqual(book['iopv']['nav'], '2.010')


if __name__ == '__main__':
    unittest.main()
