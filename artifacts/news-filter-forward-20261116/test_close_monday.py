"""Synthetic Monday valuation, accounting and evidence cases."""
import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import close_monday

from close_monday import (
    DAY, AmbiguousPartialLedgers, close_report, daily_ledger_path, etf_decision, prior_records,
    save_report, verify_mark_provenance,
)

SYMBOL = '601988.SH'
MARK = {'date': DAY, 'close': '6.88', 'source_url': 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get',
        'price_type': 'unadjusted_close'}


def opening():
    return {
        'date': DAY, 'mode': 'forward', 'planned_symbols': [SYMBOL, '513100.SH'],
        'received_symbols': [SYMBOL, '513100.SH'], 'fresh_symbols': [SYMBOL],
        'baseline': {
            'cash': '99309.99', 'positions': {SYMBOL: 100},
            'fills': [{'side': 'buy', 'symbol': SYMBOL, 'units': 100, 'price': '6.85',
                       'commission': '5.00', 'stamp_tax': '0', 'transfer_fee': '0.01',
                       'fees': '5.01', 'cash_delta': '-690.01'}],
            'decisions': [{'symbol': '513100.SH', 'reason': 'iopv_intraday_update_unverified',
                           'premium': None}],
        },
        'filtered': {'cash': '100000', 'positions': {}, 'fills': [],
                     'decisions': [{'symbol': SYMBOL, 'reason': 'news_data_unavailable',
                                    'evaluation': {'news_error': 'saved snapshot is late'}}]},
    }


class CloseMondayTests(unittest.TestCase):
    def test_missing_opening_is_not_zero_fills(self):
        daily, summary = close_report(None)
        self.assertIsNone(daily)
        self.assertEqual(summary['status'], 'opening_unverified')
        self.assertTrue(all(count is None for count in summary['coverage']['counts'].values()))
        for group in summary['groups'].values():
            self.assertIsNone(group['buy_count'])
            self.assertIsNone(group['sell_count'])
            self.assertIsNone(group['fees'])
            self.assertIsNone(group['close_nav'])

    def test_valued_positions_and_known_zero_are_reconciled(self):
        daily, summary = close_report(opening(), {'date': DAY, 'marks': {SYMBOL: MARK}})
        baseline, filtered = (summary['groups'][name] for name in ('baseline', 'filtered'))
        self.assertEqual(summary['status'], 'reconciled')
        self.assertEqual((baseline['buy_count'], baseline['sell_count'], baseline['fees'], baseline['close_nav']),
                         (1, 0, '5.01', '99997.99'))
        self.assertEqual((filtered['buy_count'], filtered['sell_count'], filtered['fees'], filtered['close_nav']),
                         (0, 0, '0', '100000'))
        self.assertEqual(daily['baseline']['close_evidence'][SYMBOL], MARK)
        self.assertEqual(baseline['513100.SH']['status'], 'iopv_intraday_update_unverified')
        self.assertEqual(summary['news_evidence']['unavailable_symbols'], [SYMBOL])
        self.assertEqual(summary['news_evidence']['unavailable_details'], {SYMBOL: 'saved snapshot is late'})
        self.assertEqual(baseline['unfilled_orders'][0]['symbol'], '513100.SH')
        self.assertEqual(baseline['unfilled_orders'][0]['reason'], 'iopv_intraday_update_unverified')
        self.assertIsNone(baseline['unfilled_orders'][0]['available_units'])

    def test_close_shows_buy_evidence_error_without_borrowing_sell_error(self):
        record = opening()
        etf = '513100.SH'
        record['baseline']['decisions'][0].update(side='buy', reason='quote_evidence_unverified')
        record['book_evidence_errors'] = {
            'selected': {etf: 'raw response checksum mismatch'},
            'exit_selected': {etf: 'different sell error'},
        }
        _, summary = close_report(record, {'date': DAY, 'marks': {SYMBOL: MARK}})
        group = summary['groups']['baseline']
        self.assertEqual(group['buy_count'], 1)
        self.assertEqual(group['fees'], '5.01')
        self.assertEqual(group[etf]['evidence_error'], 'raw response checksum mismatch')
        self.assertEqual(group['unfilled_orders'][0]['evidence_error'], 'raw response checksum mismatch')

    def test_coverage_distinguishes_raw_quotes_direction_evidence_and_fills(self):
        record = opening()
        etf = '513100.SH'
        record['quote_coverage'] = {
            'planned_symbols': [SYMBOL, etf], 'response_row_symbols': [SYMBOL, etf],
            'received_symbols': [SYMBOL, etf], 'fresh_symbols': [SYMBOL, etf],
            'two_sided_symbols': [SYMBOL, etf], 'buy_book_symbols': [SYMBOL], 'sell_book_symbols': [],
        }
        record['baseline']['decisions'][0]['last_observation'] = {
            'at': f'{DAY}T09:45:04+08:00', 'received_at': f'{DAY}T09:45:05.500+08:00',
            'ask_units': 3000,
        }
        _, summary = close_report(record, {'date': DAY, 'marks': {SYMBOL: MARK}})
        self.assertEqual(summary['coverage']['counts'],
                         {'planned': 2, 'response_row': 2, 'received': 2, 'fresh': 2,
                          'two_sided': 2, 'buy_book': 1, 'sell_book': 0})
        report = summary['groups']['baseline'][etf]
        self.assertEqual((report['quote_age_seconds'], report['diagnostic_quote_only']), (1.5, True))
        self.assertEqual(report['diagnostic_visible_units'], 3000)
        self.assertIsNone(report['available_units'])

    def test_etf_summary_uses_the_buy_decision_and_legacy_book_depth(self):
        sale = {'symbol': '513100.SH', 'side': 'sell', 'reason': 'filled',
                'required_units': 2100, 'book': {'bid_units': 2200}}
        buy = {'symbol': '513100.SH', 'side': 'buy', 'reason': 'insufficient_ask_units',
               'premium': '0.01', 'required_units': 2300, 'book': {'ask_units': 2200}}
        sale_summary = etf_decision({'decisions': [sale]})
        self.assertEqual((sale_summary['side'], sale_summary['available_units']), ('sell', 2200))
        self.assertFalse(sale_summary['iopv_update_verified'])
        summary = etf_decision({'decisions': [sale, buy]})
        self.assertEqual((summary['side'], summary['premium']), ('buy', '0.01'))
        self.assertTrue(summary['simulated_buy_premium_within_limit'])
        self.assertIsNone(summary['verified_vendor_premium_pct'])
        self.assertEqual((summary['required_units'], summary['available_units']), (2300, 2200))
        diagnostic = {**buy, 'book': None, 'premium': None, 'last_observation': {
            'raw_sha256': 'a' * 64, 'iopv': {'status': 'verified_snapshot', 'vendor_premium_pct': '0.50'},
        }}
        summary = etf_decision({'decisions': [diagnostic]})
        self.assertFalse(summary['iopv_update_verified'])
        self.assertIsNone(summary['verified_vendor_premium_pct'])
        self.assertIsNone(summary['simulated_buy_premium_within_limit'])

    def test_mark_failure_preserves_known_execution_metrics(self):
        for mark, reason in [
            (None, 'close_mark_missing'),
            ({**MARK, 'date': '2026-11-13'}, 'close_mark_wrong_date'),
            ({**MARK, 'source_url': 'https://gtimg.cn.example.com/'}, 'close_market_source_unverified'),
            ({**MARK, 'close': 'NaN'}, 'close_price_invalid'),
            ({**MARK, 'price_type': 'adjusted'}, 'unadjusted_close_unverified'),
            ({**MARK, 'valuation_gap': 'corporate_action_unknown'}, 'close_evidence_incomplete'),
        ]:
            with self.subTest(reason=reason):
                daily, summary = close_report(opening(), {'date': DAY, 'marks': {SYMBOL: mark}})
                self.assertEqual(daily['baseline']['valuation_gap'], {SYMBOL: reason})
                self.assertIsNone(daily['baseline']['nav'])
                self.assertEqual(summary['groups']['baseline']['fees'], '5.01')
                self.assertEqual(summary['groups']['filtered']['close_nav'], '100000')

    def test_provider_gap_retains_specific_reason(self):
        daily, _ = close_report(opening(), {'date': DAY, 'marks': {}, 'errors': {SYMBOL: 'provider_unavailable'}})
        self.assertEqual(daily['baseline']['valuation_gap'], {SYMBOL: 'provider_unavailable'})

    def test_validates_raw_bytes_and_after_close_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw = b'example response'
            (directory / 'raw.json').write_bytes(raw)
            mark = {**MARK, 'raw_file': 'raw.json', 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                    'quote_at': DAY + 'T15:01:00+08:00', 'observed_at': DAY + 'T18:00:00+08:00'}
            self.assertIsNone(verify_mark_provenance(mark, directory, require_raw=True))
            self.assertEqual(verify_mark_provenance({**mark, 'quote_at': DAY + 'T14:59:00+08:00'}, directory, True),
                             'close_timestamp_unverified')
            self.assertEqual(verify_mark_provenance({**mark, 'raw_file': '../raw.json'}, directory, True),
                             'close_raw_path_invalid')
            self.assertEqual(verify_mark_provenance(MARK, directory, True), 'close_raw_provenance_missing')
            (directory / 'raw.json').write_bytes(b'changed')
            self.assertEqual(verify_mark_provenance(mark, directory, True), 'close_raw_hash_mismatch')

    def test_reparses_market_bytes_and_dual_source_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            quote = [''] * 31
            quote[2], quote[3], quote[4], quote[30] = '601988', '6.88', '6.85', '20261116150100'
            raw = json.dumps({'code': 0, 'data': {'sh601988': {
                'day': [['2026-11-13', '0', '6.85'], [DAY, '0', '6.88', '0', '0', '200']],
                'qt': {'sh601988': quote},
            }}}).encode()
            sina = json.dumps([
                {'day': '2026-11-13', 'close': '6.85', 'volume': '100'},
                {'day': DAY, 'close': '6.88', 'volume': '200'},
            ]).encode()
            (directory / 'raw.json').write_bytes(raw)
            (directory / 'sina.json').write_bytes(sina)
            mark = {**MARK, 'raw_file': 'raw.json', 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                    'quote_at': DAY + 'T15:01:00+08:00', 'observed_at': DAY + 'T18:00:00+08:00',
                    'previous_close': '6.85'}
            verify = lambda record, previous: verify_mark_provenance(
                record, directory, True, symbol=SYMBOL, previous_close=previous,
            )
            self.assertIsNone(verify(mark, '6.85'))
            self.assertEqual(verify({**mark, 'close': '6.99'}, '6.85'), 'close_mark_raw_mismatch')
            self.assertEqual(verify(mark, '6.86'), 'prior_close_reference_mismatch')
            self.assertEqual(verify(mark, None), 'prior_close_evidence_missing')
            mark.update(previous_close_basis='dual_source_recovery', secondary_check={
                'status': 'agreed', 'source_url': 'https://quotes.sina.cn/daily',
                'raw_file': 'sina.json', 'raw_sha256': hashlib.sha256(sina).hexdigest(),
            })
            self.assertIsNone(verify(mark, None))
            (directory / 'sina.json').write_bytes(b'changed')
            self.assertEqual(verify(mark, None), 'prior_close_secondary_close_raw_hash_mismatch')

    def test_rejects_cash_mismatch_and_readiness_mode(self):
        for change in ('cash', 'mode'):
            record = opening()
            if change == 'cash':
                record['baseline']['cash'] = '100000'
            else:
                record['mode'] = 'readiness'
            with self.subTest(change=change), self.assertRaises(ValueError):
                close_report(record)

    def test_cli_reproduces_output_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            opening_path, marks_path = out / 'opening.json', out / 'marks.json'
            opening_path.write_text(json.dumps(opening()))
            marks_path.write_text(json.dumps({'date': DAY, 'marks': {SYMBOL: MARK}}))
            command = [sys.executable, str(Path(__file__).with_name('close_monday.py')),
                       '--opening', str(opening_path), '--marks', str(marks_path), '--out', str(out)]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            second = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(json.loads((out / 'daily.json').read_text())['baseline']['nav'], '99997.99')
            marks_path.write_text(json.dumps({'date': DAY, 'marks': {SYMBOL: {**MARK, 'close': '7.00'}}}))
            conflict = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(conflict.returncode, 0)
            self.assertIn('refusing to overwrite', conflict.stderr)
            self.assertEqual(json.loads((out / 'daily.json').read_text())['baseline']['nav'], '99997.99')

    def test_official_cli_reports_damaged_inputs_and_preserves_evidence(self):
        class AfterClose(datetime):
            @classmethod
            def now(cls, tz):
                return cls(2026, 11, 16, 18, tzinfo=tz)

        bad_cash = opening()
        bad_cash['baseline']['cash'] = '100000'
        bad_openings = (
            ('{', 'Expecting property name'),
            (json.dumps({**opening(), 'mode': 'readiness'}), 'requires a forward opening ledger'),
            (json.dumps(bad_cash), 'cash or shares do not reconcile'),
        )
        cases = [(raw, '{}', 'opening', error) for raw, error in bad_openings]
        cases += [(json.dumps(opening()), raw, 'close_marks', error)
                  for raw, error in (
                      ('{', 'Expecting property name'),
                      ('{"date":"2026-11-15","marks":{}}', 'session date and a marks object'),
                  )]
        for raw_open, raw_marks, failed_input, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                directory = base / DAY
                directory.mkdir()
                opening_path, marks_path = directory / 'opening.json', directory / 'close_marks.json'
                opening_path.write_text(raw_open)
                marks_path.write_text(raw_marks)
                output = StringIO()
                with patch.object(close_monday, 'BASE', base), patch.object(close_monday, 'datetime', AfterClose), \
                        patch.object(sys, 'argv', ['close_monday.py']), redirect_stdout(output):
                    close_monday.main()
                summary = json.loads(output.getvalue())
                self.assertIn(error, summary['input_errors'][failed_input]['error'])
                self.assertEqual(Path(summary['artifacts']['summary']).parent.parent,
                                 directory.resolve() / 'close-partials')
                self.assertFalse((directory / 'daily.json').exists())
                self.assertEqual(opening_path.read_text(), raw_open)
                self.assertEqual(marks_path.read_text(), raw_marks)
                if failed_input == 'opening':
                    self.assertEqual(summary['status'], 'opening_unverified')
                    self.assertIsNone(summary['artifacts']['daily'])
                    for group in summary['groups'].values():
                        for field in ('buy_count', 'sell_count', 'fees', 'close_nav'):
                            self.assertIsNone(group[field])
                        self.assertEqual(group['513100.SH'],
                                         {'status': 'opening_unverified', 'premium': None})
                else:
                    self.assertEqual(summary['status'], 'valuation_incomplete')
                    baseline, filtered = (summary['groups'][name] for name in ('baseline', 'filtered'))
                    self.assertEqual((baseline['buy_count'], baseline['sell_count'], baseline['fees']), (1, 0, '5.01'))
                    self.assertIsNone(baseline['close_nav'])
                    self.assertEqual(baseline['valuation_gap'], {SYMBOL: 'close_marks_unverified'})
                    self.assertEqual(filtered['close_nav'], '100000')

    def test_partial_output_does_not_block_complete_canonical_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            daily, summary = close_report(opening())
            first = save_report(directory, daily, summary, official=True)
            self.assertEqual(save_report(directory, daily, summary, official=True), first)
            partial_daily = Path(first['artifacts']['daily'])
            self.assertEqual(partial_daily.parent.parent, directory / 'close-partials')
            self.assertIsNone(json.loads(partial_daily.read_text())['baseline']['nav'])
            self.assertFalse((directory / 'daily.json').exists())

            daily, summary = close_report(opening(), {'date': DAY, 'marks': {SYMBOL: MARK}})
            final = save_report(directory, daily, summary, official=True)
            self.assertEqual(Path(final['artifacts']['daily']), directory / 'daily.json')
            self.assertEqual(json.loads((directory / 'daily.json').read_text())['baseline']['nav'], '99997.99')
            self.assertTrue(partial_daily.exists())

    def test_unverified_open_writes_only_partial_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            daily, summary = close_report(None)
            saved = save_report(directory, daily, summary, official=True)
            self.assertIsNone(saved['artifacts']['daily'])
            self.assertTrue(Path(saved['artifacts']['summary']).exists())
            self.assertFalse((directory / 'close-summary.json').exists())

    def test_later_session_reconciles_sale_and_retains_earlier_gap(self):
        earlier, _ = close_report(opening())
        tuesday = '2026-11-17'
        next_opening = {
            'date': tuesday, 'mode': 'forward', 'news_errors': [],
            'baseline': {
                'cash': '99994.66', 'positions': {},
                'fills': [{'side': 'sell', 'symbol': SYMBOL, 'units': 100, 'price': '6.90',
                           'commission': '5.00', 'stamp_tax': '0.32', 'transfer_fee': '0.01',
                           'fees': '5.33', 'cash_delta': '684.67'}],
            },
            'filtered': {'cash': '100000', 'positions': {}, 'fills': []},
        }
        daily, summary = close_report(next_opening, day=tuesday, prior=[earlier])
        baseline = summary['groups']['baseline']
        self.assertEqual(summary['status'], 'reconciled')
        self.assertEqual((baseline['buy_count'], baseline['sell_count'], baseline['fees'], baseline['close_nav']),
                         (0, 1, '5.33', '99994.66'))
        self.assertEqual(baseline['cumulative_fees'], '10.34')
        self.assertEqual(baseline['missing_valuation_dates'], [DAY])
        self.assertEqual(daily['mode'], 'forward')
        self.assertEqual(daily['news_errors'], [])
        with self.assertRaisesRegex(ValueError, 'prior ledger dates'):
            close_report(next_opening, day=tuesday)

    def test_prior_records_pick_existing_retry_with_most_compatible_marks(self):
        initial = opening()
        account = initial['baseline']
        extra_symbol = '513100.SH'
        account['positions'][extra_symbol] = 100
        account['fills'].append({'side': 'buy', 'symbol': extra_symbol, 'units': 100, 'price': '2.00',
                                 'commission': '5', 'stamp_tax': '0', 'transfer_fee': '0',
                                 'fees': '5', 'cash_delta': '-205.00'})
        account['cash'] = '99104.99'
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            day_dir = directory / DAY
            daily, summary = close_report(initial)
            save_report(day_dir, daily, summary, official=True)
            marks = {SYMBOL: MARK}
            improved, summary = close_report(initial, {'date': DAY, 'marks': marks})
            selected = save_report(day_dir, improved, summary, official=True)
            self.assertEqual(daily_ledger_path(day_dir), Path(selected['artifacts']['daily']))
            self.assertEqual(prior_records(directory, '2026-11-17'), [improved])
            self.assertIsNone(improved['baseline']['nav'])

            equivalent, summary = close_report(initial, {'date': DAY, 'marks': {SYMBOL: {**MARK, 'close': '6.880'}},
                                                         'errors': {extra_symbol: 'unavailable'}})
            alternative = save_report(day_dir, equivalent, summary, official=True)
            self.assertEqual(daily_ledger_path(day_dir), Path(min(selected['artifacts']['daily'],
                                                                alternative['artifacts']['daily'])))

    def test_prior_records_refuse_conflicting_or_complementary_retries(self):
        initial = opening()
        account = initial['baseline']
        extra_symbol = '513100.SH'
        account['positions'][extra_symbol] = 100
        account['fills'].append({'side': 'buy', 'symbol': extra_symbol, 'units': 100, 'price': '2.00',
                                 'commission': '5', 'stamp_tax': '0', 'transfer_fee': '0',
                                 'fees': '5', 'cash_delta': '-205.00'})
        account['cash'] = '99104.99'
        for other_marks, expected in (
            ({SYMBOL: {**MARK, 'close': '6.89'}}, 'close prices disagree'),
            ({extra_symbol: {**MARK, 'close': '2.00'}}, 'no ledger contains all verified valuations'),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                day_dir = Path(tmp) / DAY
                for marks in ({SYMBOL: MARK}, other_marks):
                    daily, summary = close_report(initial, {'date': DAY, 'marks': marks})
                    save_report(day_dir, daily, summary, official=True)
                with self.assertRaisesRegex(AmbiguousPartialLedgers, expected) as refused:
                    daily_ledger_path(day_dir)
                self.assertEqual(refused.exception.count, 2)

    def test_prior_record_prefers_canonical_and_rejects_ambiguous_partials(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            day_dir = directory / DAY
            daily, summary = close_report(opening())
            save_report(day_dir, daily, summary, official=True)
            self.assertEqual(prior_records(directory, '2026-11-17'), [daily])
            save_report(day_dir, {**daily, 'another_attempt': True}, summary, official=True)
            with self.assertRaisesRegex(ValueError, '2 partial ledgers'):
                prior_records(directory, '2026-11-17')
            daily, summary = close_report(opening(), {'date': DAY, 'marks': {SYMBOL: MARK}})
            save_report(day_dir, daily, summary, official=True)
            self.assertEqual(prior_records(directory, '2026-11-17'), [daily])


if __name__ == '__main__':
    unittest.main()
