"""Replay real provider response layouts with controlled intraday timestamps."""

import copy
import hashlib
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from book_evidence import verify_capture
from collect_books import IOPV_SYMBOL, SOURCES, confirm_iopv_update, parse_books
from continue_open import run_day
from monday_open import run


STOCK = '601988.SH'
ETF = IOPV_SYMBOL
DAY = date(2026, 11, 16)
SYMBOLS = {f'sh{symbol[:6]}': symbol for symbol in (STOCK, ETF)}


def tencent_row(symbol, second, *, nav='2.000', latest='2.020', premium='1.00',
                limits=('2.20', '1.80'), ask=None, ask_lots='20'):
    fields = [''] * 83
    fields[1], fields[2], fields[3:7] = 'fixture', symbol[:6], [latest, '2.00', '2.00', '200']
    fields[9:11], fields[19:21] = ['2.010', '10'], [latest if ask is None else ask, ask_lots]
    fields[30] = f'202611160945{second:02d}'
    fields[47], fields[48] = limits
    fields[61], fields[77], fields[78], fields[82] = 'ETF', premium, nav, 'CNY'
    return f'v_sh{symbol[:6]}="' + '~'.join(fields) + '";\n'


def save_attempt(root, filename, payload, receipt, codes=None):
    url = SOURCES['tencent'] + ','.join(codes or SYMBOLS)
    raw = payload.encode('gb18030')
    (root / filename).write_bytes(raw)
    books, errors = parse_books('tencent', raw, SYMBOLS)
    return {'source': 'tencent', 'url': url, 'retrieved_at': f'{DAY}T{receipt}+08:00',
            'raw_file': filename, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'books': books, 'parse_errors': errors}


def select(attempt, symbol):
    return {**attempt['books'][symbol], 'source_url': attempt['url'],
            'raw_file': attempt['raw_file'], 'received_at': attempt['retrieved_at']}


class BookEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        first = save_attempt(self.root, 'first-tencent.txt', tencent_row(STOCK, 10) + tencent_row(ETF, 10),
                             '09:45:11')
        second = save_attempt(self.root, 'second-tencent.txt',
                              tencent_row(ETF, 20, nav='2.010', latest='2.020', premium='0.50'), '09:45:21',
                              ['sh513100'])
        first_etf, current_etf = select(first, ETF), select(second, ETF)
        self.capture = {
            'mode': 'forward', 'date': str(DAY), 'attempts': [first, second],
            'selected': {STOCK: select(first, STOCK),
                         ETF: {**current_etf, 'iopv_update': confirm_iopv_update(first_etf, current_etf)}},
            'exit_selected': {ETF: first_etf}, 'unavailable': {},
        }

    def verify(self, value=None):
        return verify_capture(value or self.capture, self.root, DAY, SYMBOLS.values())

    def test_replay_rejects_a_journal_with_only_malformed_attempts(self):
        for attempts in ([], [None], [{}], [{'error': 'provider unavailable'}]):
            with self.subTest(attempts=attempts):
                captured = {**self.capture, 'attempts': attempts, 'selected': {}, 'exit_selected': {}}
                with self.assertRaisesRegex(ValueError, 'no verifiable quote attempts'):
                    self.verify(captured)

    def test_quote_coverage_tracks_raw_stages_before_trade_gates(self):
        stale, invalid, one_side, absent, zero_depth = (
            '600000.SH', '600001.SH', '600002.SH', '600003.SH', '600004.SH')
        symbols = [STOCK, ETF, stale, invalid, one_side, absent, zero_depth]
        by_code = {f'sh{symbol[:6]}': symbol for symbol in symbols}
        raw = (
            tencent_row(STOCK, 44, latest='2.000', ask_lots='100') + tencent_row(ETF, 44)
            + tencent_row(stale, 10) + tencent_row(invalid, 44).replace('v_sh', 'var hq_str_sh')
            + tencent_row(one_side, 44, ask='0', ask_lots='0')
            + tencent_row(zero_depth, 44, ask_lots='0')
        )
        attempt = save_attempt(self.root, 'mixed-coverage.txt', raw, '09:45:45', by_code)
        attempt['books'], _ = parse_books('tencent', raw.encode('gb18030'), by_code)
        failed = {'source': 'tencent', 'url': SOURCES['tencent'] + 'sh600003', 'error': 'provider unavailable'}
        capture = {'mode': 'forward', 'date': str(DAY), 'attempts': [attempt, failed],
                   'selected': {symbol: select(attempt, symbol) for symbol in (STOCK, one_side, zero_depth)},
                   'exit_selected': {}, 'unavailable': {ETF: 'iopv_intraday_update_unverified',
                                                        stale: 'quote_stale', invalid: 'invalid_quote'}}
        audited = verify_capture(capture, self.root, DAY, symbols)
        self.assertEqual(audited['quote_coverage'], {
            'planned_symbols': sorted(symbols),
            'response_row_symbols': sorted(set(symbols) - {absent}),
            'received_symbols': sorted({STOCK, ETF, stale, one_side, zero_depth}),
            'fresh_symbols': sorted({STOCK, ETF, one_side, zero_depth}),
            'two_sided_symbols': sorted({STOCK, ETF, zero_depth}),
            'buy_book_symbols': sorted({STOCK, one_side, zero_depth}),
            'sell_book_symbols': [],
        })
        self.assertNotIn(invalid, audited['last_observations'])
        self.assertNotIn(absent, audited['last_observations'])
        candidates = [{'symbol': symbol, 'previous_close': '2.00', 'tick': '.01',
                       'kind': 'stock', 'cross_border': symbol == ETF} for symbol in symbols]
        opening = run({'candidates': candidates, 'coverage': 'fixture'}, audited, [])
        self.assertEqual(opening['quote_coverage'], audited['quote_coverage'])
        self.assertEqual(opening['fresh_symbols'], opening['quote_coverage']['fresh_symbols'])
        for group in ('baseline', 'filtered'):
            self.assertEqual([fill['symbol'] for fill in opening[group]['fills']], [STOCK])
            decisions = {row['symbol']: row for row in opening[group]['decisions']}
            for symbol, reason in ((ETF, 'iopv_intraday_update_unverified'), (stale, 'quote_stale')):
                self.assertEqual(decisions[symbol]['reason'], reason)
                self.assertIsNone(decisions[symbol]['book'])
                self.assertIsNone(decisions[symbol]['available_units'])
                self.assertEqual(decisions[symbol]['last_observation']['raw_sha256'], attempt['raw_sha256'])
            self.assertEqual(decisions[zero_depth]['reason'], 'no_ask_liquidity')
            self.assertEqual(decisions[zero_depth]['available_units'], 0)
            self.assertNotIn('last_observation', decisions[invalid])
            self.assertNotIn('last_observation', decisions[absent])

    def test_accepts_complete_raw_journal_and_distinct_entry_exit_snapshots(self):
        result = self.verify()
        self.assertEqual(set(result['selected']), {STOCK, ETF})
        book = result['selected'][ETF]
        self.assertEqual(book['iopv']['nav'], '2.010')
        update = book['iopv_update']
        self.assertEqual(update['previous_nav'], '2.000')
        self.assertEqual(update['previous_vendor_premium_pct'], '1.00')
        self.assertEqual(update['previous_last'], '2.020')
        self.assertEqual(update['previous_raw_sha256'],
                         hashlib.sha256((self.root / 'first-tencent.txt').read_bytes()).hexdigest())
        self.assertEqual(book['raw_sha256'],
                         hashlib.sha256((self.root / 'second-tencent.txt').read_bytes()).hexdigest())
        self.assertEqual(result['exit_selected'][ETF]['raw_file'], 'first-tencent.txt')
        self.assertEqual(result['evidence_errors'], {'selected': {}, 'exit_selected': {}})
        self.assertEqual(self.verify(result)['selected'][ETF], book)

    def test_etf_pair_rejects_tampered_provenance_or_previous_raw_response(self):
        for field, wrong_value in (
            ('previous_raw_sha256', '0' * 64),
            ('previous_vendor_premium_pct', '0.00'),
            ('previous_last', '3.000'),
        ):
            with self.subTest(field=field):
                audited = self.verify()
                audited['selected'][ETF]['iopv_update'][field] = wrong_value
                result = self.verify(audited)
                self.assertNotIn(ETF, result['selected'])
                self.assertEqual(result['unavailable'][ETF], 'iopv_evidence_unverified')
                self.assertIn(STOCK, result['selected'])
                self.assertIn(ETF, result['exit_selected'])

        archived = self.root / 'first-tencent.txt'
        archived.write_bytes(archived.read_bytes() + b'\n')
        result = self.verify()
        self.assertNotIn(ETF, result['selected'])
        self.assertIn('checksum mismatch', result['attempts'][0]['evidence_error'])

    def test_bad_entry_claim_does_not_stop_other_candidates_or_verified_exit(self):
        for change, reason in (
            ({'ask_units': 5000}, 'quote_evidence_unverified'),
            ({'received_at': f'{DAY}T09:45:25+08:00'}, 'quote_evidence_unverified'),
            ({'iopv_update': {'previous_nav': '1.900'}}, 'iopv_evidence_unverified'),
            ({'iopv': {'status': 'verified_snapshot', 'nav': '2.020'}}, 'iopv_evidence_unverified'),
        ):
            with self.subTest(change=change):
                captured = copy.deepcopy(self.capture)
                captured['selected'][ETF].update(change)
                result = self.verify(captured)
                self.assertEqual(set(result['selected']), {STOCK})
                self.assertEqual(result['unavailable'][ETF], reason)
                self.assertIn(ETF, result['exit_selected'])
                self.assertNotIn(ETF, result['exit_unavailable'])

    def test_raw_checksum_mismatch_only_affects_claims_depending_on_it(self):
        captured = copy.deepcopy(self.capture)
        captured['attempts'][1]['raw_sha256'] = '0' * 64
        result = self.verify(captured)
        self.assertEqual(set(result['selected']), {STOCK})
        self.assertIn(ETF, result['exit_selected'])
        self.assertEqual(result['evidence_errors']['selected'], {ETF: 'raw response checksum mismatch'})
        self.assertIn('checksum mismatch', result['attempts'][1]['evidence_error'])

    def test_earlier_missing_attempt_cannot_be_bypassed_with_later_claim(self):
        captured = copy.deepcopy(self.capture)
        earlier = {**captured['attempts'][0], 'raw_file': 'missing-tencent.txt'}
        captured['attempts'].insert(0, earlier)
        result = self.verify(captured)
        self.assertFalse(result['selected'])
        self.assertFalse(result['exit_selected'])
        self.assertIn('missing', result['attempts'][0]['evidence_error'])

    def test_rejects_raw_path_escape_and_mismatched_provider(self):
        for key, value in (
            ('raw_file', '../second-tencent.txt'),
            ('url', 'https://example.org/q=sh513100'),
            ('url', SOURCES['tencent'] + 'sh513100?x=1'),
            ('source', 'sina'),
        ):
            with self.subTest(key=key, value=value):
                captured = copy.deepcopy(self.capture)
                captured['attempts'][1][key] = value
                result = self.verify(captured)
                self.assertEqual(set(result['selected']), {STOCK})
                self.assertIn('evidence_error', result['attempts'][1])

    def test_rejects_symlink_to_raw_file(self):
        (self.root / 'link.txt').symlink_to(self.root / 'second-tencent.txt')
        captured = copy.deepcopy(self.capture)
        captured['attempts'][1]['raw_file'] = 'link.txt'
        captured['selected'][ETF]['raw_file'] = 'link.txt'
        result = self.verify(captured)
        self.assertNotIn(ETF, result['selected'])
        self.assertIn('unsafe', result['attempts'][1]['evidence_error'])

    def test_later_exit_cannot_replace_first_qualifying_snapshot(self):
        captured = copy.deepcopy(self.capture)
        captured['exit_selected'][ETF] = select(captured['attempts'][1], ETF)
        result = self.verify(captured)
        self.assertNotIn(ETF, result['exit_selected'])
        self.assertEqual(result['exit_unavailable'][ETF], 'quote_evidence_unverified')
        self.assertIn(ETF, result['selected'])
        self.assertNotIn(ETF, result['unavailable'])

    def test_later_exit_is_first_qualifying_when_earlier_limits_are_missing(self):
        first = save_attempt(self.root, 'missing-limits.txt',
                             tencent_row(STOCK, 10, limits=('', '0')), '09:45:11', ['sh601988'])
        second = save_attempt(self.root, 'usable-limits.txt',
                              tencent_row(STOCK, 20), '09:45:21', ['sh601988'])
        book = select(second, STOCK)
        captured = {'mode': 'forward', 'date': str(DAY), 'attempts': [first, second],
                    'selected': {STOCK: book}, 'exit_selected': {STOCK: book}, 'unavailable': {}}
        result = verify_capture(captured, self.root, DAY, [STOCK], held=[STOCK], buy_symbols=[])
        self.assertEqual(result['quote_coverage']['sell_book_symbols'], [STOCK])
        self.assertEqual(result['quote_coverage']['buy_book_symbols'], [])
        self.assertEqual(result['exit_selected'][STOCK]['raw_file'], 'usable-limits.txt')
        self.assertEqual(result['selected'][STOCK]['raw_file'], 'usable-limits.txt')

    def test_provider_row_prefix_cannot_be_misattributed(self):
        captured = copy.deepcopy(self.capture)
        attempt = captured['attempts'][1]
        path = self.root / attempt['raw_file']
        raw = path.read_bytes().replace(b'v_sh513100=', b'var hq_str_sh513100=')
        path.write_bytes(raw)
        attempt['raw_sha256'] = hashlib.sha256(raw).hexdigest()
        result = self.verify(captured)
        self.assertEqual(set(result['selected']), {STOCK})
        self.assertEqual(result['attempts'][1]['parse_errors'][ETF], 'raw row belongs to a different provider')

    def test_missing_prior_update_cannot_be_faked_using_only_books_json(self):
        captured = copy.deepcopy(self.capture)
        captured['attempts'][0]['url'] = SOURCES['tencent'] + 'sh601988'
        captured['selected'][STOCK]['source_url'] = captured['attempts'][0]['url']
        result = self.verify(captured)
        self.assertNotIn(ETF, result['selected'])
        self.assertEqual(result['unavailable'][ETF], 'iopv_evidence_unverified')
        self.assertIn(STOCK, result['selected'])

    def test_single_iopv_keeps_update_unverified_reason(self):
        captured = copy.deepcopy(self.capture)
        captured['attempts'].pop()
        del captured['selected'][ETF]
        captured['unavailable'][ETF] = 'iopv_intraday_update_unverified'
        result = self.verify(captured)
        self.assertNotIn(ETF, result['selected'])
        self.assertEqual(result['unavailable'][ETF], 'iopv_intraday_update_unverified')
        self.assertNotIn(ETF, result['evidence_errors']['selected'])
        self.assertIn(STOCK, result['selected'])

    def test_eligible_raw_quote_without_claim_remains_unselected(self):
        captured = copy.deepcopy(self.capture)
        captured['selected'].clear()
        captured['exit_selected'].clear()
        captured['unavailable'][ETF] = 'iopv_intraday_update_unverified'
        result = verify_capture(captured, self.root, DAY, SYMBOLS.values(), held=[ETF])
        self.assertEqual(result['selected'], {})
        self.assertEqual(result['exit_selected'], {})
        self.assertEqual(result['unavailable'],
                         {STOCK: 'quote_evidence_unverified', ETF: 'quote_evidence_unverified'})
        self.assertEqual(result['exit_unavailable'], {ETF: 'quote_evidence_unverified'})
        self.assertEqual(result['evidence_errors'], {
            'selected': dict.fromkeys((STOCK, ETF), 'eligible archived entry quote is missing the capture claim'),
            'exit_selected': {ETF: 'eligible archived exit quote is missing the capture claim'},
        })

    def test_missing_claim_uses_first_raw_audit_error_without_affecting_other_symbol(self):
        captured = copy.deepcopy(self.capture)
        del captured['selected'][ETF]
        captured['unavailable'][ETF] = 'quote_stale'
        captured['attempts'][1]['raw_sha256'] = '0' * 64
        missing = dict(captured['attempts'][1], raw_file='missing-tencent.txt')
        captured['attempts'].append(missing)
        result = self.verify(captured)
        self.assertNotIn(ETF, result['selected'])
        self.assertEqual(result['unavailable'][ETF], 'quote_evidence_unverified')
        self.assertEqual(result['evidence_errors']['selected'], {ETF: 'raw response checksum mismatch'})
        self.assertEqual(set(result['selected']), {STOCK})
        self.assertIn(ETF, result['exit_selected'])
        self.assertEqual(result['evidence_errors']['exit_selected'], {})

        captured['attempts'][0]['url'] = SOURCES['tencent'] + 'sh601988'
        captured['selected'][STOCK]['source_url'] = captured['attempts'][0]['url']
        captured['exit_selected'].clear()
        result = verify_capture(captured, self.root, DAY, SYMBOLS.values(), held=[ETF])
        self.assertEqual(set(result['selected']), {STOCK})
        self.assertEqual(result['exit_unavailable'], {ETF: 'quote_evidence_unverified'})
        self.assertEqual(result['evidence_errors']['exit_selected'], {ETF: 'raw response checksum mismatch'})

    def test_held_etf_exit_diagnostics_are_independent_of_entry_iopv(self):
        previous = {
            'baseline': {'cash': '99800', 'nav': '100000', 'positions': {ETF: 100},
                         'close_prices': {ETF: '2.00'}},
            'filtered': {'cash': '100000', 'nav': '100000', 'positions': {}},
        }
        prepared = {'coverage': 'scenario', 'candidates': [{'symbol': ETF, 'previous_close': '2.00'}]}
        news = {'batches': [], 'errors': [], 'sources': []}
        for receipt, limits, claimed, reason in (
            ('09:45:11', ('', '0'), False, 'price_limits_unverified'),
            ('09:45:50', ('2.20', '1.80'), False, 'quote_stale'),
            ('09:45:11', ('2.20', '1.80'), False, 'quote_evidence_unverified'),
            ('09:45:11', ('2.20', '1.80'), True, 'filled'),
        ):
            with self.subTest(reason=reason):
                attempt = save_attempt(self.root, 'exit.txt', tencent_row(ETF, 10, limits=limits),
                                       receipt, ['sh513100'])
                captured = {'mode': 'forward', 'date': str(DAY), 'attempts': [attempt], 'selected': {},
                            'exit_selected': {ETF: select(attempt, ETF)} if claimed else {},
                            'unavailable': {ETF: 'iopv_intraday_update_unverified'}}
                verified = verify_capture(captured, self.root, DAY, [ETF], held=[ETF])
                self.assertEqual(verified['exit_unavailable'], {} if claimed else {ETF: reason})
                opening = run_day(DAY, previous, prepared, verified, news)
                baseline, filtered = (opening[group] for group in ('baseline', 'filtered'))
                self.assertEqual(baseline['decisions'][0]['reason'], reason)
                if not claimed:
                    self.assertEqual(baseline['decisions'][0]['last_observation']['received_at'],
                                     attempt['retrieved_at'])
                    self.assertIsNone(baseline['decisions'][0]['book'])
                self.assertEqual(baseline['positions'], {} if claimed else {ETF: 100})
                self.assertEqual([f['side'] for f in baseline['fills']], ['sell'] if claimed else [])
                self.assertEqual(filtered['decisions'][0]['reason'], 'iopv_intraday_update_unverified')


if __name__ == '__main__':
    unittest.main()
