"""Reconcile a session's opening ledger and report sourced close valuations."""
import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from reconcile import reconcile

DAY = '2026-11-16'
BASE = Path(__file__).resolve().parent
DIRECTORY = BASE / DAY
SESSIONS = tuple((date.fromisoformat(DAY) + timedelta(days=i)).isoformat() for i in range(5))
GROUPS = ('baseline', 'filtered')
MARKET_DOMAINS = ('gtimg.cn', 'qq.com', 'sina.com.cn', 'sina.cn', 'sinajs.cn',
                  'eastmoney.com', 'sse.com.cn', 'szse.cn')
SECURITY_CODE = re.compile(r'[0-9]{6}[.](SH|SZ)')
CST = ZoneInfo('Asia/Shanghai')
INPUT_ERRORS = (OSError, ValueError, KeyError, TypeError, ArithmeticError, AttributeError, IndexError)


def market_source(value):
    if not isinstance(value, str):
        return False
    try:
        url = urlsplit(value)
        host = url.hostname or ''
    except ValueError:
        return False
    return (
        url.scheme == 'https' and url.username is None and url.password is None
        and any(host == domain or host.endswith('.' + domain) for domain in MARKET_DOMAINS)
    )


def closing_price(mark, day=DAY):
    if not isinstance(mark, dict):
        return None, 'close_mark_missing'
    if mark.get('date') != day:
        return None, 'close_mark_wrong_date'
    if mark.get('price_type') != 'unadjusted_close':
        return None, 'unadjusted_close_unverified'
    if not market_source(mark.get('source_url')):
        return None, 'close_market_source_unverified'
    if mark.get('valuation_gap'):
        return None, 'close_evidence_incomplete'
    raw_price = mark.get('close')
    if not isinstance(raw_price, str):
        return None, 'close_price_invalid'
    try:
        price = Decimal(raw_price)
    except InvalidOperation:
        return None, 'close_price_invalid'
    if not price.is_finite() or price <= 0:
        return None, 'close_price_invalid'
    return price, None


def execution_details(row):
    side = row.get('side')
    book = row.get('book') or {}
    observed = book or row.get('last_observation') or {}
    quote_age = None
    try:
        received = datetime.fromisoformat(observed.get('received_at'))
        quoted = datetime.fromisoformat(observed.get('at'))
        if received.utcoffset() is not None and quoted.utcoffset() is not None:
            quote_age = (received - quoted).total_seconds()
    except (TypeError, ValueError):
        pass
    quantity_key = 'bid_units' if side == 'sell' else 'ask_units'
    request_failure = row.get('quote_request_failure')
    return {'side': side, 'required_units': row.get('required_units'),
            'available_units': row.get('available_units', book.get(quantity_key)),
            'quote_age_seconds': quote_age, 'diagnostic_quote_only': bool(observed and not book),
            **({'quote_request_failure': request_failure} if request_failure else {}),
            **({'diagnostic_visible_units': observed.get(quantity_key)} if observed and not book else {})}


def quote_evidence_detail(errors, decision):
    side = decision.get('side')
    key = {'buy': 'selected', 'sell': 'exit_selected'}.get(side)
    return errors.get(key, {}).get(decision['symbol']) if key else None


def etf_decision(account, evidence_errors=None):
    rows = [d for d in account.get('decisions', []) if d['symbol'] == '513100.SH']
    if not rows:
        return {'status': 'decision_unverified', 'premium': None}
    row = next((d for d in rows if d.get('side') == 'buy'), rows[0])
    book = row.get('book') or {}
    premium = row.get('premium')
    iopv = book.get('iopv') or {}
    update = book.get('iopv_update') or {}
    verified_update = (
        row.get('side') == 'buy' and isinstance(iopv, dict) and isinstance(update, dict)
        and iopv.get('status') == 'verified_snapshot'
        and book.get('raw_sha256') != update.get('previous_raw_sha256')
        and all(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest)
                for digest in (book.get('raw_sha256'), update.get('previous_raw_sha256')))
    )
    detail = quote_evidence_detail(evidence_errors or {}, row) if row['reason'] != 'filled' else None
    premium_within_limit = None
    if premium is not None:
        try:
            value = Decimal(premium)
            if value.is_finite():
                premium_within_limit = value <= Decimal('.01')
        except (ArithmeticError, TypeError, ValueError):
            pass
    return {'status': row['reason'], 'premium': premium,
            **({'evidence_error': detail} if detail else {}),
            'simulated_buy_premium_within_limit': premium_within_limit,
            'iopv_update_verified': verified_update,
            'verified_vendor_premium_pct': iopv.get('vendor_premium_pct') if verified_update else None,
            'current_quote_evidence': {
                'received_at': book.get('received_at'), 'raw_file': book.get('raw_file'),
                'raw_sha256': book.get('raw_sha256'), 'latest_price': book.get('last'),
            } if verified_update else None,
            **execution_details(row),
            'quote_at': book.get('at'), 'quote_source': book.get('source_url'),
            'iopv': book.get('iopv'), 'iopv_update': book.get('iopv_update'),
            'last_unverified_observation': row.get('last_observation')}


def read_raw_provenance(mark, directory):
    raw_file, sha256 = mark.get('raw_file'), mark.get('raw_sha256')
    if not isinstance(raw_file, str) or not isinstance(sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', sha256):
        return None, 'close_raw_provenance_missing'
    raw_path = (directory / raw_file).resolve()
    if Path(raw_file).is_absolute() or not raw_path.is_relative_to(directory.resolve()):
        return None, 'close_raw_path_invalid'
    try:
        raw = raw_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            return None, 'close_raw_hash_mismatch'
    except OSError:
        return None, 'close_raw_unavailable'
    return raw, None


def verify_mark_provenance(mark, directory, require_raw, day=DAY, symbol=None, previous_close=None):
    if not require_raw and 'raw_file' not in mark and 'raw_sha256' not in mark:
        return None
    raw, error = read_raw_provenance(mark, directory)
    if error:
        return error
    try:
        quoted = datetime.fromisoformat(mark['quote_at'])
        observed = datetime.fromisoformat(mark['observed_at'])
        if quoted.utcoffset() is None or observed.utcoffset() is None:
            return 'close_timestamp_unverified'
        local = quoted.astimezone(CST)
        if local.date().isoformat() != day or local.time() < time(15) or observed < quoted:
            return 'close_timestamp_unverified'
    except (ValueError, KeyError, TypeError):
        return 'close_timestamp_unverified'
    if require_raw and symbol:
        try:
            from collect_close_marks import parse_sina, parse_tencent, prior_session

            parsed_day = date.fromisoformat(day)
            if (not isinstance(mark.get('source_url'), str)
                    or urlsplit(mark['source_url']).hostname != 'web.ifzq.gtimg.cn'
                    or not SECURITY_CODE.fullmatch(symbol)):
                return 'close_market_source_unverified'
            prior_day = prior_session(parsed_day)
            if previous_close is None:
                secondary = mark.get('secondary_check')
                if (mark.get('previous_close_basis') != 'dual_source_recovery'
                        or not isinstance(secondary, dict) or secondary.get('status') != 'agreed'
                        or not market_source(secondary.get('source_url'))
                        or urlsplit(secondary['source_url']).hostname != 'quotes.sina.cn'):
                    return 'prior_close_evidence_missing'
                secondary_raw, error = read_raw_provenance(secondary, directory)
                if error:
                    return 'prior_close_secondary_' + error
                reference = parse_sina(secondary_raw, day=prior_day, require_latest=False)
                if reference != Decimal(mark['previous_close']):
                    return 'close_mark_raw_mismatch'
            else:
                reference = Decimal(previous_close)
                if reference != Decimal(mark['previous_close']):
                    return 'prior_close_reference_mismatch'
            raw_price, raw_quote_at = parse_tencent(raw, symbol[-2:].lower() + symbol[:6],
                                                     reference, observed, day=parsed_day, prior_day=prior_day)
            if raw_price != Decimal(mark['close']) or raw_quote_at != quoted:
                return 'close_mark_raw_mismatch'
            if previous_close is None and parse_sina(secondary_raw, day=parsed_day) != raw_price:
                return 'close_mark_raw_mismatch'
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError, IndexError):
            return 'close_raw_payload_unverified'
    return None


def validate_prior(prior, day):
    expected = SESSIONS[:SESSIONS.index(day)]
    if tuple(record.get('date') for record in prior) != expected:
        raise ValueError(f'prior ledger dates must be {list(expected)}')
    if any(record.get('mode') != 'forward' for record in prior):
        raise ValueError('prior ledger must be a forward record')


class AmbiguousPartialLedgers(ValueError):
    def __init__(self, day, count, detail):
        self.count = count
        super().__init__(f'{day} has {count} partial ledgers that cannot be resolved: {detail}')


def partial_ledger_state(path, day):
    """Extract fixed execution and the validated valuation represented by this file."""
    row = json.loads(path.read_text(encoding='utf-8'))
    opening_identity(row, day)
    execution = {key: value for key, value in row.items() if key not in GROUPS}
    marks, known_nav = {}, set()
    valuation_fields = {'nav', 'close_prices', 'close_evidence', 'valuation_gap', 'missing_valuation_dates'}
    for group in GROUPS:
        account = row[group]
        execution[group] = {key: value for key, value in account.items() if key not in valuation_fields}
        positions, prices = account['positions'], account['close_prices']
        gaps = account.get('valuation_gap', {})
        if not all(isinstance(field, dict) for field in (positions, prices, gaps)):
            raise ValueError(f'{group}: invalid position or valuation mapping')
        if (not prices.keys() <= positions.keys() or not gaps.keys() <= positions.keys()
                or any(type(units) is not int or units <= 0 for units in positions.values())):
            raise ValueError(f'{group}: valuation does not match positions')
        cash = Decimal(account['cash'])
        if not cash.is_finite():
            raise ValueError(f'{group}: nonfinite cash')
        total = cash
        for symbol, value in prices.items():
            if value is None:
                continue
            price = Decimal(value)
            if not price.is_finite() or price <= 0 or symbol in gaps:
                raise ValueError(f'{group}: invalid or contradictory mark for {symbol}')
            marks[group, symbol] = price
            total += price * positions[symbol]
        fully_valued = all((group, symbol) in marks for symbol in positions)
        nav = account['nav']
        if (nav is not None) != fully_valued or (nav is not None and Decimal(nav) != total):
            raise ValueError(f'{group}: inconsistent close NAV')
        if nav is not None:
            known_nav.add(group)
    return json.dumps(execution, sort_keys=True, allow_nan=False), marks, known_nav


def daily_ledger_path(folder):
    """Choose an existing ledger whose known valuations cover all compatible retries."""
    day = folder.name
    anchored = folder.parent.resolve() / day
    canonical = folder / 'daily.json'
    paths = ([canonical] if canonical.exists() or canonical.is_symlink()
             else sorted(folder.glob('close-partials/*/daily.json')))
    for path in paths:
        if path.resolve() != anchored / path.relative_to(folder) or not path.is_file():
            raise ValueError(f'{day}: daily ledger path is not canonical')
        if path != canonical and not re.fullmatch(r'[a-f0-9]{64}', path.parent.name):
            raise ValueError(f'{day}: partial ledger directory is not a content digest')
    if len(paths) <= 1:
        return paths[0] if paths else None

    try:
        states = [partial_ledger_state(path, day) for path in paths]
    except INPUT_ERRORS as exc:
        raise AmbiguousPartialLedgers(day, len(paths), f'invalid partial ledger: {exc}') from exc
    if any(execution != states[0][0] for execution, _, _ in states[1:]):
        raise AmbiguousPartialLedgers(day, len(paths), 'execution records disagree')
    all_marks, prices_by_symbol, all_known_nav = set(), {}, set()
    for _, marks, known_nav in states:
        for (_, symbol), price in marks.items():
            previous = prices_by_symbol.setdefault(symbol, price)
            if previous != price:
                raise AmbiguousPartialLedgers(day, len(paths), f'close prices disagree for {symbol}')
        all_marks.update(marks)
        all_known_nav.update(known_nav)
    for path, (_, marks, known_nav) in zip(paths, states):
        if marks.keys() == all_marks and known_nav == all_known_nav:
            return path
    raise AmbiguousPartialLedgers(day, len(paths), 'no ledger contains all verified valuations')


def prior_paths(directory, day):
    """Find official ledgers or compatible partials for previous sessions."""
    paths = []
    for session in SESSIONS[:SESSIONS.index(day)]:
        path = daily_ledger_path(directory / session)
        if path is None:
            raise ValueError(f'{session} has no daily ledger')
        paths.append(path)
    return paths


def prior_records(directory, day):
    records = [json.loads(path.read_text(encoding='utf-8')) for path in prior_paths(directory, day)]
    validate_prior(records, day)
    return records


def opening_identity(opening, day):
    if opening.get('date') != day or opening.get('mode') != 'forward':
        raise ValueError('close requires a forward opening ledger for ' + day)


def session_marks(marks, day):
    if marks is None:
        return {}, {}
    if not isinstance(marks, dict) or marks.get('date') != day or not isinstance(marks.get('marks'), dict):
        raise ValueError('close marks must contain the session date and a marks object')
    errors = marks.get('errors', {})
    if not isinstance(errors, dict):
        raise ValueError('close mark errors must map symbols to failure reasons')
    return marks['marks'], errors


def quote_coverage_summary(opening):
    opening = opening or {}
    audited = opening.get('quote_coverage') or {}
    keys = ('planned_symbols', 'response_row_symbols', 'received_symbols', 'fresh_symbols',
            'two_sided_symbols', 'buy_book_symbols', 'sell_book_symbols')
    symbols = {key: audited.get(key) for key in keys}
    for key in ('planned_symbols', 'received_symbols'):
        if key not in audited:
            symbols[key] = opening.get(key)
    counts = {key.removesuffix('_symbols'): len(values) if values is not None else None
              for key, values in symbols.items()}
    return {**symbols, 'counts': counts}


def close_report(opening, marks=None, *, day=DAY, prior=()):
    """Return (daily ledger or None, report) for one pair of session accounts."""
    if opening is None:
        groups = {
            name: {'buy_count': None, 'sell_count': None, 'fees': None, 'close_nav': None,
                   '513100.SH': {'status': 'opening_unverified', 'premium': None}}
            for name in GROUPS
        }
        return None, {'date': day, 'status': 'opening_unverified',
                      'coverage': quote_coverage_summary(None), 'groups': groups}
    opening_identity(opening, day)
    validate_prior(prior, day)
    marks_by_symbol, mark_errors = session_marks(marks, day)
    daily = {key: value for key, value in opening.items() if key not in GROUPS}
    summaries = {}
    evidence_errors = opening.get('book_evidence_errors') or {}
    for name in GROUPS:
        account = opening[name]
        fills = account['fills']
        positions = account['positions']
        if day == DAY and any(fill['side'] != 'buy' for fill in fills):
            raise ValueError('the experiment opens from cash on Monday and has no sell orders')
        prices, gaps, evidence = {}, {}, {}
        for symbol, units in positions.items():
            if not SECURITY_CODE.fullmatch(symbol) or type(units) is not int or units <= 0:
                raise ValueError(f'invalid position: {symbol}')
            mark = marks_by_symbol.get(symbol)
            price, gap = closing_price(mark, day)
            if not gap and mark.get('provenance_error'):
                gap = mark['provenance_error']
            if gap:
                gaps[symbol] = mark_errors.get(symbol, gap) if mark is None else gap
            else:
                prices[symbol] = str(price)
                evidence[symbol] = mark
        close_nav = None if gaps else str(
            Decimal(account['cash']) + sum(Decimal(prices[s]) * units for s, units in positions.items())
        )
        daily[name] = {**account, 'nav': close_nav, 'close_prices': prices,
                       'close_evidence': evidence, 'valuation_gap': gaps}
        summaries[name] = {
            'buy_count': sum(fill['side'] == 'buy' for fill in fills),
            'sell_count': sum(fill['side'] == 'sell' for fill in fills), 'close_nav': close_nav,
            'valuation_gap': gaps, '513100.SH': etf_decision(account, evidence_errors),
            'unfilled_orders': [],
        }
        for row in account.get('decisions', []):
            if row['reason'] == 'filled':
                continue
            order = {'symbol': row['symbol'], 'reason': row['reason'], **execution_details(row)}
            if detail := quote_evidence_detail(evidence_errors, row):
                order['evidence_error'] = detail
            summaries[name]['unfilled_orders'].append(order)
    reconciled = reconcile([*prior, daily])['groups']
    news_errors = opening.get('news_errors', [])
    news_archive = opening.get('news_archive')
    archive_gap = news_archive and any(news_archive.get(key) for key in
                                       ('missing_dates', 'invalid_dates', 'incomplete_dates'))
    unavailable = [d for d in opening['filtered'].get('decisions', [])
                   if d['reason'] == 'news_data_unavailable']
    unavailable_details = {
        d['symbol']: d['evaluation']['news_error']
        for d in unavailable if (d.get('evaluation') or {}).get('news_error')
    }
    for name in GROUPS:
        total = str(sum((Decimal(fill['fees']) for fill in daily[name]['fills']), Decimal(0)))
        summary_fields = {
            'fees': total, 'cumulative_fees': reconciled[name]['fees'],
            'missing_valuation_dates': reconciled[name]['missing_valuation_dates'],
        }
        summaries[name].update(summary_fields)
        daily[name].update(summary_fields)
    return daily, {
        'date': day, 'status': 'valuation_incomplete' if any(g['valuation_gap'] for g in summaries.values())
        else 'reconciled',
        'coverage': quote_coverage_summary(opening),
        'news_evidence': {'degraded': bool(news_errors or unavailable or archive_gap),
                          'errors': news_errors, 'unavailable_symbols': [d['symbol'] for d in unavailable],
                          **({'unavailable_details': unavailable_details} if unavailable_details else {}),
                          **({'archive': news_archive} if news_archive is not None else {})},
        'groups': summaries,
    }


def write_new_outputs(directory, outputs):
    """Allow identical reruns and refuse to replace different evidence."""
    encoded = {directory / name: json.dumps(value, ensure_ascii=False, indent=2) + '\n'
               for name, value in outputs.items()}
    for path, value in encoded.items():
        if path.exists() and path.read_text(encoding='utf-8') != value:
            raise FileExistsError(f'refusing to overwrite {path}; use another --out directory')
    directory.mkdir(parents=True, exist_ok=True)
    for path, value in encoded.items():
        if path.exists():
            continue
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                         delete=False) as temp:
            temp.write(value)
            staged = Path(temp.name)
        try:
            try:
                os.link(staged, path)
            except FileExistsError:
                if path.read_text(encoding='utf-8') != value:
                    raise
        finally:
            staged.unlink()


def save_report(directory, daily, summary, official):
    """Keep incomplete attempts available without occupying the official ledger paths."""
    if official and summary['status'] != 'reconciled':
        payload = json.dumps({'daily': daily, 'summary': summary}, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        directory = directory / 'close-partials' / digest
    stored_summary = {
        **summary,
        'artifacts': {
            'summary': str(directory / 'close-summary.json'),
            'daily': str(directory / 'daily.json') if daily is not None else None,
        },
    }
    outputs = {'close-summary.json': stored_summary}
    if daily is not None:
        outputs = {'daily.json': daily, **outputs}
    write_new_outputs(directory, outputs)
    return stored_summary


def read_optional_object(path, label):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{label} must contain a JSON object')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default=DAY, choices=SESSIONS)
    parser.add_argument('--opening', type=Path)
    parser.add_argument('--marks', type=Path)
    parser.add_argument('--prior-daily', type=Path, action='append')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    directory = (BASE / args.date).resolve()
    args.opening = args.opening or directory / 'opening.json'
    args.marks = args.marks or directory / 'close_marks.json'
    args.out = (args.out or directory).resolve()
    official = args.out == directory
    if official:
        if args.opening.resolve() != directory / 'opening.json' or args.marks.resolve() != directory / 'close_marks.json':
            parser.error('custom evidence paths require a separate --out directory')
        now = datetime.now(CST)
        if now < datetime.combine(date.fromisoformat(args.date), time(15), tzinfo=CST):
            parser.error('official close is not due')
    try:
        input_errors = {}
        prior_closes = {}
        try:
            opening = read_optional_object(args.opening, 'opening ledger')
            if opening is not None:
                opening_identity(opening, args.date)
                prior_closes = opening.get('prior_closes', {})
                if not isinstance(prior_closes, dict):
                    raise ValueError('opening prior_closes must map symbols to prices')
        except INPUT_ERRORS as exc:
            if not official:
                raise
            opening = None
            input_errors['opening'] = {'path': str(args.opening), 'error': str(exc)}
        prior = []
        if opening is not None:
            if args.prior_daily is None:
                prior = prior_records(BASE, args.date)
            else:
                prior = [json.loads(path.read_text(encoding='utf-8')) for path in args.prior_daily]
                if official and any(not path.resolve().is_relative_to(BASE / record['date'])
                                    for path, record in zip(args.prior_daily, prior)):
                    raise ValueError('official prior ledger must be inside its session directory')
            validate_prior(prior, args.date)
        try:
            marks = read_optional_object(args.marks, 'close marks')
            prices, _ = session_marks(marks, args.date)
        except INPUT_ERRORS as exc:
            if not official:
                raise
            marks, prices = None, {}
            input_errors['close_marks'] = {'path': str(args.marks), 'error': str(exc)}
        if opening is not None and prices:
            if official and args.date == DAY and not prior_closes:
                frozen = json.loads((BASE / 'monday-preparation/candidates.json').read_text(encoding='utf-8'))
                prior_closes = {row['symbol']: row['previous_close'] for row in frozen['candidates']}
            for symbol, mark in prices.items():
                if isinstance(mark, dict):
                    error = verify_mark_provenance(mark, args.marks.parent, require_raw=official,
                                                   day=args.date, symbol=symbol,
                                                   previous_close=prior_closes.get(symbol))
                    if error:
                        mark['provenance_error'] = error
        try:
            if opening is not None and 'close_marks' in input_errors:
                held = {symbol for group in GROUPS for symbol in opening[group]['positions']}
                marks = {'date': args.date, 'marks': {},
                         'errors': dict.fromkeys(held, 'close_marks_unverified')}
            daily, summary = close_report(opening, marks, day=args.date, prior=prior)
        except INPUT_ERRORS as exc:
            if not official:
                raise
            # Do not attribute a prior accounting failure to today's opening.
            if prior:
                reconcile(prior)
            daily, summary = close_report(None, day=args.date)
            input_errors['opening'] = {'path': str(args.opening), 'error': str(exc)}
        if input_errors:
            summary['input_errors'] = input_errors
            if daily is not None:
                summary['status'] = 'valuation_incomplete'
        stored_summary = save_report(args.out, daily, summary, official)
        print(json.dumps(stored_summary, ensure_ascii=False, indent=2))
    except INPUT_ERRORS as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
