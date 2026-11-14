"""Continue the frozen paper study from the preceding reconciled ledger."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_FLOOR

from collect_books import CST, quote_status
from book_evidence import record_quote_request_failure, verify_capture
from close_monday import SESSIONS, prior_paths, validate_prior
from monday_open import BASE, DAY, IOPV_SYMBOL, PROTOCOL, decide, news_snapshot, publish_once, tick_price
from quant_research.ledger import Costs
from reconcile import reconcile

GROUPS = ('baseline', 'filtered')


def positive(value):
    number = Decimal(value)
    if not number.is_finite() or number <= 0:
        raise ValueError('expected a positive price')
    return number


def sale(day, symbol, units, book, previous_close):
    if book is None:
        return 'quote_unavailable', None
    try:
        received = datetime.fromisoformat(book['received_at'])
        if received.utcoffset() is None:
            return 'invalid_quote', None
        received = received.astimezone(CST)
        status = quote_status(book, received, day)
        if status != 'fresh':
            return status, None
        if received.date() != day or not time(9, 45) <= received.time() <= time(9, 50):
            return 'outside_window', None
        if book.get('traded_volume') == 0:
            return 'no_session_trades', None
        available = book['bid_units']
        if type(available) is not int or available < 0:
            return 'invalid_quote', None
        if available == 0:
            return 'no_bid_liquidity', None
        if available < units:
            return 'insufficient_bid_units', None
        if previous_close is None or positive(book['previous']) != positive(previous_close):
            return 'corporate_action_or_previous_close_unverified', None
        if not book.get('source_url', '').startswith('https://qt.gtimg.cn/q='):
            return 'price_limits_unverified', None
        upper, lower = positive(book['upper_limit']), positive(book['lower_limit'])
        kind = 'etf' if symbol.startswith(('15', '51', '56', '58')) else 'stock'
        tick = Decimal('.001' if kind == 'etf' else '.01')
        bid = positive(book['bid'])
        px = tick_price(bid * Decimal('.9995'), tick, ROUND_FLOOR)
        if lower > upper or not lower <= bid <= upper or not lower <= px <= upper:
            return 'sell_outside_price_limits', None
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return 'invalid_quote_or_price_limits', None
    gross = px * units
    fees = [Decimal(str(fee)) for fee in Costs().charges(day, kind, 'sell', float(gross))]
    total = sum(fees)
    return 'filled', dict(symbol=symbol, side='sell', units=units, price=str(px), fees=str(total),
                          cash_delta=str(gross-total), commission=str(fees[0]), stamp_tax=str(fees[1]),
                          transfer_fee=str(fees[2]))


def held_prior_close(symbol, previous, prepared_references):
    references = list(prepared_references.get(symbol, ()))
    for group in GROUPS:
        account = previous[group]
        if symbol in account['positions'] and symbol not in (account.get('valuation_gap') or {}):
            mark = account.get('close_prices', {}).get(symbol)
            if mark is not None:
                references.append(mark)
    try:
        values = {positive(reference) for reference in references}
    except (ValueError, TypeError, ArithmeticError):
        return None
    return str(values.pop()) if len(values) == 1 else None


def run_day(day, previous, prepared, captured, snapshot):
    accounts = {}
    candidates = prepared['candidates'] if prepared else []
    books, unavailable = captured['selected'], captured['unavailable']
    exit_books = captured.get('exit_selected', books)
    last_observations = captured.get('last_observations', {})
    quote_coverage = captured.get('quote_coverage')
    prior_closes = {row['symbol']: row['previous_close'] for row in candidates}
    prepared_references = {symbol: [close] for symbol, close in prior_closes.items()}
    for row in prepared.get('reviewed', ()) if prepared else ():
        if 'previous_close' in row and row.get('sources'):
            prepared_references.setdefault(row['symbol'], []).append(row['previous_close'])
    held = {symbol for group in GROUPS for symbol in previous[group]['positions']}
    for symbol in held:
        prior_closes[symbol] = held_prior_close(symbol, previous, prepared_references)
    for group in GROUPS:
        prior = previous[group]
        cash, positions = Decimal(prior['cash']), dict(prior['positions'])
        fills, decisions = [], []
        last_exit = None
        for symbol, units in sorted(positions.items()):
            book = exit_books.get(symbol)
            reason, fill = sale(day, symbol, units, book, prior_closes[symbol])
            if book is None:
                reason = captured.get('exit_unavailable', {}).get(symbol, reason)
            if fill:
                cash += Decimal(fill['cash_delta'])
                del positions[symbol]
                fills.append(fill)
                received = datetime.fromisoformat(book['received_at'])
                last_exit = received if last_exit is None else max(last_exit, received)
            decision = dict(symbol=symbol, side='sell', reason=reason, book=book, required_units=units,
                            available_units=book.get('bid_units') if book else None)
            if book is None and symbol in last_observations:
                decision['last_observation'] = last_observations[symbol]
            record_quote_request_failure(captured, decision)
            decisions.append(decision)
        nav = prior.get('nav')
        budget = min(Decimal(5000), Decimal(nav) * Decimal('.05')) if nav is not None else None
        for row in candidates:
            symbol = row['symbol']
            book = books.get(symbol)
            premium, evaluation, fill = None, None, None
            if symbol in positions:
                reason = 'already_held'
            elif budget is None:
                reason = 'previous_nav_unverified'
            elif book is None:
                reason = unavailable.get(symbol, 'quote_unavailable')
            else:
                try:
                    received = datetime.fromisoformat(book['received_at'])
                    if last_exit is not None and received < last_exit:
                        reason = 'entry_quote_precedes_exit'
                    else:
                        reason, evaluation, fill, premium = decide(
                            row, book, snapshot['batches'], snapshot['errors'], day=day,
                            news_captured_at=snapshot.get('news_archive', {}).get('captured_at'),
                            prior_day=day-timedelta(days=1), budget=budget)
                except (ValueError, KeyError, TypeError, ArithmeticError):
                    reason = 'invalid_quote'
                if evaluation is not None:
                    verdict = evaluation['baseline'] if group == 'baseline' else evaluation
                    reason = verdict['reason']
                    if verdict['eligible']:
                        balance = cash + Decimal(fill['cash_delta'])
                        if balance < 0:
                            reason = 'insufficient_cash'
                        else:
                            reason, cash = 'filled', balance
                            positions[symbol] = fill['units']
                            fills.append(fill)
            decision = dict(symbol=symbol, side='buy', reason=reason, book=book, premium=premium,
                            required_units=fill['units'] if fill else None,
                            available_units=book.get('ask_units') if book else None, evaluation=evaluation)
            if book is None and symbol in last_observations:
                decision['last_observation'] = last_observations[symbol]
            record_quote_request_failure(captured, decision)
            decisions.append(decision)
        accounts[group] = dict(cash=str(cash), positions=positions, fills=fills, decisions=decisions)
    return {'mode': 'forward', 'date': str(day), 'prior_closes': prior_closes,
            'candidate_coverage': prepared.get('coverage') if prepared else 'unavailable',
            'candidate_data_missing': not prepared or (
                'reviewed' in prepared and not any(row.get('status') in ('eligible', 'signal_not_triggered')
                                                   for row in prepared['reviewed'])
            ),
            'candidate_symbols': [row['symbol'] for row in candidates],
            'expected_sell_symbols': {group: sorted(previous[group]['positions']) for group in GROUPS},
            'planned_symbols': sorted(held | {row['symbol'] for row in candidates}),
            'received_symbols': sorted({s for a in captured.get('attempts', []) for s in a.get('books', {})}),
            'fresh_symbols': quote_coverage['fresh_symbols'] if quote_coverage is not None
                             else sorted(books.keys() | exit_books.keys()),
            **({'quote_coverage': quote_coverage} if quote_coverage is not None else {}),
            'news_sources': snapshot['sources'], 'news_errors': snapshot['errors'],
            **({'news_archive': snapshot['news_archive']} if 'news_archive' in snapshot else {}),
            **accounts}


def verify_saved_opening(path, day):
    """Check the recorded prior ledger without rerunning a published opening."""
    opening = json.loads(path.read_bytes())
    previous_day = str(day - timedelta(days=1))
    if not isinstance(opening, dict) or opening.get('date') != str(day) or opening.get('mode') != 'forward':
        raise ValueError('existing opening is not a forward ledger for the requested date')
    for group in GROUPS:
        account = opening.get(group)
        if (not isinstance(account, dict) or not isinstance(account.get('positions'), dict)
                or not isinstance(account.get('fills'), list) or not isinstance(account.get('cash'), str)):
            raise ValueError(f'existing opening has an invalid {group} account')
    source = opening.get('starting_from')
    relative = source.get('path') if isinstance(source, dict) else None
    expected = rf'{previous_day}/(?:close-partials/[a-f0-9]{{64}}/)?daily[.]json'
    if not isinstance(relative, str) or not re.fullmatch(expected, relative) or source.get('date') != previous_day:
        raise ValueError('existing opening has an invalid prior ledger reference')
    prior = BASE / relative
    if not prior.is_file() or prior.resolve() != prior or hashlib.sha256(prior.read_bytes()).hexdigest() != source.get('sha256'):
        raise ValueError('existing opening prior ledger is missing or its hash has changed')


def verified_input(reference, allowed_path, label):
    relative = reference.get('path') if isinstance(reference, dict) else None
    if not isinstance(relative, str) or not re.fullmatch(allowed_path, relative):
        raise ValueError(f'invalid pinned {label} path')
    path = BASE / relative
    if path.resolve() != path or not path.is_file():
        raise ValueError(f'pinned {label} file is missing or noncanonical')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference.get('sha256'):
        raise ValueError(f'pinned {label} hash has changed')
    return json.loads(payload)


def load_opening_inputs(manifest, day):
    start = datetime.combine(day, time(9, 45), CST)
    end = datetime.combine(day, time(9, 50), CST)
    if not isinstance(manifest, dict) or manifest.get('mode') != 'forward' or manifest.get('date') != str(day):
        raise ValueError('invalid opening inputs manifest for the requested date')
    stamp = manifest.get('pinned_at')
    if not isinstance(stamp, str):
        raise ValueError('opening inputs manifest has no pin time')
    pinned_at = datetime.fromisoformat(stamp)
    if pinned_at.utcoffset() is None or not start <= pinned_at <= end:
        raise ValueError('opening inputs must be pinned in the capture window')
    expected = SESSIONS[:SESSIONS.index(str(day))]
    references = manifest.get('prior_ledgers')
    if not isinstance(references, list) or len(references) != len(expected):
        raise ValueError('opening inputs manifest has invalid prior ledger coverage')
    records = [
        verified_input(ref, rf'{session}/(?:close-partials/[a-f0-9]{{64}}/)?daily[.]json', 'prior ledger')
        for session, ref in zip(expected, references)
    ]
    validate_prior(records, str(day))
    reconcile(records)
    if 'prepared' not in manifest:
        raise ValueError('opening inputs manifest has no candidate reference')
    prepared = None
    if manifest['prepared'] is not None:
        prepared = verified_input(manifest['prepared'], rf'{day}-preparation/candidates[.]json', 'candidate')
        if (not isinstance(prepared, dict) or prepared.get('date') != str(day)
                or not isinstance(prepared.get('captured_at'), str)
                or not isinstance(prepared.get('candidates'), list)):
            raise ValueError('invalid pinned candidate evidence')
        captured_at = datetime.fromisoformat(prepared['captured_at'])
        if captured_at.utcoffset() is None or captured_at >= start:
            raise ValueError('candidate evidence was not available before opening')
    return records[-1], prepared, pinned_at


def opening_inputs(path, day, books):
    """Publish inputs before sampling. Recover only with the first recorded inputs."""
    def read_saved():
        if path.resolve() != path:
            raise ValueError('opening inputs manifest path is not canonical')
        return json.loads(path.read_bytes())

    if path.exists() or path.is_symlink():
        return read_saved()
    if (datetime.now(CST) > datetime.combine(day, time(9, 50), CST)
            or any((books / name).exists() for name in ('books.json', 'capture-progress.json'))):
        raise ValueError('opening inputs were not pinned before capture')

    def reference(input_path):
        if input_path.resolve() != input_path or not input_path.is_file():
            raise ValueError('opening input path is not a canonical file')
        return {'path': str(input_path.relative_to(BASE)),
                'sha256': hashlib.sha256(input_path.read_bytes()).hexdigest()}

    prior = [reference(prior_path) for prior_path in prior_paths(BASE, str(day))]
    candidate = BASE / f'{day}-preparation/candidates.json'
    prepared = reference(candidate) if candidate.exists() or candidate.is_symlink() else None
    manifest = {'mode': 'forward', 'date': str(day), 'pinned_at': datetime.now(CST).isoformat(),
                'prior_ledgers': prior, 'prepared': prepared}
    load_opening_inputs(manifest, day)
    return publish_once(path, manifest, read_existing=read_saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True, type=date.fromisoformat)
    day = parser.parse_args().date
    if not DAY < day <= date(2026, 11, 20):
        parser.error('use this entry point for November 17–20')
    if hashlib.sha256((BASE/'protocol.md').read_bytes()).hexdigest() != PROTOCOL:
        parser.error('frozen protocol hash mismatch')
    now = datetime.now(CST)
    if now < datetime.combine(day, time(9, 45), CST):
        parser.error('opening is not due')
    out = BASE / str(day)
    published = out / 'opening.json'
    if published.exists() or published.is_symlink():
        if published.resolve() != published:
            raise ValueError('existing opening path is not canonical')
        verify_saved_opening(published, day)
        return
    books = out/'books'
    out.mkdir(exist_ok=True)
    manifest = opening_inputs(out/'opening-inputs.json', day, books)
    previous, prepared, pinned_at = load_opening_inputs(manifest, day)
    snapshot = news_snapshot(out/'opening-news.json', day)
    held = {symbol for group in GROUPS for symbol in previous[group]['positions']}
    buy_symbols = {row['symbol'] for row in (prepared['candidates'] if prepared else [])}
    symbols = sorted(held | buy_symbols)
    end = datetime.combine(day, time(9, 50), CST)
    if not symbols:
        captured = {'mode': 'forward', 'date': str(day), 'selected': {}, 'unavailable': {}, 'attempts': []}
    else:
        if not (books/'books.json').is_file():
            if datetime.now(CST) > end and not (books/'capture-progress.json').is_file():
                raise ValueError('no intraday capture to replay')
            limits = [arg for symbol in sorted(held) for arg in ('--require-limits', symbol)]
            subprocess.run([sys.executable, str(BASE/'collect_books.py'), '--date', str(day), '--out', str(books),
                            '--require-513100-iopv', *limits, *symbols], check=True)
        captured = json.loads((books/'books.json').read_text())
    captured = verify_capture(captured, books, day, symbols, held=held, buy_symbols=buy_symbols)
    start = datetime.combine(day, time(9, 45), CST)
    for attempt in captured['attempts']:
        if attempt.get('evidence_error') or not attempt.get('books'):
            continue
        received_at = datetime.fromisoformat(attempt['retrieved_at'])
        if start <= received_at < pinned_at:
            raise ValueError('archived opening quote was received before the inputs were pinned')
    result = run_day(day, previous, prepared, captured, snapshot)
    result['book_evidence_errors'] = captured.get('evidence_errors', {})
    prior = manifest['prior_ledgers'][-1]
    result['starting_from'] = {'date': previous['date'], 'path': prior['path'], 'sha256': prior['sha256']}
    publish_once(out/'opening.json', result)


if __name__ == '__main__':
    main()
