"""Run the frozen first-day simulation from the precomputed common shortlist."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

from collect_books import CST, IOPV_SYMBOL, quote_status
from book_evidence import record_quote_request_failure, verify_capture
from scheduled_open import capture_directory
from quant_research.ledger import Costs
from quant_research.news_entry import compare_news_entry
from quant_research.paper_entry import evaluate_paper_buy

BASE = Path(__file__).resolve().parent
DAY = date(2026, 11, 16)
INITIAL = Decimal('100000')
PROTOCOL = '477e3dcfcf51b4b21ddb3fd51ebee38cdfb126c4fcb5c9fc8b715da4dcf692fe'


def tick_price(value, tick, rounding):
    return (value / tick).to_integral_value(rounding=rounding) * tick


def iopv_update_verified(book, nav, day=DAY):
    prior = book.get('iopv_update')
    if not isinstance(prior, dict):
        return False
    try:
        earlier = datetime.fromisoformat(prior['previous_at'])
        earlier_received = datetime.fromisoformat(prior['previous_received_at'])
        current = datetime.fromisoformat(book['at'])
        received = datetime.fromisoformat(book['received_at'])
        earlier_nav = Decimal(prior['previous_nav'])
        if quote_status({'at': prior['previous_at']}, earlier_received, day) != 'fresh':
            return False
        earlier_file = prior['previous_raw_file']
        current_file = book['raw_file']
        return (
            earlier < current and earlier_received < received
            and earlier_nav.is_finite() and earlier_nav > 0 and earlier_nav != nav
            and bool(earlier_file and current_file) and earlier_file != current_file
        )
    except (ValueError, TypeError, KeyError, ArithmeticError):
        return False


def decide(row, book, news, news_errors=(), *, news_captured_at=None,
           day=DAY, prior_day=date(2026, 11, 13), budget=Decimal(5000)):
    received = datetime.fromisoformat(book['received_at'])
    if received.utcoffset() is None:
        return 'invalid_quote', None, None, None
    received = received.astimezone(CST)
    if not time(9, 45) <= received.time() <= time(9, 50) or received.date() != day:
        return 'outside_window', None, None, None
    status = quote_status(book, received, day)
    if status != 'fresh':
        return status, None, None, None
    name = str(book.get('name', '')).upper().replace(' ', '')
    if name.startswith(('ST', '*ST', 'SST', 'S*ST', '退市')) or name.endswith('退'):
        return 'risk_warning_or_delisting', None, None, None
    if book.get('traded_volume') == 0:
        return 'no_session_trades', None, None, None
    ask_units = book['ask_units']
    if type(ask_units) is not int or ask_units < 0:
        return 'invalid_quote', None, None, None
    if ask_units == 0:
        return 'no_ask_liquidity', None, None, None
    tick = Decimal(row['tick'])
    ask = Decimal(book['ask'])
    previous = Decimal(row['previous_close'])
    if not ask.is_finite():
        return 'invalid_quote', None, None, None
    if ask <= 0 or Decimal(book['previous']) != previous:
        return 'invalid_ask_or_previous_close_mismatch', None, None, None
    price = tick_price(ask * Decimal('1.0005'), tick, ROUND_CEILING)
    premium = None
    if row['cross_border']:
        if row['symbol'] != IOPV_SYMBOL or not book.get('source_url', '').startswith('https://qt.gtimg.cn/q='):
            return 'premium_not_verified', None, None, None
        evidence = book.get('iopv')
        if not isinstance(evidence, dict):
            return 'iopv_missing', None, None, None
        if evidence.get('status') != 'verified_snapshot':
            return evidence.get('status', 'iopv_missing'), None, None, None
        nav = Decimal(evidence['nav'])
        if not nav.is_finite() or nav <= 0:
            return 'iopv_invalid', None, None, None
        if not iopv_update_verified(book, nav, day):
            return 'iopv_intraday_update_unverified', None, None, None
        premium = str(price / nav - 1)
        if price > nav * Decimal('1.01'):
            return 'premium_above_limit', None, None, premium
    limit = tick_price(previous * Decimal('1.01'), tick, ROUND_FLOOR)
    for units in range(int(budget // (price * 100)) * 100, 0, -100):
        gross = price * units
        fees = tuple(Decimal(str(v)) for v in Costs().charges(day, row['kind'], 'buy', float(gross)))
        total = sum(fees)
        if gross + total <= budget:
            break
    else:
        return 'budget_below_one_lot', None, None, premium
    observation = dict(symbol=row['symbol'], as_of=book['received_at'], signal=True,
                       signal_at=f'{prior_day}T15:00:00+08:00', expires_at=f'{day}T09:50:01+08:00',
                       limit_price=str(limit), ordered_units=units, max_quote_age_seconds=30,
                       quote=dict(symbol=row['symbol'], at=book['at'], ask_price=str(price),
                                  ask_units=ask_units))
    baseline = evaluate_paper_buy(observation)
    try:
        if news_errors:
            raise ValueError(', '.join(map(str, news_errors)))
        if news_captured_at is not None and datetime.fromisoformat(news_captured_at) > received:
            raise ValueError('saved news snapshot was captured after the quote was received')
        decision = compare_news_entry(observation, news, available_by=news_captured_at)
    except (ValueError, TypeError, KeyError) as exc:
        decision = {
            **baseline,
            'baseline': {key: baseline[key] for key in ('eligible', 'reason')},
            'eligible': False,
            'reason': 'news_data_unavailable' if baseline['eligible'] else baseline['reason'],
            'news_error': str(exc),
        }
    fill = dict(symbol=row['symbol'], units=units, side='buy', price=str(price),
                commission=str(fees[0]), stamp_tax=str(fees[1]), transfer_fee=str(fees[2]),
                fees=str(total), cash_delta=str(-gross-total))
    return decision['reason'], decision, fill, premium


def run(prepared, books, news, *, news_errors=(), news_captured_at=None):
    output = {group: dict(cash=str(INITIAL), positions={}, fills=[], decisions=[])
              for group in ('baseline', 'filtered')}
    attempts = books.get('attempts', [])
    quote_coverage = books.get('quote_coverage')
    last_observations = books.get('last_observations', {})
    for row in prepared['candidates']:
        symbol = row['symbol']
        book = books['selected'].get(symbol)
        premium = None
        quote_error = None
        if book is None:
            reason, decision, fill = books['unavailable'].get(symbol, 'quote_unavailable'), None, None
        else:
            try:
                reason, decision, fill, premium = decide(
                    row, book, news, news_errors, news_captured_at=news_captured_at)
            except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
                reason, decision, fill = 'invalid_quote', None, None
                quote_error = str(exc)
        for group, account in output.items():
            verdict = decision['baseline'] if decision and group == 'baseline' else decision
            label = verdict['reason'] if verdict else reason
            if verdict and verdict['eligible']:
                cash = Decimal(account['cash']) + Decimal(fill['cash_delta'])
                if cash < 0:
                    label = 'insufficient_cash'
                else:
                    account['cash'] = str(cash)
                    account['positions'][symbol] = fill['units']
                    account['fills'].append(fill)
                    label = 'filled'
            entry = dict(symbol=symbol, side='buy', reason=label, book=book, premium=premium,
                         required_units=fill['units'] if fill else None,
                         available_units=book.get('ask_units') if book else None,
                         evaluation=verdict, quote_error=quote_error)
            if book is None and symbol in last_observations:
                entry['last_observation'] = last_observations[symbol]
            record_quote_request_failure(books, entry)
            account['decisions'].append(entry)
    return {'date': str(DAY), 'coverage': prepared['coverage'], 'news_errors': list(news_errors),
            'prior_closes': {row['symbol']: row['previous_close'] for row in prepared['candidates']},
            'candidate_symbols': [row['symbol'] for row in prepared['candidates']],
            'expected_sell_symbols': {group: [] for group in output},
            'planned_symbols': [row['symbol'] for row in prepared['candidates']],
            'received_symbols': sorted({symbol for a in attempts
                                        for symbol in a.get('books', {})}),
            'fresh_symbols': quote_coverage['fresh_symbols'] if quote_coverage is not None
                             else sorted(books['selected']),
            **({'quote_coverage': quote_coverage} if quote_coverage is not None else {}), **output}


def publish_once(output, data, *, read_existing=None):
    """Save complete JSON without replacing a previously published artifact."""
    with tempfile.NamedTemporaryFile(dir=output.parent, mode='w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.flush()
        try:
            os.link(file.name, output)
        except FileExistsError:
            if read_existing is not None:
                return read_existing()
            if json.loads(output.read_text()) != data:
                raise ValueError(f'{output.name} already exists with different content')
    return data


def unavailable_news(output, error):
    return {'batches': [], 'errors': [f'{output.name}: {error}'], 'sources': []}


def read_news_snapshot(output, day):
    """Read the recorded snapshot without replacing damaged evidence."""
    try:
        snapshot = json.loads(output.read_text(encoding='utf-8'))
        if not isinstance(snapshot, dict) or not all(
            isinstance(snapshot.get(key), list) for key in ('batches', 'errors', 'sources')
        ):
            raise ValueError('expected batches, errors and sources lists')
        batches, errors = snapshot['batches'], snapshot['errors']
        if not batches and not errors:
            raise ValueError('snapshot has no news batches or recorded errors')
        if not all(isinstance(batch, dict) and isinstance(batch.get('events'), list) for batch in batches):
            raise ValueError('expected a news batch with an events list')
        if not all(isinstance(error, str) and error.strip() for error in errors):
            raise ValueError('expected nonempty news error messages')
        archive = snapshot.get('news_archive', {})
        if not isinstance(archive, dict):
            raise ValueError('expected a news_archive object')
        captured_at = archive.get('captured_at')
        if captured_at is None and errors and not batches:
            return snapshot
        if not isinstance(captured_at, str):
            raise ValueError('snapshot capture timestamp is missing')
        captured = datetime.fromisoformat(captured_at)
        if captured.utcoffset() is None:
            raise ValueError('snapshot capture timestamp lacks a time zone')
        captured = captured.astimezone(CST)
        if captured.date() != day or captured.time() > time(9, 50):
            raise ValueError('snapshot capture timestamp is outside the opening session')
        return snapshot
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError) as exc:
        return unavailable_news(output, exc)


def news_snapshot(output, day=DAY):
    saved = read_news_snapshot(output, day)
    if saved is not None:
        return saved
    first_day = date(2026, 11, 13)
    now = datetime.now(CST)
    deadline = datetime.combine(day, time(9, 50), CST)
    expected = [
        str(previous) for offset in (3, 2, 1)
        if (previous := day - timedelta(days=offset)) >= first_day
    ]
    archive = {'captured_at': now.isoformat(), 'expected_dates': expected,
               'loaded_dates': [], 'empty_dates': [], 'missing_dates': [],
               'invalid_dates': [], 'incomplete_dates': [], 'collection_status': {}}
    snapshot = {'batches': [], 'errors': [], 'sources': [], 'news_archive': archive}
    def publish():
        def read_first_writer():
            saved = read_news_snapshot(output, day)
            return saved if saved is not None else unavailable_news(output, 'saved snapshot disappeared')

        try:
            return publish_once(output, snapshot, read_existing=read_first_writer)
        except OSError as exc:
            return unavailable_news(output, exc)

    if now > deadline:
        archive.update(captured_at=None, checked_at=now.isoformat())
        snapshot['errors'].append('no news snapshot was saved during the opening window')
        return publish()

    directory = BASE.parent / 'news-forward'
    for offset in range((day - first_day).days + 1):
        batch_date = str(first_day + timedelta(days=offset))
        path = directory / f'{batch_date}.json'
        try:
            payload = path.read_bytes()
            batch = json.loads(payload)
            if not isinstance(batch, dict) or batch.get('date') != batch_date or not isinstance(batch.get('events'), list):
                raise ValueError('news batch requires matching date and an events list')
            status = batch.get('collection_status', 'legacy')
            if 'collection_status' in batch and status not in ('complete', 'partial', 'failed'):
                raise ValueError('unrecognized collection_status')
        except FileNotFoundError:
            if batch_date in expected:
                archive['missing_dates'].append(batch_date)
            continue
        except (OSError, ValueError) as exc:
            archive['invalid_dates'].append(batch_date)
            if batch_date in expected or batch_date == str(day):
                snapshot['errors'].append(f'{path.name}: {exc}')
            continue
        archive['loaded_dates'].append(batch_date)
        archive['collection_status'][batch_date] = status
        snapshot['sources'].append({'file': path.name, 'sha256': hashlib.sha256(payload).hexdigest()})
        if status in ('legacy', 'partial', 'failed'):
            archive['incomplete_dates'].append(batch_date)
        elif not batch['events']:
            archive['empty_dates'].append(batch_date)
        if status != 'failed':
            snapshot['batches'].append(batch)

    captured = datetime.now(CST)
    if captured.date() != day or captured > deadline:
        archive.update(captured_at=None, checked_at=captured.isoformat())
        snapshot['batches'].clear()
        snapshot['errors'].append('archive reads did not finish during the opening window')
    else:
        archive['captured_at'] = captured.isoformat()
    if not snapshot['batches'] and not snapshot['errors']:
        snapshot['errors'].append('no news batches available')
    return publish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-ready', action='store_true')
    args = parser.parse_args()
    if hashlib.sha256((BASE/'protocol.md').read_bytes()).hexdigest() != PROTOCOL:
        raise ValueError('frozen protocol hash mismatch')
    shortlist = (BASE/'monday-preparation/candidates.json').read_bytes()
    if hashlib.sha256(shortlist).hexdigest() != '137ed04d32777fd23db94bad04be9562d17a8262531f34f71f9272bb24439d7c':
        raise ValueError('Monday shortlist hash mismatch')
    prepared = json.loads(shortlist)
    if prepared['date'] != str(DAY) or not 1 <= len(prepared['candidates']) <= 10:
        raise ValueError('Monday shortlist unavailable')
    symbols = [row['symbol'] for row in prepared['candidates']]
    if len(set(symbols)) != len(symbols):
        raise ValueError('duplicate shortlist symbol')
    if args.check_ready:
        print(json.dumps({'mode': 'readiness', **run(prepared, {'selected': {}, 'unavailable': {}}, [])}, indent=2))
        return
    now = datetime.now(CST)
    if now.date() < DAY or (now.date() == DAY and now.time() < time(9, 45)):
        raise ValueError('outside Monday capture window')
    out = BASE / str(DAY)
    out.mkdir(exist_ok=True)
    snapshot = news_snapshot(out/'opening-news.json')
    books = capture_directory(out)
    if not (books/'books.json').exists():
        if not (books/'capture-progress.json').is_file() and (now.date() != DAY or now.time() > time(9, 50)):
            raise ValueError('no completed Monday capture to replay')
        subprocess.run([sys.executable, str(BASE/'collect_books.py'), '--date', str(DAY),
                        '--out', str(books), '--require-513100-iopv', *symbols], check=True)
    captured = json.loads((books/'books.json').read_text())
    captured = verify_capture(captured, books, DAY, symbols)
    result = {'mode': 'forward', 'capture_directory': books.name, 'news_sources': snapshot['sources'],
              'book_evidence_errors': captured['evidence_errors'],
              **({'news_archive': snapshot['news_archive']} if 'news_archive' in snapshot else {}),
              **run(prepared, captured, snapshot['batches'], news_errors=snapshot['errors'],
                    news_captured_at=snapshot.get('news_archive', {}).get('captured_at'))}
    publish_once(out/'opening.json', result)


if __name__ == '__main__':
    main()
