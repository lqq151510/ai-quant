"""Run the local original-rule and cash-only paper pilot."""

import argparse
import fcntl
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time
from decimal import Decimal

from config import CST, DAYS, ROOT, candidate_path, load_candidates, previous_session, sha256_file

import collect_books
from book_evidence import verify_capture
from close_monday import closing_price, verify_mark_provenance
from collect_close_marks import collect
from continue_open import run_day
from monday_open import publish_once
from reconcile import INITIAL, reconcile


OPEN = time(9, 45)
OPEN_END = time(9, 50)
CLOSE = time(15)
EMPTY_ACCOUNT = {'cash': str(INITIAL), 'positions': {}, 'fills': [], 'decisions': []}
CASH_CONTROL = {**EMPTY_ACCOUNT, 'nav': str(INITIAL), 'close_prices': {}, 'valuation_gap': {}}


def read_json(path):
    if path.resolve() != path or not path.is_file():
        raise ValueError(f'missing or noncanonical evidence: {path.relative_to(ROOT)}')
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise ValueError(f'{path.name} must be a JSON object')
    return data


def metrics(records):
    if not records:
        return None
    groups = reconcile([dict(date=row['date'], baseline=row['baseline'], filtered=row['cash_control'])
                        for row in records])['groups']
    return {'baseline': groups['baseline'], 'cash_control': groups['filtered']}


def session_candidate(path, folder):
    if not path.is_relative_to(folder) or path.resolve() != path:
        raise ValueError(f'{folder.name}: candidate file must be inside the session directory')
    return path


def candidate_gaps(prepared):
    return sorted(row['symbol'] for row in prepared['reviewed']
                  if row['status'] in ('data_gap', 'turnover_threshold_uncertain'))


def read_daily(day, folder, prior_hashes):
    daily = read_json(folder / 'daily.json')
    if daily['date'] != str(day) or daily['mode'] != 'forward':
        raise ValueError(f'{day}: wrong daily ledger identity')
    for filename, digest in daily['inputs'].items():
        if filename not in {'opening.json', 'close-progress.json'}:
            raise ValueError(f'{day}: unexpected daily input {filename}')
        path = folder / filename
        read_json(path)
        if sha256_file(path) != digest:
            raise ValueError(f'{day}: {filename} changed after settlement')
    if set(daily['inputs']) != {'opening.json', 'close-progress.json'}:
        raise ValueError(f'{day}: missing daily input hashes')
    opening = read_json(folder / 'opening.json')
    if opening['date'] != str(day) or opening['mode'] != 'forward':
        raise ValueError(f'{day}: wrong opening ledger identity')
    manifest = read_json(folder / 'opening-inputs.json')
    if manifest['date'] != str(day) or manifest['prior_daily_sha256'] != prior_hashes:
        raise ValueError(f'{day}: opening manifest does not match the prior ledger chain')
    candidates = session_candidate(folder / manifest['candidates_file'], folder)
    read_json(candidates)
    if sha256_file(candidates) != manifest['candidates_sha256']:
        raise ValueError(f'{day}: candidates changed after input pinning')
    required = {'opening-inputs.json', 'books/books.json'} if manifest['symbols'] else {'opening-inputs.json'}
    if set(opening['inputs']) != required:
        raise ValueError(f'{day}: missing opening input hashes')
    for filename, digest in opening['inputs'].items():
        if filename not in {'opening-inputs.json', 'books/books.json'}:
            raise ValueError(f'{day}: unexpected opening input {filename}')
        path = folder / filename
        read_json(path)
        if sha256_file(path) != digest:
            raise ValueError(f'{day}: {filename} changed after opening')
    if daily['book_gaps'] != opening['book_gaps'] or daily['candidate_gaps'] != opening['candidate_gaps']:
        raise ValueError(f'{day}: opening and daily evidence gaps differ')
    if any(daily['baseline'][key] != opening['baseline'][key]
           for key in ('cash', 'positions', 'fills', 'decisions')):
        raise ValueError(f'{day}: opening and daily account differ')
    if daily['cash_control'] != CASH_CONTROL:
        raise ValueError(f'{day}: cash control differs from initial cash')
    return daily


def prior_days(day):
    return DAYS[:DAYS.index(day)]


def prior_state(day):
    records, hashes = [], {}
    for prior in prior_days(day):
        folder = ROOT / str(prior)
        records.append(read_daily(prior, folder, hashes))
        hashes[str(prior)] = sha256_file(folder / 'daily.json')
    metrics(records)
    return records, hashes


def open_session(day, folder, now):
    start, end = (datetime.combine(day, value, CST) for value in (OPEN, OPEN_END))
    books_dir = folder / 'books'
    book_path = books_dir / 'books.json'
    manifest_path = folder / 'opening-inputs.json'
    if now < start:
        raise ValueError(f'{day}: waiting for the 09:45–09:50 opening window')
    saved_capture = book_path.is_file() or (books_dir / 'capture-progress.json').is_file()
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    if now > end and manifest is None:
        raise ValueError(f'{day}: opening window missed, no pinned inputs')
    if now > end and not saved_capture and manifest['symbols']:
        raise ValueError(f'{day}: opening window missed, no saved market capture')

    candidates_file = folder / manifest['candidates_file'] if manifest else candidate_path(day)
    candidates_file = session_candidate(candidates_file, folder)
    prepared = load_candidates(day, candidates_file)
    records, prior_hashes = prior_state(day)
    prior = records[-1]['baseline'] if records else CASH_CONTROL
    held = sorted(prior['positions'])
    buy = sorted(row['symbol'] for row in prepared['candidates'])
    symbols = sorted(set(held) | set(buy))
    gaps = candidate_gaps(prepared)
    if not symbols and gaps:
        raise ValueError(f'{day}: incomplete candidate data cannot establish a no-trade session')
    inputs = dict(date=str(day), candidates_file=candidates_file.relative_to(folder).as_posix(),
                  candidates_sha256=sha256_file(candidates_file),
                  prior_daily_sha256=prior_hashes, symbols=symbols, held=held, candidates=buy)
    if manifest is not None:
        pinned = datetime.fromisoformat(manifest['pinned_at'])
        if pinned.utcoffset() is None or not start <= pinned <= end:
            raise ValueError(f'{day}: input manifest timestamp outside the opening window')
        if manifest != {**inputs, 'pinned_at': manifest['pinned_at']}:
            raise ValueError(f'{day}: candidate or prior ledger inputs changed after pinning')
    else:
        if saved_capture:
            raise ValueError(f'{day}: no input manifest predating the saved market capture')
        pinned = datetime.now(CST)
        if not start <= pinned <= end:
            raise ValueError(f'{day}: opening window elapsed before inputs could be pinned')
        publish_once(manifest_path, {**inputs, 'pinned_at': pinned.isoformat()})

    if symbols:
        if not book_path.is_file():
            exit_flags = [value for symbol in held for value in ('--require-limits', symbol)]
            collect_books.main(['--date', str(day), '--out', str(books_dir), '--require-513100-iopv',
                                *exit_flags, *symbols], allowed_days=DAYS)
        capture = verify_capture(read_json(book_path), books_dir, day, symbols, held=held, buy_symbols=buy)
        coverage = capture['quote_coverage']
        if not coverage['fresh_symbols']:
            raise ValueError(f'{day}: capture has no fresh verified quote, no opening ledger published')
        for attempt in capture['attempts']:
            if not attempt.get('evidence_error') and attempt.get('books'):
                received = datetime.fromisoformat(attempt['retrieved_at'])
                if start <= received < pinned:
                    raise ValueError(f'{day}: market quote predates input pinning')
        missing_buys = sorted(set(buy) - set(coverage['buy_book_symbols']))
        missing_sells = sorted(set(held) - set(coverage['sell_book_symbols']))
    else:
        capture = {'selected': {}, 'exit_selected': {}, 'unavailable': {}, 'attempts': []}
        missing_buys, missing_sells = [], []
    result = run_day(day, {'baseline': prior}, prepared, capture, {'batches': [], 'errors': [], 'sources': []},
                     prior_day=previous_session(day), groups=('baseline',))
    result['candidate_gaps'] = gaps
    result['book_gaps'] = {'buy': missing_buys, 'sell': missing_sells}
    result['inputs'] = {'opening-inputs.json': sha256_file(manifest_path)}
    if symbols:
        result['inputs']['books/books.json'] = sha256_file(book_path)
    return publish_once(folder / 'opening.json', result)


def close_session(day, folder, now):
    if now < datetime.combine(day, CLOSE, CST):
        raise ValueError(f'{day}: waiting for the 15:00 close')
    opening = open_session(day, folder, now)
    progress_path = folder / 'close-progress.json'
    account = opening['baseline']
    held = account['positions']
    references = {symbol: Decimal(price) if price is not None else None
                  for symbol, price in opening['prior_closes'].items()}
    previous = previous_session(day)
    progress = read_json(progress_path) if progress_path.is_file() else dict(date=str(day), marks={}, errors={})
    if progress['date'] != str(day) or not progress['marks'].keys() <= held.keys():
        raise ValueError(f'{day}: close evidence does not match opening positions')

    if now.date() == day and not (folder / 'daily.json').is_file():
        missing = sorted(held.keys() - progress['marks'].keys())
        with ThreadPoolExecutor(max_workers=4) as pool:
            for symbol, mark, error, _ in pool.map(
                    lambda code: collect(folder, code, references[code], day=day, prior_day=previous), missing):
                if error:
                    progress['errors'][symbol] = error
                else:
                    progress['marks'][symbol] = mark
                    progress['errors'].pop(symbol, None)
        collect_books.write_json(progress_path, progress)
    if not progress_path.is_file():
        raise ValueError(f'{day}: same-session close evidence is missing')

    prices, gaps = {}, {}
    for symbol in held:
        mark = progress['marks'].get(symbol)
        price, error = closing_price(mark, str(day))
        if not error:
            error = verify_mark_provenance(mark, folder, True, str(day), symbol, references[symbol],
                                           prior_day=previous)
        if error:
            gaps[symbol] = progress['errors'].get(symbol, error)
        else:
            prices[symbol] = str(price)
    if gaps:
        raise ValueError(f'{day}: close valuation gap: {gaps}')
    nav = Decimal(account['cash']) + sum(Decimal(prices[symbol]) * quantity for symbol, quantity in held.items())
    daily = {'mode': 'forward', 'date': str(day),
             'inputs': {'opening.json': sha256_file(folder / 'opening.json'),
                        'close-progress.json': sha256_file(progress_path)},
             'book_gaps': opening['book_gaps'], 'candidate_gaps': opening['candidate_gaps'],
             'baseline': {**account, 'nav': str(nav), 'close_prices': prices, 'valuation_gap': {}},
             'cash_control': CASH_CONTROL}
    before, _ = prior_state(day)
    metrics([*before, daily])
    return publish_once(folder / 'daily.json', daily)


def status(now):
    sessions, records, hashes = [], [], {}
    for day in DAYS:
        folder = ROOT / str(day)
        quotes = folder / 'books' / 'books.json'
        row = {'date': str(day), 'candidates_saved': False,
               'market_capture_saved': quotes.is_file(),
               'opening_saved': (folder / 'opening.json').is_file()}
        manifest_path = folder / 'opening-inputs.json'
        try:
            chosen = (session_candidate(folder / read_json(manifest_path)['candidates_file'], folder)
                      if manifest_path.is_file() else candidate_path(day))
        except FileNotFoundError:
            chosen = None
        if chosen is not None:
            prepared = load_candidates(day, chosen)
            gaps = candidate_gaps(prepared)
            row.update(candidates_saved=True, candidate_file=chosen.relative_to(folder).as_posix(),
                       eligible_candidates=len(prepared['candidates']), reviewed_symbols=len(prepared['reviewed']),
                       candidate_gaps=gaps, candidate_coverage_status='partial' if gaps else 'complete')
        if (folder / 'daily.json').is_file():
            if len(records) != DAYS.index(day):
                raise ValueError(f'{day}: a daily ledger exists after a missing session')
            daily = read_daily(day, folder, hashes)
            hashes[str(day)] = sha256_file(folder / 'daily.json')
            records.append(daily)
            row['fills'] = len(daily['baseline']['fills'])
            row['book_gaps'] = daily['book_gaps']
            row['candidate_gaps'] = daily['candidate_gaps']
            row['state'] = ('settled_with_evidence_gaps' if row['candidate_gaps'] or any(row['book_gaps'].values())
                            else 'settled_with_fills' if row['fills'] else 'settled_no_fills')
        elif now < datetime.combine(day, OPEN, CST):
            row['state'] = 'awaiting_open' if row['candidates_saved'] else 'awaiting_preparation'
        elif not row['opening_saved']:
            if now <= datetime.combine(day, OPEN_END, CST):
                row['state'] = 'opening_window'
            else:
                row['state'] = 'opening_unsettled' if row['market_capture_saved'] else 'opening_evidence_missing'
        else:
            row['state'] = 'awaiting_close' if now < datetime.combine(day, CLOSE, CST) else 'close_unsettled'
        sessions.append(row)
    return {'sessions': sessions, 'settled_sessions': len(records), 'paired_metrics': metrics(records),
            'pilot_outcome': 'pending' if len(records) < len(DAYS) else 'descriptive_only'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('open', 'close'):
        action = actions.add_parser(name)
        action.add_argument('--date', type=date.fromisoformat, required=True, choices=DAYS)
    actions.add_parser('status')
    args = parser.parse_args(argv)
    try:
        now = datetime.now(CST)
        if args.action == 'status':
            result = status(now)
        else:
            start = OPEN if args.action == 'open' else CLOSE
            if now < datetime.combine(args.date, start, CST):
                raise ValueError(f'{args.action} is not due for {args.date}')
            folder = ROOT / str(args.date)
            if not folder.is_dir() or folder.resolve() != folder:
                raise ValueError(f'{args.date}: run candidate preparation first')
            with (folder / '.session.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                action = open_session if args.action == 'open' else close_session
                result = action(args.date, folder, now)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
