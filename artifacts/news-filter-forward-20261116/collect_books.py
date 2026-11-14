"""Capture public order books for the frozen forward experiment."""
import argparse
from base64 import b64decode, b64encode
import fcntl
import hashlib
import json
import os
import re
import signal
import time
from datetime import date, datetime, time as wall_time
from decimal import Decimal, InvalidOperation
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

CST = ZoneInfo('Asia/Shanghai')
SOURCES = {'sina': 'https://hq.sinajs.cn/list=', 'tencent': 'https://qt.gtimg.cn/q='}
IOPV_SYMBOL = '513100.SH'


def tencent_iopv(fields):
    """Recognize the observed 513100 layout and cross-check its paired values."""
    if len(fields) <= 82 or (fields[2], fields[61], fields[82]) != ('513100', 'ETF', 'CNY'):
        return {'status': 'iopv_layout_unverified'}
    if not fields[77] or not fields[78]:
        return {'status': 'iopv_missing'}
    try:
        latest, premium_pct, nav = map(Decimal, (fields[3], fields[77], fields[78]))
        if not all(value.is_finite() for value in (latest, premium_pct, nav)) or latest <= 0 or nav <= 0:
            return {'status': 'iopv_invalid'}
        difference = abs((latest / nav - 1) * 100 - premium_pct)
    except InvalidOperation:
        return {'status': 'iopv_invalid'}
    if difference > Decimal('.01'):
        return {'status': 'iopv_premium_mismatch'}
    return {'status': 'verified_snapshot', 'nav': str(nav), 'vendor_premium_pct': str(premium_pct)}


def capture(url, timeout=8):
    """Time one request on the standalone POSIX collector's main thread."""
    if timeout <= 0:
        raise TimeoutError('market request deadline already elapsed')
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('market capture requires an idle process alarm')

    def expire(_signum, _frame):
        raise TimeoutError('market request exceeded its deadline')

    req = Request(url, headers={'Referer': 'https://finance.sina.com.cn/', 'User-Agent': 'Mozilla/5.0'})
    previous_handler = signal.signal(signal.SIGALRM, expire)
    try:
        # The socket timeout alone restarts as each portion of the response arrives.
        signal.setitimer(signal.ITIMER_REAL, timeout)
        with urlopen(req, timeout=timeout) as reply:
            payload = reply.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError('market payload too large')
            if reply.length not in (0, None):
                raise IncompleteRead(payload, reply.length)
            return datetime.now(CST).isoformat(), payload
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        finally:
            signal.signal(signal.SIGALRM, previous_handler)


def parse_books(vendor, payload, symbols):
    books, errors = {}, {}
    for code, body in re.findall(r'(?:hq_str_|v_)((?:sh|sz)\d+)="([^"]*)"', payload.decode('gb18030')):
        if code not in symbols:
            continue
        symbol = symbols[code]
        try:
            if symbol in books or symbol in errors:
                raise ValueError('duplicate quote row')
            if vendor == 'sina':
                f = body.split(',')
                name = f[0]
                traded_volume = int(f[8])
                stamp = datetime.fromisoformat(f[30] + 'T' + f[31])
                bid, bid_units, ask, ask_units, last, previous = f[11], int(f[10]), f[21], int(f[20]), f[3], f[2]
            else:
                f = body.split('~')
                name = f[1]
                traded_volume = int(f[6])*100
                if f[2] != code[2:]:
                    raise ValueError('quote symbol mismatch')
                stamp = datetime.strptime(f[30], '%Y%m%d%H%M%S')
                bid, bid_units, ask, ask_units, last, previous = f[9], int(f[10])*100, f[19], int(f[20])*100, f[3], f[4]
            if any(not p.is_finite() or p < 0 for p in map(Decimal, (bid, ask, last, previous))):
                raise ValueError('invalid price')
            if min(bid_units, ask_units, traded_volume) < 0:
                raise ValueError('invalid displayed quantity')
            if (bid_units > 0 and Decimal(bid) == 0) or (ask_units > 0 and Decimal(ask) == 0):
                raise ValueError('positive quantity without a price')
            books[symbol] = dict(at=stamp.replace(tzinfo=CST).isoformat(), bid=bid, bid_units=bid_units,
                                 ask=ask, ask_units=ask_units, last=last, previous=previous, name=name,
                                 traded_volume=traded_volume)
            if vendor == 'tencent' and symbol == IOPV_SYMBOL:
                books[symbol]['iopv'] = tencent_iopv(f)
            if vendor == 'tencent' and len(f) > 48:
                books[symbol].update(upper_limit=f[47], lower_limit=f[48])
        except (ValueError, IndexError, ArithmeticError) as exc:
            books.pop(symbol, None)
            errors[symbol] = str(exc)
    return books, errors


def quote_status(book, received, day):
    if book is None:
        return 'quote_unavailable'
    try:
        at = datetime.fromisoformat(book['at'])
        if at.utcoffset() is None or received.utcoffset() is None:
            return 'invalid_quote'
        at = at.astimezone(CST)
    except (ValueError, TypeError, KeyError):
        return 'invalid_quote'
    age = (received - at).total_seconds()
    if age < 0:
        return 'quote_from_future'
    if at.date() != day or at.time() < wall_time(9, 45) or age > 30:
        return 'quote_stale'
    return 'fresh'


def confirm_iopv_update(previous, current):
    """Require a visible value change between two coherent intraday snapshots."""
    if previous is None or current['at'] <= previous['at']:
        return None
    if Decimal(current['iopv']['nav']) == Decimal(previous['iopv']['nav']):
        return None
    return {'previous_at': previous['at'], 'previous_nav': previous['iopv']['nav'],
            'previous_raw_file': previous['raw_file'],
            'previous_received_at': previous['received_at']}


def usable_price_limits(book):
    try:
        lower, upper = (Decimal(book[key]) for key in ('lower_limit', 'upper_limit'))
        return lower.is_finite() and upper.is_finite() and 0 < lower < upper
    except (KeyError, TypeError, ArithmeticError):
        return False


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=date.fromisoformat, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe', action='store_true', help='one diagnostic pass, never eligible for study fills')
    parser.add_argument('--require-513100-iopv', action='store_true')
    parser.add_argument('--require-limits', action='append', default=[])
    parser.add_argument('symbols', nargs='+')
    args = parser.parse_args()
    symbols = {}
    for symbol in args.symbols:
        if not re.fullmatch(r'[0-9]{6}[.](SH|SZ)', symbol):
            parser.error(f'invalid security code: {symbol}')
        symbols[symbol[-2:].lower() + symbol[:6]] = symbol
    now = datetime.now(CST)
    start = datetime.combine(args.date, wall_time(9, 45), CST)
    end = datetime.combine(args.date, wall_time(9, 50), CST)
    if not args.probe and (not date(2026, 11, 16) <= args.date <= date(2026, 11, 20) or now < start):
        parser.error('outside frozen study capture window')
    checkpoint = args.out / 'capture-progress.json'
    identity = {'mode': 'diagnostic' if args.probe else 'forward', 'date': str(args.date),
                'symbols': sorted(symbols.values()), 'require_513100_iopv': args.require_513100_iopv,
                'require_limits': sorted(set(args.require_limits))}
    if (args.out/'books.json').exists():
        parser.error('completed books already exist')
    if not args.probe and now > end and not checkpoint.is_file():
        parser.error('no saved intraday capture to finalize')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.capture.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('capture already running for this output directory')
        if (args.out/'books.json').exists():
            parser.error('completed books already exist')
        selected, exit_selected, last_status, attempts = {}, {}, {}, []
        prior_iopv = None
        pending_response = None
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text())
            if saved['identity'] != identity:
                parser.error('capture checkpoint belongs to a different run')
            selected, last_status, attempts, prior_iopv = (
                saved[k] for k in ('selected', 'last_status', 'attempts', 'prior_iopv')
            )
            exit_selected = saved.get('exit_selected', {})
            pending_response = saved.get('pending_response')
        remaining = set(symbols.values()) - selected.keys()

        def persist(pending=None):
            state = dict(identity=identity, selected=selected, exit_selected=exit_selected,
                         last_status=last_status, attempts=attempts, prior_iopv=prior_iopv)
            if pending is not None:
                state['pending_response'] = pending
            write_json(checkpoint, state)

        def commit_response(pending):
            nonlocal prior_iopv
            index = len(attempts) + 1
            vendor, url = pending['source'], pending['url']
            prefix = SOURCES[vendor]
            if pending['index'] != index or not url.startswith(prefix):
                raise ValueError('pending response does not match the next request')
            codes = url.removeprefix(prefix).split(',')
            request_symbols = {code: symbols[code] for code in codes}
            if len(request_symbols) != len(codes) or not set(request_symbols.values()) <= remaining:
                raise ValueError('pending response has invalid requested symbols')
            payload = b64decode(pending['payload'], validate=True)
            checksum = hashlib.sha256(payload).hexdigest()
            if len(payload) > 2_000_000 or checksum != pending['raw_sha256']:
                raise ValueError('pending response payload failed validation')
            received = datetime.fromisoformat(pending['retrieved_at'])
            if received.utcoffset() is None:
                raise ValueError('pending response has no receipt timezone')
            raw_file = pending['raw_file']
            if not re.fullmatch(rf'(?:retry-)*{index:03d}-{vendor}[.]txt', raw_file):
                raise ValueError('pending response has an invalid raw filename')
            path = args.out / raw_file
            if path.is_symlink():
                raise ValueError('pending response has an unsafe raw filename')
            attempt = {'source': vendor, 'url': url}
            attempts.append(attempt)
            try:
                books, parse_errors = parse_books(vendor, payload, request_symbols)
            except UnicodeError as exc:
                attempt['error'] = str(exc)
                persist()
                return
            path.write_bytes(payload)
            attempt.update(raw_file=raw_file, raw_sha256=checksum, retrieved_at=pending['retrieved_at'],
                           books=books, parse_errors=parse_errors, statuses={})
            for symbol in sorted(request_symbols.values()):
                book = books.get(symbol)
                status = quote_status(book, received, args.date)
                if symbol in parse_errors:
                    status = 'invalid_quote'
                if status == 'fresh' and received > end and not args.probe:
                    status = 'quote_received_after_window'
                if status == 'fresh' and symbol in args.require_limits:
                    if usable_price_limits(book):
                        if not args.probe:
                            exit_selected.setdefault(symbol, {**book, 'received_at': received.isoformat(),
                                                              'source_url': url, 'raw_file': path.name})
                    elif symbol not in exit_selected:
                        status = 'price_limits_unverified'
                if status == 'fresh' and args.require_513100_iopv and symbol == IOPV_SYMBOL:
                    iopv_status = book['iopv']['status']
                    if iopv_status != 'verified_snapshot':
                        status = iopv_status
                    elif received <= end:
                        current = {**book, 'raw_file': path.name, 'received_at': received.isoformat()}
                        confirmation = confirm_iopv_update(prior_iopv, current)
                        if prior_iopv is None or current['at'] > prior_iopv['at']:
                            prior_iopv = current
                        if confirmation is None:
                            status = 'iopv_intraday_update_unverified'
                        else:
                            book['iopv_update'] = confirmation
                attempt['statuses'][symbol] = status
                if status != 'quote_unavailable' or symbol not in last_status:
                    last_status[symbol] = status
                if status == 'fresh' and not args.probe and received <= end:
                    selected[symbol] = {**book, 'received_at': received.isoformat(), 'source_url': url,
                                        'raw_file': path.name}
            remaining.difference_update(selected)
            persist()

        if pending_response is not None:
            commit_response(pending_response)
        elif not checkpoint.is_file():
            persist()

        # Capture scheduled exits before the ordinary entry requests.
        route = [('tencent', True)] if args.require_limits else []
        route += [('sina', False), ('tencent', False)]
        while remaining and (args.probe or datetime.now(CST) <= end):
            for vendor, exits_only in route:
                if not remaining or (not args.probe and datetime.now(CST) > end):
                    break
                request_symbols = {
                    k: v for k, v in symbols.items() if v in remaining
                    and (not exits_only or v in args.require_limits)
                    and not (args.require_513100_iopv and v == IOPV_SYMBOL and vendor == 'sina')
                    and not (v in args.require_limits and vendor == 'sina')
                }
                if not request_symbols:
                    continue
                url = SOURCES[vendor] + ','.join(request_symbols)
                try:
                    timeout = 8 if args.probe else min(8, (end - datetime.now(CST)).total_seconds())
                    if timeout <= 0:
                        break
                    received_at, payload = capture(url, timeout)
                except (OSError, HTTPException, ValueError, IndexError, KeyError) as exc:
                    attempts.append({'source': vendor, 'url': url, 'error': str(exc)})
                    persist()
                    continue
                index = len(attempts) + 1
                path = args.out / f'{index:03d}-{vendor}.txt'
                # Legacy checkpoints may have left an unindexed response at this path.
                while path.exists() or path.is_symlink():
                    path = path.with_name('retry-' + path.name)
                pending = {'index': index, 'source': vendor, 'url': url, 'retrieved_at': received_at,
                           'raw_file': path.name, 'raw_sha256': hashlib.sha256(payload).hexdigest(),
                           'payload': b64encode(payload).decode('ascii')}
                persist(pending)
                commit_response(pending)
            if args.probe:
                break
            if remaining and datetime.now(CST) < end:
                time.sleep(min(3, max(0, (end - datetime.now(CST)).total_seconds())))
        result = {'mode': 'diagnostic' if args.probe else 'forward', 'date': str(args.date),
                  'selected': selected, 'exit_selected': exit_selected,
                  'unavailable': {s: last_status.get(s, 'quote_unavailable')
                                                       for s in sorted(remaining)}, 'attempts': attempts}
        write_json(args.out / 'books.json', result)


if __name__ == '__main__':
    main()
