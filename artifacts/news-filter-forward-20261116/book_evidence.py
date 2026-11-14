"""Rebuild accepted books from the raw capture journal before execution."""

import hashlib
import re
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path

from collect_books import CST, IOPV_SYMBOL, SOURCES, confirm_iopv_update, parse_books, quote_status, usable_price_limits


class EvidenceError(ValueError):
    """A capture claim could not be corroborated by its archived response."""


def market_code(symbol):
    if not isinstance(symbol, str) or not re.fullmatch(r'[0-9]{6}[.](SH|SZ)', symbol):
        raise ValueError('invalid capture symbol')
    return symbol[-2:].lower() + symbol[:6]


def request_symbols(attempt, universe):
    vendor, url = attempt.get('source'), attempt.get('url')
    if not isinstance(vendor, str) or vendor not in SOURCES:
        raise EvidenceError('unknown quote provider')
    prefix = SOURCES[vendor]
    if not isinstance(url, str) or not url.startswith(prefix):
        raise EvidenceError('request URL does not match the quote provider')
    codes = url[len(prefix):].split(',')
    if len(set(codes)) != len(codes) or any(code not in universe for code in codes):
        raise EvidenceError('request URL contains an unexpected security')
    return {code: universe[code] for code in codes}


def read_attempt(attempt, directory, requested, day):
    name = attempt.get('raw_file')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+[.]txt', name):
        raise EvidenceError('invalid raw response filename')
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise EvidenceError('raw response missing or unsafe')
    if path.stat().st_size > 2_000_000:
        raise EvidenceError('raw response exceeds the collector size limit')
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != attempt.get('raw_sha256'):
        raise EvidenceError('raw response checksum mismatch')
    stamp = attempt.get('retrieved_at')
    try:
        received = datetime.fromisoformat(stamp)
        if received.utcoffset() is None:
            raise ValueError('receipt timezone missing')
        received = received.astimezone(CST)
    except (TypeError, ValueError) as exc:
        raise EvidenceError('invalid receipt timestamp') from exc
    if received.date() != day:
        raise EvidenceError('receipt is not from the study session')
    # Both providers use assignment syntax. Reject a known row from the other
    # protocol rather than letting its fields be interpreted under this URL.
    prefix = rb'v_' if attempt['source'] == 'sina' else rb'hq_str_'
    mismatched = {code.decode('ascii') for code in re.findall(prefix + rb'((?:sh|sz)\d+)="', payload)}
    requested_for_parse = {code: symbol for code, symbol in requested.items() if code not in mismatched}
    books, errors = parse_books(attempt['source'], payload, requested_for_parse)
    errors.update({requested[code]: 'raw row belongs to a different provider'
                   for code in mismatched & requested.keys()})
    return received, books, errors


def rebuild_attempts(attempts, directory, day, universe, held):
    """Replay the earliest qualifying books, independently of saved decisions."""
    audited, entries, exits, by_file = [], {}, {}, {}
    failed_before_entry, failed_before_exit = {}, {}
    exit_failures = {}
    response_rows, fresh, two_sided = set(), set(), set()
    last_observations = {}
    request_failures = {'buy': {}, 'sell': {}}
    prior_iopv, last_received = None, None
    for attempt in attempts:
        if not isinstance(attempt, dict):
            audited.append({'books': {}, 'evidence_error': 'invalid attempt record'})
            continue
        observed = {**attempt, 'books': {}}
        audited.append(observed)
        requested = {}
        try:
            requested = request_symbols(attempt, universe)
            if 'raw_file' not in attempt and isinstance(attempt.get('error'), str):
                source, error = attempt['source'], attempt['error']
                if error.strip():
                    for symbol in requested.values():
                        detail = {'source': source, 'error': error}
                        if source == 'tencent' or (symbol != IOPV_SYMBOL and symbol not in held):
                            request_failures['buy'][symbol] = detail
                        if source == 'tencent' and symbol in held:
                            request_failures['sell'][symbol] = detail
                continue
            received, books, errors = read_attempt(attempt, directory, requested, day)
            if last_received is not None and received < last_received:
                raise EvidenceError('receipt timestamps reverse capture order')
        except (OSError, ValueError, ArithmeticError) as exc:
            detail = observed['evidence_error'] = str(exc)
            for symbol in requested.values():
                if symbol not in entries:
                    failed_before_entry.setdefault(symbol, detail)
                if attempt.get('source') == 'tencent' and symbol not in exits:
                    failed_before_exit.setdefault(symbol, detail)
            continue
        last_received = received
        observed.update(books=books, parse_errors=errors)
        response_rows.update(books.keys() | errors.keys())
        name = attempt['raw_file']
        if name in by_file:
            by_file[name] = None
        else:
            by_file[name] = observed
        in_window = time(9, 45) <= received.time() <= time(9, 50)
        tencent = attempt['source'] == 'tencent'
        if tencent:
            for symbol in held & errors.keys():
                exit_failures[symbol] = 'invalid_quote'
        for symbol, book in books.items():
            current = {**book, 'received_at': attempt['retrieved_at'],
                       'source_url': attempt['url'], 'raw_file': name, 'raw_sha256': attempt['raw_sha256']}
            last_observations[symbol] = current
            status = quote_status(book, received, day)
            eligible_quote = in_window and status == 'fresh'
            limits_ok = eligible_quote and tencent and usable_price_limits(book)
            if tencent and symbol in held:
                if status != 'fresh':
                    exit_failures[symbol] = status
                elif not in_window:
                    exit_failures[symbol] = ('quote_received_after_window' if received.time() > time(9, 50)
                                             else 'outside_window')
                elif not limits_ok:
                    exit_failures[symbol] = 'price_limits_unverified'
            if not eligible_quote:
                continue
            fresh.add(symbol)
            if Decimal(book['bid']) > 0 and Decimal(book['ask']) > 0:
                two_sided.add(symbol)
            if limits_ok and symbol not in failed_before_exit:
                exits.setdefault(symbol, current)
            if symbol in entries or symbol in failed_before_entry:
                continue
            if symbol in held and symbol not in exits:
                continue
            if symbol == IOPV_SYMBOL:
                if attempt['source'] != 'tencent' or book.get('iopv', {}).get('status') != 'verified_snapshot':
                    continue
                confirmation = confirm_iopv_update(prior_iopv, current)
                if prior_iopv is None or current['at'] > prior_iopv['at']:
                    old = prior_iopv
                    prior_iopv = current
                else:
                    old = None
                if confirmation is None or old is None:
                    continue
                if datetime.fromisoformat(old['received_at']) >= received:
                    continue
                current = {**current, 'iopv_update': confirmation}
            entries[symbol] = current
    coverage = {'response_row_symbols': sorted(response_rows), 'received_symbols': sorted(last_observations),
                'fresh_symbols': sorted(fresh), 'two_sided_symbols': sorted(two_sided)}
    audit_failures = {'selected': failed_before_entry, 'exit_selected': failed_before_exit}
    return audited, entries, exits, by_file, exit_failures, coverage, last_observations, audit_failures, request_failures


def verify_capture(captured, directory, day, symbols, *, held=(), buy_symbols=None):
    """Return a sanitized copy of one formal capture, retaining per-symbol failures."""
    if not isinstance(captured, dict) or captured.get('mode') != 'forward' or captured.get('date') != str(day):
        raise ValueError('capture is not a formal record for the requested session')
    attempts = captured.get('attempts')
    if not isinstance(attempts, list) or any(
        not isinstance(captured.get(key, {}), dict) for key in ('selected', 'exit_selected', 'unavailable')
    ):
        raise ValueError('invalid capture journal structure')
    directory = Path(directory)
    universe = {market_code(symbol): symbol for symbol in symbols}
    held = set(held).intersection(universe.values())
    audited, entries, exits, by_file, exit_failures, coverage, observations, audit_failures, request_failures = (
        rebuild_attempts(attempts, directory, day, universe, held))
    if universe and not any('evidence_error' not in attempt for attempt in audited):
        raise ValueError('capture has no verifiable quote attempts')
    # An eligible raw quote still needs a matching saved exit claim.
    exit_unavailable = {
        symbol: 'quote_evidence_unverified' if symbol in exits else exit_failures.get(symbol, 'quote_unavailable')
        for symbol in sorted(held)
    }
    unavailable = {symbol: reason if isinstance(reason, str) and reason else 'quote_evidence_unverified'
                   for symbol, reason in captured.get('unavailable', {}).items()
                   if symbol in universe.values()}
    output = {**captured, 'attempts': audited, 'unavailable': unavailable,
              'exit_unavailable': exit_unavailable, 'evidence_errors': {}}
    for key, eligible in (('selected', entries), ('exit_selected', exits)):
        clean, errors = {}, {}
        reasons = output['unavailable' if key == 'selected' else 'exit_unavailable']
        claims = captured.get(key, {})
        failures = audit_failures[key]
        for symbol, claim in claims.items():
            expected = eligible.get(symbol)
            if symbol not in universe.values():
                errors[symbol] = 'security outside the execution universe'
                continue
            is_iopv_entry = symbol == IOPV_SYMBOL and key == 'selected'
            reason = 'quote_evidence_unverified'
            if expected is None:
                if is_iopv_entry and isinstance(claim, dict):
                    name = claim.get('raw_file')
                    attempt = by_file.get(name) if isinstance(name, str) else None
                    if attempt and symbol in attempt['books']:
                        reason = 'iopv_evidence_unverified'
                errors[symbol] = failures.get(symbol, 'no independently eligible archived quote')
            elif not isinstance(claim, dict):
                errors[symbol] = 'selected quote is not an object'
            elif by_file.get(expected['raw_file']) is None:
                errors[symbol] = 'raw filename is reused by multiple attempts'
            elif is_iopv_entry and by_file.get(expected['iopv_update']['previous_raw_file']) is None:
                reason = 'iopv_evidence_unverified'
                errors[symbol] = 'prior IOPV filename is reused by multiple attempts'
            else:
                differences = [field for field, value in expected.items()
                               if (field != 'raw_sha256' or field in claim) and claim.get(field) != value]
                if is_iopv_entry:
                    update = expected['iopv_update']
                    previous_attempt = by_file[update['previous_raw_file']]
                    previous_book = previous_attempt['books'][symbol]
                    previous_provenance = {
                        'previous_raw_sha256': previous_attempt['raw_sha256'],
                        'previous_vendor_premium_pct': previous_book['iopv']['vendor_premium_pct'],
                        'previous_last': previous_book['last'],
                    }
                    claimed_update = claim.get('iopv_update')
                    # Collector journals omit the added provenance. A verified replay
                    # can include it, provided each supplied value matches the raw response.
                    if isinstance(claimed_update, dict) and 'iopv_update' in differences:
                        optional_claim_fields = {field: value for field, value in previous_provenance.items()
                                                 if field in claimed_update}
                        if claimed_update == {**update, **optional_claim_fields}:
                            differences.remove('iopv_update')
                    if not differences:
                        expected = {**expected, 'iopv_update': {**update, **previous_provenance}}
                if not differences:
                    clean[symbol] = expected
                    reasons.pop(symbol, None)
                    continue
                if is_iopv_entry and set(differences) <= {'iopv', 'iopv_update'}:
                    reason = 'iopv_evidence_unverified'
                errors[symbol] = 'selected quote differs from archived evidence: ' + ', '.join(differences)
            reasons[symbol] = reason
        missing_claims = (eligible.keys() | failures.keys()) - claims.keys()
        direction = 'entry' if key == 'selected' else 'exit'
        if key == 'exit_selected':
            missing_claims.intersection_update(held)
        for symbol in sorted(missing_claims):
            reasons[symbol] = 'quote_evidence_unverified'
            errors[symbol] = failures.get(symbol, f'eligible archived {direction} quote is missing the capture claim')
        output[key], output['evidence_errors'][key] = clean, errors
    planned = set(universe.values())
    buy_candidates = planned if buy_symbols is None else planned.intersection(buy_symbols)
    output['quote_coverage'] = {
        'planned_symbols': sorted(planned), **coverage,
        'buy_book_symbols': sorted(output['selected'].keys() & buy_candidates),
        'sell_book_symbols': sorted(output['exit_selected'].keys() & held),
    }
    output['last_observations'] = observations
    output['quote_request_failures'] = request_failures
    return output


def record_quote_request_failure(capture, decision):
    """Retain the latest request failure when a decision lacks a specific quote reason."""
    if decision['reason'] == 'quote_unavailable':
        failures = capture.get('quote_request_failures', {}).get(decision['side'], {})
        if failure := failures.get(decision['symbol']):
            decision['quote_request_failure'] = failure
