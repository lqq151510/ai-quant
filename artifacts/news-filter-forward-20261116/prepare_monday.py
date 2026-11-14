"""Build a bounded Monday shortlist from captured adjusted and unadjusted daily bars."""
import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
SH = '600036 601166 601628 600109 600028 600886 601006 601899 600362 600019 600660 600741 600498 600893 600372 601766 600196 600085 600900 600941 600276 601988 603135 603833 603979 600048 600143 603920 600350'
SZ = '000776 002594 000725 002008 000002 000333 002371 000858 002415 000651 000001 002475'
ETF_SH = '510050 510300 510500 510880 512480 515880 515790 518880 511010 588000 512100 512660 512010 512690 512200 512800 513100'
ETF_SZ = '159915 159949 159920 159922 159919'
CROSS_BORDER = {'513100', '159920'}


def get_bars(symbol, adjusted, raw, days):
    code = symbol[-2:].lower() + symbol[:6]
    url = ('https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param='
           f'{code},day,{days[0]},{days[-1]},12,' + ('qfq' if adjusted else ''))
    payload = download(url)
    (raw / f'{symbol}-{int(adjusted)}.json').write_bytes(payload)
    data = json.loads(payload)['data'][code]
    rows = data.get('qfqday' if adjusted else 'day')
    if not rows or [row[0] for row in rows] != days:
        raise ValueError('missing contiguous six sessions')
    if any(Decimal(r[2]) <= 0 or Decimal(r[5]) <= 0 for r in rows):
        raise ValueError('missing positive close or volume')
    return data, rows, url


def download(url):
    for attempt in range(2):
        try:
            with urlopen(Request(url, headers={'Referer': 'https://finance.sina.com.cn/'}), timeout=5) as response:
                payload = response.read(2_000_001)
                if len(payload) > 2_000_000:
                    raise ValueError('market payload too large')
                return payload
        except OSError:
            if attempt:
                raise


def inspect(item, raw, days):
    symbol, kind = item
    result = dict(symbol=symbol, kind=kind)
    try:
        unadjusted, rows, raw_url = get_bars(symbol, False, raw, days)
        code = symbol[-2:].lower() + symbol[:6]
        quote = unadjusted['qt'][code]
        if quote[2] != symbol[:6]:
            raise ValueError('quote symbol mismatch')
        result['name'] = quote[1]
        if result['name'].upper().startswith(('ST', '*ST', 'SST', 'S*ST')) or '退' in result['name']:
            return {**result, 'status': 'risk_warning_or_delisting'}
        adjusted, adjusted_rows, adjusted_url = get_bars(symbol, True, raw, days)
        if rows[-1][2] != adjusted_rows[-1][2]:
            raise ValueError('adjusted close does not anchor to the latest raw close')
        tick = Decimal('.001' if kind == 'etf' else '.01')
        close = Decimal(rows[-1][2])
        gain = Decimal(adjusted_rows[-1][2]) / Decimal(adjusted_rows[0][2]) - 1
        if quote[30][:8] == days[-1].replace('-', '') and Decimal(quote[3]) != close:
            raise ValueError('closing quote does not match raw close')
        if gain <= 0:
            return {**result, 'status': 'signal_not_triggered', 'five_day_return': str(gain)}
        sina_url = ('https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_a=/'
                    f'CN_MarketDataService.getKLineData?symbol={code}&scale=240&ma=no&datalen=8')
        payload = download(sina_url)
        (raw / f'{symbol}-sina.txt').write_bytes(payload)
        text = payload.decode()
        sina = [bar for bar in json.loads(text[text.index('[{'):text.rindex(']')+1]) if bar['day'] in days]
        if [bar['day'] for bar in sina] != days:
            raise ValueError('Sina session dates do not match')
        for a, b in zip(rows, sina):
            if (list(map(Decimal, a[1:5])) != [Decimal(b[f]) for f in ('open','close','high','low')]
                or abs(Decimal(a[5])*100-Decimal(b['volume'])) >= 100):
                raise ValueError('raw OHLC or volume differs across sources')
        floor = sum(Decimal(b['low'])*Decimal(b['volume']) for b in sina[-5:]) / 5
        ceiling = sum(Decimal(b['high'])*Decimal(b['volume']) for b in sina[-5:]) / 5
        status = ('eligible' if floor >= 30000000 else
                  'signal_not_triggered' if ceiling < 30000000 else 'turnover_threshold_uncertain')
        return {**result, 'status': status, 'five_day_return': str(gain), 'turnover_lower_bound': str(floor),
                'turnover_upper_bound': str(ceiling), 'previous_close': str(close), 'tick': str(tick),
                'cross_border': symbol[:6] in CROSS_BORDER, 'sources': [raw_url, adjusted_url, sina_url]}
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError, IndexError) as exc:
        return {**result, 'status': 'data_gap', 'reason': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=date.fromisoformat, required=True)
    day = parser.parse_args().date
    if not date(2026, 11, 17) <= day <= date(2026, 11, 20):
        parser.error('Monday uses the immutable precomputed shortlist; prepare dates November 17–20 only')
    prior = day - timedelta(days=1)
    timezone = ZoneInfo('Asia/Shanghai')
    now = datetime.now(timezone)
    if not datetime.combine(prior, time(15), timezone) <= now < datetime.combine(day, time(9, 45), timezone):
        parser.error('shortlist preparation requires the previous close and must precede entry')
    sessions = []
    session = prior
    while len(sessions) < 6:
        if session.weekday() < 5:
            sessions.append(str(session))
        session -= timedelta(days=1)
    days = sessions[::-1]
    out = BASE / f'{day}-preparation'
    out.mkdir(exist_ok=True)
    output = out / 'candidates.json'
    if output.is_file():
        print(f'{day}: using existing shortlist {output}')
        return
    raw = Path(tempfile.mkdtemp(prefix='bars-', dir=out))
    universe = [(f'{code}.{exchange}', kind) for codes, exchange, kind in
                [(SH, 'SH', 'stock'), (SZ, 'SZ', 'stock'), (ETF_SH, 'SH', 'etf'), (ETF_SZ, 'SZ', 'etf')]
                for code in codes.split()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda item: inspect(item, raw, days), universe))
    ranked = sorted((row for row in rows if row['status'] == 'eligible'),
                    key=lambda row: (-Decimal(row['five_day_return']), row['symbol']))
    captured_at = datetime.now(timezone)
    if captured_at >= datetime.combine(day, time(9, 45), timezone):
        parser.error('candidate data collection ended after the entry cutoff')
    payload = dict(date=str(day), captured_at=captured_at.isoformat(), raw_directory=raw.name,
                   coverage='bounded_sample_not_full_market', sessions=days, reviewed=rows,
                   candidates=ranked[:10])
    with tempfile.NamedTemporaryFile(dir=out, mode='w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        try:
            os.link(stream.name, output)
        except FileExistsError:
            print(f'{day}: another run already published {output}')
            return
    print(len(rows), {s: sum(r['status']==s for r in rows) for s in {r['status'] for r in rows}})
    for r in ranked[:10]:
        print(r['symbol'],r['name'],r['five_day_return'],r['turnover_lower_bound'])


if __name__ == '__main__':
    main()
