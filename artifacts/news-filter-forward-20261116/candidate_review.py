"""Friday-close volume and trend review for the frozen Monday shortlist."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from collect_books import CST
from prepare_monday import download

ROOT = Path(__file__).resolve().parent
CUTOFF = date(2026, 11, 13)
SHORTLIST = '137ed04d32777fd23db94bad04be9562d17a8262531f34f71f9272bb24439d7c'
NEWS = {
    '603135.SH': ('11 月 13 日披露交易异常波动公告。公告称此前三个交易日涨幅偏离值累计超过 20%，换手率增加，并澄清公司未开展通用人形机器人本体研发。', 'https://pdf.dfcfw.com/pdf/H2_AN202611121831664588_1.pdf'),
    '513100.SH': ('11 月 9 日发布过二级市场溢价风险提示及临时停牌公告。周一仍以当时的参考净值核验为准。', 'https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-11-09/513100_20261109_FW5A.pdf'),
    '603920.SH': ('11 月 12 日披露交易异常波动公告，称此前三个交易日涨幅偏离值累计超过 20%。公告提及拟回购股份，并称截至公告时尚未实施。', 'https://pdf.dfcfw.com/pdf/H2_AN202611111831632005_1.pdf'),
}


def sessions(count):
    dates = []
    day = CUTOFF
    while len(dates) < count:
        if day.weekday() < 5:
            dates.append(str(day))
        day -= timedelta(days=1)
    return dates[::-1]


def rows_of(payload, code, field, expected):
    rows = json.loads(payload)['data'][code][field]
    if [r[0] for r in rows] != expected:
        raise ValueError('expected consecutive sessions were not supplied')
    closes, volumes = ([Decimal(r[i]) for r in rows] for i in (2, 5))
    if any(not v.is_finite() or v <= 0 for v in closes + volumes):
        raise ValueError('invalid close or volume')
    return closes, volumes


def analyze(row, out):
    symbol = row['symbol']
    code = symbol[-2:].lower() + symbol[:6]
    frozen = ROOT/'monday-preparation'/'bars'
    dated = sessions(20)
    raw_file = frozen/f'{symbol}-0.json'
    adj_file = frozen/f'{symbol}-1.json'
    result = {'symbol': symbol, 'name': row['name'], 'sources': []}
    try:
        raw, volume = rows_of(raw_file.read_bytes(), code, 'day', dated[-6:])
        close, _ = rows_of(adj_file.read_bytes(), code, 'qfqday', dated[-6:])
        if raw[-1] != Decimal(row['previous_close']) or close[-1] != raw[-1]:
            raise ValueError('Friday close does not match frozen shortlist')
        average = sum(volume[:-1]) / 5
        result.update(five_day_return=row['five_day_return'], friday_change=str(raw[-1]/raw[-2]-1),
                      raw_volume_units=str(volume[-1]*100), average_volume_units=str(average*100),
                      volume_ratio=str(volume[-1]/average), close=str(close[-1]),
                      ma={'5': str(sum(close[-5:])/5)})
        result['sources'].extend(str(p.relative_to(ROOT)) for p in (raw_file, adj_file))
        url = ('https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param='
               f'{code},day,{dated[0]},{dated[-1]},30,qfq')
        payload = download(url)
        path = out/f'{symbol}-qfq-20.json'
        path.write_bytes(payload)
        result.update(extended_url=url, extended_raw=path.name,
                      extended_sha256=hashlib.sha256(payload).hexdigest())
        full, _ = rows_of(payload, code, 'qfqday', dated)
        if full[-6:] != close:
            raise ValueError('20-session adjusted history conflicts with archived last six')
        for n in (10, 20):
            result['ma'][str(n)] = str(sum(full[-n:])/n)
    except (OSError, ValueError, IndexError, KeyError, TypeError, ArithmeticError) as exc:
        result['gap'] = str(exc)
    return result


def render(rows):
    def price(value):
        return '—' if value is None else f'{Decimal(value):.3f}'

    def pct(value):
        return '—' if value is None else f'{Decimal(value):+.2%}'

    lines = [
        '# 周一冻结候选的成交量、消息与均线',
        '',
        '行情截至北京时间 2026-11-13 收盘。量比为周五成交量除以前五个交易日平均成交量。'
        '均线采用前复权收盘价，20 日数据须与已封存的六日行情重合。该观察单独记录，正式模拟沿用冻结规则。',
        '',
        '| 代码 | 标的 | 周五涨跌 | 五日涨幅 | 周五量比 | 收盘 | MA5 | MA10 | MA20 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        ratio = row.get('volume_ratio')
        ma = row.get('ma', {})
        volume_label = '—' if ratio is None else f'{Decimal(ratio):.2f}×'
        lines.append(f"| {row['symbol']} | {row['name']} | {pct(row.get('friday_change'))} | "
                     f"{pct(row.get('five_day_return'))} | {volume_label} | {price(row.get('close'))} | "
                     f"{price(ma.get('5'))} | {price(ma.get('10'))} | {price(ma.get('20'))} |")
    lines.extend(['', '## 消息面', ''])
    for symbol, (content, url) in NEWS.items():
        lines.append(f'- {symbol}。{content} [来源]({url})')
    lines.append('- 世运电路公告是本轮补查取得，保留在本报告。冻结实验仍从正式新闻归档读取，并按记录的首次采集时间判断。')
    lines.extend(['- 其余候选缺少本次已核实的单标的消息。', '',
                  '## 数据缺口', ''])
    gaps = [f"- {row['symbol']}。{row['gap']}" for row in rows if row.get('gap')]
    lines.extend(gaps or ['十只候选的成交量和三条均线均已计算。'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out
    raw = (ROOT/'monday-preparation'/'candidates.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHORTLIST:
        parser.error('frozen shortlist hash mismatch')
    out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda row: analyze(row, out), json.loads(raw)['candidates']))
    (out/'candidate-review.json').write_text(json.dumps(
        {'market_close': str(CUTOFF), 'captured_at': datetime.now(CST).isoformat(), 'rows': rows},
        ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'candidate-review.md').write_text(render(rows), encoding='utf-8')
    print(render(rows))


if __name__ == '__main__':
    main()
