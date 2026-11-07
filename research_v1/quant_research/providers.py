"""Collect small, attributable samples from public market-data providers."""

from __future__ import annotations

import contextlib
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

_REQUEST_TIMEOUT = 30
_BAOSTOCK_TIMEOUT = 15
_WORKER_COUNT = 2
_ETF_BARS = {"date", "open", "high", "low", "close", "volume", "amount"}
_F10_DIVIDENDS = {"权益登记日", "除息日", "每10份分红", "分红发放日"}
_F10_SPLITS = {"拆分折算日", "拆分类型", "拆分折算比例"}


class ProviderResponseError(RuntimeError):
    """The provider responded with an error or unexpected result structure."""


@dataclass(frozen=True)
class _Request:
    key: str
    provider: str
    endpoint: str
    arguments: dict[str, str]
    date_column: str | None = None
    required_columns: tuple[str, ...] = ()


def _request_plan(start: date, end: date) -> list[_Request]:
    from .protocol import ETF_SYMBOLS

    requests = []
    for symbol in ETF_SYMBOLS:
        requests.extend([
            _Request(
                f"etf_bars_{symbol}", "akshare_sina", "fund_etf_hist_sina", {"symbol": f"sh{symbol}"},
                "date", tuple(sorted(_ETF_BARS)),
            ),
            _Request(
                f"etf_cumulative_dividends_{symbol}", "akshare_sina", "fund_etf_dividend_sina",
                {"symbol": f"sh{symbol}"}, "日期", ("日期", "累计分红"),
            ),
            _Request(
                f"etf_actions_{symbol}", "eastmoney_f10",
                f"https://fundf10.eastmoney.com/fhsp_{symbol}.html", {},
            ),
        ])

    for year in range(start.year, end.year + 1):
        requests.append(_Request(
            f"trading_calendar_{year}", "baostock", "query_trade_dates",
            {"start_date": max(start, date(year, 1, 1)).isoformat(),
             "end_date": min(end, date(year, 12, 31)).isoformat()},
            "calendar_date", ("calendar_date", "is_trading_day"),
        ))
    for sample_date in ("2015-01-05", "2024-01-02"):
        requests.append(_Request(
            f"csi300_membership_{sample_date}", "baostock", "query_hs300_stocks", {"date": sample_date},
            "updateDate", ("updateDate", "code"),
        ))
    requests.extend([
        _Request(
            "stock_bars_600000_sample", "baostock", "query_history_k_data_plus",
            {
                "code": "sh.600000", "start_date": "2024-01-02", "end_date": "2024-01-12",
                "frequency": "d", "adjustflag": "3",
                "fields": "date,code,open,high,low,close,preclose,volume,amount,adjustflag,tradestatus,isST",
            },
            "date", ("date", "open", "close", "adjustflag", "tradestatus"),
        ),
        _Request(
            "stock_actions_600000_sample", "baostock", "query_dividend_data",
            {"code": "sh.600000", "year": "2024", "yearType": "operate"}, "dividOperateDate",
            ("code", "dividOperateDate", "dividRegistDate", "dividCashPsBeforeTax"),
        ),
    ])
    return requests


def _fetch_f10(url: str) -> dict[str, tuple[pd.DataFrame, str]]:
    import pandas as pd
    import requests

    response = requests.get(url, timeout=(5, 15))
    response.raise_for_status()
    results = {}
    for table in pd.read_html(BytesIO(response.content), encoding="utf-8", keep_default_na=False):
        columns = set(table.columns)
        if _F10_DIVIDENDS <= columns:
            results["dividends"] = (table, "除息日")
        elif _F10_SPLITS <= columns:
            results["splits"] = (table, "拆分折算日")
    if not results:
        raise ProviderResponseError("F10 page has no recognized corporate-action tables")
    return results


def _fetch(request: _Request) -> tuple[dict[str, tuple[pd.DataFrame, str | None]], str | None]:
    if request.provider == "eastmoney_f10":
        return _fetch_f10(request.endpoint), None
    if request.provider == "akshare_sina":
        import akshare

        table = getattr(akshare, request.endpoint)(**request.arguments)
        return {"": (table, request.date_column)}, akshare.__version__
    if request.provider == "baostock":
        import baostock

        session = baostock.login()
        if session.error_code != "0":
            raise ProviderResponseError(f"BaoStock login {session.error_code}: {session.error_msg}")
        try:
            response = getattr(baostock, request.endpoint)(**request.arguments)
            if response.error_code != "0":
                raise ProviderResponseError(f"BaoStock query {response.error_code}: {response.error_msg}")
            table = response.get_data()
            if response.error_code != "0":
                raise ProviderResponseError(f"BaoStock download {response.error_code}: {response.error_msg}")
        finally:
            baostock.logout()
        return {"": (table, request.date_column)}, getattr(baostock, "__version__", None)
    raise ValueError(f"unknown provider {request.provider}")


def _capture_tables(request: _Request, scratch: Path) -> dict[str, Any]:
    import pandas as pd

    socket.setdefaulttimeout(12)
    with contextlib.redirect_stdout(sys.stderr):
        tables, version = _fetch(request)

    files = []
    for suffix, (frame, date_column) in tables.items():
        name = f"{request.key}_{suffix}.csv" if suffix else f"{request.key}.csv"
        path = scratch / name
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        missing = sorted(set(request.required_columns) - set(frame))
        dates = pd.to_datetime(frame[date_column], errors="coerce") if date_column in frame else None
        valid = dates.dropna() if dates is not None else None
        files.append({
            "name": name, "sha256": digest, "rows": len(frame),
            "columns": [str(column) for column in frame], "missing_columns": missing,
            "date_column": date_column, "invalid_date_rows": int(dates.isna().sum()) if dates is not None else None,
            "first_date": valid.min().date().isoformat() if valid is not None and not valid.empty else None,
            "last_date": valid.max().date().isoformat() if valid is not None and not valid.empty else None,
        })
    populated = [item for item in files if item["rows"]]
    status = "captured" if populated else "empty"
    if any(item["missing_columns"] or item["invalid_date_rows"] for item in populated):
        status = "invalid_schema"
    return {"status": status, "provider_version": version, "files": files}


def _worker(request_json: str, output: str) -> None:
    try:
        result = _capture_tables(_Request(**json.loads(request_json)), Path(output))
    except Exception as exc:
        # Arbitrary HTTP exception text may expose local proxy credentials.
        error = {"type": type(exc).__name__}
        if isinstance(exc, (ProviderResponseError, ImportError)):
            error["message"] = str(exc)[:300]
        result = {"status": "failed", "error": error, "files": []}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def _collect(request: _Request, raw: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"request": asdict(request)}
    timeout = _BAOSTOCK_TIMEOUT if request.provider == "baostock" else _REQUEST_TIMEOUT
    with tempfile.TemporaryDirectory(prefix=f".{request.key}-", dir=raw) as scratch:
        try:
            process = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                 json.dumps(asdict(request), ensure_ascii=False), scratch],
                capture_output=True, encoding="utf-8", check=True, timeout=timeout,
            )
            payload = json.loads(process.stdout)
            for item in payload["files"]:
                name = item.pop("name")
                (Path(scratch) / name).rename(raw / name)
                item["path"] = f"raw/{name}"
            evidence.update(payload)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            error: dict[str, Any] = {"type": type(exc).__name__}
            if isinstance(exc, subprocess.TimeoutExpired):
                error["seconds"] = timeout
            elif isinstance(exc, subprocess.CalledProcessError):
                error["exit_code"] = exc.returncode
            evidence.update({"status": "failed", "error": error, "files": []})
    evidence["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return evidence


def _session_alignment(output: Path, records: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    import pandas as pd

    captured = {record["request"]["key"]: record for record in records if record["status"] == "captured"}
    calendar_records = [record for key, record in captured.items() if key.startswith("trading_calendar")]
    if not calendar_records:
        return {"status": "calendar_unavailable", "etfs": {}}
    calendar = pd.concat([
        pd.read_csv(output / record["files"][0]["path"], dtype=str) for record in calendar_records
    ], ignore_index=True)
    calendar_dates = pd.to_datetime(calendar["calendar_date"], errors="raise")
    day_span = pd.date_range(start, end, freq="D")
    if (
        calendar_dates.duplicated().any() or set(calendar_dates) != set(day_span)
        or not calendar["is_trading_day"].isin(("0", "1")).all()
    ):
        return {"status": "calendar_incomplete", "etfs": {}}

    sessions = set(calendar_dates[calendar["is_trading_day"].eq("1")].dt.date)
    etfs = {}
    for key, record in captured.items():
        if not key.startswith("etf_bars_"):
            continue
        prices = pd.read_csv(output / record["files"][0]["path"], usecols=["date"])
        dates = pd.to_datetime(prices["date"], errors="raise").dt.date
        window = dates[(dates >= start) & (dates <= end)]
        observed = set(window)
        missing = sorted(sessions - observed)
        extra = sorted(observed - sessions)
        etfs[key.removeprefix("etf_bars_")] = {
            "rows_in_window": len(window), "missing_sessions": len(missing),
            "non_session_dates": len(extra), "duplicate_rows": int(window.duplicated().sum()),
            "first_missing_sessions": [day.isoformat() for day in missing[:5]],
            "first_non_session_dates": [day.isoformat() for day in extra[:5]],
        }
    joined_path = output / "exchange_calendar_joined.csv"
    calendar.to_csv(joined_path, index=False, encoding="utf-8", lineterminator="\n", mode="x")
    return {
        "status": "provider_calendar_compared", "sessions": len(sessions), "etfs": etfs,
        "joined_calendar": {"path": joined_path.name, "rows": len(calendar),
                            "sha256": hashlib.sha256(joined_path.read_bytes()).hexdigest()},
    }


def run_probe(output: Path, *, start: date, end: date) -> dict:
    """Capture public provider evidence once in a new directory, with no synthetic fallback."""
    from .protocol import ETF_SYMBOLS, ResearchPeriod

    ResearchPeriod(start, end)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    requests = _request_plan(start, end)
    public_http = [request for request in requests if request.provider != "baostock"]
    anonymous_session = [request for request in requests if request.provider == "baostock"]
    with ThreadPoolExecutor(max_workers=_WORKER_COUNT) as pool:
        records = list(pool.map(lambda request: _collect(request, raw), public_http))
    # BaoStock anonymous logins may share backend session state across clients.
    session_failed = False
    for request in anonymous_session:
        if session_failed:
            records.append({"request": asdict(request), "status": "skipped_after_failure", "files": []})
            continue
        record = _collect(request, raw)
        records.append(record)
        session_failed = record["status"] != "captured"
    counts = sum(record["status"] == "captured" for record in records)
    captured = {record["request"]["key"] for record in records if record["status"] == "captured"}
    report = {
        "schema_version": 1, "origin": "market_provider_probe", "status": "data_insufficient",
        "start": start.isoformat(), "end": end.isoformat(),
        "acquisition_status": "complete" if counts == len(records) else "partial" if counts else "failed",
        "captured_requests": counts, "total_requests": len(records), "sources": records,
        "coverage": {
            "exchange_calendar": "provider_observed" if all(
                request.key in captured for request in requests if request.endpoint == "query_trade_dates"
            ) else "incomplete",
            "etf_raw_bars": "provider_observed" if all(f"etf_bars_{s}" in captured for s in ETF_SYMBOLS) else "incomplete",
            "etf_actions": "third_party_tables" if all(f"etf_actions_{s}" in captured for s in ETF_SYMBOLS) else "incomplete",
            "etf_limits": "missing", "stock_raw_bars": "sample_only" if "stock_bars_600000_sample" in captured else "missing",
            "stock_actions": "sample_only" if "stock_actions_600000_sample" in captured else "missing",
            "stock_membership": "dated_samples_only" if any(k.startswith("csi300_membership_") for k in captured) else "missing",
            "stock_limits": "missing",
        },
        "calendar_alignment": _session_alignment(output, records, start, end),
        "issues": [
            "行情和日历为供应商响应，待核对交易所来源及缺失交易日对应停牌等事件。",
            "新浪累计分红仅作线索，东财分红及折算表待用基金公告交叉核验。",
            "未采集并独立验证历史涨跌停价格、可交易状态及全量公司行动。",
            "个股仅采样单股和两个历史成分日期，未核验成分公告获知时间及退市覆盖。",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (output / "audit.json").write_text(serialized + "\n", encoding="utf-8")
    return json.loads(serialized)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Use the quant_research CLI to run a provider probe.")
    _worker(sys.argv[2], sys.argv[3])
