"""Capture public stock and ETF fund-flow evidence for forward research."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .providers import _Request, _collect
from .snapshot import file_sha256

SOURCE_FIELDS = {
    "主力净流入-净占比": "main_flow_fraction",
    "超大单净流入-净占比": "extra_large_flow_fraction",
    "大单净流入-净占比": "large_flow_fraction",
}
FLOW_COLUMNS = ["date", "symbol", "kind", "known_at", "source", *SOURCE_FIELDS.values()]


def _valid_codes(codes: list[str], prefixes: str) -> tuple[str, ...]:
    if any(not re.fullmatch(rf"[{prefixes}][0-9]{{5}}", code) for code in codes):
        raise ValueError("expected six-digit Shanghai or Shenzhen codes")
    return tuple(sorted(set(codes)))


def capture_fund_flow(out: Path, *, stocks: list[str], etfs: list[str]) -> dict:
    stocks, etfs = _valid_codes(stocks, "036"), _valid_codes(etfs, "15")
    if not 1 <= len(stocks) + len(etfs) <= 20 or set(stocks).intersection(etfs):
        raise ValueError("request 1 to 20 distinct stock or ETF codes")
    plan = [
        _Request(f"stock_flow_{code}", "akshare_eastmoney", "stock_individual_fund_flow",
                 {"stock": code, "market": "sh" if code.startswith("6") else "sz"},
                 "日期", tuple(SOURCE_FIELDS))
        for code in stocks
    ]
    if etfs:
        plan.append(_Request("etf_flow_spot", "akshare_eastmoney", "fund_etf_spot_em", {}, "数据日期",
                             ("代码", "更新时间", *SOURCE_FIELDS)))
    out.mkdir(parents=True, exist_ok=False)
    raw = out / "raw"
    raw.mkdir()
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(lambda request: _collect(request, raw), plan))
    evidence = {"schema_version": 1, "stocks": stocks, "etfs": etfs, "records": records}
    (out / "capture.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    observations = read_fund_flow(out)
    observations.to_parquet(out / "observations.parquet", index=False)
    counts = {code: int(observations["symbol"].eq(code).sum()) for code in (*stocks, *etfs)}
    return {"status": "captured" if all(counts.values()) else "data_insufficient",
            "observations_by_symbol": counts,
            "endpoints": {item["request"]["key"]: item["status"] for item in records}}


def read_fund_flow(capture: Path) -> pd.DataFrame:
    """Read raw hash-checked observations, timed by actual capture, never backdated."""
    meta = json.loads((capture / "capture.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1:
        raise ValueError("unsupported fund-flow capture schema")
    frames = []
    for record in meta["records"]:
        if record["status"] != "captured":
            continue
        request = record["request"]
        endpoint = request["endpoint"]
        if endpoint not in {"stock_individual_fund_flow", "fund_etf_spot_em"}:
            raise ValueError("unsupported fund-flow endpoint")
        known = pd.Timestamp(record["observed_at_utc"])
        if pd.isna(known) or known.tzinfo is None:
            raise ValueError("fund-flow capture time must have a timezone")
        known = known.tz_convert("UTC")
        local = known.tz_convert("Asia/Shanghai")
        today = local.tz_localize(None).normalize()
        [evidence] = record["files"]
        path = capture / "raw" / f"{request['key']}.csv"
        if path.resolve().parent != (capture / "raw").resolve() or file_sha256(path) != evidence["sha256"]:
            raise ValueError("fund-flow evidence hash or path mismatch")
        raw = pd.read_csv(path, dtype={"代码": str})
        if endpoint == "stock_individual_fund_flow":
            code = request["arguments"]["stock"]
            if code not in meta["stocks"]:
                raise ValueError("unexpected stock in fund-flow capture")
            frame = raw.assign(symbol=code, kind="stock")
            frame["date"] = pd.to_datetime(frame["日期"], errors="raise").dt.normalize()
            start = pd.Timestamp(record.get("started_at_utc"))
            close = (today + pd.Timedelta(hours=15)).tz_localize("Asia/Shanghai")
            # A request issued before close can return after close with a partial current row.
            started_after_close = not pd.isna(start) and start.tzinfo is not None and close <= start <= known
            frame = frame.loc[frame["date"].lt(today) | (frame["date"].eq(today) & started_after_close)]
        else:
            frame = raw.loc[raw["代码"].isin(meta["etfs"])].copy()
            frame = frame.assign(symbol=frame["代码"], kind="etf")
            frame["date"] = pd.to_datetime(frame["数据日期"], errors="raise").dt.normalize()
            update = pd.to_datetime(frame["更新时间"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")
            same_day = update.dt.tz_localize(None).dt.normalize().eq(frame["date"])
            frame = frame.loc[same_day & update.dt.hour.ge(15) & update.le(known)]
        frame = frame.assign(known_at=known, source=f"akshare_eastmoney/{endpoint}")
        for name, target in SOURCE_FIELDS.items():
            fraction = pd.to_numeric(frame[name], errors="coerce") / 100.0
            frame[target] = fraction.where(np.isfinite(fraction) & fraction.between(-1, 1))
        frame = frame.dropna(subset=list(SOURCE_FIELDS.values()), how="all")
        if frame.duplicated(["date", "symbol", "kind"]).any():
            raise ValueError("ambiguous duplicate fund-flow observations")
        frames.append(frame[FLOW_COLUMNS])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FLOW_COLUMNS)


def capture_features_asof(
    observations: pd.DataFrame, decisions: pd.DataFrame, features: list[str],
) -> pd.DataFrame:
    """Join only same-session observations first captured by the decision time."""
    keys = ["date", "symbol", "kind"]
    if decisions.duplicated(keys).any() or not isinstance(decisions["as_of"].dtype, pd.DatetimeTZDtype):
        raise ValueError("feature decisions require unique keys and a timezone-aware cutoff")
    if observations.empty:
        return decisions.assign(**dict.fromkeys(features, np.nan))
    candidates = decisions.merge(observations[keys + ["known_at", *features]], on=keys, how="inner")
    eligible = candidates.loc[candidates["known_at"].le(candidates["as_of"])].copy()
    if eligible.duplicated([*keys, "known_at"]).any():
        raise ValueError("ambiguous feature versions at identical timestamp")
    selected = eligible.sort_values("known_at").drop_duplicates(keys, keep="last")
    return decisions.merge(selected[keys + features], on=keys, how="left", validate="one_to_one")


def fund_flow_asof(observations: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    return capture_features_asof(observations, decisions, list(SOURCE_FIELDS.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stock", action="append", default=[])
    parser.add_argument("--etf", action="append", default=[])
    args = parser.parse_args()
    report = capture_fund_flow(args.out, stocks=args.stock, etfs=args.etf)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "captured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
