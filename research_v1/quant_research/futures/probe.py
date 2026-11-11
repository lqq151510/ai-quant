"""Capture a bounded public historical sample without pretending it is audited."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .engine import FuturesError
from .output import sha256

_CHILD = r'''
import json, sys
try:
    import akshare as ak
    frame = ak.futures_zh_daily_sina(symbol=sys.argv[1])
    if frame is None or frame.empty:
        print(json.dumps({"status": "empty"}))
    else:
        frame.to_csv(sys.argv[2], index=False, encoding="utf-8")
        print(json.dumps({"status": "captured", "rows": len(frame), "columns": list(frame.columns)}))
except Exception as exc:
    print(json.dumps({"status": "provider_error", "error_type": type(exc).__name__}))
'''


def capture_sina(symbol: str, output: Path, *, timeout: int = 30, attempts: int = 1) -> dict:
    match = re.fullmatch(r"[A-Za-z]{1,5}(\d{3,4})", symbol)
    if not match or match.group(1) in {"000", "888", "999", "0000", "8888", "9999"}:
        raise FuturesError("supply a dated Sina contract code, not a continuous/index symbol")
    if not 1 <= timeout <= 60 or attempts not in (1, 2):
        raise FuturesError("timeout must be 1–60 seconds and attempts must be 1 or 2")
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw.csv"
    audit = {"source": "akshare.futures_zh_daily_sina", "source_page": "https://finance.sina.com.cn/futures/",
             "symbol": symbol, "started_utc": datetime.now(timezone.utc).isoformat(),
             "qualification": "data_insufficient", "attempts": [],
             "missing_evidence": ["historical_upper_lower_limits", "contract_and_fee_schedule",
                                  "exchange_calendar_and_historical_availability"]}
    for _ in range(attempts):
        try:
            child = subprocess.run([sys.executable, "-c", _CHILD, symbol, str(raw)], capture_output=True,
                                   text=True, timeout=timeout, check=False)
            try:
                response = json.loads(child.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                response = {"status": "process_error", "exit_code": child.returncode}
        except subprocess.TimeoutExpired:
            response = {"status": "timeout"}
        audit["attempts"].append(response)
        if response.get("status") == "captured" and raw.is_file():
            audit["raw_sha256"] = sha256(raw)
            break
        raw.unlink(missing_ok=True)  # Only the fresh, partially written output of this attempt.
    audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit
