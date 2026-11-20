#!/usr/bin/env python3
"""Run the whole repository's checks in one command.

The project has three independent test suites plus an evidence archive, and
until now they were run from different directories with different interpreters.
This gate runs all of them, verifies the archived evidence hashes, audits the
frozen forward window, and writes a timestamped JSON record.

Exit code is non-zero only when a test or the evidence archive fails. Missing
forward records are reported as warnings unless ``--strict-forward`` is passed,
because data collection depends on the market calendar rather than on the code.

Usage:
    python3 tools/quality_gate.py
    python3 tools/quality_gate.py --strict-forward
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_VENV = ROOT / "research_v1" / ".venv" / "bin" / "python"
PROTOTYPE_VENV = ROOT / ".venv" / "bin" / "python"
FORWARD_DIR = ROOT / "artifacts" / "news-filter-forward-20261116"
EVIDENCE = ROOT / "research_v1" / "evidence"
SNAPSHOT = ROOT / "research_v1" / "artifacts" / "market-etf-verified-20261108"


def run(cmd: list[str], cwd: Path, extra_env: dict[str, str] | None = None,
        timeout: int = 900) -> dict:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        env.update(extra_env)
    started = time.monotonic()
    completed = subprocess.run(cmd, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout)
    return {
        "cmd": " ".join(str(part) for part in cmd),
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "seconds": round(time.monotonic() - started, 2),
        "output": completed.stdout,
    }


def last_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def pytest_summary(result: dict) -> str:
    match = re.search(r"(\d+) passed", result["output"])
    if match:
        return f"{match.group(1)} passed"
    return last_line(result["output"]) or "failed"


def unittest_summary(result: dict) -> str:
    match = re.search(r"Ran (\d+) tests?", result["output"])
    count = f"{match.group(1)} tests" if match else "tests"
    return count + (" OK" if result["returncode"] == 0 else " FAILED")


def audit_summary(result: dict) -> str:
    match = re.search(r'"status": "([a-z_]+)"', result["output"])
    return match.group(1) if match else last_line(result["output"])


def observed_counts(results: list[dict]) -> tuple[int, int, int, int]:
    research = int(re.search(r"(\d+) passed", results[0]["output"]).group(1))
    prototype = int(re.search(r"(\d+) passed", results[1]["output"]).group(1))
    subtests_match = re.search(r"(\d+) subtests passed", results[1]["output"])
    subtests = int(subtests_match.group(1)) if subtests_match else 0
    forward = int(re.search(r"Ran (\d+) tests?", results[2]["output"]).group(1))
    return research, prototype, subtests, forward


def documented_counts() -> tuple[int, int, int, int] | None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"分别通过\s*\*\*(\d+)\s*项、(\d+)\s*项（另有\s*(\d+)\s*个子测试）和\s*(\d+)\s*项测试\*\*",
        text,
    )
    if not match:
        return None
    return tuple(int(value) for value in match.groups())


def git_state() -> dict:
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], ROOT)
    head = run(["git", "rev-parse", "--short", "HEAD"], ROOT)
    status = run(["git", "status", "--porcelain"], ROOT)
    dirty = len([line for line in status["output"].splitlines() if line.strip()])
    return {
        "branch": last_line(branch["output"]),
        "head": last_line(head["output"]),
        "dirty_entries": dirty,
    }


def protocol_window() -> tuple[date, date] | None:
    protocol = FORWARD_DIR / "protocol.md"
    if not protocol.is_file():
        return None
    text = protocol.read_text(encoding="utf-8")
    match = re.search(r"#\s*(\d{4}-\d{2}-\d{2})\s*至\s*(\d{4}-\d{2}-\d{2})", text)
    if not match:
        return None
    return date.fromisoformat(match.group(1)), date.fromisoformat(match.group(2))


def audit_forward(today: date) -> dict:
    window = protocol_window()
    if window is None:
        return {"status": "missing_protocol", "days": []}
    start, end = window
    days = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            day_dir = FORWARD_DIR / cursor.isoformat()
            opening = day_dir / "opening.json"
            daily = day_dir / "daily.json"
            summary = day_dir / "close-summary.json"
            record = {
                "date": cursor.isoformat(),
                "opening": opening.is_file(),
                "daily": daily.is_file(),
                "close_summary": summary.is_file(),
            }
            if daily.is_file():
                try:
                    payload = json.loads(daily.read_text(encoding="utf-8"))
                    record["daily_date_matches"] = payload.get("date") == cursor.isoformat()
                except json.JSONDecodeError:
                    record["daily_date_matches"] = False
            complete = record["opening"] and record["daily"] and record.get("daily_date_matches")
            if complete:
                state = "complete"
            elif cursor == today:
                state = "in_progress"
            elif cursor > today:
                state = "pending"
            elif record["opening"]:
                state = "opening_only"
            else:
                state = "missing"
            record["state"] = state
            days.append(record)
        cursor += timedelta(days=1)
    incomplete = [day["date"] for day in days if day["state"] in ("missing", "opening_only")]
    return {"status": "ok" if not incomplete else "gaps", "window": [start.isoformat(), end.isoformat()],
            "days": days, "gaps": incomplete}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-forward", action="store_true",
                        help="treat missing forward records as a failure")
    parser.add_argument("--no-record", action="store_true", help="do not write the JSON record")
    args = parser.parse_args()

    if not RESEARCH_VENV.is_file() or not PROTOTYPE_VENV.is_file():
        print("missing virtualenv: run `uv sync` in research_v1 and the repository root", file=sys.stderr)
        raise SystemExit(2)

    steps = [
        ("research tests",
         [str(RESEARCH_VENV), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
         ROOT / "research_v1", None, pytest_summary),
        ("prototype tests",
         [str(PROTOTYPE_VENV), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
         ROOT, None, pytest_summary),
        ("forward tests",
         [str(RESEARCH_VENV), "-m", "unittest", "discover", "-s", str(FORWARD_DIR), "-p", "test_*.py"],
         ROOT, {"PYTHONPATH": str(ROOT / "research_v1")}, unittest_summary),
        ("evidence archive",
         [str(RESEARCH_VENV), str(EVIDENCE / "build_manifest.py"), "--check"],
         ROOT, None, lambda result: last_line(result["output"])),
    ]
    if SNAPSHOT.is_dir():
        steps.append(
            ("data audit",
             [str(RESEARCH_VENV), "-m", "quant_research.cli", "audit", str(SNAPSHOT),
              "--end", "2024-12-31", "--kind", "etf"],
             ROOT / "research_v1", None, audit_summary)
        )

    results = []
    failed = 0
    print(f"AI量化 quality gate  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    for name, cmd, cwd, env, summarizer in steps:
        try:
            result = run(cmd, cwd, env)
        except subprocess.TimeoutExpired:
            result = {"cmd": " ".join(cmd), "cwd": str(cwd), "returncode": 124,
                      "seconds": 0.0, "output": "timeout"}
        result["name"] = name
        result["summary"] = summarizer(result)
        result["ok"] = result["returncode"] == 0
        results.append(result)
        if not result["ok"]:
            failed += 1
        print(f"{'PASS' if result['ok'] else 'FAIL'}  {name:<18} {result['summary']:<28} {result['seconds']:>6.1f}s")
        if not result["ok"]:
            for line in result["output"].splitlines()[-12:]:
                print(f"      {line}")

    forward = audit_forward(date.today())
    forward_ok = forward["status"] == "ok"
    if forward["status"] == "missing_protocol":
        print("WARN  forward window     no protocol.md found")
    else:
        label = "PASS" if forward_ok else "WARN"
        detail = "、".join(forward["gaps"]) if forward["gaps"] else "all records complete"
        print(f"{label}  {'forward window':<18} {detail:<28}")
        if args.strict_forward and not forward_ok:
            failed += 1

    doc_ok = None
    try:
        observed = observed_counts(results)
        claimed = documented_counts()
        if claimed is None:
            print("WARN  docs counts        README claim not parseable")
        else:
            doc_ok = observed == claimed
            detail = "README 一致" if doc_ok else f"README {claimed} vs 实测 {observed}"
            print(f"{'PASS' if doc_ok else 'WARN'}  {'docs counts':<18} {detail:<28}")
    except (AttributeError, ValueError, IndexError):
        print("WARN  docs counts        could not derive test counts")

    state = git_state()
    print(f"      git                {state['branch']} @ {state['head']}, {state['dirty_entries']} dirty entries")

    record = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git": state,
        "steps": [{key: value for key, value in result.items() if key != "output"} for result in results],
        "forward": forward,
        "docs_counts_match": doc_ok,
        "ok": failed == 0,
    }
    if not args.no_record:
        output_dir = ROOT / "artifacts" / "quality-gate"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"      record             {path.relative_to(ROOT)}")

    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
