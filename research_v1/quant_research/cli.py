"""Auditable local CLI. Market ingestion and synthetic demos are separate commands."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from .protocol import HOLDOUT_START, RESEARCH_START, ResearchPeriod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant-research")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="capture a bounded set of real provider samples")
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--start", type=date.fromisoformat, default=RESEARCH_START)
    probe.add_argument("--end", type=date.fromisoformat, required=True)
    demo = commands.add_parser("demo", help="run engineering checks on a marked synthetic fixture")
    demo.add_argument("--out", type=Path, required=True)
    audit = commands.add_parser("audit", help="verify immutable Parquet and the market study gate")
    backtest = commands.add_parser("baseline", help="run the baselines only before the sealed test interval")
    for sub in (audit, backtest):
        sub.add_argument("snapshot", type=Path)
        sub.add_argument("--end", type=date.fromisoformat, required=True)
        sub.add_argument("--kind", choices=("all", "etf", "stock"), default="all")
    backtest.add_argument("--out", type=Path, required=True)
    evidence = commands.add_parser("evidence", help="calculate forward evidence features at an explicit cutoff")
    evidence.add_argument("bundle", type=Path)
    evidence.add_argument("--as-of", type=datetime.fromisoformat, required=True)
    evidence.add_argument("--target", action="append", required=True)
    evidence.add_argument("--lookback-days", type=int, default=7)
    prepare = commands.add_parser("prepare-etf", help="normalize captured tables, keeping unverified evidence blocked")
    prepare.add_argument("--probe", type=Path, required=True)
    prepare.add_argument("--calendar", type=Path, required=True)
    prepare.add_argument("--bulk", type=Path, default=None, help="cross-checked multi-source bulk evidence directory")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--start", type=date.fromisoformat, default=RESEARCH_START)
    prepare.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)

    def emit(value: object) -> None:
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

    try:
        if args.command == "probe":
            from .providers import run_probe

            report = run_probe(args.out, start=args.start, end=args.end)
            emit({"status": report["status"], "acquisition_status": report["acquisition_status"],
                  "audit": str((args.out / "audit.json").resolve())})
            return 0
        if args.command == "evidence":
            from .evidence import evidence_features, load_evidence

            rows = evidence_features(load_evidence(args.bundle), as_of=args.as_of,
                                     lookback=timedelta(days=args.lookback_days), targets=args.target)
            emit([r.model_dump(mode="json") for r in rows])
            return 0
        if args.command == "prepare-etf":
            from .prepare import prepare_etf_snapshot

            manifest = prepare_etf_snapshot(
                args.probe,
                args.calendar,
                args.out,
                ResearchPeriod(args.start, args.end),
                bulk_dir=args.bulk,
            )
            emit({"origin": manifest["origin"], "coverage": manifest["coverage"], "snapshot": str(args.out.resolve())})
            return 0

        from .snapshot import audit_snapshot, read_snapshot, write_snapshot

        if args.command == "demo":
            from .demo import fixture

            args.out.mkdir(parents=True, exist_ok=False)
            frames = fixture()
            write_snapshot(args.out / "fixture", origin="synthetic_fixture", frames=frames, sources=[])
            manifest, frames = read_snapshot(args.out / "fixture")
            start, end = frames["calendar"]["date"].iloc[[0, -1]]
            origin = manifest["origin"]
            kinds = ("etf", "stock")
        else:
            if args.command == "baseline" and args.end >= HOLDOUT_START:
                raise ValueError("baseline command does not open the sealed test interval")
            manifest, frames = read_snapshot(args.snapshot)
            period = ResearchPeriod(RESEARCH_START, args.end)
            kinds = ("etf", "stock") if args.kind == "all" else (args.kind,)
            gates = [audit_snapshot(manifest, frames, period, kind) for kind in kinds]
            ready = all(row["status"] == "ready" for row in gates)
            if args.command == "audit" or not ready:
                emit(gates)
                return 0 if ready else 2
            args.out.mkdir(parents=True, exist_ok=False)
            start, end = period.start, period.end
            origin = "audited_market"
            (args.out / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        from .baseline import run_baselines
        from .report import save_report

        import platform
        from .snapshot import file_sha256

        code_dir = Path(__file__).parent
        lock = code_dir.parent / "uv.lock"
        archive = args.out / "source"
        archive.mkdir()
        for source in sorted(code_dir.glob("*.py")):
            (archive / source.name).write_bytes(source.read_bytes())
        fingerprint = {
            "origin": origin, "python": sys.version, "platform": platform.platform(),
            "window": {"start": str(start), "end": str(end)},
            "code_sha256": {p.name: file_sha256(p) for p in sorted(archive.glob("*.py"))},
            "lock_sha256": file_sha256(lock) if lock.exists() else None,
        }
        (args.out / "run_manifest.json").write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")

        results = run_baselines(frames, args.out, start=start, end=end, origin=origin, kinds=kinds)
        save_report(args.out, results, origin=origin, period=f"{start:%Y-%m-%d} — {end:%Y-%m-%d}")
        emit({"origin": origin, "ledger_runs": len(results), "report": str((args.out / "report.html").resolve())})
        return 0
    except (ValueError, FileExistsError, OSError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
