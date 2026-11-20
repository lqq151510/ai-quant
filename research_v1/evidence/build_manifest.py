"""Archive the evidence files cited by REJECTED.md and verify their hashes.

The candidate registry points at report files that live under ignored
``artifacts/`` directories, so a fresh clone cannot check any of the numbers.
This script copies the cited files byte-for-byte into ``research_v1/evidence``
and records the original path, size and SHA-256 in ``MANIFEST.json``.

Rules:
- an existing archive file is never overwritten; a hash mismatch is an error;
- the archive keeps the artifact directory layout, so links stay readable;
- ``--check`` only verifies, and exits non-zero on any missing or changed file.

Usage:
    python research_v1/evidence/build_manifest.py
    python research_v1/evidence/build_manifest.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "research_v1" / "evidence"
MANIFEST = EVIDENCE_DIR / "MANIFEST.json"
CST = timezone(timedelta(hours=8))

# (candidate ids, repo-relative source path)
SOURCES: list[tuple[tuple[str, ...], str]] = [
    (("R-01",), "research_v1/artifacts/momentum-study-20261111-a/report.md"),
    (("R-02", "R-13"), "research_v1/artifacts/control-study-20261114/report.md"),
    (("R-02", "R-04", "R-13"), "research_v1/artifacts/control-study-20261114-h20/report.md"),
    (("R-02", "R-13"), "research_v1/artifacts/control-study-20261114-h60/report.md"),
    (("R-03",), "research_v1/artifacts/reversal-5-sessions-20261114/report.md"),
    (("R-05", "R-06"), "research_v1/artifacts/reversal-60-exposure-20261114/report.md"),
    (("R-05",), "research_v1/artifacts/reversal-60-subwindows-20261114/report.md"),
    (("R-07",), "artifacts/net-edge-study-20261111/report.md"),
    (("R-08",), "artifacts/entry-timing-study-20261111/report.md"),
    (("R-09", "R-10"), "artifacts/rl-self-review-20261111/report.md"),
    (("R-11",), "research_v1/artifacts/momentum-rl-overlay-20261111-e/report.md"),
    (("R-12",), "artifacts/etf-deadband-study-20261110/results/report.json"),
    (("R-14",), "research_v1/artifacts/etf-vol-target-12pct-20261114/report.md"),
    (("R-15",), "research_v1/artifacts/vol-target-20261114/results.json"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_path(source: str) -> str:
    """Map an artifact path to the archive path, preserving the layout below it."""
    parts = Path(source).parts
    if parts and parts[0] == "research_v1":
        parts = parts[1:]
    if not parts or parts[0] != "artifacts":
        raise ValueError(f"source is outside artifacts/: {source}")
    return (Path("research_v1") / "evidence" / Path(*parts[1:])).as_posix()


def build() -> int:
    entries = []
    changed = 0
    for ids, source in SOURCES:
        source_path = ROOT / source
        if not source_path.is_file():
            print(f"MISSING SOURCE  {source}", file=sys.stderr)
            return 1
        target_rel = evidence_path(source)
        target = ROOT / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(source_path)
        if target.exists():
            if sha256(target) != source_hash:
                print(f"HASH MISMATCH   {target_rel} != {source}", file=sys.stderr)
                return 1
        else:
            shutil.copy2(source_path, target)
            changed += 1
        text = target.read_bytes()
        entries.append(
            {
                "candidate_ids": list(ids),
                "source_path": source,
                "evidence_path": target_rel,
                "sha256": source_hash,
                "bytes": target.stat().st_size,
                "source_mtime": datetime.fromtimestamp(source_path.stat().st_mtime, CST).isoformat(),
                "contains_machine_paths": "/Users/" in text.decode("utf-8", "replace"),
            }
        )
    manifest = {
        "archived_at": datetime.now(CST).isoformat(timespec="seconds"),
        "note": (
            "REJECTED.md 引用文件的仓库内副本；字节与原件一致，哈希按原件计算。"
            "含本机绝对路径的报告原样保留，以便核对哈希，路径映射见 source_path。"
        ),
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"archived {len(entries)} files ({changed} copied now) -> {MANIFEST.relative_to(ROOT)}")
    return 0


def check() -> int:
    if not MANIFEST.is_file():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = 0
    for entry in manifest["entries"]:
        path = ROOT / entry["evidence_path"]
        if not path.is_file():
            print(f"MISSING  {entry['evidence_path']}", file=sys.stderr)
            problems += 1
        elif sha256(path) != entry["sha256"]:
            print(f"CHANGED  {entry['evidence_path']}", file=sys.stderr)
            problems += 1
    print(f"checked {len(manifest['entries'])} files, {problems} problem(s)")
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the archive without writing")
    args = parser.parse_args()
    raise SystemExit(check() if args.check else build())


if __name__ == "__main__":
    main()
