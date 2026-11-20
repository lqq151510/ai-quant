"""Guards the archived evidence behind REJECTED.md.

A fresh clone only has the files under ``research_v1/evidence``; the original
reports live in ignored ``artifacts`` directories. These tests keep the links,
hashes and manifest in sync so a registry row cannot silently point at a file
that a reader cannot open.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research_v1" / "evidence" / "MANIFEST.json"
REGISTRY = ROOT / "research_v1" / "REJECTED.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def manifest_entries() -> list[dict]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["entries"], "evidence manifest is empty"
    return data["entries"]


def registry_rows() -> list[tuple[str, str]]:
    rows = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| R-"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append((cells[0], cells[-1]))
    assert rows, "no candidate rows found in REJECTED.md"
    return rows


def test_archived_files_match_manifest_hashes():
    for entry in manifest_entries():
        path = ROOT / entry["evidence_path"]
        assert entry["evidence_path"].startswith("research_v1/evidence/")
        assert path.is_file(), entry["evidence_path"]
        assert sha256(path) == entry["sha256"], entry["evidence_path"]
        assert entry["candidate_ids"]


def test_every_registry_row_links_to_an_archived_file():
    known = {entry["evidence_path"] for entry in manifest_entries()}
    for candidate, evidence_cell in registry_rows():
        links = re.findall(r"`([^`]+)`", evidence_cell)
        assert links, f"{candidate} has no evidence link"
        for link in links:
            assert link in known, f"{candidate} links to unarchived {link}"
            assert (ROOT / link).is_file(), f"{candidate} links to missing {link}"


def test_registry_candidates_and_manifest_are_the_same_set():
    registry_ids = {candidate for candidate, _ in registry_rows()}
    manifest_ids = {
        candidate for entry in manifest_entries() for candidate in entry["candidate_ids"]
    }
    assert registry_ids == manifest_ids
