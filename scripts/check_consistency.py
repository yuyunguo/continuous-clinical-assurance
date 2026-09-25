"""Cross-document consistency checks for the CCA project.

Checks: relative file references resolve; indicator, scenario, requirement, and reference IDs cited in
documents exist; `file.md section N.M` cross-references point at real headings; stale terms and
phrases are absent; numeric claims match values recomputed from the data; scenario hypothesis tags
are used by the Phase II tests; the indicator instrument covers the matrix constructs.

Usage: python scripts/check_consistency.py
Exit status is non-zero when any error is found. Warnings do not fail the run.
"""

import glob
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_indicators import INDICATOR_DIR, load_records  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["Indicator-Instrument.md", "Instrument-Specification.md", "phase1/Phase-I-Synthesis.md",
        "phase1/Assurance-Control-Matrix.md", "phase1/Citation-Audit-Report.md", "phase2/Phase-II-Simulation-Design.md",
        "phase2/Scenario-Coverage.md", "docs/superpowers/specs/2026-09-19-indicator-instrument-design.md",
        "Research-Alignment-Review.md"]
SOURCE_DOCS = ("Continuous_Clinical_Assurance_Research_Plan.md", "Evidence-and-measurement-matrix.md")
FILE_REF = re.compile(r"`([A-Za-z0-9_\-./*]+\.(?:md|yaml|py|bib|json|txt))`")
SECTION_REF = re.compile(r"`([A-Za-z0-9_\-./]+\.md)`\s+sections?\s+([0-9]+(?:\.[0-9]+)?)")
STALE = ["Phase II tunes", "tuned on calibration streams only", "Time to Assurance Failure (TAF)\n"]
RETIRED_SCENARIOS = {"SC03"}  # retired 2026-09-19 as redundant with SC08; history may still mention it
STALE_EXEMPT = {("phase2/Phase-II-Simulation-Design.md", "tuned on calibration streams only")}  # quoted in the revision log


def _read(rel: str) -> str:
    """Read a repo-relative text file."""
    return (ROOT / rel).read_text(encoding="utf-8")


def check_paths(errors: List[str]) -> None:
    """Every backticked file reference must resolve from the document's folder or the repo root."""
    for doc in DOCS:
        if not (ROOT / doc).exists():
            continue
        base = (ROOT / doc).parent
        for ref in sorted(set(FILE_REF.findall(_read(doc)))):
            candidates = [base / ref, ROOT / ref]
            if "*" in ref:
                found = any(glob.glob(str(c)) for c in candidates)
            else:
                found = any(c.exists() for c in candidates) or any(
                    "archive" not in str(hit) for hit in ROOT.rglob(Path(ref).name))
            if not found and not ref.startswith(("src/", "tests/", "run/")):
                errors.append(f"{doc}: file reference `{ref}` does not resolve")


def check_ids(errors: List[str], ids: Dict[str, Set[str]]) -> None:
    """Indicator, scenario, requirement, and reference-key IDs cited in documents must exist."""
    patterns = {"indicator": (r"\b[CPUHOLG][1-9]\b", ids["indicator"]), "scenario": (r"\bSC\d{2}\b", ids["scenario"] | RETIRED_SCENARIOS),
                "requirement": (r"\bR\d{2}\b", ids["requirement"])}
    for doc in DOCS:
        if not (ROOT / doc).exists():
            continue
        text = _read(doc)
        for kind, (pattern, valid) in patterns.items():
            bad = sorted(set(re.findall(pattern, text)) - valid)
            if bad:
                errors.append(f"{doc}: unknown {kind} IDs {bad}")
        bad_keys = sorted(set(re.findall(r"\[([a-z]+\d{4})\]", text)) - ids["bibkey"])
        if bad_keys:
            errors.append(f"{doc}: reference keys not in references.bib {bad_keys}")


def check_sections(errors: List[str]) -> None:
    """`file.md section N.M` must point at an existing heading."""
    for doc in DOCS:
        if not (ROOT / doc).exists():
            continue
        for target, num in SECTION_REF.findall(_read(doc)):
            path = next((p for p in [(ROOT / doc).parent / target, ROOT / target] if p.exists()), None)
            if path is None:
                continue
            heads = re.findall(r"(?m)^#{2,4}\s+([0-9]+(?:\.[0-9]+)?)[.\s]", path.read_text(encoding="utf-8"))
            if num not in heads:
                errors.append(f"{doc}: cross-reference to {target} section {num} has no such heading")


def check_stale(errors: List[str]) -> None:
    """Stale terms and phrases from earlier versions."""
    for doc in DOCS:
        if not (ROOT / doc).exists():
            continue
        text = _read(doc)
        errors += [f"{doc}: stale phrase {p.strip()!r}" for p in STALE if p in text and (doc, p) not in STALE_EXEMPT]
        for n, line in enumerate(text.splitlines(), 1):
            if re.search(r"\bTAF\b", line) and not re.search(r"TAFD|plan|old name|renamed|Terminology", line):
                errors.append(f"{doc}:{n}: standalone 'TAF' (metric is TAFD)")


def check_numbers(errors: List[str], facts: Dict[str, Any]) -> None:
    """Numeric statements in documents must match values recomputed from data."""
    rules: List[Tuple[str, str, int]] = [
        ("Indicator-Instrument.md", r"(\d+) candidate indicators", facts["indicators"]),
        ("Instrument-Specification.md", r"the (\d+) indicator records", facts["indicators"]),
        ("Instrument-Specification.md", r"The (\d+) indicators distribute", facts["indicators"]),
        ("phase2/Phase-II-Simulation-Design.md", r"Of the (\d+) indicators", facts["indicators"]),
        ("phase2/Phase-II-Simulation-Design.md", r"(\d+) carry the `predictive` tag", facts["predictive"]),
        ("phase2/Scenario-Coverage.md", r"(\d+) of \d+ indicators apply to predictive AI", facts["predictive"]),
        ("phase2/Phase-II-Simulation-Design.md", r"holds (\d+) scenarios", facts["scenarios"]),
        ("phase2/Phase-II-Simulation-Design.md", r"giving (\d+) scenario-level-shape cells", facts["cells"]),
        ("phase2/Phase-II-Simulation-Design.md", r"\((\d+) cells before sub-threshold exclusions", facts["rh1_cells"]),
        ("phase2/Phase-II-Simulation-Design.md", r"Total \| \| \| ([\d,]+) ", facts["streams"]),
        ("phase1/Phase-I-Synthesis.md", r"Of (\d+) coded requirements", facts["requirements"]),
        ("phase1/Phase-I-Synthesis.md", r"All (\d+) references passed", facts["refs"]),
        ("phase1/Phase-I-Synthesis.md", r"(\d+) VALID \(two or more", facts["refs_valid"]),
        ("phase1/Citation-Audit-Report.md", r"(\d+) candidates:", facts["refs"]),
        ("Instrument-Specification.md", r"(\d+) of the matrix's 20 constructs", facts["matrix_covered"]),
    ]
    for doc, pattern, expected in rules:
        if not (ROOT / doc).exists():
            continue
        m = re.search(pattern, _read(doc))
        if not m:
            errors.append(f"{doc}: expected a statement matching {pattern!r}, none found")
        elif int(m.group(1).replace(",", "")) != expected:
            errors.append(f"{doc}: states {m.group(1)} for {pattern!r}, data gives {expected}")


def check_records(errors: List[str], records: List[Dict[str, Any]]) -> None:
    """Records that do not apply to predictive AI must not claim a Phase II test."""
    for r in records:
        if "predictive" not in r["ai_types"] and re.search(r"(?<!outside )Phase II\b", r["validation"]):
            errors.append(f"{r['id']}: validation cites Phase II but the indicator is outside the predictive anchor")


def check_spec_tables(errors: List[str], records: List[Dict[str, Any]]) -> None:
    """The component and evidence-level tables of the specification must list exactly the records' IDs."""
    by_component: Dict[str, Set[str]] = {}
    by_level: Dict[int, Set[str]] = {}
    for r in records:
        by_component.setdefault(r["state_component"], set()).add(r["id"])
        by_level.setdefault(r["evidence_level"], set()).add(r["id"])
    for row in _read("Instrument-Specification.md").splitlines():
        cells = [c.strip() for c in row.strip("|").split("|")]
        if not row.startswith("| ") or len(cells) != 4 or not cells[3].isdigit():
            continue
        listed = set(re.findall(r"\b[A-Z]\d\b", cells[2]))
        expected = by_component.get(cells[0][:1]) if cells[0][1:2] == "," else by_level.get(int(cells[0])) if cells[0].isdigit() else None
        if expected is None:
            continue
        if listed != expected or int(cells[3]) != len(expected):
            errors.append(f"Instrument-Specification.md: table row {cells[0]!r} lists {sorted(listed)} (n={cells[3]}), records give {sorted(expected)}")


def check_hypothesis_use(errors: List[str], warnings: List[str], scenarios: List[Dict[str, Any]]) -> None:
    """Every scenario tagged with RHk and a harm-defined onset must appear in the Phase II RHk row."""
    text = _read("phase2/Phase-II-Simulation-Design.md")
    rows = {m.group(1): m.group(0) for m in re.finditer(r"(?m)^\| (RH[1-4]) \|.*$", text)}
    for s in scenarios:
        if s["family"] == "control":
            continue
        for h in s["hypotheses"]:
            if h in rows and s["id"] not in rows[h]:
                (warnings if not s["primary_tafd"] else errors).append(
                    f"{s['id']} is tagged {h} but is not used in the {h} test row of Phase II section 9.1")


def check_matrix(errors: List[str], warnings: List[str], records: List[Dict[str, Any]]) -> None:
    """Every record needs a matrix_construct, and every matrix construct should have an indicator."""
    matrix = _read("Evidence-and-measurement-matrix.md")
    rows = re.findall(r"(?m)^\|\s*\*\*[^|]+\*\*\s*\|[^|]*\|[^|]*\|\s*([^|]+?)\s*\|", matrix)
    constructs = list(dict.fromkeys(rows))
    used = {r.get("matrix_construct") for r in records}
    schema_list = yaml.safe_load(_read("indicators/schema.yaml"))["matrix_constructs"]
    if schema_list != constructs:
        errors.append("indicators/schema.yaml matrix_constructs differ from the matrix master table")
    errors += [f"{r['id']}: no matrix_construct" for r in records if not r.get("matrix_construct")]
    errors += [f"{r['id']}: matrix_construct {r['matrix_construct']!r} is not a matrix construct"
               for r in records if r.get("matrix_construct") and r["matrix_construct"] not in constructs]
    out_of_scope = yaml.safe_load(_read("indicators/schema.yaml")).get("out_of_scope_constructs") or {}
    warnings += [f"matrix construct {c!r} has no indicator" for c in constructs if c not in used and c not in out_of_scope]


def main() -> int:
    """Run all checks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    records = load_records(INDICATOR_DIR)
    scenarios = yaml.safe_load(_read("phase2/scenarios.yaml"))
    reqs = yaml.safe_load(_read("phase1/requirements.yaml"))
    audit = json.loads(_read("phase1/citation_audit.json"))
    perturb = [s for s in scenarios if s["family"] != "control"]
    cells = sum(len(s["perturbation"]["shapes"]) * 3 for s in perturb)
    core = {s["id"] for s in perturb if "RH1" in s["hypotheses"] and s["primary_tafd"] and s["purpose"] == "fair_test"}
    sens_cells = 4 * cells + sum(len(s["perturbation"]["shapes"]) * 3 for s in perturb if s["id"] in ("SC06", "SC09"))
    facts = {"indicators": len(records), "predictive": sum("predictive" in r["ai_types"] for r in records),
             "scenarios": len(scenarios), "cells": cells,
             "rh1_cells": sum(len(s["perturbation"]["shapes"]) * 3 for s in perturb if s["id"] in core),
             "streams": cells * 500 + sens_cells * 200 + 10000 + 2000, "requirements": len(reqs), "refs": len(audit),
             "refs_valid": sum(v["status"] == "VALID" for v in audit),
             "matrix_covered": len({r["matrix_construct"] for r in records})}
    ids = {"indicator": {r["id"] for r in records}, "scenario": {s["id"] for s in scenarios},
           "requirement": {r["id"] for r in reqs},
           "bibkey": set(re.findall(r"@\w+\{([^,\s]+),", _read("phase1/references.bib")))}
    errors: List[str] = []
    warnings: List[str] = []
    check_paths(errors)
    check_ids(errors, ids)
    check_sections(errors)
    check_stale(errors)
    check_numbers(errors, facts)
    check_records(errors, records)
    check_spec_tables(errors, records)
    check_hypothesis_use(errors, warnings, scenarios)
    check_matrix(errors, warnings, records)
    for w in warnings:
        logger.warning(w)
    for e in errors:
        logger.error(e)
    logger.info("%d error(s), %d warning(s); facts: %s", len(errors), len(warnings), facts)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
