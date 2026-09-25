"""Validate the Phase II scenario catalog against the indicator instrument.

Checks: unique IDs; required keys; indicator IDs exist and apply to predictive AI; the response
sets of a scenario are mutually consistent; hypotheses are valid; every predictive-applicable
indicator is a primary responder in at least one scenario; each Phase II hypothesis has at least
two scenarios. Renders phase2/Scenario-Coverage.md.

Usage: python scripts/validate_scenarios.py
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_indicators import INDICATOR_DIR, load_records  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "phase2" / "scenarios.yaml"
OUTPUT = ROOT / "phase2" / "Scenario-Coverage.md"
PURPOSES = ("control", "fair_test", "blind_spot", "harmless_drift", "process")
REQUIRED = ("id", "name", "family", "purpose", "hypotheses", "primary_tafd", "label_dependence", "perturbation",
            "primary", "secondary", "held_stable", "expected_null", "note")
PHASE2_HYPOTHESES = ("RH1", "RH2", "RH3", "RH4")
ALL_HYPOTHESES = PHASE2_HYPOTHESES + ("RH5",)
LABEL_MODES = ("label_free", "label_dependent", "none")
SETS = ("primary", "secondary", "held_stable", "expected_null")


def _ids(scenario: Dict[str, Any], key: str) -> Set[str]:
    """Return the indicator IDs in a response set ('all' expands to nothing here)."""
    value = scenario[key]
    return set(value) if isinstance(value, list) else set()


def validate(scenarios: List[Dict[str, Any]], applicable: Set[str], known: Set[str]) -> List[str]:
    """Return validation errors for the catalog."""
    errors: List[str] = []
    seen: Set[str] = set()
    for s in scenarios:
        sid = s.get("id", "?")
        if sid in seen:
            errors.append(f"duplicate id {sid}")
        seen.add(sid)
        errors += [f"{sid}: missing key {k}" for k in REQUIRED if k not in s]
        if any(k not in s for k in REQUIRED):
            continue
        for key in SETS:
            if s[key] == "all" and key == "expected_null" and s["family"] == "control":
                continue
            if not isinstance(s[key], list):
                errors.append(f"{sid}: {key} must be a list")
                continue
            for i in s[key]:
                if i not in known:
                    errors.append(f"{sid}: {key} has unknown indicator {i}")
                elif i not in applicable:
                    errors.append(f"{sid}: {key} indicator {i} does not apply to predictive AI")
        if all(isinstance(s[k], list) for k in SETS):
            pairs = [("primary", "expected_null"), ("primary", "held_stable"), ("secondary", "expected_null"),
                     ("held_stable", "expected_null"), ("primary", "secondary")]
            errors += [f"{sid}: {a} and {b} overlap on {sorted(_ids(s, a) & _ids(s, b))}"
                       for a, b in pairs if _ids(s, a) & _ids(s, b)]
        if s["family"] != "control" and not s.get("primary"):
            errors.append(f"{sid}: non-control scenario needs at least one primary indicator")
        errors += [f"{sid}: unknown hypothesis {h}" for h in s["hypotheses"] if h not in ALL_HYPOTHESES]
        if s["purpose"] not in PURPOSES:
            errors.append(f"{sid}: bad purpose {s['purpose']}")
        if (s["purpose"] == "control") != (s["family"] == "control"):
            errors.append(f"{sid}: purpose control and family control must go together")
        if s["purpose"] == "harmless_drift" and s["primary_tafd"] is not False:
            errors.append(f"{sid}: a harmless-drift scenario has no harm-defined onset, so primary_tafd must be false")
        if s["label_dependence"] not in LABEL_MODES:
            errors.append(f"{sid}: bad label_dependence {s['label_dependence']}")
        if not isinstance(s["primary_tafd"], bool):
            errors.append(f"{sid}: primary_tafd must be true or false")
        pert = s["perturbation"]
        if set(pert.get("levels", {})) != {"low", "medium", "high"} or not pert.get("shapes"):
            errors.append(f"{sid}: perturbation needs low/medium/high levels and at least one shape")
    for h in ("RH1", "RH2"):
        n = sum(h in s.get("hypotheses", []) and s.get("purpose") == "fair_test" and s.get("primary_tafd") is True for s in scenarios)
        if n < 3:
            errors.append(f"hypothesis {h} has {n} fair-test scenario(s) with a harm-defined onset; at least 3 required")
    covered = {i for s in scenarios if isinstance(s.get("primary"), list) for i in s["primary"]}
    errors += [f"indicator {i} is not a primary responder in any scenario" for i in sorted(applicable - covered)]
    for h in PHASE2_HYPOTHESES:
        n = sum(h in s.get("hypotheses", []) for s in scenarios if s.get("family") != "control")
        if n < 2:
            errors.append(f"hypothesis {h} has {n} scenario(s); at least 2 required")
    return errors


def render(scenarios: List[Dict[str, Any]], records: List[Dict[str, Any]], applicable: Set[str]) -> str:
    """Render coverage tables to markdown."""
    roles: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for s in scenarios:
        for key in SETS:
            for i in _ids(s, key):
                roles[i][key].append(s["id"])
    lines = ["# Phase II scenario coverage", "",
             "Generated by `scripts/validate_scenarios.py` from `phase2/scenarios.yaml`.", "",
             f"{len(scenarios)} scenarios. {len(applicable)} of {len(records)} indicators apply to predictive AI. "
             f"The other {len(records) - len(applicable)} apply only to generative or agentic systems and are outside Phase II.", "",
             "## Scenarios per hypothesis", "", "| Hypothesis | Scenarios | n |", "|---|---|---|"]
    for h in ALL_HYPOTHESES:
        ids = [s["id"] for s in scenarios if h in s["hypotheses"] and s["family"] != "control"]
        lines.append(f"| {h} | {', '.join(ids) or '-'} | {len(ids)} |")
    by_purpose = {p: [s["id"] for s in scenarios if s["purpose"] == p] for p in PURPOSES}
    excluded_tafd = [s["id"] for s in scenarios if not s["primary_tafd"]]
    lines += ["", f"Scenarios excluded from the primary TAFD analysis (no harm-defined onset): {', '.join(excluded_tafd)}.", "",
              "## Scenarios by purpose", "", "| Purpose | Scenarios |", "|---|---|"]
    lines += [f"| {p} | {', '.join(ids) or '-'} |" for p, ids in by_purpose.items()]
    lines += ["", "## Indicator roles across scenarios", "",
              "| Indicator | Primary responder in | Secondary in | Held stable in | Negative control in |", "|---|---|---|---|---|"]
    for r in records:
        if r["id"] not in applicable:
            continue
        role = roles[r["id"]]
        cell = lambda k: ", ".join(role[k]) or "-"  # noqa: E731
        lines.append(f"| {r['id']} | {cell('primary')} | {cell('secondary')} | {cell('held_stable')} | {cell('expected_null')} |")
    lines += ["", "Outside Phase II (no `predictive` tag): " + ", ".join(r["id"] for r in records if r["id"] not in applicable) + "."]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Validate the catalog and render coverage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    records = load_records(INDICATOR_DIR)
    applicable = {r["id"] for r in records if "predictive" in r["ai_types"]}
    scenarios = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    errors = validate(scenarios, applicable, {r["id"] for r in records})
    for e in errors:
        logger.error(e)
    if errors:
        return 1
    OUTPUT.write_text(render(scenarios, records, applicable), encoding="utf-8")
    logger.info("Validated %d scenarios against %d predictive-applicable indicators; wrote %s",
                len(scenarios), len(applicable), OUTPUT.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
