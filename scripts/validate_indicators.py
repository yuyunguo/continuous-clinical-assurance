"""Validate the indicator instrument and render it to markdown.

Usage:
    python scripts/validate_indicators.py [--expected 40] [--render]

Exit status is non-zero when any validation error is found.
"""

import argparse
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
INDICATOR_DIR = ROOT / "indicators"
OUTPUT_PATH = ROOT / "Indicator-Instrument.md"
DEFAULT_EXPECTED = 41
TEXT_FIELDS = (
    "name",
    "definition",
    "formula",
    "frequency",
    "baseline",
    "alert_rule",
    "detection_method",
    "clinical_importance",
    "validation",
)
LIST_FIELDS = ("data_required", "ai_types")


@dataclass(frozen=True)
class Schema:
    """Controlled vocabularies loaded from indicators/schema.yaml."""

    required_fields: List[str]
    domains: Dict[str, str]
    state_components: Dict[str, str]
    evidence_levels: Dict[int, str]
    ai_types: List[str]
    trigger_levels: List[str]
    domain_components: Dict[str, List[str]]
    matrix_constructs: List[str]
    out_of_scope_constructs: Dict[str, str]


def load_schema(path: Path) -> Schema:
    """Load the schema file.

    Args:
        path: Path to schema.yaml.

    Returns:
        Parsed schema.

    Raises:
        FileNotFoundError: When the schema file is missing.
        KeyError: When a required schema key is absent.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("Schema file not found: %s", path)
        raise
    return Schema(
        required_fields=raw["required_fields"],
        domains=raw["domains"],
        state_components=raw["state_components"],
        evidence_levels=raw["evidence_levels"],
        ai_types=raw["ai_types"],
        trigger_levels=raw["trigger_levels"],
        domain_components=raw["domain_components"],
        matrix_constructs=raw["matrix_constructs"],
        out_of_scope_constructs=raw.get("out_of_scope_constructs") or {},
    )


def load_records(directory: Path) -> List[Dict[str, Any]]:
    """Load all indicator records from domain_*.yaml files, in file order."""
    records: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("domain_*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            logger.error("Invalid YAML in %s", path)
            raise
        for record in data or []:
            record["_source"] = path.name
            records.append(record)
    return records


def _is_blank(value: Any) -> bool:
    """Return True for missing, empty, or whitespace-only values."""
    return value is None or (isinstance(value, str) and not value.strip())


def _check_vocab(rec: Dict[str, Any], schema: Schema, where: str) -> List[str]:
    """Check controlled-vocabulary fields of one record."""
    errors: List[str] = []
    domain = rec.get("domain")
    component = rec.get("state_component")
    if domain not in schema.domains:
        errors.append(f"{where}: unknown domain {domain!r}")
    if component not in schema.state_components:
        errors.append(f"{where}: unknown state_component {component!r}")
    elif domain in schema.domain_components and component not in schema.domain_components[domain]:
        errors.append(f"{where}: component {component} not allowed in domain {domain}")
    if rec.get("matrix_construct") not in schema.matrix_constructs:
        errors.append(f"{where}: unknown matrix_construct {rec.get('matrix_construct')!r}")
    if rec.get("evidence_level") not in schema.evidence_levels:
        errors.append(f"{where}: invalid evidence_level {rec.get('evidence_level')!r}")
    ai_types = rec.get("ai_types") or []
    if not ai_types or any(t not in schema.ai_types for t in ai_types):
        errors.append(f"{where}: ai_types must be a non-empty subset of {schema.ai_types}")
    return errors


def _check_actions(rec: Dict[str, Any], schema: Schema, where: str) -> List[str]:
    """Check that governance_action has exactly the trigger levels, all non-empty."""
    actions = rec.get("governance_action")
    if not isinstance(actions, dict):
        return [f"{where}: governance_action must be a mapping"]
    errors: List[str] = []
    if set(actions) != set(schema.trigger_levels):
        errors.append(f"{where}: governance_action keys {sorted(actions)} != {schema.trigger_levels}")
    errors.extend(
        f"{where}: empty governance_action for {level}"
        for level in schema.trigger_levels
        if _is_blank(actions.get(level))
    )
    return errors


def validate(records: List[Dict[str, Any]], schema: Schema, expected: int) -> List[str]:
    """Validate all records and return a list of error messages."""
    errors: List[str] = []
    if len(records) != expected:
        errors.append(f"expected {expected} records, found {len(records)}")
    duplicates = [i for i, n in Counter(r.get("id") for r in records).items() if n > 1]
    errors.extend(f"duplicate id {i!r}" for i in duplicates)
    for rec in records:
        where = f"{rec.get('id', '?')} ({rec['_source']})"
        errors.extend(f"{where}: missing field {f!r}" for f in schema.required_fields if f not in rec)
        errors.extend(f"{where}: blank field {f!r}" for f in TEXT_FIELDS if _is_blank(rec.get(f)))
        errors.extend(
            f"{where}: {f} must be a non-empty list"
            for f in LIST_FIELDS
            if not isinstance(rec.get(f), list) or not rec.get(f)
        )
        errors.extend(_check_vocab(rec, schema, where))
        errors.extend(_check_actions(rec, schema, where))
    return errors


def _cell(text: Any) -> str:
    """Escape a value for use in a markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def load_matrix(path: Path) -> Dict[str, Dict[str, str]]:
    """Parse the master table of the Evidence and Measurement Matrix: construct -> domain, objective, risk."""
    rows = re.findall(r"(?m)^\|\s*\*\*([^|]+)\*\*\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", path.read_text(encoding="utf-8"))
    return {c.strip(): {"domain": d.strip(), "objective": o.strip(), "risk": r.strip()} for d, o, r, c in rows}


def load_scenario_roles(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Map indicator -> role -> Phase II scenario IDs, from phase2/scenarios.yaml (empty if absent)."""
    if not path.exists():
        return {}
    roles: Dict[str, Dict[str, List[str]]] = {}
    for scenario in yaml.safe_load(path.read_text(encoding="utf-8")):
        for role in ("primary", "expected_null"):
            ids = scenario.get(role)
            for ind in ids if isinstance(ids, list) else []:
                roles.setdefault(ind, {}).setdefault(role, []).append(scenario["id"])
    return roles


def matrix_lines(rec: Dict[str, Any], matrix: Dict[str, Dict[str, str]]) -> List[str]:
    """Render the governance objective and clinical risk carried by the record's matrix construct."""
    row = matrix.get(rec.get("matrix_construct", ""))
    if not row:
        return []
    return [f"- **Matrix row:** {row['domain']} ({rec['matrix_construct']})",
            f"- **Governance objective:** {row['objective']}", f"- **Clinical or operational risk:** {row['risk']}"]


def scenario_line(ind_id: str, roles: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """Render the Phase II scenarios where the indicator is a primary responder or a negative control."""
    role = roles.get(ind_id)
    if not role:
        return ["- **Phase II scenarios:** none (outside the Phase II predictive anchor)"]
    return ["- **Phase II scenarios:** primary responder in " + (", ".join(role.get("primary", [])) or "none")
            + "; negative control in " + (", ".join(role.get("expected_null", [])) or "none")]


def coverage_section(records: List[Dict[str, Any]], schema: Schema, matrix: Dict[str, Dict[str, str]]) -> List[str]:
    """Render which matrix constructs have indicators and which do not."""
    by_construct: Dict[str, List[str]] = {c: [] for c in schema.matrix_constructs}
    for r in records:
        by_construct[r["matrix_construct"]].append(r["id"])
    lines = ["## Coverage of the Evidence and Measurement Matrix", "", "| Matrix construct | Matrix row | Indicators |", "|---|---|---|"]
    lines += [f"| {c} | {matrix.get(c, {}).get('domain', '-')} | {', '.join(ids) or ('out of scope' if c in schema.out_of_scope_constructs else '**none**')} |"
              for c, ids in by_construct.items()]
    gaps = [c for c, ids in by_construct.items() if not ids]
    open_gaps = [c for c in gaps if c not in schema.out_of_scope_constructs]
    lines += ["", f"{len(schema.matrix_constructs) - len(gaps)} of {len(schema.matrix_constructs)} matrix constructs have at least one indicator."
              + (" No indicator yet: " + ", ".join(open_gaps) + "." if open_gaps else "")]
    lines += [f"Out of scope, {c}: {schema.out_of_scope_constructs[c]}" for c in gaps if c in schema.out_of_scope_constructs]
    return lines


def render(records: List[Dict[str, Any]], schema: Schema, matrix: Optional[Dict[str, Dict[str, str]]] = None,
           roles: Optional[Dict[str, Dict[str, List[str]]]] = None) -> str:
    """Render records to markdown: summary tables followed by full detail."""
    lines = [
        "# Clinical AI Assurance Indicator Instrument",
        "",
        "Generated by `scripts/validate_indicators.py` from `indicators/*.yaml`. Do not edit by hand.",
        "",
        f"{len(records)} candidate indicators. Thresholds are rules over VOE-specified margins "
        "m_Y < m_O < m_R (yellow, orange, red). Green means within the validated envelope. "
        "The indicators are candidates until tested in Phase II/III.",
        "",
    ]
    for domain, title in schema.domains.items():
        group = [r for r in records if r["domain"] == domain]
        lines += [f"## Domain {domain}: {title}", "", "| ID | Indicator | State | Evidence level | AI types | Detection |", "|---|---|---|---|---|---|"]
        for r in group:
            lines.append(
                f"| {r['id']} | {_cell(r['name'])} | {r['state_component']} | {r['evidence_level']} "
                f"| {', '.join(r['ai_types'])} | {_cell(r['detection_method'])} |"
            )
        lines.append("")
        for r in group:
            actions = r["governance_action"]
            lines += [
                f"### {r['id']}. {r['name']}",
                "",
                *matrix_lines(r, matrix or {}),
                f"- **Definition:** {r['definition']}",
                f"- **Formula:** `{r['formula']}`",
                f"- **Data required:** {', '.join(r['data_required'])}",
                f"- **Frequency:** {r['frequency']}",
                f"- **Baseline:** {r['baseline']}",
                f"- **Alert rule:** {r['alert_rule']}",
                f"- **Detection method:** {r['detection_method']}",
                f"- **Governance action:** yellow: {actions['yellow']}; orange: {actions['orange']}; red: {actions['red']}",
                f"- **Clinical importance:** {r['clinical_importance']}",
                f"- **Validation:** {r['validation']}",
                *scenario_line(r["id"], roles or {}),
                "",
            ]
    lines += coverage_section(records, schema, matrix or {})
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run validation and optional rendering; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=int, default=DEFAULT_EXPECTED)
    parser.add_argument("--render", action="store_true", help="write Indicator-Instrument.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    schema = load_schema(INDICATOR_DIR / "schema.yaml")
    records = load_records(INDICATOR_DIR)
    errors = validate(records, schema, args.expected)
    for message in errors:
        logger.error(message)
    if errors:
        logger.error("Validation failed: %d error(s)", len(errors))
        return 1

    by_domain = Counter(r["domain"] for r in records)
    logger.info("Validation passed: %d records %s", len(records), dict(sorted(by_domain.items())))
    if args.render:
        matrix = load_matrix(ROOT / "Evidence-and-measurement-matrix.md")
        roles = load_scenario_roles(ROOT / "phase2" / "scenarios.yaml")
        OUTPUT_PATH.write_text(render(records, schema, matrix, roles), encoding="utf-8")
        logger.info("Rendered %s", OUTPUT_PATH.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
