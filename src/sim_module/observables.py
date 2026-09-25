"""Observables map (milestone M0): which simulator output supplies each data element an indicator needs."""

from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
OBSERVABLES_PATH = ROOT / "phase2" / "observables.yaml"
INDICATOR_GLOB = "indicators/domain_*.yaml"


def load_observables(path: Path = OBSERVABLES_PATH) -> Dict[str, Any]:
    """Load the observables map."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_indicators() -> List[Dict[str, Any]]:
    """Load the indicator records that apply to predictive AI."""
    records: List[Dict[str, Any]] = []
    for path in sorted(ROOT.glob(INDICATOR_GLOB)):
        records += [r for r in yaml.safe_load(path.read_text(encoding="utf-8")) if "predictive" in r["ai_types"]]
    return records


def validate_observables(records: List[Dict[str, Any]], obs: Dict[str, Any]) -> List[str]:
    """Return errors: unmapped data elements, mappings to unknown outputs, and unused mapping entries."""
    outputs, elements = obs["outputs"], obs["elements"]
    needed = {e for r in records for e in r["data_required"]}
    errors = [f"unmapped data element: {e!r}" for e in sorted(needed - set(elements))]
    errors += [f"element {e!r} maps to unknown output {o!r}" for e, o in sorted(elements.items()) if o not in outputs]
    errors += [f"mapping entry {e!r} matches no indicator" for e in sorted(set(elements) - needed)]
    errors += [f"output {o!r} has status {v.get('status')!r}" for o, v in outputs.items() if v.get("status") not in ("implemented", "planned")]
    return errors


def indicator_status(records: List[Dict[str, Any]], obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """For each indicator, the outputs it needs and whether all of them exist yet."""
    outputs, elements = obs["outputs"], obs["elements"]
    result: Dict[str, Dict[str, Any]] = {}
    for r in records:
        needed = sorted({elements[e] for e in r["data_required"] if e in elements})
        missing = [o for o in needed if outputs.get(o, {}).get("status") != "implemented"]
        result[r["id"]] = {"outputs": needed, "missing_outputs": missing, "testable_now": not missing}
    return result
