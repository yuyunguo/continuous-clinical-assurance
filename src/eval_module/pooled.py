"""One-sided bootstrap for a scenario-pooled estimand (design section 9.1)."""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def pooled_bootstrap(values: Dict[str, List[np.ndarray]], margin: float, rng: np.random.Generator,
                     n_boot: int = 2000) -> Tuple[float, float, float, float]:
    """Estimate, 95 percent interval, and one-sided p for a pooled mean, with null: the estimand is at most `margin`.

    `values` maps a scenario to its cells, each an array of per-replicate values (for a paired reduction, the difference between the
    reference and candidate). Cells are averaged within a scenario, and scenarios have equal weight. The bootstrap resamples replicates
    within each cell. The estimate is NaN when there are no cells.
    """
    scenarios = [cells for cells in values.values() if cells]
    if not scenarios:
        return float("nan"), float("nan"), float("nan"), float("nan")

    def pooled(draw: bool) -> float:
        per = []
        for cells in scenarios:
            per.append(np.mean([(v[rng.integers(0, len(v), len(v))] if draw else v).mean() for v in cells]))
        return float(np.mean(per))

    boots = np.array([pooled(True) for _ in range(n_boot)])
    return pooled(False), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(np.mean(boots <= margin))


def collect_diffs(store: Mapping[str, np.ndarray], scenarios: Sequence[str], lag: int, reference: str,
                  candidate: Optional[str]) -> Dict[str, List[np.ndarray]]:
    """Per scenario, the cells' per-replicate values of `reference` minus `candidate` at a label lag.

    Store keys are `scenario|level|shape|lag|name`. With `candidate=None` the reference values themselves are returned.
    """
    out: Dict[str, List[np.ndarray]] = {s: [] for s in scenarios}
    for key in sorted(store):
        parts = key.split("|")
        if len(parts) != 5 or parts[0] not in out or int(parts[3]) != lag or parts[4] != reference:
            continue
        values = np.asarray(store[key], dtype=float)
        out[parts[0]].append(values if candidate is None else values - np.asarray(store["|".join(parts[:4] + [candidate])], dtype=float))
    return out
