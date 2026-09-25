"""Holm step-down adjustment across the pre-specified hypotheses (design section 9.1)."""

from typing import Dict, Tuple

import numpy as np


def holm(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Tuple[float, bool]]:
    """Holm-adjusted p-values and rejection decisions at family-wise level `alpha`.

    A NaN p-value is never rejected and does not count toward the family size.
    """
    valid = {k: v for k, v in p_values.items() if not np.isnan(v)}
    order = sorted(valid, key=lambda k: valid[k])
    m, running = len(order), 0.0
    out: Dict[str, Tuple[float, bool]] = {k: (float("nan"), False) for k in p_values}
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * valid[name]))
        out[name] = (running, running <= alpha)
    return out
