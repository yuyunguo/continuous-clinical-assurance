"""Computational margin analysis (decision D5): harm-equivalent margins, pass curves, and the resolution floor."""

import re
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats import norm

_NUM = r"([+-]?\d+(?:\.\d+)?)"
_INTERVAL = rf"{_NUM} \[{_NUM}, {_NUM}\]"


@dataclass(frozen=True)
class PilotRow:
    """One cell of the pilot's detection comparison at one label lag."""

    lag: int
    scenario: str
    level: str
    shape: str
    n: int
    reduction: float
    reduction_lo: float
    reduction_hi: float
    exposure: float
    exposure_lo: float
    exposure_hi: float


def implied_margin(excess_per_window: float, baseline_rhe: float, delta: float) -> float:
    """Windows of earlier detection that avoid `delta` baseline-windows of harm, given the excess harm accrued per window of delay."""
    return float(delta * baseline_rhe / excess_per_window) if excess_per_window > 0 else float("inf")


def pass_share(lowers: np.ndarray, margin: float) -> float:
    """Share of cells whose interval's lower bound is at least the margin (NaN cells are ignored)."""
    valid = lowers[~np.isnan(lowers)]
    return float(np.mean(valid >= margin)) if len(valid) else float("nan")


def se_from_interval(lo: float, hi: float) -> float:
    """Standard error implied by a 95 percent interval."""
    return (hi - lo) / (2 * norm.ppf(0.975))


def min_detectable(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true reduction a one-sided test detects with the given power, for a standard error of `se`."""
    return float((norm.ppf(1 - alpha) + norm.ppf(power)) * se)


def parse_pilot(text: str) -> List[PilotRow]:
    """Read the detection-comparison tables of `phase2/M3-M4-Pilot.md` (all label lags)."""
    rows: List[PilotRow] = []
    lag = -1
    for line in text.splitlines():
        head = re.match(r"### Label lag (\d+) windows", line)
        if head:
            lag = int(head.group(1))
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if lag < 0 or len(cells) < 12 or re.fullmatch(r"SC\d+", cells[0]) is None or not cells[3].isdigit():
            continue
        red, expo = re.fullmatch(_INTERVAL, cells[9]), re.fullmatch(_INTERVAL, cells[10])
        if red is None or expo is None:
            continue
        r, e = (float(x) for x in red.groups()), (float(x) for x in expo.groups())
        rows.append(PilotRow(lag, cells[0], cells[1], cells[2], int(cells[3]), *r, *e))
    return rows
