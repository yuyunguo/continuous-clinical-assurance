"""Decision D2 sensitivity: the RH1 and RH2 rehearsal under several false-alarm budgets.

Usage: python scripts/budget_sweep.py
Runs `scripts/m5_rehearsal.py` at each budget (alert episodes per 1,000 monitored windows) with the environment override `M5_BUDGET`, in parallel, storing
results under `phase2/results/budget/<budget>/`. Then run `scripts/budget_report.py`. The primary budget (from the config) is the default run in `phase2/results/`, and the earlier primary (5) is archived beside the sweep.
Exploratory. With 7 to 14 day windows a budget of 5 allows one false alarm every 3.8 to 7.7 years, far stricter than a governance committee needs.
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "phase2" / "results" / "budget"
BUDGETS: List[float] = [1.0, 20.0, 50.0, 150.0]


def main() -> int:
    """Start one rehearsal per budget and wait for all of them."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    procs = []
    for budget in BUDGETS:
        folder = BASE / f"{budget:g}"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("rh3.json", "rh4.json"):
            shutil.copy(ROOT / "phase2" / "results" / name, folder / name)
        env = {**os.environ, "M5_BUDGET": str(budget), "PHASE2_RESULTS": str(folder)}
        log = open(folder / "rehearsal.log", "w", encoding="utf-8")
        procs.append((budget, subprocess.Popen([sys.executable, str(ROOT / "scripts" / "m5_rehearsal.py")], env=env, stdout=log, stderr=subprocess.STDOUT)))
        logger.info("started budget %g", budget)
    for budget, proc in procs:
        logger.info("budget %g finished with exit code %d", budget, proc.wait())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
