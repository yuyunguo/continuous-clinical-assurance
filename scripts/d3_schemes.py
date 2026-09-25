"""Decision D3 sensitivity: set up and run the RH1 and RH2 rehearsal under alternative severity weightings.

Usage: python scripts/d3_schemes.py [scheme ...]
For each scheme it writes a config with the scheme's weights, re-levels a copy of the scenario catalog against the same harm targets (so each scheme
sees the same harm ratios), and runs `scripts/m5_rehearsal.py` with the environment overrides `PHASE2_CONFIG`, `PHASE2_SCENARIOS`, and `PHASE2_RESULTS`.
The runs go in parallel. Results are stored under `phase2/results/d3/<scheme>/`. Then run `scripts/d3_report.py`. Exploratory. Schemes whose harm targets
cannot be reached are recorded and skipped.
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim_module.config import DEFAULT_CONFIG  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "phase2" / "results" / "d3"
# name -> (class weights for an unneeded alert, factor for a missed deterioration). The primary scheme (1, 7.1, 100 and 2.4, the geometric mean of the two raters) is the default config.
# rater_1 and rater_2 are each rater's own values, range_high is the top of the second rater's class ranges, and the factor runs across the pooled range 1.5 to 5.
SCHEMES: Dict[str, Tuple[List[float], float]] = {
    "rater_1": ([1.0, 5.0, 50.0], 3.0), "rater_2": ([1.0, 10.0, 200.0], 2.0), "range_high": ([1.0, 20.0, 1000.0], 2.4),
    "fn_factor_1_5": ([1.0, 7.1, 100.0], 1.5), "fn_factor_5": ([1.0, 7.1, 100.0], 5.0), "flat": ([1.0, 1.0, 1.0], 1.0)}


def setup(name: str, weights: List[float], fn_factor: float) -> Dict[str, str]:
    """Write the scheme's config and re-leveled catalog. Returns the environment for its runs, or an empty dict if its targets are unreachable."""
    folder = BASE / name
    folder.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["harm"]["weights"], raw["harm"]["fn_factor"] = weights, fn_factor
    (folder / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    shutil.copy(ROOT / "phase2" / "scenarios.yaml", folder / "scenarios.yaml")
    env = {**os.environ, "PHASE2_CONFIG": str(folder / "config.yaml"), "PHASE2_SCENARIOS": str(folder / "scenarios.yaml"), "PHASE2_RESULTS": str(folder)}
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "relevel_catalog.py"), "--write", "--allow-cap"], env=env, capture_output=True, text=True)
    (folder / "relevel.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    if done.returncode != 0:
        logger.error("%s: harm targets are not reachable under this scheme (see %s)", name, folder / "relevel.log")
        return {}
    for name_json in ("rh3.json", "rh4.json"):            # RH3 and RH4 are not re-run here, so the primary files stand in for the family table
        shutil.copy(ROOT / "phase2" / "results" / name_json, folder / name_json)
    return env


def main() -> int:
    """Set up every scheme, run the rehearsals in parallel, and wait."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    procs = []
    chosen = sys.argv[1:] or list(SCHEMES)
    for name, (weights, fn_factor) in ((n, SCHEMES[n]) for n in chosen):
        env = setup(name, weights, fn_factor)
        if not env:
            continue
        log = open(BASE / name / "rehearsal.log", "w", encoding="utf-8")
        procs.append((name, subprocess.Popen([sys.executable, str(ROOT / "scripts" / "m5_rehearsal.py")], env=env, stdout=log, stderr=subprocess.STDOUT)))
        logger.info("started %s", name)
    for name, proc in procs:
        code = proc.wait()
        logger.info("%s finished with exit code %d", name, code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
