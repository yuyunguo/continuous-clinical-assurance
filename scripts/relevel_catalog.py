"""Re-solve the catalog's harm-scaled levels against each scenario's own `harm_targets` (decision D15) and rewrite the level strings.

Usage: python scripts/relevel_catalog.py [--write] [--allow-cap]
Run this after any change to the harm weights, the false-negative factor, or the generator (decisions D3, D10). Without --write it prints the proposed
strings. Only the first number in each level string changes, so the wording, units, and modes stay as they are. Targets are never changed here.
With --allow-cap (used for the D3 sensitivity schemes) a target the plausibility cap cannot reach gets the cap value, and the shortfall is logged.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim_module import AXES, DeployedModel, SyntheticWorld, load_config, solve_level  # noqa: E402
from src.sim_module.config import scenarios_path  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
CAP_TOLERANCE = 0.02


def format_level(old: str, value: float) -> str:
    """Replace the first number of `old` by the solved value (in percent when `old` says pct), to three significant digits."""
    shown = value * 100.0 if "pct" in old else value
    text = f"{float(f'{shown:.3g}'):g}"
    return NUMBER.sub(text, old, count=1)


def main() -> int:
    """Solve, print, and optionally write."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config()
    world = SyntheticWorld(cfg.generator, cfg.harm)
    model = DeployedModel(cfg.generator).fit(world, np.random.default_rng(cfg.seed))
    catalog = scenarios_path()
    raw = catalog.read_text(encoding="utf-8")
    scenarios: Dict[str, Any] = {s["id"]: s for s in yaml.safe_load(raw)}
    replacements = []
    for sid, s in scenarios.items():
        if "harm_targets" not in s:
            continue
        spec = s["perturbation"]
        for level, target in zip(("low", "medium", "high"), s["harm_targets"]):
            solved = solve_level(spec["type"], target, world, model, cfg, mode=spec.get("mode"))
            old = str(spec["levels"][level])
            value = solved.value
            if value is None and ("--allow-cap" in sys.argv or target - solved.max_ratio <= CAP_TOLERANCE):
                value = AXES[spec["type"]].cap        # the target was set at what the plausibility cap can reach, and noise puts it a hair above
                logger.warning("%s %s: target %.2f is %.3f above the maximum %.3f, using the cap %g", sid, level, target, target - solved.max_ratio, solved.max_ratio, value)
            if value is None:
                logger.error("%s %s: target %.2f is not reachable (max %.2f)", sid, level, target, solved.max_ratio)
                return 1
            new = format_level(old, value)
            replacements.append((sid, level, old, new))
            logger.info("%s %-6s target %.2f: %r -> %r", sid, level, target, old, new)
    if "--write" in sys.argv:
        for sid, level, old, new in replacements:
            block_start = raw.index(f"{{id: {sid},")
            block_end = raw.index("\n- {id:", block_start) if "\n- {id:" in raw[block_start:] else len(raw)
            block = raw[block_start:block_end]
            needle = f"{level}: {old}"
            if block.count(needle) != 1:
                raise ValueError(f"{sid} {level}: cannot find {needle!r} exactly once")
            raw = raw[:block_start] + block.replace(needle, f"{level}: {new}") + raw[block_end:]
        catalog.write_text(raw, encoding="utf-8")
        logger.info("Wrote %s", catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
