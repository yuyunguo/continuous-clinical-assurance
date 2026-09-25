"""Count governance-relevant terms in the saved Phase I source texts.

Counts are raw regular-expression matches (case-insensitive) in the extracted text of each
source. They show where a source uses a term at all. They do not measure how well it treats it.
Usage: python scripts/term_coverage.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "phase1" / "sources"
OUTPUT = ROOT / "phase1" / "term_coverage.md"

TERMS: Dict[str, str] = {
    "monitor*": r"monitor",
    "calibration (excluding recalibration)": r"(?<![a-z])calibrat",
    "recalibration (an action)": r"recalibrat",
    "uncertainty": r"uncertaint",
    "drift": r"drift",
    "dataset/distribution shift": r"(dataset|distribution|data)\s+shift",
    "automation bias": r"automation\s+bias|over-relying",
    "override": r"overrid",
    "near miss": r"near[\s-]miss",
    "threshold": r"threshold",
    "detection time": r"time[\s-]to[\s-]detect|detection\s+time|time\s+to\s+identif",
}
GROUPS: Dict[str, List[str]] = {
    "S1 NIST AI RMF 1.0 (full text)": ["nist_ai_100_1.txt"],
    "S2 EU AI Act Art. 14 + 72 (mirror; 2 articles only)": ["eu_ai_act_art14.txt", "eu_ai_act_art72.txt"],
    "S3 FDA PCCP guidance (full text)": ["fda_pccp_guidance.txt"],
}


def count(text: str, pattern: str) -> int:
    """Return the number of case-insensitive matches of pattern in text."""
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def main() -> int:
    """Compute the term-coverage table and write it to phase1/term_coverage.md."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest = {m["file"]: m for m in json.loads((SOURCE_DIR / "manifest.json").read_text(encoding="utf-8"))}
    texts = {g: " ".join((SOURCE_DIR / f).read_text(encoding="utf-8") for f in files) for g, files in GROUPS.items()}
    header = "| Term (regex) | " + " | ".join(f"{g}" for g in GROUPS) + " |"
    lines = ["# Term coverage in retrieved governance sources", "",
             "Raw case-insensitive match counts in the extracted text saved under `phase1/sources/`. "
             "A zero means the term does not occur in the retrieved text. A nonzero count does not show "
             "adequate treatment. Extraction details and hashes are in `phase1/sources/manifest.json`.", "",
             "Characters: " + "; ".join(f"{g.split(' (')[0]} {len(t):,}" for g, t in texts.items()), "",
             header, "|---|" + "---|" * len(GROUPS)]
    for label, pattern in TERMS.items():
        lines.append(f"| {label} (`{pattern.replace('|', chr(92) + '|')}`) | " + " | ".join(str(count(t, pattern)) for t in texts.values()) + " |")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d sources in manifest)", OUTPUT.name, len(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
