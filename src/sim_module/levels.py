"""Parse the free-text level values of the scenario catalog into numbers."""

import re


def parse_level(text: str) -> float:
    """Return the number in a level string. Percentages become fractions.

    Examples: "0.25 SD" -> 0.25, "1.5x" -> 1.5, "slope 0.9" -> 0.9, "5 pct" -> 0.05.

    Raises:
        ValueError: When the string contains no number.
    """
    match = re.search(r"-?\d+(?:\.\d+)?", str(text))
    if match is None:
        raise ValueError(f"no number in level {text!r}")
    value = float(match.group(0))
    return value / 100.0 if "pct" in str(text) else value
