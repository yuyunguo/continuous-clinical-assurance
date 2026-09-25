"""Rolling calibration statistics over trailing windows (decision D21, a pre-specified sensitivity for RH2)."""

from typing import Dict, Sequence

import numpy as np

from .detectors import apply_label_lag, standardize
from .statistics import _ece, _slope_deviation
from .streams import StreamData

ROLLING_ORDER = ("U1r", "U2r", "U3r")   # ECE, Brier score, and slope deviation on the pooled cases of the trailing windows


def rolling_statistics(stream: StreamData, span: int = 4) -> Dict[str, np.ndarray]:
    """Calibration statistics on the cases of the last `span` windows pooled (length W). The first `span - 1` windows are NaN."""
    w, m = stream.y.shape
    n = w - span + 1
    y = np.stack([stream.y[i:i + n] for i in range(span)], axis=1).reshape(n, span * m)
    p = np.stack([stream.reported_p[i:i + n] for i in range(span)], axis=1).reshape(n, span * m)
    pad = np.full(span - 1, np.nan)
    return {"U1r": np.concatenate([pad, _ece(y, p)]), "U2r": np.concatenate([pad, ((p - y) ** 2).mean(axis=1)]),
            "U3r": np.concatenate([pad, _slope_deviation(y, p)])}


def rolling_scores_multi(stream: StreamData, lags: Sequence[int], span: int = 4) -> Dict[int, np.ndarray]:
    """Standardized rolling scores per label lag, shape (T, 3) each. All three need outcomes, so all are lagged."""
    stats = rolling_statistics(stream, span)
    series = np.stack([stats[i] for i in ROLLING_ORDER], axis=1)[None]
    z_full = standardize(series, stream.n_baseline, np.ones(len(ROLLING_ORDER)), full=True)
    return {lag: apply_label_lag(z_full, stream.n_baseline, lag)[0] for lag in lags}
