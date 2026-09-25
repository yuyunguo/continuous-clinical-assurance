"""Reproducibility helpers: seeding and environment recording."""

import logging
import os
import platform
import random
from typing import Dict

import numpy as np
import scipy
import sklearn

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed the standard library and NumPy generators and fix the hash seed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Seeds set to %d", seed)


def log_environment() -> Dict[str, str]:
    """Return the software versions that affect results, for the experiment record."""
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "platform": platform.platform()}
