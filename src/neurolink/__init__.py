"""NeuroLink -- decoding visual perception from human intracranial LFP."""
from __future__ import annotations

import warnings

# Apple's Accelerate BLAS raises spurious divide-by-zero / overflow / invalid
# FP flags inside `matmul` on Apple Silicon. The inputs and outputs involved are
# verified finite at every point where we depend on them (see the explicit
# np.isfinite assertions in neural/spectral.py and align/evaluate.py), and the
# flood otherwise drowns out real warnings from sklearn internals. Suppress only
# this exact message class -- never RuntimeWarning as a whole.
warnings.filterwarnings(
    "ignore", message=r".*encountered in matmul", category=RuntimeWarning)

__all__ = ["__version__"]
__version__ = "0.1.0"
