"""A hand-built model of primary visual cortex, as the scientific control.

DINOv2 is a 86M-parameter ViT trained on 142M natural images. The stimuli here
are gratings and 1/f noise -- exactly the inputs a 1960s oriented-energy model of
V1 was designed for. If a Gabor filter bank explains the neural geometry as well
as the ViT does, the foundation model is adding nothing on this stimulus class,
and that is worth knowing rather than assuming.

Implements a quadrature-pair Gabor energy bank: complex Gabors at `n_orientations`
orientations and `n_scales` log-spaced spatial frequencies, with the phase-invariant
energy (the complex-filter modulus) pooled over the image, matching the classic
V1 complex-cell model.
"""
from __future__ import annotations

import numpy as np

from .params import load_gray


def _gabor_bank_fourier(size: int, deg: float, n_orient: int, n_scale: int,
                        f_lo: float, f_hi: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Build the bank directly in the Fourier domain (fast and exact).

    Each filter is a log-Gabor: Gaussian in log-radial frequency, Gaussian in
    orientation. Returns (n_filters, size, size) transfer functions.
    """
    fy = np.fft.fftfreq(size)[:, None] * size / deg      # cycles per degree
    fx = np.fft.fftfreq(size)[None, :] * size / deg
    r = np.hypot(fy, fx)
    th = np.arctan2(fy, fx)
    r[0, 0] = 1e-6

    freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), n_scale)
    orients = np.linspace(0, np.pi, n_orient, endpoint=False)
    sigma_r = 0.55                                        # log-radial bandwidth (~1.5 octaves)
    sigma_t = np.pi / n_orient / 1.5

    filters, meta = [], []
    for f0 in freqs:
        radial = np.exp(-(np.log(r / f0) ** 2) / (2 * sigma_r ** 2))
        radial[0, 0] = 0.0
        for t0 in orients:
            dth = np.angle(np.exp(1j * (th - t0)))
            ang = np.exp(-(dth ** 2) / (2 * sigma_t ** 2))
            dth2 = np.angle(np.exp(1j * (th - t0 + np.pi)))
            ang = ang + np.exp(-(dth2 ** 2) / (2 * sigma_t ** 2))
            filters.append(radial * ang)
            meta.append((float(f0), float(np.degrees(t0))))
    return np.stack(filters), meta


def embed_images(paths: list[str], cfg: dict, size: int = 256,
                 n_pool: int = 4) -> tuple[np.ndarray, list[str]]:
    """Log Gabor energy, pooled over an `n_pool` x `n_pool` spatial grid.

    Returns (n_images, n_filters * n_pool^2) features and their names.
    """
    g = cfg["vision"]["gabor"]
    deg = cfg["stimuli"]["visual_angle_deg"]
    bank, meta = _gabor_bank_fourier(size, deg, g["n_orientations"], g["n_scales"],
                                     g["freq_lo_cpd"], g["freq_hi_cpd"])
    edges = np.linspace(0, size, n_pool + 1).astype(int)

    feats = []
    for p in paths:
        a = load_gray(p, size)
        A = np.fft.fft2(a - a.mean())
        # Phase-invariant energy = modulus of the analytic (one-sided) response.
        resp = np.abs(np.fft.ifft2(A[None, :, :] * bank))          # (F, H, W)
        pooled = np.stack([
            resp[:, edges[i]:edges[i + 1], edges[j]:edges[j + 1]].mean(axis=(1, 2))
            for i in range(n_pool) for j in range(n_pool)
        ], axis=1)                                                  # (F, n_pool^2)
        feats.append(np.log(pooled.ravel() + 1e-8))

    names = [f"gabor_f{f0:.2f}_o{t0:.0f}_p{k}"
             for (f0, t0) in meta for k in range(n_pool ** 2)]
    return np.stack(feats).astype(np.float32), names
