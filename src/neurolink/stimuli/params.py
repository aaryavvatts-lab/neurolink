"""Recover each stimulus image's generative parameters directly from its pixels.

These are the ground truth for parametric reconstruction: if we can decode
spatial frequency, orientation, phase, noise exponent and contrast from the
brain, we can re-render the image from those numbers and compare it to what the
subject actually saw. Measuring them from pixels (rather than trusting a
condition label) means the reconstruction target is verifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image


def load_gray(path: str, size: int | None = None) -> np.ndarray:
    """Load as float in [0, 1]."""
    im = Image.open(path).convert("L")
    if size is not None and im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def _power_spectrum(a: np.ndarray) -> np.ndarray:
    a = a - a.mean()
    n = a.shape[0]
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    return np.abs(np.fft.fftshift(np.fft.fft2(a * w))) ** 2


def _radial_profile(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = P.shape[0]; c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - c, xx - c).astype(int)
    tot = np.bincount(r.ravel(), P.ravel())
    cnt = np.bincount(r.ravel())
    k = np.arange(1, min(n // 2, len(tot)))
    return k, tot[k] / np.maximum(cnt[k], 1)


def measure(path: str, deg: float, size: int = 512) -> dict:
    """Measure one image. `deg` is the stimulus width in degrees of visual angle."""
    a = load_gray(path, size)
    rms = float(a.std())
    mean_lum = float(a.mean())

    if rms < 1e-6:                                   # the blank gray screen
        return dict(rms_contrast=0.0, mean_lum=mean_lum, spatial_freq_cpd=np.nan,
                    orientation_deg=np.nan, phase_rad=np.nan, noise_exponent=np.nan,
                    orient_concentration=0.0, is_blank=True)

    P = _power_spectrum(a)
    k, prof = _radial_profile(P)

    # 1/f^alpha exponent from a log-log fit over mid frequencies (avoids the DC
    # shoulder and the anti-aliasing rolloff at the top).
    lo, hi = 4, min(150, len(k) - 1)
    alpha = -float(np.polyfit(np.log(k[lo:hi]), np.log(prof[lo:hi] + 1e-30), 1)[0])

    # Dominant spatial frequency: peak of the radial profile, in cycles/image.
    cyc_per_img = float(k[int(np.argmax(prof))])
    sf_cpd = cyc_per_img / deg

    # Orientation: energy-weighted circular mean (mod 180 deg) in the dominant ring.
    n = a.shape[0]; c = n // 2
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(yy - c, xx - c)
    ring = (rr > cyc_per_img - 4) & (rr < cyc_per_img + 4)
    th = np.arctan2(yy - c, xx - c) % np.pi
    wgt = P[ring]
    z = (wgt * np.exp(2j * th[ring])).sum() / (wgt.sum() + 1e-30)
    orientation = float((np.angle(z) / 2.0) % np.pi)
    concentration = float(np.abs(z))                 # 0 = isotropic, 1 = pure orientation

    # Phase of the dominant sinusoid, read off the complex FFT at the peak component.
    A = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    band = ring & (np.cos(2 * (th - orientation)) > 0.5)
    if band.any():
        idx = np.argmax(np.abs(A) * band)
        phase = float(np.angle(A.ravel()[idx]))
    else:
        phase = np.nan

    return dict(rms_contrast=rms, mean_lum=mean_lum, spatial_freq_cpd=sf_cpd,
                cycles_per_image=cyc_per_img, orientation_deg=np.degrees(orientation),
                phase_rad=phase, noise_exponent=alpha,
                orient_concentration=concentration, is_blank=False)


def measure_all(stim_table: pd.DataFrame, deg: float, size: int = 512) -> pd.DataFrame:
    rows = []
    for _, r in stim_table.iterrows():
        m = measure(r["path"], deg=deg, size=size)
        m["stim_id"] = int(r["stim_id"]); m["trial_type"] = int(r["trial_type"])
        rows.append(m)
    return pd.DataFrame(rows).set_index("stim_id").sort_index()


CONDITION_LABELS = {
    1: "noise k/f^0", 2: "noise k/f^2", 3: "noise k/f^4",
    4: "grating 0.16 cpd", 5: "grating 0.32 cpd",
    6: "grating 0.64 cpd", 7: "grating 1.28 cpd", 8: "blank",
}
GRATING_CONDITIONS = (4, 5, 6, 7)
NOISE_CONDITIONS = (1, 2, 3)


def render_grating(sf_cpd: float, orientation_deg: float, phase_rad: float,
                   contrast: float, deg: float, size: int = 512,
                   square: bool = True) -> np.ndarray:
    """Re-render a square-wave grating from decoded parameters."""
    x = (np.arange(size) - size / 2) / size * deg
    xx, yy = np.meshgrid(x, x)
    th = np.radians(orientation_deg)
    # The measured orientation is that of the spectral energy, which is normal to
    # the grating bars; project along it so the rendered bars match the original.
    proj = xx * np.cos(th) + yy * np.sin(th)
    w = np.sin(2 * np.pi * sf_cpd * proj + phase_rad)
    if square:
        w = np.sign(w)
    return np.clip(0.5 + contrast * w, 0, 1)


def render_noise(alpha: float, contrast: float, size: int = 512,
                 seed: int = 0) -> np.ndarray:
    """Re-render a 1/f^alpha noise pattern from decoded parameters."""
    rng = np.random.default_rng(seed)
    n = size
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    f = np.hypot(fy, fx)
    f[0, 0] = 1.0
    amp = f ** (-alpha / 2.0)
    amp[0, 0] = 0.0
    ph = rng.uniform(0, 2 * np.pi, (n, n))
    img = np.real(np.fft.ifft2(amp * np.exp(1j * ph)))
    img = img / (img.std() + 1e-30) * contrast
    return np.clip(0.5 + img, 0, 1)
