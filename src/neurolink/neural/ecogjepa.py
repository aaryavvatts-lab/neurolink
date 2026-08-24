"""Encoder C -- ECoG-JEPA: a masked-spectrogram transformer pretrained on this
dataset's own raw LFP.

Why this exists. The off-the-shelf "brain foundation models" (Encoder B) are all
pretrained on scalp EEG, bandlimited to 40-125 Hz. In ECoG the visual signal
lives in broadband gamma at 70-200 Hz, so those models cannot see it no matter
how large they are. This encoder follows the same self-supervised recipe as
BrainBERT (Wang et al., ICLR 2023) -- masked reconstruction of time-frequency
patches -- but is pretrained at the recording's full sample rate, so gamma
survives.

Pretraining uses no stimulus labels of any kind: it consumes the continuous
recording and never sees an event file. That keeps the downstream decoding
evaluation honest.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from scipy import signal as sps


# --------------------------------------------------------------------------- #
# Spectrogram front end
# --------------------------------------------------------------------------- #

@dataclass
class SpecConfig:
    sfreq_target: float = 512.0
    nperseg: int = 128
    hop: int = 16              # 31.25 ms at 512 Hz
    fmin: float = 4.0
    fmax: float = 200.0

    @property
    def frame_s(self) -> float:
        return self.hop / self.sfreq_target


def resample_to(x: np.ndarray, sf_in: float, sf_out: float) -> np.ndarray:
    """Polyphase resample along the last axis."""
    if abs(sf_in - sf_out) < 1e-6:
        return x
    from math import gcd
    a, b = int(round(sf_out)), int(round(sf_in))
    g = gcd(a, b)
    return sps.resample_poly(x, a // g, b // g, axis=-1)


def spectrogram(x: np.ndarray, sf: float, sc: SpecConfig) -> tuple[np.ndarray, np.ndarray]:
    """Log-magnitude STFT. x is (..., n_times) at `sf` Hz.

    Returns (spec (..., T, F), freqs (F,)).
    """
    xr = resample_to(np.asarray(x, dtype=np.float64), sf, sc.sfreq_target)
    f, _, Z = sps.stft(xr, fs=sc.sfreq_target, nperseg=sc.nperseg,
                       noverlap=sc.nperseg - sc.hop, window="hann",
                       boundary=None, padded=False, axis=-1)
    # scipy puts frequency on axis -2 and time on axis -1; swap to (..., T, F).
    S = np.moveaxis(np.abs(Z), -1, -2)
    m = (f >= sc.fmin) & (f <= sc.fmax)
    return np.log10(S[..., m] + 1e-12), f[m]


def normalize(spec: np.ndarray, stats: tuple[np.ndarray, np.ndarray] | None = None):
    """Per-channel z-score over time. spec is (n_ch, T, F)."""
    if stats is None:
        mu = spec.mean(axis=1, keepdims=True)
        sd = spec.std(axis=1, keepdims=True) + 1e-6
    else:
        mu, sd = stats
    return (spec - mu) / sd, (mu, sd)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class ECoGJEPA(nn.Module):
    """Transformer encoder over spectrogram time frames.

    Each token is one time frame's log-power across frequency. The input carries
    the binary mask alongside the (zeroed) values so the model knows which cells
    it is being asked to fill in.
    """

    def __init__(self, n_freq: int, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, max_len: int = 128, dropout: float = 0.0):
        super().__init__()
        self.n_freq = n_freq
        self.d_model = d_model
        self.inp = nn.Linear(2 * n_freq, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        # dropout defaults to 0: masked-reconstruction pretraining is already
        # heavily regularised by the masking itself (as in MAE), and any non-zero
        # dropout forces PyTorch off the fused attention kernel on MPS.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_freq)

    def encode(self, spec: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """spec (B, T, F) -> (B, T, d_model)."""
        if mask is None:
            mask = torch.zeros_like(spec)
        x = torch.cat([spec * (1 - mask), mask], dim=-1)
        h = self.inp(x) + self.pos[:, : x.shape[1]]
        return self.norm(self.enc(h))

    def forward(self, spec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(spec, mask))


def make_mask(shape: tuple[int, int, int], ratio: float, gen: torch.Generator,
              device, n_time_spans: int = 3, span_len: int = 5,
              n_freq_bands: int = 2, band_len: int = 8) -> torch.Tensor:
    """BrainBERT-style structured masking: contiguous time spans and frequency bands.

    Random per-cell masking is far too easy -- neighbouring time-frequency cells
    are highly correlated, so the model can interpolate without learning anything.
    Masking contiguous blocks forces it to use longer-range structure.

    Fully vectorised over the batch; a per-sample Python loop here dominated the
    training step and left the GPU idle.
    """
    B, T, F = shape
    t = torch.arange(T, device=device)[None, :]
    f = torch.arange(F, device=device)[None, :]

    mt = torch.zeros((B, T), device=device, dtype=torch.bool)
    for _ in range(n_time_spans):
        st = torch.randint(0, max(1, T - span_len), (B, 1), generator=gen, device=device)
        mt |= (t >= st) & (t < st + span_len)

    mf = torch.zeros((B, F), device=device, dtype=torch.bool)
    for _ in range(n_freq_bands):
        st = torch.randint(0, max(1, F - band_len), (B, 1), generator=gen, device=device)
        mf |= (f >= st) & (f < st + band_len)

    m = (mt[:, :, None] | mf[:, None, :]).float()
    # Top up with scattered cells to reach the target ratio.
    cur = m.mean()
    extra_p = torch.clamp((ratio - cur) / (1 - cur + 1e-9), min=0.0)
    if float(extra_p) > 0:
        extra = torch.rand(shape, generator=gen, device=device) < extra_p
        m = torch.clamp(m + extra.float(), max=1.0)
    return m


@dataclass
class TrainReport:
    losses: list[float]
    val_losses: list[float]
    baseline_val: float          # loss of predicting the per-frequency mean
    n_params: int
    spec: dict


# --------------------------------------------------------------------------- #
# Pretraining
# --------------------------------------------------------------------------- #

def build_corpus(runs: list[tuple[str, np.ndarray, float]], sc: SpecConfig,
                 val_frac: float = 0.1) -> dict:
    """Spectrogram every continuous recording and split off a held-out tail.

    `runs` is a list of (key, data (n_ch, n_times), sfreq). Normalisation
    statistics are computed on the training portion only.
    """
    train, val, meta = [], [], []
    for key, data, sf in runs:
        S, freqs = spectrogram(data, sf, sc)              # (n_ch, T, F)
        cut = int(S.shape[1] * (1 - val_frac))
        _, stats = normalize(S[:, :cut], None)
        tr, _ = normalize(S[:, :cut], stats)
        va, _ = normalize(S[:, cut:], stats)
        train.append(tr.astype(np.float32))
        val.append(va.astype(np.float32))
        meta.append({"key": key, "n_ch": S.shape[0], "T": S.shape[1]})
    return {"train": train, "val": val, "freqs": freqs, "meta": meta}


def _sample(chunks: list[np.ndarray], n: int, win: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n` random (channel, time-window) spectrogram patches.

    Vectorised per chunk via advanced indexing; the previous per-sample loop was
    the training bottleneck.
    """
    sizes = np.array([c.shape[0] * max(1, c.shape[1] - win) for c in chunks], dtype=np.float64)
    which = rng.choice(len(chunks), size=n, p=sizes / sizes.sum())
    offs = np.arange(win)
    out = np.empty((n, win, chunks[0].shape[2]), dtype=np.float32)
    for w in np.unique(which):
        sel = np.flatnonzero(which == w)
        c = chunks[w]
        ch = rng.integers(0, c.shape[0], size=sel.size)
        t0 = rng.integers(0, max(1, c.shape[1] - win), size=sel.size)
        out[sel] = c[ch[:, None], t0[:, None] + offs[None, :]]
    return out


def pretrain(corpus: dict, cfg: dict, device: str = "cpu", win: int = 64,
             steps_per_epoch: int = 250, log_every: int = 10) -> tuple[ECoGJEPA, TrainReport]:
    e = cfg["ecogjepa"]
    torch.manual_seed(e["seed"])
    rng = np.random.default_rng(e["seed"])
    gen = torch.Generator(device=device).manual_seed(e["seed"])

    n_freq = corpus["train"][0].shape[2]
    model = ECoGJEPA(n_freq=n_freq, d_model=e["d_model"], n_layers=e["n_layers"],
                     n_heads=e["n_heads"], max_len=max(128, win),
                     dropout=e.get("dropout", 0.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=e["lr"], weight_decay=e["weight_decay"])
    total = e["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=e["lr"], total_steps=total,
                                                pct_start=0.1)

    vb = torch.from_numpy(_sample(corpus["val"], 512, win, np.random.default_rng(999))).to(device)
    vmask = make_mask(tuple(vb.shape), e["mask_ratio"], torch.Generator(device=device).manual_seed(7), device)
    # Trivial baseline: predict the channel mean (zero, post z-scoring).
    baseline = float((vb * vmask).abs().sum() / vmask.sum())

    losses, val_losses = [], []
    for ep in range(e["epochs"]):
        model.train()
        run = 0.0
        for _ in range(steps_per_epoch):
            xb = torch.from_numpy(_sample(corpus["train"], e["batch_size"], win, rng)).to(device)
            mk = make_mask(tuple(xb.shape), e["mask_ratio"], gen, device)
            pred = model(xb, mk)
            loss = ((pred - xb).abs() * mk).sum() / mk.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.detach()
        losses.append(float(run) / steps_per_epoch)

        model.eval()
        with torch.no_grad():
            vp = model(vb, vmask)
            val_losses.append(float(((vp - vb).abs() * vmask).sum() / vmask.sum()))
        if ep % log_every == 0 or ep == e["epochs"] - 1:
            print(f"  epoch {ep:3d}  train L1={losses[-1]:.4f}  val L1={val_losses[-1]:.4f} "
                  f"(trivial baseline {baseline:.4f})")

    rep = TrainReport(losses=losses, val_losses=val_losses, baseline_val=baseline,
                      n_params=sum(p.numel() for p in model.parameters()),
                      spec=asdict(SpecConfig()))
    return model, rep


@torch.no_grad()
def embed_epochs(model: ECoGJEPA, X: np.ndarray, sfreq: float, sc: SpecConfig,
                 window: tuple[float, float], tmin: float, device: str = "cpu",
                 stats: tuple | None = None, batch: int = 8) -> np.ndarray:
    """Per-trial embeddings. X is (n_trials, n_ch, n_times).

    Returns (n_trials, n_ch * d_model): each channel encoded independently and
    mean-pooled over the response window, so downstream models retain access to
    which electrode carried which information.
    """
    model.eval()
    n_tr, n_ch = X.shape[:2]
    S, _ = spectrogram(X.reshape(n_tr * n_ch, -1), sfreq, sc)      # (N, T, F)
    S = S.reshape(n_tr, n_ch, S.shape[1], S.shape[2])
    Sn, _ = normalize(S.transpose(1, 0, 2, 3).reshape(n_ch, -1, S.shape[-1]), stats)
    Sn = Sn.reshape(n_ch, n_tr, S.shape[2], S.shape[3]).transpose(1, 0, 2, 3)

    frame_t = tmin + np.arange(S.shape[2]) * sc.frame_s + sc.nperseg / sc.sfreq_target / 2
    keep = (frame_t >= window[0]) & (frame_t < window[1])
    if keep.sum() == 0:
        keep = np.ones_like(frame_t, dtype=bool)

    out = np.empty((n_tr, n_ch * model.d_model), dtype=np.float32)
    flat = Sn.reshape(n_tr * n_ch, S.shape[2], S.shape[3]).astype(np.float32)
    embs = []
    for i in range(0, flat.shape[0], batch * n_ch):
        xb = torch.from_numpy(flat[i:i + batch * n_ch]).to(device)
        h = model.encode(xb)                                       # (B, T, d)
        embs.append(h[:, keep].mean(dim=1).float().cpu().numpy())
    E = np.concatenate(embs).reshape(n_tr, n_ch * model.d_model)
    out[:] = E
    return out
