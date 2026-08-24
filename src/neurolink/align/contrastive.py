"""CLIP-style contrastive alignment of the neural and image latent spaces.

This is the method the project brief describes: project both modalities into a
shared space and train with InfoNCE so a trial's neural embedding sits closest to
the embedding of the image that produced it.

It is included because it is the named method, and reported honestly against
ridge regression. At ~175 training trials a two-tower MLP has far more capacity
than the data supports, so it is regularised hard (dropout, weight decay, early
stopping on a group-disjoint validation split) and is not expected to win.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self, n_in: int, hidden: int, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_in),
            nn.Dropout(dropout),
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_neural: int, n_image: int, hidden: int, dim: int, dropout: float):
        super().__init__()
        self.neural = Tower(n_neural, hidden, dim, dropout)
        self.image = Tower(n_image, hidden, dim, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, xn, xi):
        return self.neural(xn), self.image(xi)


def info_nce(zn: torch.Tensor, zi: torch.Tensor, scale: torch.Tensor,
             groups: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric InfoNCE.

    When `groups` is given, other trials showing the *same* image are masked out
    of the negatives -- treating them as negatives would train the model to
    separate identical stimuli.
    """
    logits = scale.exp().clamp(max=100.0) * zn @ zi.T
    n = logits.shape[0]
    tgt = torch.arange(n, device=logits.device)
    if groups is not None:
        same = groups[:, None] == groups[None, :]
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, -1e4)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))


def fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, cfg: dict,
                groups_tr: np.ndarray | None = None, device: str = "cpu",
                verbose: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train the two towers and return (test neural embeddings, image projector, log).

    Returns predictions in the *shared* space, so downstream metrics compare
    projected neural embeddings against projected image embeddings.
    """
    c = cfg["align"]["contrastive"]
    seed = cfg["align"]["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Standardise using training statistics only.
    mn, sn = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    mi, si = Ytr.mean(0, keepdims=True), Ytr.std(0, keepdims=True) + 1e-8
    Xtr_, Xte_ = (Xtr - mn) / sn, (Xte - mn) / sn
    Ytr_ = (Ytr - mi) / si

    # Group-disjoint validation split for early stopping.
    g = groups_tr if groups_tr is not None else np.arange(len(Xtr_))
    uniq = rng.permutation(np.unique(g))
    n_val = max(1, int(0.2 * len(uniq)))
    val_g = set(uniq[:n_val].tolist())
    vm = np.array([x in val_g for x in g])
    tr_idx, va_idx = np.flatnonzero(~vm), np.flatnonzero(vm)

    dev = torch.device(device)
    xt = torch.tensor(Xtr_[tr_idx], dtype=torch.float32, device=dev)
    yt = torch.tensor(Ytr_[tr_idx], dtype=torch.float32, device=dev)
    gt = torch.tensor(g[tr_idx].astype(np.int64), device=dev)
    xv = torch.tensor(Xtr_[va_idx], dtype=torch.float32, device=dev)
    yv = torch.tensor(Ytr_[va_idx], dtype=torch.float32, device=dev)
    gv = torch.tensor(g[va_idx].astype(np.int64), device=dev)

    model = TwoTower(Xtr_.shape[1], Ytr_.shape[1], c["hidden"], c["dim"], c["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])

    best, best_state, bad, log = np.inf, None, 0, []
    for ep in range(c["epochs"]):
        model.train()
        zn, zi = model(xt, yt)
        loss = info_nce(zn, zi, model.logit_scale, gt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vn, vi = model(xv, yv)
            vloss = float(info_nce(vn, vi, model.logit_scale, gv))
        log.append({"epoch": ep, "train": float(loss.detach()), "val": vloss})
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c["patience"]:
                break
        if verbose and ep % 50 == 0:
            print(f"    ep{ep:4d} train={float(loss):.3f} val={vloss:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model.neural(torch.tensor(Xte_, dtype=torch.float32, device=dev)).cpu().numpy()

    def project_images(Y: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model.image(torch.tensor((Y - mi) / si, dtype=torch.float32,
                                            device=dev)).cpu().numpy()

    return pred, project_images, {"best_val": best, "epochs_run": len(log)}
