"""Decoding metrics, with the statistics needed to believe them.

Every headline number is accompanied by a permutation p-value (the null obtained
by shuffling the stimulus-to-trial assignment) and a bootstrap confidence
interval. Chance levels are stated explicitly rather than assumed.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _zs(A: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=1, keepdims=True)
    return A / (A.std(axis=1, keepdims=True) + 1e-12)


def corr_matrix(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Pearson correlation between every predicted latent and every target latent."""
    Pz, Tz = _zs(P), _zs(T)
    if not (np.isfinite(Pz).all() and np.isfinite(Tz).all()):
        raise ValueError("non-finite latents entering correlation")
    # Apple's Accelerate BLAS raises spurious FP flags inside matmul; inputs are
    # verified finite above, so the flags carry no information here.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return Pz @ Tz.T / P.shape[1]


def two_way_identification(P: np.ndarray, T: np.ndarray) -> float:
    """Fraction of distractors the true target beats. Chance = 0.5.

    Computed exhaustively over all n-1 distractors per trial rather than by
    sampling pairs, so there is no Monte-Carlo noise in the estimate.
    """
    C = corr_matrix(P, T)
    n = C.shape[0]
    true = np.diag(C)
    wins = (true[:, None] > C).sum(axis=1) - 0        # diagonal comparison is False
    return float(wins.mean() / (n - 1))


def retrieval(P: np.ndarray, cand: np.ndarray, true_idx: np.ndarray,
              ks=(1, 5, 10)) -> dict[str, float]:
    """Rank a fixed candidate pool by similarity to each predicted latent.

    `cand` is (n_candidates, d); `true_idx` gives each trial's correct row.
    """
    C = corr_matrix(P, cand)                          # (n_trials, n_cand)
    order = np.argsort(-C, axis=1)
    ranks = np.array([int(np.flatnonzero(order[i] == t)[0]) for i, t in enumerate(true_idx)])
    out = {f"top{k}": float((ranks < k).mean()) for k in ks}
    out["median_rank"] = float(np.median(ranks) + 1)
    out["n_candidates"] = int(cand.shape[0])
    # Percentile rank: 1.0 = perfect, 0.5 = chance. Robust to pool size.
    out["percentile_rank"] = float(1.0 - ranks.mean() / (cand.shape[0] - 1))
    return out


def _rdm_parts(X: np.ndarray):
    """Correlation matrix plus the upper-triangle indices used as the RDM vector."""
    C = np.corrcoef(X)
    iu = np.triu_indices(C.shape[0], k=1)
    return C, iu


def rsa(neural: np.ndarray, image: np.ndarray, n_perm: int = 1000,
        seed: int = 0) -> dict[str, float]:
    """Spearman correlation between neural and image representational geometry.

    Both inputs are (n_stimuli, n_features), one row per *stimulus* (trials
    already averaged). The null shuffles stimulus identity on one side.

    Permuting stimulus labels is the same as permuting the rows *and* columns of
    the already-computed correlation matrix, so the matrix is built once rather
    than recomputed per permutation -- which matters when features are ~28k-d.
    """
    Cn, iu = _rdm_parts(neural)
    Ci, _ = _rdm_parts(image)
    a = 1.0 - Cn[iu]
    b = 1.0 - Ci[iu]
    rho = float(stats.spearmanr(a, b).statistic)

    rng = np.random.default_rng(seed)
    n = neural.shape[0]
    br = stats.rankdata(b)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        ap = 1.0 - Cn[np.ix_(p, p)][iu]
        null[i] = stats.spearmanr(ap, br).statistic
    pval = float(((np.abs(null) >= abs(rho)).sum() + 1) / (n_perm + 1))
    return {"rho": rho, "p": pval, "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "n_stimuli": int(n)}


def partial_rsa(neural: np.ndarray, image: np.ndarray, control: np.ndarray,
                n_perm: int = 1000, seed: int = 0) -> dict[str, float]:
    """RSA between neural and image geometry after regressing out a control model.

    Used to ask whether DINOv2 explains neural structure *beyond* what a simple
    V1 oriented-energy model already explains.
    """
    def resid(y_ranked, x_ranked):
        A = np.stack([np.ones_like(x_ranked), x_ranked], axis=1)
        beta, *_ = np.linalg.lstsq(A, y_ranked, rcond=None)
        return y_ranked - A @ beta

    Cn, iu = _rdm_parts(neural)
    Ci, _ = _rdm_parts(image)
    Cc, _ = _rdm_parts(control)
    a = 1.0 - Cn[iu]
    b = 1.0 - Ci[iu]
    c = stats.rankdata(1.0 - Cc[iu])
    b_res = resid(stats.rankdata(b), c)
    rho = float(stats.spearmanr(resid(stats.rankdata(a), c), b_res).statistic)

    rng = np.random.default_rng(seed)
    n = neural.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        ap = 1.0 - Cn[np.ix_(p, p)][iu]
        null[i] = stats.spearmanr(resid(stats.rankdata(ap), c), b_res).statistic
    pval = float(((np.abs(null) >= abs(rho)).sum() + 1) / (n_perm + 1))
    return {"rho_partial": rho, "p": pval, "n_stimuli": int(n)}


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    b = np.array([stat(values[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def permutation_p(observed: float, null: np.ndarray, tail: str = "greater") -> float:
    null = np.asarray(null)
    if tail == "greater":
        k = (null >= observed).sum()
    else:
        k = (np.abs(null) >= abs(observed)).sum()
    return float((k + 1) / (len(null) + 1))


def noise_ceiling(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    """Split-half reliability between two independent measurements of the same stimuli.

    A and B are (n_stimuli, n_features) from two runs. The mean per-stimulus
    correlation across runs upper-bounds what any decoder could achieve.

    Each feature is centred across stimuli first. Without that step the statistic
    is dominated by the encoder's constant offset -- a representation identical for
    every stimulus scores a perfect 1.0, which says nothing about how reliably it
    encodes the stimulus. Centring makes this measure stimulus-specific reliability,
    which is what a noise ceiling is supposed to mean.
    """
    mu = 0.5 * (A.mean(axis=0, keepdims=True) + B.mean(axis=0, keepdims=True))
    Az, Bz = _zs(A - mu), _zs(B - mu)
    per_stim = (Az * Bz).sum(axis=1) / A.shape[1]
    lo, hi = bootstrap_ci(per_stim)
    return {"mean_r": float(per_stim.mean()), "ci_lo": lo, "ci_hi": hi,
            "n_stimuli": int(A.shape[0])}


def within_condition_retrieval(P: np.ndarray, cand: np.ndarray, true_idx: np.ndarray,
                               trial_cond: np.ndarray, cand_cond: np.ndarray,
                               ks=(1, 5)) -> dict[str, float]:
    """Retrieval restricted to same-condition exemplars.

    This is the decisive test of whether a decoder identifies the *image* or only
    its category. With 30 exemplars per condition, chance top-1 is 1/30. Overall
    retrieval can look well above chance purely by getting the condition right,
    so that number alone cannot distinguish the two.
    """
    C = corr_matrix(P, cand)
    ranks, pool = [], []
    for i, t in enumerate(true_idx):
        m = np.flatnonzero(cand_cond == trial_cond[i])
        if len(m) < 2:
            continue
        sub = C[i, m]
        tpos = int(np.flatnonzero(m == t)[0])
        ranks.append(int((sub > sub[tpos]).sum()))
        pool.append(len(m))
    ranks = np.asarray(ranks); pool = np.asarray(pool)
    out = {f"top{k}": float((ranks < k).mean()) for k in ks}
    out["chance_top1"] = float((1.0 / pool).mean())
    out["chance_top5"] = float((np.minimum(5, pool) / pool).mean())
    out["percentile_rank"] = float(1.0 - (ranks / (pool - 1)).mean())
    out["mean_pool"] = float(pool.mean())
    out["n_trials"] = int(len(ranks))
    return out
