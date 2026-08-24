"""Publication-style figures. Every panel is driven by computed results only."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from ..stimuli.params import CONDITION_LABELS

COND_COLORS = {
    1: "#4C72B0", 2: "#55A868", 3: "#8172B2",       # noise
    4: "#C44E52", 5: "#DD8452", 6: "#DA8BC3", 7: "#937860",  # gratings
}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "figure.facecolor": "white",
})


def fig_spectra(freqs, psd_by_cond, psd_blank, out, title=""):
    """The Hermes 2015 dissociation: gratings show a narrowband gamma bump, noise does not."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    base = np.log10(psd_blank + 1e-30)
    for ax, conds, name in [(axes[0], (1, 2, 3), "Noise patterns"),
                            (axes[1], (4, 5, 6, 7), "Gratings")]:
        for c in conds:
            y = np.log10(psd_by_cond[c] + 1e-30) - base
            ax.plot(freqs, y, color=COND_COLORS[c], lw=1.6, label=CONDITION_LABELS[c])
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.axvspan(30, 80, color="0.9", zorder=0)
        ax.set_xscale("log"); ax.set_xlim(8, 200)
        ax.set_xlabel("Frequency (Hz)"); ax.set_title(name)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("log$_{10}$ power change\nfrom blank baseline")
    lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    fig.suptitle(title or "Stimulus dependence of gamma (shaded: 30-80 Hz)", y=1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_encoder_comparison(rows, out, metric="two_way", chance=0.5,
                           ylabel="2-way identification", title=""):
    """rows: list of dicts with keys subject, encoder, label, family, value, ci(optional)."""
    subs = sorted({r["subject"] for r in rows})
    encs = list(dict.fromkeys(r["encoder"] for r in rows))
    fam_color = {"A: hand-crafted": "#2F4B7C",
                 "B: off-the-shelf EEG foundation model": "#C44E52",
                 "C: self-pretrained on this ECoG": "#55A868"}
    fig, axes = plt.subplots(1, len(subs), figsize=(5.2 * len(subs), 3.8), squeeze=False)
    for ax, sub in zip(axes[0], subs):
        rs = [r for r in rows if r["subject"] == sub]
        rs = sorted(rs, key=lambda r: encs.index(r["encoder"]))
        x = np.arange(len(rs))
        vals = [r["value"] for r in rs]
        cols = [fam_color.get(r["family"], "0.5") for r in rs]
        bars = ax.bar(x, vals, color=cols, width=0.68)
        for r, b in zip(rs, bars):
            if r.get("ci"):
                ax.plot([b.get_x() + b.get_width() / 2] * 2, r["ci"], color="0.2", lw=1.2)
        ax.axhline(chance, color="0.35", ls="--", lw=1.0)
        ax.text(len(rs) - 0.4, chance, " chance", va="bottom", ha="right",
                fontsize=7.5, color="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in rs], rotation=28, ha="right", fontsize=7.5)
        ax.set_title(sub); ax.set_ylabel(ylabel if sub == subs[0] else "")
        ax.set_ylim(0, max(1.0, max(vals) * 1.15) if metric != "two_way" else 1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
    fig.legend(handles, list(fam_color), loc="upper center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle(title, y=1.16 if title else 1.0)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_confusion(cm, out, title="", labels=None):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = labels or [CONDITION_LABELS[i] for i in range(1, 8)]
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(norm)):
        for j in range(len(norm)):
            if norm[i, j] > 0.01:
                ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.55 else "0.2")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Decoded"); ax.set_ylabel("Presented"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="proportion")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_param_scatter(Y, P, cols, trial_type, out, title=""):
    """True vs decoded for each continuous stimulus parameter."""
    show = [("log_spatial_freq", "log$_{10}$ spatial freq (cpd)"),
            ("noise_exponent", r"noise exponent $\alpha$"),
            ("rms_contrast", "RMS contrast")]
    fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.2))
    for ax, (c, lbl) in zip(np.atleast_1d(axes), show):
        i = cols.index(c)
        for cond in range(1, 8):
            m = trial_type == cond
            ax.scatter(Y[m, i], P[m, i], s=11, alpha=0.65,
                       color=COND_COLORS[cond], edgecolors="none",
                       label=CONDITION_LABELS[cond])
        lim = [min(Y[:, i].min(), P[:, i].min()), max(Y[:, i].max(), P[:, i].max())]
        ax.plot(lim, lim, color="0.4", ls="--", lw=0.9)
        r = np.corrcoef(Y[:, i], P[:, i])[0, 1]
        ax.set_title(f"{lbl}\nr = {r:+.3f}")
        ax.set_xlabel("presented"); ax.set_ylabel("decoded from brain")
    axes[-1].legend(frameon=False, fontsize=6.2, loc="lower right")
    fig.suptitle(title, y=1.04)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_reconstruction_gallery(items, out, title=""):
    """items: list of (true_image_array, recon_array, caption)."""
    n = len(items)
    fig = plt.figure(figsize=(1.75 * n, 4.0))
    gs = gridspec.GridSpec(2, n, hspace=0.06, wspace=0.06)
    for k, (t, r, cap) in enumerate(items):
        a = fig.add_subplot(gs[0, k]); a.imshow(t, cmap="gray", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([]); a.set_title(cap, fontsize=6.8)
        if k == 0:
            a.set_ylabel("presented", fontsize=8)
        b = fig.add_subplot(gs[1, k]); b.imshow(r, cmap="gray", vmin=0, vmax=1)
        b.set_xticks([]); b.set_yticks([])
        if k == 0:
            b.set_ylabel("reconstructed\nfrom brain", fontsize=8)
    fig.suptitle(title, y=0.97)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_training_curve(report, out, title="ECoG-JEPA self-supervised pretraining"):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(report["losses"], label="train", color="#4C72B0", lw=1.4)
    ax.plot(report["val_losses"], label="held-out", color="#C44E52", lw=1.4)
    ax.axhline(report["baseline_val"], color="0.4", ls="--", lw=1.0,
               label="trivial baseline (predict mean)")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked-reconstruction L1")
    ax.set_title(title); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_electrode_profile(accs, ch_names, out, chance=1 / 7, title=""):
    order = np.argsort(-np.asarray(accs))
    a = np.asarray(accs)[order]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(np.arange(len(a)), a, color="#2F4B7C", width=1.0)
    ax.axhline(chance, color="#C44E52", ls="--", lw=1.1, label="chance (1/7)")
    ax.set_xlabel("electrode (sorted)"); ax.set_ylabel("7-way accuracy")
    ax.set_title(title); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
