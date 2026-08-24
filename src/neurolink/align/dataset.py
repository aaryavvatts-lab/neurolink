"""Assemble per-subject trial matrices across encoders and image latent spaces."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENCODERS = ["spectral", "ecogjepa_all", "ecogjepa_sub-02", "signal_jepa", "cbramod"]
IMAGE_SPACES = ["dino224", "dino518", "gabor"]

ENCODER_LABELS = {
    "spectral": "Spectral (Hermes)",
    "ecogjepa_all": "ECoG-JEPA (ours)",
    "ecogjepa_sub-02": "ECoG-JEPA (sub-02 only)",
    "signal_jepa": "SignalJEPA (EEG)",
    "cbramod": "CBraMod (EEG)",
}
ENCODER_FAMILY = {
    "spectral": "A: hand-crafted",
    "ecogjepa_all": "C: self-pretrained on this ECoG",
    "ecogjepa_sub-02": "C: self-pretrained on this ECoG",
    "signal_jepa": "B: off-the-shelf EEG foundation model",
    "cbramod": "B: off-the-shelf EEG foundation model",
}


@dataclass
class SubjectData:
    sub: str
    stim_id: np.ndarray
    trial_type: np.ndarray
    run: np.ndarray
    feats: dict[str, np.ndarray]      # encoder -> (n_trials, d)
    ch_names: list[str]
    coords_mm: np.ndarray

    def available(self) -> list[str]:
        return [e for e in ENCODERS if e in self.feats]


def load_subject(sub: str, runs: list[str], paths, verbose: bool = True) -> SubjectData:
    """Pool a subject's runs, keeping only trials every encoder could produce."""
    per_run = []
    for run in runs:
        prep = np.load(paths.cache / f"prep_{sub}_run-{run}.npz", allow_pickle=True)
        enc_path = paths.cache / f"enc_{sub}_run-{run}.npz"
        # Step 3 is optional: with only step 2 run, the spectral encoder alone is
        # still a complete, usable pipeline.
        enc = np.load(enc_path, allow_pickle=True) if enc_path.exists() else None
        n = len(prep["stim_id"])

        feats = {"spectral": prep["feat_X"].astype(np.float32)}
        valid = np.ones(n, dtype=bool)
        for key in ENCODERS[1:]:
            if enc is None or key not in enc.files:
                continue
            E = enc[key].astype(np.float32)
            kept_key = f"{key}_kept"
            if kept_key in enc.files:
                kept = enc[kept_key]
                full = np.full((n, E.shape[1]), np.nan, dtype=np.float32)
                full[kept] = E
                mask = np.zeros(n, dtype=bool); mask[kept] = True
                valid &= mask
                feats[key] = full
            else:
                assert E.shape[0] == n, f"{key} trial count {E.shape[0]} != {n}"
                feats[key] = E

        per_run.append({
            "feats": {k: v[valid] for k, v in feats.items()},
            "stim_id": prep["stim_id"][valid],
            "trial_type": prep["trial_type"][valid],
            "run": np.full(valid.sum(), run),
            "ch_names": ([str(c) for c in enc["ch_names"]] if enc is not None
                         else [str(c) for c in prep["ch_names"]]),
            "coords_mm": enc["coords_mm"] if enc is not None else None,
            "n_dropped": int((~valid).sum()),
        })
        if verbose and per_run[-1]["n_dropped"]:
            print(f"  [{sub} run-{run}] dropped {per_run[-1]['n_dropped']} trials "
                  f"lacking a full encoder window")

    common = sorted(set.intersection(*[set(r["feats"]) for r in per_run]))
    feats = {k: np.concatenate([r["feats"][k] for r in per_run]) for k in common}
    return SubjectData(
        sub=sub,
        stim_id=np.concatenate([r["stim_id"] for r in per_run]),
        trial_type=np.concatenate([r["trial_type"] for r in per_run]),
        run=np.concatenate([r["run"] for r in per_run]),
        feats=feats,
        ch_names=per_run[0]["ch_names"],
        coords_mm=per_run[0]["coords_mm"],
    )


def load_image_spaces(paths) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Image latents indexed by stimulus id, plus the measured parameter table."""
    z = np.load(paths.cache / "stimuli.npz", allow_pickle=True)
    stim_id = z["stim_id"]
    spaces = {
        "dino224": z["dino224_concat"],
        "dino518": z["dino518_concat"],
        "gabor": z["gabor"],
    }
    params = {"cols": [str(c) for c in z["param_cols"]], "X": z["params"],
              "trial_type": z["trial_type"]}
    return spaces, stim_id, params


def image_targets(space: np.ndarray, space_stim_id: np.ndarray,
                  trial_stim_id: np.ndarray) -> np.ndarray:
    """Expand a per-image latent matrix to one row per trial."""
    pos = {int(s): i for i, s in enumerate(space_stim_id)}
    return space[np.array([pos[int(s)] for s in trial_stim_id])]
