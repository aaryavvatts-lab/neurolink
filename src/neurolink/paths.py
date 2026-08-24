"""Path resolution and config loading. Single source of truth for where things live."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(p)
    return cfg


@dataclass(frozen=True)
class Paths:
    bids_root: Path
    outputs: Path

    @property
    def stimuli(self) -> Path:
        return self.bids_root / "stimuli"

    @property
    def cache(self) -> Path:
        return self.outputs / "cache"

    @property
    def models(self) -> Path:
        return self.outputs / "models"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def video(self) -> Path:
        return self.outputs / "video"

    @property
    def site_public(self) -> Path:
        return REPO_ROOT / "site" / "public"

    def ieeg_dir(self, sub: str, ses: str = "01") -> Path:
        return self.bids_root / sub / f"ses-{ses}" / "ieeg"

    def run_stem(self, sub: str, run: str, ses: str = "01") -> Path:
        return self.ieeg_dir(sub, ses) / f"{sub}_ses-{ses}_task-visual_run-{run}"

    def electrodes_tsv(self, sub: str, ses: str = "01") -> Path:
        return self.ieeg_dir(sub, ses) / f"{sub}_ses-{ses}_electrodes.tsv"

    def pial_gii(self, sub: str, hemi: str, ses: str = "01") -> Path:
        return (
            self.bids_root
            / "derivatives" / "surface" / sub / f"ses-{ses}" / "anat"
            / f"{sub}_ses-{ses}_T1w_pial.{hemi}.surf.gii"
        )

    def ensure(self) -> "Paths":
        for d in (self.cache, self.models, self.figures, self.video):
            d.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=4)
def get_paths(config_path: str | None = None) -> Paths:
    cfg = load_config(config_path)
    bids_root = (REPO_ROOT / cfg["paths"]["bids_root"]).resolve()
    outputs = (REPO_ROOT / cfg["paths"]["outputs"]).resolve()
    return Paths(bids_root=bids_root, outputs=outputs).ensure()
