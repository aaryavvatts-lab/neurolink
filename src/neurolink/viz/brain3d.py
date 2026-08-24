"""Render electrode decoding contribution on the subject's own pial surface.

A biological sanity check: if the decodable signal is real visual information,
the informative electrodes should sit over early visual cortex, not scatter at
random across the array.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
import pyvista as pv

pv.OFF_SCREEN = True

VIEWS = {
    "R": {"posterior": (0, -1, 0), "lateral": (1, 0, 0), "ventral": (0, 0, -1)},
    "L": {"posterior": (0, -1, 0), "lateral": (-1, 0, 0), "ventral": (0, 0, -1)},
}


def load_pial(path) -> pv.PolyData:
    g = nib.load(str(path))
    v = np.asarray(g.agg_data("NIFTI_INTENT_POINTSET"), dtype=np.float64)
    f = np.asarray(g.agg_data("NIFTI_INTENT_TRIANGLE"), dtype=np.int64)
    faces = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel()
    return pv.PolyData(v, faces)


def render(pial_path, coords_mm: np.ndarray, values: np.ndarray, hemi: str,
          out_prefix, labels: np.ndarray | None = None, radius: float = 2.2,
          cmap: str = "inferno", window=(1000, 800), title: str | None = None
          ) -> list[str]:
    """Write one PNG per anatomical view. `values` colours the electrodes."""
    brain = load_pial(pial_path)
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    lo, hi = (np.nanmin(v[finite]), np.nanmax(v[finite])) if finite.any() else (0.0, 1.0)

    written = []
    for name, direction in VIEWS[hemi].items():
        p = pv.Plotter(off_screen=True, window_size=window)
        p.add_mesh(brain, color="#d9d2cc", opacity=1.0, smooth_shading=True,
                   specular=0.15, diffuse=0.9, ambient=0.25)
        pts = pv.PolyData(np.asarray(coords_mm, dtype=float))
        pts["value"] = v
        glyph = pts.glyph(geom=pv.Sphere(radius=radius, theta_resolution=18,
                                         phi_resolution=18), scale=False, orient=False)
        p.add_mesh(glyph, scalars="value", cmap=cmap, clim=(lo, hi),
                   scalar_bar_args={"title": title or "contribution",
                                    "vertical": True, "position_x": 0.86,
                                    "position_y": 0.25, "height": 0.5,
                                    "title_font_size": 14, "label_font_size": 12,
                                    "color": "black"})
        p.set_background("white")
        p.camera_position = [
            tuple(np.array(direction, dtype=float) * 320 + brain.center),
            tuple(brain.center),
            (0, 0, 1) if name != "ventral" else (0, 1, 0),
        ]
        p.camera.zoom(1.35)
        out = f"{out_prefix}_{name}.png"
        p.screenshot(out)
        p.close()
        written.append(out)
    return written
