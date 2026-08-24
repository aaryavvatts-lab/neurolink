"""Parameter recovery: synthesise stimuli with known parameters, measure them back."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from PIL import Image

from neurolink.stimuli.params import measure, render_grating, render_noise

DEG = 25.0


def _tmp_png(arr, tmp_path, name="x.png"):
    p = tmp_path / name
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(p)
    return str(p)


@pytest.mark.parametrize("sf", [0.16, 0.32, 0.64, 1.28])
def test_grating_spatial_frequency_recovered(sf, tmp_path):
    img = render_grating(sf, 90.0, 0.0, 0.5, DEG, size=900)
    m = measure(_tmp_png(img, tmp_path), deg=DEG, size=512)
    assert m["spatial_freq_cpd"] == pytest.approx(sf, rel=0.12), m


@pytest.mark.parametrize("ori", [0.0, 30.0, 60.0, 90.0, 135.0])
def test_grating_orientation_recovered(ori, tmp_path):
    img = render_grating(0.64, ori, 0.0, 0.5, DEG, size=900)
    m = measure(_tmp_png(img, tmp_path), deg=DEG, size=512)
    # Orientation is defined mod 180 deg; compare on the circle.
    d = np.degrees(np.angle(np.exp(2j * np.radians(m["orientation_deg"] - ori)))) / 2
    assert abs(d) < 8.0, (m["orientation_deg"], ori)


@pytest.mark.parametrize("alpha", [0.0, 2.0, 4.0])
def test_noise_exponent_recovered(alpha, tmp_path):
    img = render_noise(alpha, 0.15, size=900, seed=3)
    m = measure(_tmp_png(img, tmp_path), deg=DEG, size=512)
    assert m["noise_exponent"] == pytest.approx(alpha, abs=0.6), m


def test_gratings_are_more_oriented_than_noise(tmp_path):
    g = measure(_tmp_png(render_grating(0.64, 45.0, 0.0, 0.5, DEG, 900), tmp_path, "g.png"),
                deg=DEG, size=512)
    n = measure(_tmp_png(render_noise(2.0, 0.15, 900, seed=1), tmp_path, "n.png"),
                deg=DEG, size=512)
    assert g["orient_concentration"] > n["orient_concentration"]


def test_blank_detected(tmp_path):
    m = measure(_tmp_png(np.full((900, 900), 0.5), tmp_path, "b.png"), deg=DEG, size=512)
    assert m["is_blank"] and m["rms_contrast"] == 0.0
