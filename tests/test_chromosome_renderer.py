"""Source-geometry checks for the independent visualization script."""

from pathlib import Path
import math
import runpy
import xml.etree.ElementTree as ET
import numpy as np

renderer = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/chromosome_frame.py")
)


def test_original_strand_centers_radii_and_angles():
    geometry = renderer["arc_geometry"]
    assert geometry(25, 1, 25, 100)[:3] == (150, 150, 95)
    assert geometry(25, 2, 25, 100)[:3] == (150, 150, 105)
    assert geometry(25, 3, 25, 100, True)[:3] == (450, 150, 90)
    assert geometry(25, 4, 25, 100, True)[:3] == (450, 150, 110)
    np.testing.assert_allclose(geometry(25, 1, 25, 100)[3:], [0, math.pi / 2])


def test_full_chromosome_circle_and_actual_footprint_arc(tmp_path):
    data = {
        "time_s": 60,
        "sequence_length": 100,
        "arcs": [(1, 1, 100, "black", 2), (25, 4, 10, "red", 1)],
    }
    renderer["render"](data, tmp_path / "frame")
    root = ET.parse(tmp_path / "frame.svg").getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    circle = root.find("s:circle", ns)
    arc = root.find("s:path", ns)
    assert circle.attrib["cx"] == "150" and circle.attrib["r"] == "95"
    assert arc.attrib["stroke"] == "red" and "A110,110" in arc.attrib["d"]
    assert (tmp_path / "frame.png").is_file()
