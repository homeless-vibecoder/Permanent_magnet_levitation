"""Geometry for rectangular permanent magnets used in the force-field model.

A magnet is modeled by the surface magnetic charge picture: the two faces normal
to z carry uniform sheets of magnetic charge (north face = +, south face = -).
Only the planar footprint of a face is needed here; the sign and the z-position
of each face are handled by the force pipeline in ``magnet_strength.py``.

Per the project spec, orientation is ignored: x is always lengthwise and y is
always widthwise. The ``theta`` field is kept for API completeness but unused.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RectangularMagnet:
    length: float  # extent along x (mm)
    width: float  # extent along y (mm)
    thickness: float  # extent along z (mm)
    theta: float = 0.0  # orientation in degrees; intentionally ignored

    @property
    def area(self) -> float:
        return self.length * self.width

    def footprint(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Rasterize the face footprint, centered at the origin, onto a grid.

        Returns a ``(len(ys), len(xs))`` array that is 1.0 inside the rectangle
        and 0.0 outside. This is a *charge density* (charge per unit area = 1),
        so the represented total charge is ``sum(mask) * dx * dy`` which tends to
        ``length * width`` as the grid is refined -- i.e. proportional to area,
        not to the number of grid points.
        """
        x = np.asarray(xs)
        y = np.asarray(ys)
        mask_x = np.abs(x) <= self.length / 2.0
        mask_y = np.abs(y) <= self.width / 2.0
        return np.outer(mask_y, mask_x).astype(float)
