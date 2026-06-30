"""Diagnostic: why does the optimizer see a (near) zero MEAN vertical force?

This reproduces the single-magnet Fz field exactly as optimizer.py builds it,
and then measures three different integrals so we can compare:

  (A) Fz directly above a ground magnet          -> should be POSITIVE (lift)
  (B) 2-D plane integral   sum Fz * dA            -> ~ ZERO  (Gauss/Earnshaw)
  (C) 1-D line integral    int Fz dx at fixed y   -> POSITIVE (matches graphs/)

It also prints the analytic kernel integrals that explain (B) vs (C).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from magnet_strength import (
    RectangularMagnet,
    pole_pairs,
    build_grid,
    build_ground_charges,
    compute_force_field,
)

# Same config as optimizer.py
GROUND_MAGNET = RectangularMagnet(length=10.0, width=10.0, thickness=2.0)
UNIT_MAGNET = RectangularMagnet(length=10.0, width=10.0, thickness=2.0)
GROUND_SPACING = (16.0, 16.0)
GROUND_COUNT = (12, 12)
HOVER_HEIGHT = 4.0
STRENGTH_CONSTANT = 1.0
DX = DY = 0.6
FIELD_MARGIN = 10.0


def build_single_field():
    gm, um = GROUND_MAGNET, UNIT_MAGNET
    ax, ay = GROUND_SPACING
    nx, ny = GROUND_COUNT
    centers = [(i * ax, j * ay) for i in range(nx) for j in range(ny)]
    pairs = pole_pairs(HOVER_HEIGHT, gm.thickness, um.thickness)
    xs, ys, rxs, rys = build_grid(centers, gm, DX, DY, FIELD_MARGIN)
    G = build_ground_charges(centers, gm, xs, ys, rxs, rys)
    T = um.footprint(rxs, rys)
    Fx, Fy, Fz = compute_force_field(G, T, rxs, rys, DX, DY, pairs, STRENGTH_CONSTANT)
    return xs, ys, Fz, centers


def main():
    xs, ys, Fz, centers = build_single_field()
    dA = DX * DY

    # (A) value directly above a central ground magnet
    cx, cy = centers[len(centers) // 2]
    i = int(round((cx - xs[0]) / DX))
    j = int(round((cy - ys[0]) / DY))
    print("Field shape:", Fz.shape, " Fz range:", (Fz.min(), Fz.max()))
    print(f"(A) Fz directly above a magnet center = {Fz[j, i]:+.4f}   (expect > 0, lift)")

    # (B) full 2-D plane integral and the spatial mean
    plane_integral = Fz.sum() * dA
    print(f"(B) 2-D plane integral  sum Fz*dA      = {plane_integral:+.4e}")
    print(f"    spatial mean Fz                    = {Fz.mean():+.4e}   (expect ~ 0)")

    # interior-only mean (what the optimizer effectively averages over)
    mx = int(FIELD_MARGIN / DX) + 5
    interior = Fz[mx:-mx, mx:-mx]
    print(f"    interior mean Fz                   = {interior.mean():+.4e}")

    # (C) 1-D line integral through a magnet center (matches graphs/ I(s))
    line_integral = Fz[j, :].sum() * DX
    print(f"(C) 1-D line integral   int Fz dx (y=cy)= {line_integral:+.4f}   (expect > 0)")

    # Analytic kernel integrals that explain (B) vs (C):
    # for a +1 source, the single-charge Fz kernel = C*dz/(x^2+y^2+dz^2)^(3/2).
    print("\nWhy (B)~0 but (C)>0 -- single-charge kernel integrals vs gap dz:")
    print(f"{'dz':>6} {'sign':>5} {'plane int (2pi)':>16} {'line int (2/dz)':>16}")
    pairs = pole_pairs(HOVER_HEIGHT, GROUND_MAGNET.thickness, UNIT_MAGNET.thickness)
    tot_plane = tot_line = 0.0
    for name, dz, sign in pairs:
        plane = 2.0 * np.pi          # independent of dz
        line = 2.0 / dz              # depends on dz
        tot_plane += sign * plane
        tot_line += sign * line
        print(f"{dz:>6.1f} {sign:>+5.0f} {sign*plane:>16.4f} {sign*line:>16.4f}")
    print(f"{'sum':>6} {'':>5} {tot_plane:>16.4f} {tot_line:>16.4f}")
    print("\n=> plane integral cancels (sum of signs = 0); line integral does NOT.")


if __name__ == "__main__":
    main()
