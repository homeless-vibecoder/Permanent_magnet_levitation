"""Diagnostic figure for the platform optimizer.

Panel 1: horizontal slices of the single-magnet Fz field -- shows that the
         force is repulsive (+) directly above ground magnets but attractive
         (-) when laterally offset. The field is NOT uniformly positive.
Panel 2: distribution of NET vertical force on the platform over random
         (translation, rotation) placements, with the mean and worst-15%
         cutoff marked. Mean > 0, but a sizable tail is negative.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import optimizer as O  # reuse its config + builders


def main():
    rng = np.random.default_rng(O.SEED)
    xs, ys, Fx, Fy, Fz, centers, ground_bbox = O.build_single_field()
    model = O.PlatformModel(xs, ys, Fz)

    # ---- Panel 1 data: slices through a magnet row and through a gap row ----
    ax_sp, ay_sp = O.GROUND_SPACING
    nx, ny = O.GROUND_COUNT
    # central magnet row y, and the gap row half a lattice step above it
    y_mag = (ny // 2) * ay_sp
    y_gap = y_mag + ay_sp / 2.0
    j_mag = int(round((y_mag - ys[0]) / O.DY))
    j_gap = int(round((y_gap - ys[0]) / O.DY))

    # ---- Panel 2 data: net Fz distribution at the initial layout ----
    positions = O.init_positions(np.random.default_rng(O.SEED))
    T, theta = O.sample_placements(rng, ground_bbox)
    fz, _, _ = model.evaluate(positions, T, theta)
    k = max(1, int(round(O.N_SAMPLES * O.WORST_FRACTION)))
    thr = np.sort(fz)[k - 1]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5))

    axL.axhline(0, color="k", lw=0.8)
    axL.plot(xs, Fz[j_mag, :], color="crimson", lw=1.8,
             label="through a magnet row (y = magnet centers)")
    axL.plot(xs, Fz[j_gap, :], color="steelblue", lw=1.8,
             label="through a gap row (y = halfway between rows)")
    for i in range(nx):
        axL.axvline(i * ax_sp, color="0.8", lw=0.6, zorder=0)
    axL.set_title("Single platform-magnet vertical force vs. position\n"
                  "(+ = lift / repulsion,  - = pull-down / attraction)")
    axL.set_xlabel("x (mm)   [grey lines = ground-magnet columns]")
    axL.set_ylabel("Fz on one platform magnet")
    axL.legend(loc="upper right", fontsize=8)
    axL.set_xlim(xs[0], xs[-1])

    axR.hist(fz, bins=60, color="slategray", alpha=0.85)
    axR.axvline(0.0, color="k", lw=1.0, ls=":")
    axR.axvline(fz.mean(), color="green", lw=2.0,
                label=f"mean = {fz.mean():+.2f}  (net lift > 0)")
    axR.axvline(thr, color="crimson", lw=2.0,
                label=f"worst-15% cutoff = {thr:+.2f}")
    axR.axvline(fz[fz <= thr].mean(), color="darkred", lw=2.0, ls="--",
                label=f"mean of worst 15% = {fz[fz <= thr].mean():+.2f}")
    axR.set_title("Net vertical force over random platform placements\n"
                  "(initial layout; the objective is the dashed dark-red line)")
    axR.set_xlabel("net Fz on the whole platform")
    axR.set_ylabel("count")
    axR.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    out = os.path.join(_HERE, "diagnosis.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print(f"mean Fz over placements      = {fz.mean():+.3f}")
    print(f"fraction of placements < 0   = {(fz < 0).mean()*100:.1f}%")
    print(f"mean of worst {int(O.WORST_FRACTION*100)}%          = {fz[fz<=thr].mean():+.3f}")


if __name__ == "__main__":
    main()
