"""Decisive test: line integral vs plane integral of Fz for a REAL magnet.

Same flipped dipole-dipole setup as graphs/test_dipole (h=0.5, thickness=0.2),
but in full 3D. We accumulate the vertical force two ways as we grow the cutoff:

    I_line(s)   = int_{-s}^{s} Fz(x, y=0) dx          (1-D, like your graph)
    I_plane(R)  = int int_{x^2+y^2 <= R^2} Fz dx dy   (2-D, the platform case)

Claim under test: "2D is just 1D spread over area; asymptotics are the same."
If true, both curves (suitably normalized) would tend to the same nonzero value.
What actually happens: I_line -> +0.83 (nonzero), I_plane -> 0 (cancels).
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = 0.5          # test-magnet centre height above source + pole
T = 0.2          # pole gap (thickness) of both magnets
Q = 1.0

# Flipped dipole-dipole pole pairs -> (sign = q_s*q_t, vertical gap dz):
#   src+ @0 & test+ @h-t/2 : +, dz = h - t/2
#   src+ @0 & test- @h+t/2 : -, dz = h + t/2
#   src- @-t & test+@h-t/2 : -, dz = h + t/2
#   src- @-t & test-@h+t/2 : +, dz = h + 3t/2
PAIRS = [
    (+1.0, H - T / 2),
    (-1.0, H + T / 2),
    (-1.0, H + T / 2),
    (+1.0, H + 3 * T / 2),
]


def Fz_on_line(x):
    out = np.zeros_like(x, dtype=float)
    for sign, dz in PAIRS:
        out += sign * Q * Q * dz / (x * x + dz * dz) ** 1.5
    return out


def Fz_on_grid(X, Y):
    out = np.zeros_like(X, dtype=float)
    R2 = X * X + Y * Y
    for sign, dz in PAIRS:
        out += sign * Q * Q * dz / (R2 + dz * dz) ** 1.5
    return out


def main():
    # ---- analytic asymptotes ----
    line_inf = sum(sign * Q * Q * 2.0 / dz for sign, dz in PAIRS)   # full line
    plane_inf = sum(sign * Q * Q * 2.0 * np.pi for sign, dz in PAIRS)
    print(f"analytic  I_line(inf)  = {line_inf:+.4f}   (sum of 2*sign/dz)")
    print(f"analytic  I_plane(inf) = {plane_inf:+.4f}   (sum of 2*pi*sign)")

    # The field is axially symmetric (source on the z-axis), so Fz depends only
    # on lateral distance r. Both accumulations are then fast 1-D integrals:
    #   I_line(s)  = 2 * int_0^s Fz(r) dr
    #   I_plane(R) = int_0^R Fz(r) * 2*pi*r dr
    smax = 500.0
    dr = 1e-3
    r = np.arange(dr, smax, dr)
    fz_r = Fz_on_line(r)

    I_line = 2.0 * np.cumsum(fz_r) * dr
    I_plane = np.cumsum(fz_r * 2.0 * np.pi * r) * dr

    s_axis = r
    Rbins = r

    print(f"numeric   I_line(s={smax:.0f})  = {I_line[-1]:+.4f}")
    print(f"numeric   I_plane(R={smax:.0f}) = {I_plane[-1]:+.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.axhline(0, color="k", lw=0.8)
    ax1.plot(s_axis, I_line, color="C0", lw=2.5,
             label=r"line:  $\int_{-s}^{s} F_z\,dx$   (1-D, your graph)")
    ax1.plot(Rbins, I_plane, color="C3", lw=2.5,
             label=r"plane:  $\int\!\int_{r<R} F_z\,dA$   (2-D, platform)")
    ax1.axhline(line_inf, color="C0", lw=1.0, ls="--",
                label=f"line asymptote = {line_inf:+.3f}")
    ax1.axhline(0.0, color="C3", lw=1.0, ls="--",
                label="plane asymptote = 0.000")
    ax1.set_xscale("log")
    ax1.set_xlabel("cutoff   s  or  R   (log scale)")
    ax1.set_ylabel("accumulated vertical force")
    ax1.set_title("Same magnet, same kernel: 1-D line integral survives,\n"
                  "2-D plane integral cancels to zero (Gauss / no monopole)")
    ax1.legend(loc="center left", fontsize=9)
    ax1.set_xlim(0.05, smax)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "line_vs_plane.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
