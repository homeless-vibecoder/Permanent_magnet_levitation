"""
Force on a test DIPOLE (a small magnet) near a source dipole.

Difference from the test_charge model
--------------------------------------
Previously the test object was a single + magnetic charge. Here the test
object is itself a small magnet (a dipole), so this shows the actual force
a *magnet* would feel near another magnet (including attraction).

Geometry (all unit constants)
-----------------------------
Source magnet (fixed), moment pointing +y:
    +q at (0, 0),   -q at (0, -thickness).

Test magnet FLIPPED (moment pointing -y, i.e. + charge on the bottom),
centred at (x, h), same thickness:
    -q at (x, h + thickness/2),   (top)
    +q at (x, h - thickness/2).   (bottom)

The net force on the test magnet is the sum of the Coulomb forces on each
of its two charges from each of the two source charges:

    F = sum_{test t} sum_{source s}  q_t * q_s * (P_t - P_s) / |P_t - P_s|^3.

Pairwise signs q_t*q_s: like poles repel (+), unlike attract (-). With the
test magnet flipped, the source's top +q faces the test's bottom +q (like
poles), so directly above the source the test magnet is pushed UP --
i.e. REPULSION (the levitation-relevant configuration).

Outputs (written to ../images/)
-------------------------------
1. dipole_dipole_force_field.png       setup + colour-coded net-force field
2. dipole_dipole_force_Fy.png          y-component (linear, clipped colour scale)
3. dipole_dipole_force_Fy_symlog.png   y-component on a symmetric-log scale
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, SymLogNorm
import os

# ----------------------------------------------------------------------
# Model parameters (all unit constants)
# ----------------------------------------------------------------------
thickness = 0.2     # vertical +/- gap, same for source and test magnet
q = 1.0             # magnitude of each magnetic charge

# Source magnet charges: (charge, x, y)
SOURCE = [(+q, 0.0, 0.0), (-q, 0.0, -thickness)]

# Test magnet (FLIPPED): charge offsets relative to its centre (dx, dy, charge)
TEST_OFFSETS = [(0.0, +thickness / 2, -q),   # top (-)
                (0.0, -thickness / 2, +q)]   # bottom (+)

# Plot domain for the test-magnet centre (x, y).  y plays the role of h.
X_MIN, X_MAX = -2.0, 2.0
Y_MIN, Y_MAX = -2.0, 2.0


def force_on_test(x, y):
    """Net (Fx, Fy) on the test magnet whose CENTRE is at (x, y)."""
    Fx = np.zeros_like(x, dtype=float)
    Fy = np.zeros_like(y, dtype=float)
    for ox, oy, qt in TEST_OFFSETS:
        tx = x + ox
        ty = y + oy
        for qs, sx, sy in SOURCE:
            dx = tx - sx
            dy = ty - sy
            r3 = np.power(dx * dx + dy * dy, 1.5)
            with np.errstate(divide="ignore", invalid="ignore"):
                coeff = qt * qs / r3
            Fx = Fx + coeff * dx
            Fy = Fy + coeff * dy
    return Fx, Fy


def _too_close(x, y, margin):
    return (
        (np.hypot(x - 0.0, y - 0.0) < margin)
        | (np.hypot(x - 0.0, y + thickness) < margin)
    )


def draw_source(ax):
    """Draw the source magnet and a small schematic of the test magnet."""
    ax.scatter(0.0, 0.0, s=220, c="red", edgecolors="black", zorder=6)
    ax.scatter(0.0, -thickness, s=220, c="blue", edgecolors="black", zorder=6)
    ax.annotate("source +q\n(0, 0)", (0.0, 0.0), textcoords="offset points",
                xytext=(10, 6), fontsize=9, fontweight="bold", color="red")
    ax.annotate("source -q\n(0, -t)", (0.0, -thickness),
                textcoords="offset points", xytext=(10, -24), fontsize=9,
                fontweight="bold", color="blue")

    # Schematic test magnet (FLIPPED: - on top, + on bottom) upper-right.
    cx, cy = 0.78 * X_MAX, 0.78 * Y_MAX
    ax.scatter(cx, cy + thickness / 2, s=90, c="blue", edgecolors="black",
               zorder=6)
    ax.scatter(cx, cy - thickness / 2, s=90, c="red", edgecolors="black",
               zorder=6)
    ax.annotate("test magnet\n(flipped: + below)", (cx, cy + thickness / 2),
                textcoords="offset points", xytext=(8, 2), fontsize=8)


# ----------------------------------------------------------------------
# Figure 1: setup + colour-coded net-force vector field
# ----------------------------------------------------------------------
def plot_force_field(outfile):
    n = 31
    xs = np.linspace(X_MIN, X_MAX, n)
    ys = np.linspace(Y_MIN, Y_MAX, n)
    Xg, Yg = np.meshgrid(xs, ys)

    Fx, Fy = force_on_test(Xg, Yg)
    mag = np.hypot(Fx, Fy)

    too_close = _too_close(Xg, Yg, margin=0.3)
    mag_masked = np.where(too_close, np.nan, mag)

    with np.errstate(divide="ignore", invalid="ignore"):
        U = np.where(mag > 0, Fx / mag, 0.0)
        V = np.where(mag > 0, Fy / mag, 0.0)
    U = np.where(too_close, np.nan, U)
    V = np.where(too_close, np.nan, V)

    fig, ax = plt.subplots(figsize=(8, 8))

    vmin = np.nanmin(mag_masked[mag_masked > 0])
    vmax = np.nanmax(mag_masked)
    qv = ax.quiver(Xg, Yg, U, V, mag_masked,
                   cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax),
                   pivot="mid", scale=34, width=0.004)
    cb = fig.colorbar(qv, ax=ax, shrink=0.85)
    cb.set_label("|F| on test magnet (arb. units, log scale)")

    draw_source(ax)

    ax.set_title("Net force on a test MAGNET (dipole) near a source magnet\n"
                 "(arrows = direction, colour = magnitude)")
    ax.set_xlabel("x  (test-magnet centre x)")
    ax.set_ylabel("y = h  (test-magnet centre height)")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}")


# ----------------------------------------------------------------------
# Figure 2: y-component (linear scale)
# ----------------------------------------------------------------------
def plot_force_y_projection(outfile):
    n = 400
    xs = np.linspace(X_MIN, X_MAX, n)
    ys = np.linspace(Y_MIN, Y_MAX, n)
    Xg, Yg = np.meshgrid(xs, ys)

    _, Fy = force_on_test(Xg, Yg)

    clip = 5.0
    Fy_clipped = np.clip(Fy, -clip, clip)

    fig, ax = plt.subplots(figsize=(8, 8))

    norm = TwoSlopeNorm(vmin=-clip, vcenter=0.0, vmax=clip)
    pcm = ax.pcolormesh(Xg, Yg, Fy_clipped, cmap="RdBu_r", norm=norm,
                        shading="auto")
    cb = fig.colorbar(pcm, ax=ax, shrink=0.85)
    cb.set_label("F_y on test magnet (red = up = repulsion, blue = down)")

    ax.contour(Xg, Yg, Fy, levels=[0.0], colors="black",
               linewidths=1.2, linestyles="--")

    draw_source(ax)

    ax.set_title("Vertical (y) component of the net force on the test magnet")
    ax.set_xlabel("x  (test-magnet centre x)")
    ax.set_ylabel("y = h  (test-magnet centre height)")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}")


# ----------------------------------------------------------------------
# Figure 3: y-component on a symmetric-log scale
# ----------------------------------------------------------------------
def plot_force_y_projection_symlog(outfile):
    n = 400
    xs = np.linspace(X_MIN, X_MAX, n)
    ys = np.linspace(Y_MIN, Y_MAX, n)
    Xg, Yg = np.meshgrid(xs, ys)

    _, Fy = force_on_test(Xg, Yg)

    linthresh = 1e-3
    vmax = 1e2
    Fy_capped = np.clip(Fy, -vmax, vmax)

    fig, ax = plt.subplots(figsize=(8, 8))

    norm = SymLogNorm(linthresh=linthresh, linscale=1.0,
                      vmin=-vmax, vmax=vmax, base=10)
    pcm = ax.pcolormesh(Xg, Yg, Fy_capped, cmap="RdBu_r", norm=norm,
                        shading="auto")
    cb = fig.colorbar(pcm, ax=ax, shrink=0.85)
    cb.set_label("F_y on test magnet (symlog: red = up, blue = down)")

    ax.contour(Xg, Yg, Fy, levels=[0.0], colors="black",
               linewidths=1.2, linestyles="--")

    draw_source(ax)

    ax.set_title("Vertical (y) force on the test magnet — symmetric-log scale\n"
                 "(reveals small forces far from the source)")
    ax.set_xlabel("x  (test-magnet centre x)")
    ax.set_ylabel("y = h  (test-magnet centre height)")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}")


def images_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, "..", "images")
    os.makedirs(d, exist_ok=True)
    return d


if __name__ == "__main__":
    out = images_dir()
    plot_force_field(os.path.join(out, "dipole_dipole_force_field.png"))
    plot_force_y_projection(os.path.join(out, "dipole_dipole_force_Fy.png"))
    plot_force_y_projection_symlog(
        os.path.join(out, "dipole_dipole_force_Fy_symlog.png"))
