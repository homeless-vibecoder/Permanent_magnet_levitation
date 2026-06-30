"""
Force on a test magnetic charge near a magnetic dipole (two-charge model).

Physical model (all constants set to unity)
-------------------------------------------
We model a small permanent magnet as a pair of "magnetic charges" (a
magnetic dipole made of two monopoles):

    +q  located at the origin                   P+ = (0,  0)
    -q  located a distance `thickness` below it  P- = (0, -thickness)

`thickness` is the vertical gap between the + and - faces of the magnet
(it used to be called `l`; `l`/length is now reserved for the *flat*
magnet model in flat_magnet_force_field.py).

A positive test magnetic charge sits at an arbitrary position

    Pt = (x, h)

The interaction is taken to be Coulomb-like (inverse-square). With unit
charges and unit Coulomb constant, the force on the test charge produced
by a single source charge q_s at position P_s is

    F = q_t * q_s * (Pt - P_s) / |Pt - P_s|^3            (vector form)

The total force on the test charge is the superposition of the force from
the +q charge (repulsive, since the test charge is +) and the -q charge
(attractive):

    F_total(x, h) = q_t*(+q)*(Pt - P+) / |Pt - P+|^3
                  + q_t*(-q)*(Pt - P-) / |Pt - P-|^3

Here q_t = +1, q = +1, thickness = 0.2.

Outputs (written to ../images/)
-------------------------------
1. magnetic_dipole_force_field.png   setup + colour-coded force vector field
2. magnetic_dipole_force_Fy.png      y-component (linear, clipped colour scale)
3. magnetic_dipole_force_Fy_symlog.png  y-component on a symmetric-log scale
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, SymLogNorm
import os

# ----------------------------------------------------------------------
# Model parameters (all unit constants)
# ----------------------------------------------------------------------
thickness = 0.2  # vertical gap between the + and - charges (was `l`)
q_pos = +1.0     # charge at the origin
q_neg = -1.0     # charge at (0, -thickness)
q_test = +1.0    # test charge (positive)

P_pos = np.array([0.0, 0.0])            # location of +q
P_neg = np.array([0.0, -thickness])     # location of -q

# Plot domain for the test-charge position (x, y).  y plays the role of h.
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = -thickness, 0.6


def force_on_test(x, y):
    """Return (Fx, Fy) acting on a +q_test charge located at (x, y).

    Implements F = sum_s q_t * q_s * (Pt - P_s) / |Pt - P_s|^3.
    Works on scalar or numpy-array inputs.
    """
    Fx = np.zeros_like(x, dtype=float)
    Fy = np.zeros_like(y, dtype=float)
    for q_s, P_s in ((q_pos, P_pos), (q_neg, P_neg)):
        dx = x - P_s[0]
        dy = y - P_s[1]
        r2 = dx * dx + dy * dy
        r3 = np.power(r2, 1.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            coeff = q_test * q_s / r3
        Fx = Fx + coeff * dx
        Fy = Fy + coeff * dy
    return Fx, Fy


def draw_sources(ax):
    """Draw the two source charges and annotate them."""
    ax.scatter(*P_pos, s=220, c="red", edgecolors="black",
               zorder=5, marker="o")
    ax.scatter(*P_neg, s=220, c="blue", edgecolors="black",
               zorder=5, marker="o")
    ax.annotate("+q\n(0, 0)", P_pos, textcoords="offset points",
                xytext=(10, 8), fontsize=11, fontweight="bold", color="red")
    ax.annotate("-q\n(0, -t)", P_neg, textcoords="offset points",
                xytext=(10, -22), fontsize=11, fontweight="bold", color="blue")


# ----------------------------------------------------------------------
# Figure 1: setup + colour-coded force vector field
# ----------------------------------------------------------------------
def plot_force_field(outfile):
    n = 26
    xs = np.linspace(X_MIN, X_MAX, n)
    ys = np.linspace(Y_MIN, Y_MAX, n)
    Xg, Yg = np.meshgrid(xs, ys)

    Fx, Fy = force_on_test(Xg, Yg)
    mag = np.hypot(Fx, Fy)

    too_close = (
        (np.hypot(Xg - P_pos[0], Yg - P_pos[1]) < 0.25)
        | (np.hypot(Xg - P_neg[0], Yg - P_neg[1]) < 0.25)
    )
    mag_masked = np.where(too_close, np.nan, mag)

    with np.errstate(divide="ignore", invalid="ignore"):
        U = np.where(mag > 0, Fx / mag, 0.0)
        V = np.where(mag > 0, Fy / mag, 0.0)
    U = np.where(too_close, np.nan, U)
    V = np.where(too_close, np.nan, V)

    fig, ax = plt.subplots(figsize=(8, 8))

    vmin = np.nanmin(mag_masked[mag_masked > 0])
    vmax = np.nanmax(mag_masked)
    q = ax.quiver(Xg, Yg, U, V, mag_masked,
                  cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax),
                  pivot="mid", scale=32, width=0.004)
    cb = fig.colorbar(q, ax=ax, shrink=0.85)
    cb.set_label("|F| on test charge (arb. units, log scale)")

    draw_sources(ax)

    ax.set_title("Force on a +test charge near a magnetic dipole\n"
                 "(arrows = direction, colour = magnitude)")
    ax.set_xlabel("x  (test-charge x position)")
    ax.set_ylabel("y = h  (test-charge height)")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}")


# ----------------------------------------------------------------------
# Figure 2: y-component of the force (projection onto the y-axis)
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
    cb.set_label("F_y on test charge (red = up / +y, blue = down)")

    ax.contour(Xg, Yg, Fy, levels=[0.0], colors="black",
               linewidths=1.2, linestyles="--")

    draw_sources(ax)

    ax.set_title("Vertical (y) component of the force on the test charge")
    ax.set_xlabel("x  (test-charge x position)")
    ax.set_ylabel("y = h  (test-charge height)")
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
    """Same F_y as figure 2, but on a symmetric-log (symlog) colour scale."""
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
    cb.set_label("F_y on test charge (symlog: red = up / +y, blue = down)")

    ax.contour(Xg, Yg, Fy, levels=[0.0], colors="black",
               linewidths=1.2, linestyles="--")

    draw_sources(ax)

    ax.set_title("Vertical (y) force on the test charge — symmetric-log scale\n"
                 "(reveals small forces far from the dipole)")
    ax.set_xlabel("x  (test-charge x position)")
    ax.set_ylabel("y = h  (test-charge height)")
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
    plot_force_field(os.path.join(out, "magnetic_dipole_force_field.png"))
    plot_force_y_projection(os.path.join(out, "magnetic_dipole_force_Fy.png"))
    plot_force_y_projection_symlog(
        os.path.join(out, "magnetic_dipole_force_Fy_symlog.png"))
