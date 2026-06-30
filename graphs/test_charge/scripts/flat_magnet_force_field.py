"""
Force on a test magnetic charge near a FLAT magnet (extended-charge model).

Physical model (all constants set to unity)
-------------------------------------------
Instead of a single +/- pair (a point dipole), we model a flat bar magnet
of horizontal *length* `length` and vertical *thickness* `thickness` by an
*interval* of magnetic charge on each face:

    top face   : +lambda spread along  y = 0,          x in [-length/2, +length/2]
    bottom face: -lambda spread along  y = -thickness,  x in [-length/2, +length/2]

So the previous point charges are replaced by two uniformly charged line
segments (the continuum limit of "an interval of magnets"). `thickness` is
the same vertical gap that used to be called `l` in the dipole model;
`length` is the new horizontal extent of the magnet.

Numerically each face is discretised into `N_SRC` point charges placed at
the segment midpoints, each carrying charge  lambda * ds  with
ds = length / N_SRC.  Summing them approximates the line integral

    F(Pt) = q_t * integral_face  sigma(s) * (Pt - s_hat) / |Pt - s_hat|^3 ds

where sigma = +lambda on the top face and -lambda on the bottom face.
With unit constants: q_t = +1, lambda = 1, length = 1.0, thickness = 0.2.

In the limit length -> 0 this reduces to the point-dipole model in
magnetic_dipole_force_field.py.

Outputs (written to ../images/)
-------------------------------
1. flat_magnet_force_field.png        setup + colour-coded force vector field
2. flat_magnet_force_Fy.png           y-component (linear, clipped colour scale)
3. flat_magnet_force_Fy_symlog.png    y-component on a symmetric-log scale
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, SymLogNorm
import os

# ----------------------------------------------------------------------
# Model parameters (all unit constants)
# ----------------------------------------------------------------------
length = 1.0       # horizontal length of the flat magnet (the new `l`)
thickness = 0.2    # vertical gap between the + and - faces (old `l`)
lam = 1.0          # linear magnetic-charge density on each face
q_test = +1.0      # positive test charge
N_SRC = 200        # number of point charges used to discretise each face

# Source point charges: (charge value, x, y) for both faces.
_ds = length / N_SRC
_s = -length / 2 + (np.arange(N_SRC) + 0.5) * _ds   # midpoints along x
_SRC_X = np.concatenate([_s, _s])
_SRC_Y = np.concatenate([np.zeros(N_SRC), np.full(N_SRC, -thickness)])
_SRC_Q = np.concatenate([np.full(N_SRC, +lam * _ds),
                         np.full(N_SRC, -lam * _ds)])

# Plot domain for the test-charge position (x, y).  y plays the role of h.
X_MIN, X_MAX = -2.0, 2.0
Y_MIN, Y_MAX = -1.5, 1.5


def force_on_test(x, y):
    """Return (Fx, Fy) on a +q_test charge at (x, y) from the flat magnet.

    Sums the contributions of all discretised face charges. Works on
    scalar or numpy-array inputs (loops over the source charges).
    """
    Fx = np.zeros_like(x, dtype=float)
    Fy = np.zeros_like(y, dtype=float)
    for q_s, sx, sy in zip(_SRC_Q, _SRC_X, _SRC_Y):
        dx = x - sx
        dy = y - sy
        r2 = dx * dx + dy * dy
        r3 = np.power(r2, 1.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            coeff = q_test * q_s / r3
        Fx = Fx + coeff * dx
        Fy = Fy + coeff * dy
    return Fx, Fy


def _near_magnet(x, y, margin):
    """Boolean mask: points within `margin` of either charged face."""
    in_x = np.abs(x) <= (length / 2 + margin)
    near_top = np.abs(y - 0.0) <= margin
    near_bot = np.abs(y + thickness) <= margin
    inside = (y <= margin) & (y >= -thickness - margin)
    return (in_x & (near_top | near_bot | inside))


def draw_magnet(ax):
    """Draw the two charged faces (red = +, blue = -) and the magnet body."""
    x0, x1 = -length / 2, length / 2
    # Magnet body
    ax.add_patch(plt.Rectangle((x0, -thickness), length, thickness,
                               facecolor="0.85", edgecolor="0.4",
                               zorder=3))
    # Charged faces
    ax.plot([x0, x1], [0.0, 0.0], color="red", lw=4, zorder=4)
    ax.plot([x0, x1], [-thickness, -thickness], color="blue", lw=4, zorder=4)
    ax.annotate("+ face (y = 0)", (x1, 0.0), textcoords="offset points",
                xytext=(8, 6), fontsize=10, fontweight="bold", color="red")
    ax.annotate("- face (y = -t)", (x1, -thickness),
                textcoords="offset points", xytext=(8, -16),
                fontsize=10, fontweight="bold", color="blue")


# ----------------------------------------------------------------------
# Figure 1: setup + colour-coded force vector field
# ----------------------------------------------------------------------
def plot_force_field(outfile):
    n = 31
    xs = np.linspace(X_MIN, X_MAX, n)
    ys = np.linspace(Y_MIN, Y_MAX, n)
    Xg, Yg = np.meshgrid(xs, ys)

    Fx, Fy = force_on_test(Xg, Yg)
    mag = np.hypot(Fx, Fy)

    too_close = _near_magnet(Xg, Yg, margin=0.12)
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
                  pivot="mid", scale=34, width=0.004)
    cb = fig.colorbar(q, ax=ax, shrink=0.85)
    cb.set_label("|F| on test charge (arb. units, log scale)")

    draw_magnet(ax)

    ax.set_title("Force on a +test charge near a flat magnet\n"
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

    draw_magnet(ax)

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

    draw_magnet(ax)

    ax.set_title("Vertical (y) force on the test charge — symmetric-log scale\n"
                 "(reveals small forces far from the flat magnet)")
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
    plot_force_field(os.path.join(out, "flat_magnet_force_field.png"))
    plot_force_y_projection(os.path.join(out, "flat_magnet_force_Fy.png"))
    plot_force_y_projection_symlog(
        os.path.join(out, "flat_magnet_force_Fy_symlog.png"))
