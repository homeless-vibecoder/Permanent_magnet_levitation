"""
Cumulative horizontal integral of the vertical force on a test MAGNET
(dipole), I(s) = int_0^s F_y dx, near a source magnet.

Same geometry as dipole_dipole_force_field.py (test magnet FLIPPED):
    source magnet: +q at (0, 0),                -q at (0, -thickness)
    test magnet  : -q at (x, h + thickness/2),  +q at (x, h - thickness/2)
i.e. the source moment points +y and the test moment points -y (+ charge on
the bottom), so above the source the like poles REPEL.

Along the horizontal line y = h (h = centre height of the test magnet) the
vertical net force is a sum of four monopole-pair terms,

    F_y(x, h) = sum_{pairs}  (q_t q_s) * dy / (x^2 + dy^2)^(3/2),

with dy = (test charge y) - (source charge y), constant along the line.
Each term integrates in closed form (used only to verify the numerics):

    int_0^s  dy / (x^2 + dy^2)^(3/2) dx = s / (dy * sqrt(s^2 + dy^2)),

so

    I(s)      = sum_{pairs} (q_t q_s) * s / (dy * sqrt(s^2 + dy^2)),
    I(infty)  = sum_{pairs} (q_t q_s) / dy.

A positive I means the accumulated vertical force is upward, i.e. the test
magnet is repelled from the source (the flipped/levitation configuration).

Outputs (written to ../images/)
-------------------------------
1. dipole_dipole_Fy_integral.png            I(s) vs s (+ integrand, asymptote)
2. dipole_dipole_Fy_integration_region.png  the line/region we integrate over
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import os

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
thickness = 0.2
q = 1.0
height = 0.5       # h: centre height of the test magnet
S_MAX = 3.0

SOURCE = [(+q, 0.0, 0.0), (-q, 0.0, -thickness)]
# Test magnet FLIPPED: - on top, + on bottom  -> (dy, charge)
TEST_OFFSETS = [(+thickness / 2, -q), (-thickness / 2, +q)]

# Pre-compute the (sign, dy) pairs along the line y = h.
_PAIRS = []
for oy, qt in TEST_OFFSETS:
    ty = height + oy
    for qs, sx, sy in SOURCE:
        _PAIRS.append((qt * qs, ty - sy))


def Fy_line(x):
    """F_y along the line y = h, as a function of x."""
    out = np.zeros_like(x, dtype=float)
    for sign, dy in _PAIRS:
        out = out + sign * dy / np.power(x * x + dy * dy, 1.5)
    return out


def I_analytic(s):
    out = np.zeros_like(s, dtype=float)
    for sign, dy in _PAIRS:
        out = out + sign * s / (dy * np.sqrt(s * s + dy * dy))
    return out


def I_infinity():
    return sum(sign / dy for sign, dy in _PAIRS)


def Fy_grid(x, y):
    """Full 2-D net F_y(x, y) on the test magnet centred at (x, y)."""
    out = np.zeros_like(x, dtype=float)
    for oy, qt in TEST_OFFSETS:
        ty = y + oy
        for qs, sx, sy in SOURCE:
            dx = x - sx
            dyv = ty - sy
            with np.errstate(divide="ignore", invalid="ignore"):
                out = out + qt * qs * dyv / np.power(dx * dx + dyv * dyv, 1.5)
    return out


# ----------------------------------------------------------------------
# Figure 1: I(s)
# ----------------------------------------------------------------------
def plot_integral(outfile):
    n = 4000
    xs = np.linspace(0.0, S_MAX, n)
    f = Fy_line(xs)

    dx = xs[1] - xs[0]
    I_num = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dx)))
    I_exact = I_analytic(xs)
    I_inf = I_infinity()

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax2 = ax1.twinx()
    ax2.plot(xs, f, color="0.6", lw=1.5, label="integrand  $F_y(x, h)$")
    ax2.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax2.set_ylabel("$F_y(x, h)$  (integrand, grey)", color="0.4")
    ax2.tick_params(axis="y", labelcolor="0.4")

    ax1.plot(xs, I_num, color="C0", lw=2.5,
             label=r"$I(s)=\int_0^s F_y\,dx$  (numerical)")
    ax1.plot(xs, I_exact, color="C3", lw=1.2, ls="--",
             label="closed-form check")
    ax1.axhline(I_inf, color="C2", lw=1.4, ls="-.",
                label=fr"asymptote $I(\infty)$ = {I_inf:.3f}")

    sign_change = np.where(np.diff(np.sign(f)) != 0)[0]
    for idx in sign_change[:2]:
        s_star = xs[idx]
        ax1.axvline(s_star, color="k", lw=0.8, ls=":")

    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.set_xlabel("s  (upper limit of the x-integral)")
    ax1.set_ylabel(r"$I(s)=\int_0^s F_y\,dx$  (blue)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_title("Cumulative horizontal integral of $F_y$ on a test MAGNET\n"
                  f"(flipped dipole-dipole, h={height}, thickness={thickness}; "
                  "I>0 ⇒ net repulsion)")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}")


# ----------------------------------------------------------------------
# Figure 2: integration line/region
# ----------------------------------------------------------------------
def plot_integration_region(outfile):
    x_min, x_max = -0.4, S_MAX + 0.5
    y_min, y_max = -thickness - 0.7, height + 0.9

    n = 500
    xs = np.linspace(x_min, x_max, n)
    ys = np.linspace(y_min, y_max, n)
    Xg, Yg = np.meshgrid(xs, ys)
    F = Fy_grid(Xg, Yg)

    fig, ax = plt.subplots(figsize=(9, 6))

    vmax = 1e2
    norm = SymLogNorm(linthresh=1e-3, linscale=1.0, vmin=-vmax, vmax=vmax,
                      base=10)
    pcm = ax.pcolormesh(Xg, Yg, np.clip(F, -vmax, vmax), cmap="RdBu_r",
                        norm=norm, shading="auto")
    cb = fig.colorbar(pcm, ax=ax, shrink=0.85)
    cb.set_label("$F_y$ on test magnet (symlog: red = up, blue = down)")

    xs_line = np.linspace(0.0, S_MAX, 400)
    f_line = Fy_line(xs_line)
    ax.scatter(xs_line, np.full_like(xs_line, height), c=f_line,
               cmap="RdBu_r", norm=norm, s=40, marker="s",
               edgecolors="none", zorder=4)

    ax.plot([0.0, S_MAX], [height, height], color="k", lw=2.0, zorder=5)
    ax.scatter([0.0, S_MAX], [height, height], color="k", zorder=6)
    ax.annotate("x = 0", (0.0, height), textcoords="offset points",
                xytext=(-6, 10), ha="right", fontsize=10, fontweight="bold")
    ax.annotate("x = s", (S_MAX, height), textcoords="offset points",
                xytext=(6, 10), fontsize=10, fontweight="bold")
    ax.annotate(f"integrate $F_y$ along y = h = {height}\nfrom x = 0 to x = s",
                xy=(S_MAX / 2 - 0.4, height), textcoords="offset points",
                xytext=(0, 18), ha="center", fontsize=10)

    # Source magnet.
    ax.scatter(0.0, 0.0, s=180, c="red", edgecolors="black", zorder=7)
    ax.scatter(0.0, -thickness, s=180, c="blue", edgecolors="black", zorder=7)
    ax.annotate("+q", (0.0, 0.0), textcoords="offset points", xytext=(8, -2),
                color="red", fontweight="bold")
    ax.annotate("-q", (0.0, -thickness), textcoords="offset points",
                xytext=(8, -12), color="blue", fontweight="bold")

    # Schematic test magnet on the line (FLIPPED: - on top, + on bottom).
    cx = 2.5
    ax.scatter(cx, height + thickness / 2, s=70, c="blue", edgecolors="black",
               zorder=8)
    ax.scatter(cx, height - thickness / 2, s=70, c="red", edgecolors="black",
               zorder=8)
    ax.annotate("test magnet\n(moves along the line)",
                (cx, height - thickness / 2), textcoords="offset points",
                xytext=(6, -26), fontsize=8)

    ax.set_title("Integration line for $I(s)=\\int_0^s F_y\\,dx$ (test magnet)\n"
                 "(horizontal line at centre height h; background = $F_y$)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
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
    plot_integral(os.path.join(out, "dipole_dipole_Fy_integral.png"))
    plot_integration_region(
        os.path.join(out, "dipole_dipole_Fy_integration_region.png"))
