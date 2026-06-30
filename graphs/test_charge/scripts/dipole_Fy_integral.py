"""
Cumulative horizontal integral of the vertical force, I(s) = int_0^s F_y dx,
for the point-dipole magnet (Model 1).  Flat magnet is intentionally ignored.

Setup (same dipole as magnetic_dipole_force_field.py)
-----------------------------------------------------
    +q at (0, 0),  -q at (0, -thickness),  test charge +q_test at (x, h).

The vertical force on the test charge along the horizontal line y = h is

    F_y(x, h) = h / (x^2 + h^2)^(3/2)
              - (h + thickness) / (x^2 + (h + thickness)^2)^(3/2)

(unit charges/constants).  We integrate this over x from 0 to s:

    I(s) = int_0^s F_y(x, h) dx.

This integral has a closed form (used here only to *verify* the numerics):

    int_0^s  a / (x^2 + a^2)^(3/2) dx = s / (a * sqrt(s^2 + a^2)),

so

    I(s) = s / (h * sqrt(s^2 + h^2))
         - s / ((h + thickness) * sqrt(s^2 + (h + thickness)^2)).

Outputs (written to ../images/)
-------------------------------
1. dipole_Fy_integral.png            I(s) vs s (with the integrand F_y for context)
2. dipole_Fy_integration_region.png  the line/region over which we integrate
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import os

# ----------------------------------------------------------------------
# Parameters (pick "some height and thickness")
# ----------------------------------------------------------------------
thickness = 0.2    # vertical gap between +q and -q
height = 0.3       # h: height of the integration line above +q
S_MAX = 3.0        # integrate / plot for s in [0, S_MAX]

P_pos = np.array([0.0, 0.0])
P_neg = np.array([0.0, -thickness])


def Fy(x, h):
    """Vertical force component on the test charge at (x, h)."""
    a1 = h
    a2 = h + thickness
    return (a1 / np.power(x * x + a1 * a1, 1.5)
            - a2 / np.power(x * x + a2 * a2, 1.5))


def Fy_grid(x, y):
    """Full 2-D F_y(x, y) for the background heatmap of figure 2."""
    dy1 = y - P_pos[1]
    dy2 = y - P_neg[1]
    r1 = np.power(x * x + dy1 * dy1, 1.5)
    r2 = np.power(x * x + dy2 * dy2, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (+1.0) * dy1 / r1 + (-1.0) * dy2 / r2


def I_analytic(s, h):
    a1 = h
    a2 = h + thickness
    return (s / (a1 * np.sqrt(s * s + a1 * a1))
            - s / (a2 * np.sqrt(s * s + a2 * a2)))


# ----------------------------------------------------------------------
# Figure 1: I(s) = int_0^s F_y dx
# ----------------------------------------------------------------------
def plot_integral(outfile):
    n = 4000
    xs = np.linspace(0.0, S_MAX, n)
    f = Fy(xs, height)

    # Cumulative trapezoidal integral, I[0] = 0.
    dx = xs[1] - xs[0]
    I_num = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dx)))
    I_exact = I_analytic(xs, height)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Integrand F_y(x, h) on a secondary axis for context.
    ax2 = ax1.twinx()
    ax2.plot(xs, f, color="0.6", lw=1.5, label="integrand  $F_y(x, h)$")
    ax2.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax2.set_ylabel("$F_y(x, h)$  (integrand, grey)", color="0.4")
    ax2.tick_params(axis="y", labelcolor="0.4")

    # The cumulative integral itself.
    ax1.plot(xs, I_num, color="C0", lw=2.5,
             label=r"$I(s)=\int_0^s F_y\,dx$  (numerical)")
    ax1.plot(xs, I_exact, color="C3", lw=1.2, ls="--",
             label="closed-form check")

    # Asymptote: I(s) -> 1/h - 1/(h+thickness) as s -> infinity, since
    # s / (a*sqrt(s^2+a^2)) -> 1/a.
    I_inf = 1.0 / height - 1.0 / (height + thickness)
    ax1.axhline(I_inf, color="C2", lw=1.4, ls="-.",
                label=fr"asymptote $1/h-1/(h+t)$ = {I_inf:.3f}")

    # Mark where the integrand changes sign (where I(s) peaks).
    sign_change = np.where(np.diff(np.sign(f)) != 0)[0]
    if len(sign_change):
        s_star = xs[sign_change[0]]
        ax1.axvline(s_star, color="k", lw=0.8, ls=":")
        ax1.annotate(f"$F_y=0$ at s≈{s_star:.2f}\n(I(s) is maximal here)",
                     xy=(s_star, np.interp(s_star, xs, I_num)),
                     xytext=(s_star + 0.3, np.max(I_num) * 0.6),
                     arrowprops=dict(arrowstyle="->", color="k"),
                     fontsize=9)

    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.set_xlabel("s  (upper limit of the x-integral)")
    ax1.set_ylabel(r"$I(s)=\int_0^s F_y\,dx$  (blue)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_title(f"Cumulative horizontal integral of $F_y$  "
                  f"(dipole, h={height}, thickness={thickness})")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}")


# ----------------------------------------------------------------------
# Figure 2: the line/region we integrate over
# ----------------------------------------------------------------------
def plot_integration_region(outfile):
    x_min, x_max = -0.4, S_MAX + 0.5
    y_min, y_max = -thickness - 0.5, height + 0.8

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
    cb.set_label("$F_y$ (symlog: red = up, blue = down)")

    # Sample F_y along the integration line and draw it as a coloured strip.
    xs_line = np.linspace(0.0, S_MAX, 400)
    f_line = Fy(xs_line, height)
    ax.scatter(xs_line, np.full_like(xs_line, height), c=f_line,
               cmap="RdBu_r", norm=norm, s=40, marker="s",
               edgecolors="none", zorder=4)

    # The integration line itself.
    ax.plot([0.0, S_MAX], [height, height], color="k", lw=2.0, zorder=5)
    ax.scatter([0.0, S_MAX], [height, height], color="k", zorder=6)
    ax.annotate("x = 0", (0.0, height), textcoords="offset points",
                xytext=(-6, 10), ha="right", fontsize=10, fontweight="bold")
    ax.annotate("x = s", (S_MAX, height), textcoords="offset points",
                xytext=(6, 10), fontsize=10, fontweight="bold")
    ax.annotate(f"integrate $F_y$ along y = h = {height}\n"
                f"from x = 0 to x = s",
                xy=(S_MAX / 2, height), textcoords="offset points",
                xytext=(0, 16), ha="center", fontsize=10)

    # The two source charges.
    ax.scatter(*P_pos, s=200, c="red", edgecolors="black", zorder=7)
    ax.scatter(*P_neg, s=200, c="blue", edgecolors="black", zorder=7)
    ax.annotate("+q", P_pos, textcoords="offset points", xytext=(8, -2),
                color="red", fontweight="bold")
    ax.annotate("-q", P_neg, textcoords="offset points", xytext=(8, -12),
                color="blue", fontweight="bold")

    ax.set_title("Integration line for $I(s)=\\int_0^s F_y\\,dx$\n"
                 "(horizontal line at height h, background = $F_y$ field)")
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
    plot_integral(os.path.join(out, "dipole_Fy_integral.png"))
    plot_integration_region(os.path.join(out, "dipole_Fy_integration_region.png"))
