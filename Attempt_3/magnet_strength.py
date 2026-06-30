"""Magnetic-charge force field via FFT convolution (Attempt_3).

Model
-----
Ground magnets (all north up, fixed) are laid on the plane z in [-t_ground, 0],
so each ground magnet has a +charge face at z = 0 and a -charge face at
z = -t_ground. A test magnet hovers north-side-down, occupying z in
[h, h + t_test], so it has a +charge (north) face at z = h and a -charge
(south) face at z = h + t_test.

Each face is a uniform sheet of magnetic charge (density 1 per unit area, signed
by the pole). The force between two +1 point charges is the exact Coulomb-like
law (no far-field / small-quantity approximation):

    F = C * q1 * q2 * r_vec / |r_vec|^3 ,   C = strength_constant

We want the net 3D force on the test magnet as a function of its planar position.
Because the force is linear in the charges, this is a convolution of the ground
charge distribution G, the test footprint T, and the single-charge force kernel
K (evaluated per pole-pair separation dz). The test footprint is centered and
symmetric, so cross-correlation equals convolution and the whole thing is a
triple convolution  F = K * G * T  computed with FFTs.

Four pole pairs are summed (sign = product of the two pole charge signs):

    gm_n . um_n : dz = h,                       sign = +1
    gm_n . um_s : dz = h + t_test,              sign = -1
    gm_s . um_n : dz = h + t_ground,            sign = -1
    gm_s . um_s : dz = h + t_ground + t_test,   sign = +1

Run ``python magnet_strength.py`` to validate against a brute-force direct sum
and then render the force field for the configured scenario.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from magnet_classes import RectangularMagnet


# ============================================================================
# CONFIG  -- edit these to describe your scenario
# ============================================================================

# If True, cross-check the FFT result against the slow O(N^2) brute-force sum
# before computing the full field. Set False to skip it and run fast.
run_validation = False

strength_constant = 1.0  # C: converts charge product / distance^2 into force

# Magnet geometry (mm). x = lengthwise, y = widthwise, z = thickness.
ground_magnet = RectangularMagnet(length=10.0, width=5.0, thickness=6.0)
test_magnet = RectangularMagnet(length=10.0, width=5.0, thickness=240.0)

# Hover gap (mm): distance between the ground top (+) face at z=0 and the test
# magnet's north (bottom, +) face at z=h. Thicknesses are independent.
hover_height = 2.0

# Placement of the ground magnets (centers). A regular rectangular lattice here,
# but ``ground_centers`` can be any list of (x, y) points in mm.
x_spacing = 24.0
y_spacing = 19.0
x_num_magnets = 15
y_num_magnets = 15
ground_centers = [
    (i * x_spacing, j * y_spacing)
    for i in range(x_num_magnets)
    for j in range(y_num_magnets)
]

# Grid resolution (mm) and empty margin around the magnet array (mm).
dx = 0.5
dy = 0.5
margin = 12.0

# Output image
output_png = "force_field.png"


# ============================================================================
# Core physics / numerics
# ============================================================================


def pole_pairs(h: float, t_ground: float, t_test: float):
    """Return the four (name, dz, sign) pole-pair interactions."""
    return [
        ("gm_n.um_n", h, +1.0),
        ("gm_n.um_s", h + t_test, -1.0),
        ("gm_s.um_n", h + t_ground, -1.0),
        ("gm_s.um_s", h + t_ground + t_test, +1.0),
    ]


def force_kernel(dz: float, rxs: np.ndarray, rys: np.ndarray, C: float):
    """Exact single-charge force field for a +1 source at the origin.

    Returns (Kx, Ky, Kz), each a ``(len(rys), len(rxs))`` array giving the force
    on a +1 charge located at (x, y, dz) due to a +1 charge at (0, 0, 0):

        K = C * (x, y, dz) / (x^2 + y^2 + dz^2)^(3/2)

    This is your "step 1" force field. It is exact at any distance -- no dipole
    expansion or small-quantity assumption. Since the four pole pairs all use
    dz > 0, there is no singularity.
    """
    X, Y = np.meshgrid(rxs, rys)
    r2 = X * X + Y * Y + dz * dz
    inv_r3 = r2 ** (-1.5)
    Kx = C * X * inv_r3
    Ky = C * Y * inv_r3
    Kz = C * dz * inv_r3
    return Kx, Ky, Kz


def build_grid(centers, gm: RectangularMagnet, dx: float, dy: float, margin: float):
    """Build the simulation grid.

    Returns:
        xs, ys    : absolute domain coordinates (mm) along x and y
        rxs, rys  : origin-centered coordinates (mm) for the kernel / footprints
    The grid is forced to odd size on each axis so the center cell is unambiguous,
    which keeps FFT 'same'-mode convolutions aligned with the brute-force sum.
    """
    cx = np.array([c[0] for c in centers], dtype=float)
    cy = np.array([c[1] for c in centers], dtype=float)

    x_min = cx.min() - gm.length / 2.0 - margin
    x_max = cx.max() + gm.length / 2.0 + margin
    y_min = cy.min() - gm.width / 2.0 - margin
    y_max = cy.max() + gm.width / 2.0 + margin

    nx = int(np.ceil((x_max - x_min) / dx)) + 1
    ny = int(np.ceil((y_max - y_min) / dy)) + 1
    if nx % 2 == 0:
        nx += 1
    if ny % 2 == 0:
        ny += 1

    xs = x_min + np.arange(nx) * dx
    ys = y_min + np.arange(ny) * dy

    cxi = nx // 2
    cyi = ny // 2
    rxs = (np.arange(nx) - cxi) * dx
    rys = (np.arange(ny) - cyi) * dy
    return xs, ys, rxs, rys


def build_ground_charges(centers, gm: RectangularMagnet, xs, ys, rxs, rys):
    """Ground charge footprint G = single-magnet rectangle * placement pattern.

    A delta pattern P is stamped at the grid cell nearest each magnet center,
    then convolved with the centered single-magnet footprint, exactly as the
    plan specifies. The result is a 0/1 density array over the domain.
    """
    pattern = np.zeros((len(ys), len(xs)))
    for (mx, my) in centers:
        ix = int(round((mx - xs[0]) / (xs[1] - xs[0])))
        iy = int(round((my - ys[0]) / (ys[1] - ys[0])))
        ix = min(max(ix, 0), len(xs) - 1)
        iy = min(max(iy, 0), len(ys) - 1)
        pattern[iy, ix] += 1.0

    single = gm.footprint(rxs, rys)
    G = fftconvolve(pattern, single, mode="same")
    # Clean up tiny FFT round-off so G is a crisp 0/1 mask.
    G[np.abs(G) < 1e-9] = 0.0
    return G


def compute_force_field(G, T, rxs, rys, dx, dy, pairs, C):
    """FFT force field. Returns (Fx, Fy, Fz) over the domain grid.

    F = sum_pairs sign * (dx*dy)^2 * [ K_dz * (G * T) ]

    The two convolutions (over source charges and over test charges) each carry
    a cell-area weight, hence (dx*dy)^2; this makes the result independent of the
    grid resolution (it converges to the continuous surface integral).
    """
    combined = fftconvolve(G, T, mode="same")  # T is centered/even => corr == conv
    dA2 = (dx * dy) ** 2

    # The kernel must cover the full range of source-to-output displacements, which
    # spans +/-(N-1) cells on each axis, so it is built on a (2N-1) grid. A smaller
    # kernel would truncate the long-range 1/r^2 tails and bias the result.
    ny, nx = combined.shape
    kxs = (np.arange(2 * nx - 1) - (nx - 1)) * dx
    kys = (np.arange(2 * ny - 1) - (ny - 1)) * dy

    Fx = np.zeros_like(combined)
    Fy = np.zeros_like(combined)
    Fz = np.zeros_like(combined)
    for _name, dz, sign in pairs:
        Kx, Ky, Kz = force_kernel(dz, kxs, kys, C)
        Fx += sign * fftconvolve(combined, Kx, mode="same")
        Fy += sign * fftconvolve(combined, Ky, mode="same")
        Fz += sign * fftconvolve(combined, Kz, mode="same")

    return Fx * dA2, Fy * dA2, Fz * dA2


def brute_force_field(G, T, xs, ys, rxs, rys, dx, dy, pairs, C):
    """Reference O(N^2) direct summation over every charge pair.

    Used only on small grids to verify the FFT result (signs, origin, scaling).
    Sums the exact Coulomb kernel over all ground charges (from G) and all test
    charges (from T), for every test-magnet center position on the domain grid.
    """
    dA = dx * dy
    Xout, Yout = np.meshgrid(xs, ys)

    s_idx = np.argwhere(G > 0)
    s_x = xs[s_idx[:, 1]]
    s_y = ys[s_idx[:, 0]]
    q_s = G[s_idx[:, 0], s_idx[:, 1]] * dA

    t_idx = np.argwhere(T > 0)
    t_x = rxs[t_idx[:, 1]]
    t_y = rys[t_idx[:, 0]]
    q_t = T[t_idx[:, 0], t_idx[:, 1]] * dA

    Fx = np.zeros_like(Xout)
    Fy = np.zeros_like(Xout)
    Fz = np.zeros_like(Xout)
    for si in range(len(q_s)):
        for ti in range(len(q_t)):
            ddx = Xout + t_x[ti] - s_x[si]
            ddy = Yout + t_y[ti] - s_y[si]
            qq = C * q_s[si] * q_t[ti]
            base = ddx * ddx + ddy * ddy
            for _name, dz, sign in pairs:
                inv_r3 = (base + dz * dz) ** (-1.5)
                w = sign * qq * inv_r3
                Fx += w * ddx
                Fy += w * ddy
                Fz += w * dz
    return Fx, Fy, Fz


# ============================================================================
# Validation
# ============================================================================


def validate(verbose: bool = True) -> float:
    """Compare the FFT pipeline to brute force on a small scenario.

    Returns the max relative error and raises AssertionError if it is too large.
    """
    gm = RectangularMagnet(length=4.0, width=3.0, thickness=2.0)
    tm = RectangularMagnet(length=3.0, width=2.0, thickness=1.5)
    centers = [(0.0, 0.0), (7.0, 2.0)]
    h = 3.0
    ddx = ddy = 0.5
    pairs = pole_pairs(h, gm.thickness, tm.thickness)

    xs, ys, rxs, rys = build_grid(centers, gm, ddx, ddy, margin=4.0)
    G = build_ground_charges(centers, gm, xs, ys, rxs, rys)
    T = tm.footprint(rxs, rys)

    Fx, Fy, Fz = compute_force_field(G, T, rxs, rys, ddx, ddy, pairs, strength_constant)
    Bx, By, Bz = brute_force_field(G, T, xs, ys, rxs, rys, ddx, ddy, pairs, strength_constant)

    num = np.sqrt((Fx - Bx) ** 2 + (Fy - By) ** 2 + (Fz - Bz) ** 2)
    scale = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2).max()
    rel = num.max() / scale

    if verbose:
        print("Validation (FFT vs brute force):")
        print(f"  grid              : {len(xs)} x {len(ys)}")
        print(f"  ground charges    : {int((G > 0).sum())} cells")
        print(f"  test charges      : {int((T > 0).sum())} cells")
        print(f"  peak |F| (brute)  : {scale:.6e}")
        print(f"  max abs error     : {num.max():.6e}")
        print(f"  max rel error     : {rel:.3e}")
    assert rel < 1e-9, f"FFT result disagrees with brute force (rel err {rel:.3e})"
    return rel


# ============================================================================
# Visualization
# ============================================================================


def plot_force_field(xs, ys, Fx, Fy, Fz, centers, gm, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9, 7))

    vmax = np.abs(Fz).max()
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    im = ax.imshow(
        Fz,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("F_z  (vertical force; + = repulsion / lift)")

    # Lateral force (Fx, Fy) as a quiver overlay, subsampled for legibility.
    step_x = max(1, len(xs) // 30)
    step_y = max(1, len(ys) // 30)
    Xs, Ys = np.meshgrid(xs[::step_x], ys[::step_y])
    ax.quiver(
        Xs,
        Ys,
        Fx[::step_y, ::step_x],
        Fy[::step_y, ::step_x],
        color="k",
        alpha=0.6,
        pivot="mid",
        scale_units="xy",
    )

    # Outline the ground magnets.
    for (mx, my) in centers:
        ax.add_patch(
            Rectangle(
                (mx - gm.length / 2.0, my - gm.width / 2.0),
                gm.length,
                gm.width,
                fill=False,
                edgecolor="k",
                linewidth=0.8,
                linestyle="--",
                alpha=0.5,
            )
        )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Net force on hovering test magnet vs. position\n"
                 "color = F_z, arrows = lateral (F_x, F_y)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved figure to {path}")
    plt.show()


# ============================================================================
# Main
# ============================================================================


def main():
    if run_validation:
        validate()

    pairs = pole_pairs(hover_height, ground_magnet.thickness, test_magnet.thickness)
    xs, ys, rxs, rys = build_grid(ground_centers, ground_magnet, dx, dy, margin)
    G = build_ground_charges(ground_centers, ground_magnet, xs, ys, rxs, rys)
    T = test_magnet.footprint(rxs, rys)

    Fx, Fy, Fz = compute_force_field(G, T, rxs, rys, dx, dy, pairs, strength_constant)

    print("\nForce field computed:")
    print(f"  grid            : {len(xs)} x {len(ys)}")
    print(f"  ground magnets  : {len(ground_centers)}")
    print(f"  F_z range       : [{Fz.min():.4e}, {Fz.max():.4e}]")
    print(f"  peak lift (F_z) : {Fz.max():.4e}")

    plot_force_field(xs, ys, Fx, Fy, Fz, ground_centers, ground_magnet, output_png)


if __name__ == "__main__":
    main()
