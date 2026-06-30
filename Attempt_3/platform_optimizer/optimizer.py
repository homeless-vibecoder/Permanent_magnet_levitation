"""Worst-case platform layout optimizer (CVaR on vertical force).

Idea
----
A levitating platform carries ``N`` magnets at planar offsets {r_i} from its
centroid. When the platform is dropped onto the ground array at a random
translation T and rotation theta, the net vertical force it feels is the sum of
the single-magnet force field sampled at each magnet's world position:

    Fz_sample = sum_i  Fz_single( R(theta) r_i + T )

where ``Fz_single`` is exactly the field computed by ``magnet_strength.py``
(superposition holds because the magnets are rigid and the model is linear).

We draw many random (T, theta) samples, look at the *worst* 15% of them by
vertical force (a CVaR / conditional-value-at-risk objective), and do gradient
ascent on the magnet offsets {r_i} to push that worst tail up. The single-magnet
field is precomputed on a grid once; we then sample it with bilinear
interpolation and use the analytic interpolation gradient (chained through the
rotation) to get exact gradients of the objective w.r.t. the magnet offsets.

A live matplotlib dashboard shows the layout, the learning curve, the sample
distribution, and the current worst placement, so you can watch the design
evolve and build intuition.

Run:  python optimizer.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Make the parent Attempt_3 modules importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from magnet_strength import (  # noqa: E402
    RectangularMagnet,
    pole_pairs,
    build_grid,
    build_ground_charges,
    compute_force_field,
)


# ============================================================================
# CONFIG
# ============================================================================

SEED = 0

# --- Ground array (fixed magnets, north up) ---------------------------------
GROUND_MAGNET = RectangularMagnet(length=10.0, width=10.0, thickness=2.0)
GROUND_SPACING = (16.0, 16.0)      # (x, y) lattice spacing (mm)
GROUND_COUNT = (12, 12)            # number of magnets (nx, ny)

# --- Unit magnet on the platform (north down). Square => rotation-agnostic. --
UNIT_MAGNET = RectangularMagnet(length=10.0, width=10.0, thickness=2.0)

HOVER_HEIGHT = 4.0                 # gap h (mm) between ground top and platform
STRENGTH_CONSTANT = 1.0

# Field grid resolution (mm).
DX = 0.6
DY = 0.6
FIELD_MARGIN = 10.0

# --- Platform ----------------------------------------------------------------
N_MAGNETS = 50
PLATFORM_SIZE = (120.0, 120.0)     # platform extent (mm); magnets clamped inside
INIT = "jittered_grid"             # "grid" | "random" | "jittered_grid"
INIT_JITTER = 0.25                 # jitter as fraction of grid cell (jittered_grid)

# --- Optimization ------------------------------------------------------------
N_SAMPLES = 512                    # random (T, theta) per iteration
WORST_FRACTION = 0.15              # optimize the lowest 15%
N_ITERS = 400
LEARNING_RATE = 0.5                # Adam step scale (mm)
ALLOW_ROTATION = True
ROT_RANGE = (0.0, 2.0 * np.pi)
KEEP_PLATFORM_INSIDE = True        # sample translations so platform stays on array

PLOT_EVERY = 1
OUT_DIR = _HERE


# ============================================================================
# Single-magnet field (precomputed once)
# ============================================================================


def build_single_field():
    gm = GROUND_MAGNET
    um = UNIT_MAGNET
    ax, ay = GROUND_SPACING
    nx, ny = GROUND_COUNT
    centers = [(i * ax, j * ay) for i in range(nx) for j in range(ny)]

    pairs = pole_pairs(HOVER_HEIGHT, gm.thickness, um.thickness)
    xs, ys, rxs, rys = build_grid(centers, gm, DX, DY, FIELD_MARGIN)
    G = build_ground_charges(centers, gm, xs, ys, rxs, rys)
    T = um.footprint(rxs, rys)
    Fx, Fy, Fz = compute_force_field(G, T, rxs, rys, DX, DY, pairs, STRENGTH_CONSTANT)

    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    ground_bbox = (
        cx.min() - gm.length / 2.0,
        cx.max() + gm.length / 2.0,
        cy.min() - gm.width / 2.0,
        cy.max() + gm.width / 2.0,
    )
    return xs, ys, Fx, Fy, Fz, centers, ground_bbox


# ============================================================================
# Differentiable platform response (bilinear field sampling + analytic grad)
# ============================================================================


class PlatformModel:
    """Precomputed Fz field; samples value and spatial gradient at any point."""

    def __init__(self, xs, ys, Fz):
        self.x0, self.y0 = float(xs[0]), float(ys[0])
        self.dx = float(xs[1] - xs[0])
        self.dy = float(ys[1] - ys[0])
        self.nx = Fz.shape[1]
        self.ny = Fz.shape[0]
        self.Z = Fz.astype(np.float64)

    def sample(self, X, Y):
        """Return (value, dFz/dx, dFz/dy) at world points X, Y (same shape).

        The value is the bilinear interpolant of the field, and the gradient is
        the exact analytic gradient of that same bilinear surface (so it is
        consistent with the value to machine precision within each cell).
        """
        fx = (X - self.x0) / self.dx
        fy = (Y - self.y0) / self.dy
        i0 = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)

        z00 = self.Z[j0, i0]
        z10 = self.Z[j0, i0 + 1]
        z01 = self.Z[j0 + 1, i0]
        z11 = self.Z[j0 + 1, i0 + 1]

        val = ((1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10
               + (1 - tx) * ty * z01 + tx * ty * z11)
        gx = ((1 - ty) * (z10 - z00) + ty * (z11 - z01)) / self.dx
        gy = ((1 - tx) * (z01 - z00) + tx * (z11 - z10)) / self.dy
        return val, gx, gy

    def evaluate(self, positions, T, theta):
        """Vertical force per sample and dFz_sample/d positions.

        positions : (N, 2)   platform-frame magnet offsets
        T         : (S, 2)   translations
        theta     : (S,)     rotations (rad)
        returns:
            fz    : (S,)     total vertical force per sample
            grad  : (S, N, 2) gradient of fz w.r.t. positions
            world : (S, N, 2) world coordinates of each magnet
        """
        c, s = np.cos(theta), np.sin(theta)
        R = np.stack([np.stack([c, -s], axis=-1),
                      np.stack([s, c], axis=-1)], axis=-2)  # (S, 2, 2)
        world = np.einsum("sij,nj->sni", R, positions) + T[:, None, :]  # (S, N, 2)

        val, gx, gy = self.sample(world[..., 0], world[..., 1])  # each (S, N)
        fz = val.sum(axis=1)  # (S,)

        g = np.stack([gx, gy], axis=-1)  # (S, N, 2) field gradient in world frame
        # chain rule: d fz / d r_i = R^T @ grad_world
        grad = np.einsum("sba,snb->sna", R, g)  # (S, N, 2)
        return fz, grad, world


# ============================================================================
# Initialization & sampling
# ============================================================================


def init_positions(rng):
    px, py = PLATFORM_SIZE
    if INIT == "random":
        return rng.uniform([-px / 2, -py / 2], [px / 2, py / 2],
                           size=(N_MAGNETS, 2)).astype(np.float64)

    n_side = int(round(np.sqrt(N_MAGNETS)))
    nx = max(n_side, 1)
    ny = int(np.ceil(N_MAGNETS / nx))
    gx = (np.arange(nx) - (nx - 1) / 2) / nx * px
    gy = (np.arange(ny) - (ny - 1) / 2) / ny * py
    pts = np.array([(x, y) for y in gy for x in gx], dtype=np.float64)[:N_MAGNETS]
    if INIT == "jittered_grid":
        cell = np.array([px / nx, py / ny])
        pts = pts + rng.uniform(-INIT_JITTER, INIT_JITTER, pts.shape) * cell
    return pts


def sample_placements(rng, ground_bbox):
    x0, x1, y0, y1 = ground_bbox
    if KEEP_PLATFORM_INSIDE:
        px, py = PLATFORM_SIZE
        reach = 0.5 * np.hypot(px, py) if ALLOW_ROTATION else 0.5 * max(px, py)
        x0, x1, y0, y1 = x0 + reach, x1 - reach, y0 + reach, y1 - reach
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                "Platform too large to stay inside the ground array; "
                "shrink PLATFORM_SIZE or grow GROUND_COUNT."
            )
    tx = rng.uniform(x0, x1, N_SAMPLES)
    ty = rng.uniform(y0, y1, N_SAMPLES)
    th = rng.uniform(*ROT_RANGE, N_SAMPLES) if ALLOW_ROTATION else np.zeros(N_SAMPLES)
    return np.stack([tx, ty], axis=1), th


# ============================================================================
# Adam (tiny, so we keep zero heavy deps)
# ============================================================================


class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def ascend(self, params, grad):
        """In-place gradient *ascent* step (maximize objective)."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        params += self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return params


# ============================================================================
# Live dashboard
# ============================================================================


class Dashboard:
    def __init__(self, xs, ys, Fz, live):
        self.live = live
        self.xs, self.ys, self.Fz = xs, ys, Fz
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 11))
        self.iters = []
        self.hist_obj, self.hist_mean, self.hist_min = [], [], []

    def update(self, it, positions, fz, thr, world):
        (ax_layout, ax_curve), (ax_hist, ax_field) = self.axes
        for ax in (ax_layout, ax_curve, ax_hist, ax_field):
            ax.clear()
        px, py = PLATFORM_SIZE

        ax_layout.add_patch(plt.Rectangle((-px / 2, -py / 2), px, py,
                                          fill=False, ec="k", ls="--", alpha=0.5))
        ax_layout.scatter(positions[:, 0], positions[:, 1], c="crimson", s=40, zorder=3)
        ax_layout.set_title(f"Platform layout (iter {it}, N={len(positions)})")
        ax_layout.set_xlabel("x (mm)"); ax_layout.set_ylabel("y (mm)")
        ax_layout.set_aspect("equal")
        m = 0.12 * max(px, py)
        ax_layout.set_xlim(-px / 2 - m, px / 2 + m)
        ax_layout.set_ylim(-py / 2 - m, py / 2 + m)

        self.iters.append(it)
        self.hist_obj.append(fz[fz <= thr].mean())
        self.hist_mean.append(fz.mean())
        self.hist_min.append(fz.min())
        t = self.iters
        ax_curve.plot(t, self.hist_mean, color="gray", label="mean Fz (all samples)")
        ax_curve.plot(t, self.hist_obj, color="crimson", lw=2,
                      label=f"objective: mean of worst {int(WORST_FRACTION*100)}%")
        ax_curve.plot(t, self.hist_min, color="steelblue", alpha=0.6, label="min Fz")
        ax_curve.axhline(0.0, color="k", lw=0.8, ls=":")
        ax_curve.set_title("Optimization progress")
        ax_curve.set_xlabel("iteration"); ax_curve.set_ylabel("vertical force")
        ax_curve.legend(loc="best", fontsize=8)

        ax_hist.hist(fz, bins=60, color="slategray", alpha=0.85)
        ax_hist.axvline(thr, color="crimson", lw=2,
                        label=f"worst-{int(WORST_FRACTION*100)}% cutoff")
        ax_hist.axvline(0.0, color="k", lw=0.8, ls=":")
        ax_hist.set_title("Distribution of Fz over random placements")
        ax_hist.set_xlabel("vertical force on platform"); ax_hist.set_ylabel("count")
        ax_hist.legend(loc="best", fontsize=8)

        vmax = np.abs(self.Fz).max()
        ax_field.imshow(self.Fz, origin="lower",
                        extent=[self.xs[0], self.xs[-1], self.ys[0], self.ys[-1]],
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        wi = int(np.argmin(fz))
        wpos = world[wi]
        ax_field.scatter(wpos[:, 0], wpos[:, 1], c="lime", edgecolor="k", s=30, zorder=3)
        ax_field.set_title(f"Single-magnet Fz field + worst placement (Fz={fz[wi]:.2f})")
        ax_field.set_xlabel("x (mm)"); ax_field.set_ylabel("y (mm)")

        self.fig.tight_layout()
        if self.live:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def save(self, path):
        self.fig.savefig(path, dpi=130)
        print(f"Saved dashboard to {path}")


# ============================================================================
# Main optimization loop
# ============================================================================


def main():
    rng = np.random.default_rng(SEED)

    print("Building single-magnet force field ...")
    xs, ys, Fx, Fy, Fz, centers, ground_bbox = build_single_field()
    print(f"  field grid: {Fz.shape[1]} x {Fz.shape[0]}   "
          f"Fz range [{Fz.min():.3f}, {Fz.max():.3f}]")

    model = PlatformModel(xs, ys, Fz)
    positions = init_positions(rng)
    opt = Adam(positions.shape, LEARNING_RATE)

    live = "agg" not in matplotlib.get_backend().lower()
    if live:
        plt.ion()
    dash = Dashboard(xs, ys, Fz, live)

    k = max(1, int(round(N_SAMPLES * WORST_FRACTION)))
    px, py = PLATFORM_SIZE
    lo = np.array([-px / 2, -py / 2])
    hi = np.array([px / 2, py / 2])

    for it in range(N_ITERS):
        T, theta = sample_placements(rng, ground_bbox)
        fz, grad, world = model.evaluate(positions, T, theta)

        order = np.argsort(fz)
        worst = order[:k]
        thr = fz[worst[-1]]
        # gradient of (mean of worst-k Fz) w.r.t. positions
        gobj = grad[worst].mean(axis=0)  # (N, 2)

        positions = opt.ascend(positions, gobj)
        np.clip(positions, lo, hi, out=positions)

        if it % PLOT_EVERY == 0 or it == N_ITERS - 1:
            dash.update(it, positions, fz, thr, world)
            print(f"iter {it:4d}  worst-{int(WORST_FRACTION*100)}% = {fz[worst].mean():+.4f}"
                  f"   mean = {fz.mean():+.4f}   min = {fz.min():+.4f}   max = {fz.max():+.4f}")

    dash.save(os.path.join(OUT_DIR, "optimization_result.png"))
    np.save(os.path.join(OUT_DIR, "best_layout.npy"), positions)
    print("Saved best layout to best_layout.npy")
    if live:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
