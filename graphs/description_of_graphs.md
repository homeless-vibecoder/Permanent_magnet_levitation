# Graphs

Forces near a 2D permanent magnet, split by what the **test object** is:

- **`test_charge/`** — the test object is a single `+` magnetic charge.
- **`test_dipole/`** — the test object is itself a small magnet (a dipole), so
  these show the force/**attraction a real magnet** would feel near another
  magnet.

Each force model is shown as (1) a colour-coded force vector field, (2) the
vertical component `F_y` (linear scale), (3) `F_y` (symmetric-log scale), and
where noted a cumulative integral `I(s) = int_0^s F_y dx`.

## Folder layout

```
graphs/
  description_of_graphs.md
  test_charge/                          <- test object = single + charge
    scripts/
      magnetic_dipole_force_field.py    <- point-dipole source
      flat_magnet_force_field.py        <- flat-magnet source (charged intervals)
      dipole_Fy_integral.py             <- I(s) = int_0^s F_y dx (dipole source)
    images/                             <- the matching PNGs
  test_dipole/                          <- test object = a magnet (dipole)
    scripts/
      dipole_dipole_force_field.py      <- net force on a test magnet
      dipole_dipole_Fy_integral.py      <- I(s) for the test magnet
    images/                             <- the matching PNGs
```

Regenerate (needs `numpy` + `matplotlib`), e.g.:

```bash
python3 test_charge/scripts/magnetic_dipole_force_field.py
python3 test_charge/scripts/flat_magnet_force_field.py
python3 test_charge/scripts/dipole_Fy_integral.py
python3 test_dipole/scripts/dipole_dipole_force_field.py
python3 test_dipole/scripts/dipole_dipole_Fy_integral.py
```

## Naming note: `thickness` vs `length`

The vertical gap between the `+` and `-` faces (originally called `l`) is now
called **`thickness`**. The horizontal extent of the flat magnet is called
**`length`**. All models use `thickness = 0.2`; the flat magnet uses
`length = 1.0`.

---

# PART A — test object is a single `+` charge (`test_charge/`)

## Naming note: `thickness` vs `length`

The vertical gap between the `+` and `-` faces (originally called `l`) is now
called **`thickness`**. The new horizontal extent of the flat magnet is called
**`length`**. Both models use `thickness = 0.2`; the flat magnet uses
`length = 1.0`.

---

# Model 1 — point dipole (two magnetic charges)

A magnet is modelled as two **magnetic charges**: `+q` at `(0, 0)` and `-q` at
`(0, -thickness)`. A positive test charge sits at `(x, h)`. The force is
Coulomb-like (inverse-square), all constants set to 1:

```
F(x, h) = q_test * (+q) * (P_t - P_+) / |P_t - P_+|^3
        + q_test * (-q) * (P_t - P_-) / |P_t - P_-|^3
```

with `P_+ = (0, 0)`, `P_- = (0, -thickness)`, `P_t = (x, h)`. Full derivation
and parameters live in `test_charge/scripts/magnetic_dipole_force_field.py`.

## `test_charge/images/magnetic_dipole_force_field.png`

Setup + total force as a colour-coded 2D vector field.

- Red dot = `+q` at `(0, 0)`; blue dot = `-q` at `(0, -thickness)`.
- Arrows = force **direction** (normalised to unit length for readability).
- Colour = force **magnitude** `|F|` (log scale).
- Points within radius `0.25` of either charge are masked (force diverges).
- Sanity check: arrows point away from `+q` (repulsion) and toward `-q`.

## `test_charge/images/magnetic_dipole_force_Fy.png`

Only the **y-component** `F_y` (the lift component), linear colour scale.

- **red = upward (`F_y > 0`)**, **blue = downward**, white ≈ 0; clipped to
  `[-5, 5]`.
- Dashed black curve = the **`F_y = 0` zero-crossing**.
- Sanity check: above `+q` the force is up (red), between the charges it is
  down (blue).

## `test_charge/images/magnetic_dipole_force_Fy_symlog.png`

Same `F_y` on a **symmetric-log** colour scale, to see small and large forces
together.

- Linear within `|F_y| < 1e-3`, logarithmic outside; keeps the +/- sign.
- Capped at `±10^2` so the singular cores don't dominate.
- The `F_y = 0` line asymptotes far away to the dipole **magic-angle** lines
  `y = x / sqrt(2)` (the `3 cos^2(alpha) - 1 = 0` result, `alpha ≈ 54.7°`).

---

# Model 2 — flat magnet (intervals of magnetic charge)

The single +/- pair is replaced by two uniformly charged **line segments**
(the continuum limit of "an interval of magnets"):

```
top face   : +lambda along  y = 0,          x in [-length/2, +length/2]
bottom face: -lambda along  y = -thickness,  x in [-length/2, +length/2]
```

The force on the test charge is the line integral over both faces,

```
F(P_t) = q_test * integral_face  sigma(s) * (P_t - s) / |P_t - s|^3 ds
```

with `sigma = +lambda` (top) / `-lambda` (bottom). Numerically each face is
discretised into `N_SRC = 200` point charges at the segment midpoints, each
carrying `lambda * ds`, `ds = length / N_SRC`. As `length -> 0` this reduces to
Model 1. Parameters: `length = 1.0`, `thickness = 0.2`, `lambda = q_test = 1`.
Details in `test_charge/scripts/flat_magnet_force_field.py`.

## `test_charge/images/flat_magnet_force_field.png`

Setup + total force as a colour-coded 2D vector field.

- Grey rectangle = the magnet body; **red line** = `+` face (`y = 0`),
  **blue line** = `-` face (`y = -thickness`).
- Arrows = force direction (unit length), colour = `|F|` (log scale).
- Points within `0.12` of the magnet are masked.
- Sanity check: near the faces the field looks like two extended sheets; far
  away it looks like the point dipole of Model 1.

## `test_charge/images/flat_magnet_force_Fy.png`

`F_y` (lift component), linear scale; same conventions as Model 1
(`red = up`, `blue = down`, clipped to `[-5, 5]`, dashed `F_y = 0` curve).

## `test_charge/images/flat_magnet_force_Fy_symlog.png`

`F_y` on the symmetric-log scale (linear within `|F_y| < 1e-3`, capped at
`±10^2`), revealing the weak far-field forces. Far from the magnet the
`F_y = 0` curve again approaches the dipole magic-angle lines; close in it is
flattened/widened by the finite `length` of the magnet.

---

# Extra — horizontal integral of `F_y` (dipole only)

For the point dipole we also integrate the vertical force along a horizontal
line at fixed height `h`, accumulating over `x` from `0` to `s`:

```
I(s) = int_0^s F_y(x, h) dx
```

Along `y = h` the integrand is

```
F_y(x, h) = h / (x^2 + h^2)^(3/2)  -  (h+thickness) / (x^2 + (h+thickness)^2)^(3/2)
```

which integrates in closed form (used only to verify the numerics):

```
I(s) = s / (h * sqrt(s^2 + h^2))  -  s / ((h+thickness) * sqrt(s^2 + (h+thickness)^2))
```

Parameters used: `h = 0.3`, `thickness = 0.2`, `s` in `[0, 3]`. Source:
`test_charge/scripts/dipole_Fy_integral.py`.

## `test_charge/images/dipole_Fy_integral.png`

`I(s)` vs `s`.

- **Blue** = numerical cumulative trapezoidal integral; **red dashed** = the
  closed-form expression above (they overlap, confirming correctness).
- **Grey** (right axis) = the integrand `F_y(x, h)` for context.
- Vertical dotted line marks where `F_y = 0` (s ≈ 0.55); `I(s)` is maximal
  there, then decreases slightly as the negative tail of `F_y` is added.
- **Green dash-dot** = the `s -> infinity` asymptote
  `I(inf) = 1/h - 1/(h+thickness)` (= 1.333 here), since each term
  `s / (a*sqrt(s^2+a^2)) -> 1/a`. The curve overshoots this limit and relaxes
  back down to it.

## `test_charge/images/dipole_Fy_integration_region.png`

Shows *what* is being integrated: the horizontal black line at `y = h` from
`x = 0` to `x = s`, drawn over the `F_y` field (symlog, red = up / blue = down).
`I(s)` is the signed accumulation of the field values the line passes through
(starting in the red/up region near the magnet, ending in the blue/down region).

---

# PART B — test object is a magnet/dipole (`test_dipole/`)

Same source magnet (`+q` at `(0, 0)`, `-q` at `(0, -thickness)`, moment up),
but now the **test object is itself a small magnet**, and it is **flipped**
(moment pointing **down**, i.e. its **+ charge is on the bottom**), centred at
`(x, h)`:

```
test -q at (x, h + thickness/2)   (top),
test +q at (x, h - thickness/2)   (bottom).
```

The net force is the sum of the Coulomb forces on each test charge from each
source charge:

```
F = sum_{test t} sum_{source s}  q_t * q_s * (P_t - P_s) / |P_t - P_s|^3
```

Like poles repel (`q_t q_s > 0`), unlike attract (`< 0`). With the test magnet
flipped, the source's top `+q` faces the test's bottom `+q` (like poles), so
directly above the source the test magnet is pushed **up** — **repulsion**,
the levitation-relevant configuration. (Un-flipping it — `+` on top — would
give attraction instead; it is a clean overall sign flip of `F`.) Source:
`test_dipole/scripts/dipole_dipole_force_field.py`.

## `test_dipole/images/dipole_dipole_force_field.png`

Net force on the test magnet as a colour-coded vector field. Red/blue dots =
source `+q`/`-q`; the small pair in the corner is a schematic of the flipped
test magnet (`-` on top, `+` on bottom). Arrows = direction, colour = `|F|`
(log).
- Sanity check: directly above/below the source the arrows point **away** from
  it (repulsion); off to the sides the (now anti-parallel) magnets **attract**.
  This is the dipole–dipole pattern with the test magnet flipped.

## `test_dipole/images/dipole_dipole_force_Fy.png`

`F_y` (linear, clipped to `[-5, 5]`); **`red = up = repulsion`**, `blue = down`,
dashed `F_y = 0` curve. The vertical column above the source is red (the magnet
is pushed up, away from the source).

## `test_dipole/images/dipole_dipole_force_Fy_symlog.png`

Same `F_y` on the symmetric-log scale (linear within `|F_y| < 1e-3`, capped at
`±10^2`) to reveal the weak far-field behaviour.

## `test_dipole/images/dipole_dipole_Fy_integral.png`

`I(s) = int_0^s F_y dx` along `y = h` (here `h = 0.5`). Along the line `F_y` is
a sum of four monopole-pair terms, each integrating to
`(q_t q_s) * s / (dy * sqrt(s^2 + dy^2))` with `dy` the y-gap of that pair.
- **Blue** numerical, **red dashed** closed-form (overlap = correct).
- **Green dash-dot** = asymptote `I(inf) = sum (q_t q_s)/dy = +0.417`.
- `I(s)` is **positive** (net upward / repulsion); it peaks at ≈ +0.48 then
  relaxes down to the asymptote.

## `test_dipole/images/dipole_dipole_Fy_integration_region.png`

The integration line at `y = h` over the `F_y` field, with the source magnet
and a schematic test magnet that slides along the line.
