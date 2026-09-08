# fluids

A 2D water simulation in Rust, using [Bevy](https://bevyengine.org) for the
window and rendering.

![the fluid mid dam-break](docs/dam-break.png)

```
cargo run --release
```

Release mode matters: the solver does roughly 5 ms of work per frame at the
default particle count, and a debug build is around 50x slower than that.

## Controls

| | |
|---|---|
| left mouse | push the water away from the cursor |
| right mouse | pull the water towards the cursor |
| space | pause / resume |
| `R` | reset to the starting dam break |
| `G` | flip gravity |

The window title carries a live readout: particle count, how far the fluid is
from incompressible, and the speed of the fastest particle.

## How it works

The solver is **Position Based Fluids** (Macklin & Müller, SIGGRAPH 2013) rather
than a force-based SPH scheme. Both model the same thing, but they enforce
incompressibility differently, and that difference decides the timestep:

- Force-based SPH turns a density error into a pressure force. The force has to
  be stiff enough to resist compression, and a stiff force needs a small step —
  the widely copied 2D SPH demos run at `dt = 7e-4`, so ~24 substeps per frame.
- PBF treats constant density as a *constraint* and solves it by moving
  particles directly, then reads velocity back out of the positions actually
  reached. Nothing is stiff, so a full `dt = 1/60` step is stable.

One step, in `sim.rs`:

1. Apply gravity and predict positions.
2. Build neighbour lists from the predicted positions.
3. Iterate: compute each particle's density, derive the constraint multiplier
   λ, and move particles to correct the error.
4. Set `v = (p_corrected - p_old) / dt`.
5. Apply XSPH viscosity.

Deriving velocity from the corrected positions in step 4 is what makes wall
collisions and constraint corrections energy-safe for free — a particle that
gets pushed out of a wall simply *has* less velocity afterwards, with no
restitution coefficient to tune.

Two details that are load-bearing:

**Under-relaxation.** The constraint is solved with Jacobi iteration, which
updates every particle against stale neighbours. Applying the full correction
overshoots and the fluid explodes; the first version of this did exactly that,
reaching 10⁶ units/s within two seconds. `jacobi_relax` scales the correction
down. Values at or above 0.6 diverge for some iteration counts, so 0.5 is a
ceiling rather than a starting point.

**Calibrated rest density.** Rather than hard-coding a target density, the
solver sums the kernel over an ideal lattice at startup and uses that
(`lattice_reference`). The relaxation and artificial-pressure terms are scaled
against the same reference. This means `smoothing_radius` and `spacing` can be
changed without silently invalidating every other tunable.

Neighbour search is a dense uniform grid over the (fixed) bounds, rebuilt each
step by counting sort. The bounds don't move and the cell size is the kernel
radius, so a dense grid beats a spatial hash: no modulo, no collisions, and
neighbours land contiguously in memory. Neighbour lists are built once per step
and reused across all solver iterations, which is where most of the speed comes
from.

## Layout

| file | |
|---|---|
| `src/sim.rs` | the solver — no rendering, no ECS in the hot path |
| `src/render.rs` | one sprite per particle, tinted by speed |
| `src/main.rs` | app wiring, input, window title readout |

Particle state lives in flat `Vec`s rather than as ECS components. Each solver
iteration touches whole neighbourhoods at random, which is much cheaper over
arrays than over archetype storage; the ECS holds only the sprites.

## Tests

```
cargo test --release
```

The interesting checks are physical rather than mechanical — a fluid solver can
be wrong in ways that still compile and still produce plausible-looking motion:

- `rest_density_matches_the_seed_lattice` — the calibration agrees with the
  lattice it was derived from.
- `stays_finite_and_bounded` — no NaNs, nothing tunnels through a wall.
- `settles_without_gaining_energy` — 15 s in, the fluid is nearly at rest
  instead of quietly accumulating energy.
- `settles_into_a_flat_pool` — the settled fluid spreads to the depth its own
  volume implies. This is the one that says *this behaves like water* rather
  than merely *this is stable*: a too-compressible solver settles high, a
  collapsing one settles low.
- `radial_impulse_pushes_out_and_pulls_in` — the mouse force, which is otherwise
  only reachable by hand.

Two ignored tests are diagnostics rather than assertions:

```
cargo test --release report -- --ignored --nocapture   # trace + ms/step
cargo test --release sweep  -- --ignored --nocapture   # parameter sweep
```

`sweep` is how the defaults were chosen. It reports peak and settled speed,
compression, and whether the fluid reached its expected depth, across a grid of
solver parameters — including the per-step cost, so the accuracy/speed trade is
visible in the same table.

## Toolchain

`rust-toolchain.toml` pins 1.95, because Bevy 0.19 requires it. If you'd rather
use your default toolchain, `rustup update stable` and delete the file.
