# fluids

A 3D particle water simulation in Rust and [Bevy](https://bevyengine.org), with
configurable offshore swell, a submerged reef, and a continuous water surface.

The default scene starts with flat water. An offshore generation zone introduces
a regular swell, which travels into shallower water and interacts with a steep
reef shelf. The particle solver determines the wave face, overturning, impact,
and wash. The reef mesh and collisions use the same bathymetry.

```
cargo run --release
```

Set the incoming swell in [`config.toml`](config.toml), then restart the app:

```toml
[swell]
direction = 0.0  # direction of travel: +X toward the reef; positive turns toward +Z
height = 105.0   # offshore crest-to-trough height, in world units
period = 8.0     # simulation seconds between crests
```

The old `wave.height`, `speed`, `curl`, `lip_thickness`, and `peel` controls have
been removed; use `[swell]` and the reef geometry instead.

Direction supports **-60 to +60 degrees**, relative to +X; it is a travel angle,
not a compass bearing or a meteorological “coming from” direction. The reef stays
fixed. Changing the angle changes the incoming orbital velocities and the phase
along the crest. Height is the generator's target offshore height, not the final
breaking height. `height = 0` gives still water with the same reef.

The wavemaker solves `ω² = g k tanh(k h)` for wavelength, then relaxes offshore
particle velocities toward the corresponding finite-depth orbital motion. Its
strength fades to zero before the reef. This follows the
[wave relaxation zone approach](https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation#3132-relaxation-zone-rz).
There is no prescribed crest, curl, or lip trajectory. A damping beach beyond
the reef reduces returning wash; particles remain in the tank.

The water settles for one period and the generator ramps up over two more.
Allow roughly 35 simulated seconds for the shipped swell to reach the shelf: the
ramp is 24 of them at this period, and deep-water swell carries its energy at
half the speed of its crests.
Playback starts at normal speed with automatic replay disabled; `S` slows the
simulation for inspection and `R` restarts it.

Lengths use the same arbitrary world units as the original solver, and period
uses simulation seconds. The shipped gravity is 700 units/s². Longer periods
need a deeper and wider offshore region: validation requires at least half a
wavelength of water depth, resolved wave height/wavelength, and space between
the generation zone, reef, and damping beach. For a meter-based setup, set
`world.gravity = [0, -9.81, 0]` and choose all lengths, particle spacing, and input
settings consistently. Playback speed does not change the physical period.

How big the wave can be is not a free choice. Water cannot hold a wave steeper
than about `height / wavelength = 1/7`, and wavelength is not a field: it comes
out of period and gravity as `L = g T² / 2π`. So a bigger wave needs a longer
wavelength, which needs a longer period, half a wavelength of water depth, and a
tank wide enough to hold it — the shipped preset spends 480 units of depth and
1400 of width on a 914-unit wavelength, and puts the height at 0.115 of it,
just under the 0.12 validation allows.

Gravity is the cheap half of that bill. At 700 units/s² this wavelength would
arrive every 2.9 seconds; at 90 it takes 8, without another 7000 units of
wavelength and 3500 of depth to simulate. It is the same physics at a different
scale — against real gravity the preset is 0.109 m per world unit, a 100-metre,
8-second, 11-metre swell breaking over a 3-metre shelf.

For one big wave that actually plunges, throwing a lip and a barrel along a wide
crest onto a shallow reef:

```
cargo run --release -- slab.toml
```

The swell above collapses into a bore at its ledge rather than pitching: at that
tank's proportions a deep-water wave barely shoals before it reaches the shelf,
so it arrives already as steep as it can be and has nowhere to go. `slab.toml`
sends a single solitary wave instead (`swell.waves = 1`) — one crest with no
trough, the long-period limit of a swell — up a long 1:8 reef slope. It starts
offshore already travelling, from the first-order Boussinesq solution, so there
is no generation zone and no period. Whether it plunges follows Grilli et al.
(1997): plunging for `0.025 < 1.521 · slope / sqrt(height / depth) < 0.3`, with
spilling below and a collapsing bore above. The preset sits at 0.25; the same
wave on a 1:5 slope (0.39) collapses without a lip. The lip throws about 13
simulated seconds in, first at the far wall: the reef is skewed slightly, so the
barrel peels across the 1200-unit crest toward the camera. It is about 1.3
million particles: a solver step takes about 80 ms on an M4 Pro's GPU and 400 on
its CPU, so the break takes about a minute to arrive on the GPU; `world.depth = 200` is the same break
across a narrow strip at a sixth of the cost.

For the original short-period swell across a much wider crest:

```
cargo run --release -- wide.toml
```

For the original dam break:

```
cargo run --release -- dam-break.toml
```

## Water appearance and wave shape

The renderer reconstructs a full 3D density isosurface with marching tetrahedra,
so an overhanging lip and its underside can exist in the same horizontal location.
Smooth density-gradient normals replace the old individually lit spheres.
A matching 3D density texture lets the water shader estimate refracted path
length, color absorption, and how much the lip blocks sunlight inside the barrel.
Fresnel reflections use a procedural sky. Foam follows local velocity disorder,
persists on the particles, and dissipates; detached fast particles become spray.

The optics are approximate: the shader uses a procedural sky and seabed instead
of tracing the rendered scene. Foam is a visual aeration indicator, not a bubble
or air-pressure simulation. Isotropic reconstruction still rounds sheets thinner
than a few particles. These are the main remaining limits on realism.

Useful settings in `config.toml`:

| Setting | Effect |
|---|---|
| `swell.direction` | Incoming travel angle relative to +X, in degrees |
| `swell.height` | Target offshore crest-to-trough height |
| `swell.period` | Time between crests; also sets wavelength and orbital speed |
| `swell.waves` | `0` for the endless swell; `1` for a single solitary wave of `swell.height` above still water |
| `wave.water_depth` | Still-water depth offshore |
| `wave.reef_height` | Bed rise; water above the shelf is `water_depth - reef_height` |
| `wave.reef_start` | Start of the bed rise, as a fraction of tank width |
| `wave.reef_width` | Length of the bed rise, as a fraction of tank width |
| `wave.reef_skew` | Change in reef X per unit Z; affects where the wave breaks along the shelf |
| `wave.beach_width` | Fraction of tank width used to damp shoreward wash |
| `scene.time_scale` | Playback speed, without changing the physics timestep |
| `scene.replay_after` | Replay interval in simulated seconds; `0` disables it |
| `solver.backend` | `gpu` (the default) or `cpu`; see [On the GPU](#on-the-gpu) |
| `render.surface_resolution` | Mesh voxel size / particle spacing; smaller costs more to rebuild |
| `render.foam_speed` | Foam visibility calibration; lower makes more foam visible |

Slow motion scales the simulation clock. The renderer interpolates particle
positions between fixed solver steps, so slowing playback does not increase
viscosity or artificial pressure.

The solver, surface reconstruction and spray run on the GPU by default, and all
on the CPU with `solver.backend = "cpu"`; the title names which, and reports the
solver's and the surface's costs separately. The historical solver benchmarks
below refer to the CPU solver on the dam break, before surface reconstruction
was added.

The default uses a `1400 × 720 × 140` tank with about 22,000 particles, against
68,900 in `wide.toml`. Most of that saving is the narrower crest, and the rest is
particle spacing: 14 units against 8, which is what pays for water deep enough to
carry a 914-unit wavelength. The wave grew faster than the spacing did, so it is
better resolved than the original — 65 particles along a wavelength and 7.5
across the crest, against 45 and 5. Twelve solver iterations, checked by the wave
regression tests. Surface reconstruction uses about 77,000 grid samples instead
of 548,000; thin surface features are slightly less detailed.

To tune cost, reduce `world.depth` (the span across the crest) or increase
`render.surface_resolution`. Lowering particle spacing increases cost sharply;
changing water depth or wave period also affects the physics and must pass
validation.

Everything tunable lives in [`config.toml`](config.toml), which is commented in
full. Every field is optional, so a file naming one value is a legal config, and
deleting the file gets you the defaults back. Pass a different one as an
argument: `cargo run --release -- big.toml`.

## Controls

| | |
|---|---|
| left mouse | push the water away from the cursor |
| shift + left mouse | pull the water towards the cursor |
| right drag | orbit the camera |
| scroll | zoom |
| space | pause / resume |
| `R` | replay the current scene, clearing foam and restoring gravity |
| `G` | flip gravity |
| `S` | toggle normal speed / slow motion |
| `P` | toggle water surface / particle diagnostic |
| `B` | toggle tank bounds |
| `F12` | save a screenshot in the working directory |

The fluid is on the left button, as it was in the 2D version; the camera took
the right one. A screen position names a ray rather than a point, so the push
gets its depth from the fluid itself — it lands on the frontmost water under the
cursor, which is what makes it feel direct rather than like pushing an invisible
plane floating in the tank.

Expect it to feel firmer than the 2D version did. A radial push in an
incompressible fluid is mostly cancelled by the density constraint — only the
free surface is really free to move — and in 3D there is more water in every
direction to resist it. `input.mouse_strength` is the knob, and `config.toml`
carries the measured response curve.

The window title shows simulation time and speed, particle count, frame/solver/
surface timings, compression, bulk density, and peak particle speed.

## How it works

The solver is [**Position Based Fluids**](https://matthias-research.github.io/pages/publications/pbf_sig_preprint.pdf) (Macklin & Müller, SIGGRAPH 2013) rather
than a force-based SPH scheme. Both model the same thing, but they enforce
incompressibility differently, and that difference decides the timestep:

- Force-based SPH turns a density error into a pressure force. The force has to
  be stiff enough to resist compression, and a stiff force needs a small step —
  the widely copied SPH demos run at `dt = 7e-4`, so ~24 substeps per frame.
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

Three details that are load-bearing:

**Under-relaxation.** The constraint is solved with Jacobi iteration, which
updates every particle against stale neighbours. Applying the full correction
overshoots and the fluid explodes; the first version of this did exactly that,
reaching 10⁶ units/s within two seconds. `jacobi_relax` scales the correction
down, and values above 0.55 are refused.

**Calibrated rest density.** Rather than hard-coding a target density, the
solver sums the kernel over an ideal lattice at startup and uses that
(`lattice_reference`). The relaxation and artificial-pressure terms are scaled
against the same reference. This is also what made the move from 2D to 3D
survivable: the kernel normalisation constants and the neighbour count both
changed, and the calibration recomputes the target from them rather than
carrying a stale number across.

**Parallelism.** A 3D neighbourhood holds roughly twice what a 2D one did, so
the same particle count costs about twice as much. The three hot loops run under
rayon, which is why particle state lives in flat arrays rather than as ECS
components. The neighbour build is counted first and then filled, because the
lists are variable-length and a shared output vector cannot be appended to from
several threads — that two-pass version cut roughly 5 ms off the frame, since
left serial it was the majority of it.

Neighbour search is a dense uniform grid over the (fixed) bounds, rebuilt each
step by counting sort. The bounds don't move and the cell size is the kernel
radius, so a dense grid beats a spatial hash: no modulo, no collisions, and
neighbours land contiguously in memory. Neighbour lists are built once per
substep and reused across all solver iterations.

## On the GPU

`solver.backend = "gpu"`, the default, runs the same solver as compute shaders on
the renderer's own device: `sim.wgsl` is `sim.rs` pass for pass, with the same
kernels, the same order of operations and the same arithmetic, and the CPU
solver stays as the reference it is checked against. The particles never come
back to the CPU. Each frame, during Bevy's extraction, `surface.wgsl` samples
them into the density grid, writes the water shader's volume texture, and runs
marching tetrahedra straight into the water mesh's vertex buffer; `spray.wgsl`
writes the spray droplets into another mesh the same way. Extraction sits after
the frame's solver steps and before anything draws it, so the render thread can
never draw a surface built from a later step. The mouse ray and the title's
compression, bulk density and peak speed are queried from the GPU too, as a
parallel reduction that reads back one value.

Three details decide whether the GPU agrees with the CPU at all:

**Neighbour order.** The neighbour grid is rebuilt by a counting sort: count
particles per cell atomically, prefix-sum the counts in place (a Blelloch scan),
then scatter. The scatter lands particles in their cells in whatever order the
threads ran, so each cell is then sorted back into index order. That makes the
neighbour lists identical to the CPU's, order included, and since a sum over
neighbours rounds according to its order, it is what makes a GPU run repeat
exactly and track the CPU within rounding.

**Fixed-capacity lists.** The CPU sizes its neighbour lists exactly; the GPU gives
every particle a fixed number of slots, twice the kernel ball's volume in rest
spacings, so nothing has to be read back per substep to size a buffer. The
largest neighbourhood is tracked every step, and the lists grow if it is ever
exceeded.

**Memory order.** Particles start numbered in fill order, which puts neighbours
near each other in memory, and the flow scrambles it: ten seconds into
`slab.toml` a step cost twice what the first did, with the same neighbour
counts. So every step the particles are renumbered into the previous grid's cell
order, on the GPU, and a slot-to-particle map carries them back to `Fluid`'s
numbering whenever they are read. Steps now cost the same late in the break as
at the start.

**What "matches" means.** Metal compiles with fast-math, so single steps agree to
rounding rather than bit for bit — about 5×10⁻⁵ of a particle spacing. Over
time no two runs of this flow can be compared particle for particle: nudge the
CPU's own starting positions by 2×10⁻⁵ of a spacing and it parts company with
itself by 2.5 spacings within a second, the same rate the GPU parts from it. So
the GPU is held to what does hold: a step from the CPU's exact state has to agree
all the way through the break, and a GPU run left to itself has to throw its lip
over the reef when the CPU's does.

Cost on an M4 Pro, from `cargo test --release speedup -- --ignored --nocapture`
for a solver step and `surface_cost` for rebuilding the surface, CPU against GPU:

| preset | particles | solver step | surface rebuild |
|---|---|---|---|
| `config.toml` | 22,078 | 11.8 → 3 ms | 3.5 → 2.9 ms |
| `wide.toml` | 68,909 | 30.5 → 4.5 ms | 13.6 → 4.9 ms |
| `slab.toml` | 1,306,995 | 385 → 75 ms | 80 → 18 ms |

A small tank is mostly fixed cost on the GPU — dozens of dispatches and a wait
per step — so the gain grows with the particle count. The CPU renderer also
uploads the rebuilt mesh every frame, a million triangles at `slab.toml`'s scale,
and draws its spray as a million entities; with the GPU backend neither leaves
the GPU. In the app, `slab.toml` reaches 17 simulated seconds in 90 seconds of
wall-clock time on the GPU, against 2 on the CPU. Nine tenths of a GPU step is
the Jacobi iterations (`profile_phases` breaks it down), so `solver.iterations`
is the knob that moves it.

## Turning the particle count up

There are two knobs, and they do different things.

**More water, same detail (dam break).** Raise `fluid.block`. It starts in a corner and
collapses, so a bigger block means a deeper pool. Note it is cubed, not squared.
Solver cost, measured on an M4 Pro (10 performance cores):

| block | particles | ms/step | compression |
|---|---|---|---|
| `[18, 20, 18]` | 6480 | 4.6 | 0.6% |
| `[24, 26, 24]` | 14976 | 7.1 | 1.9% |
| `[30, 29, 30]` | 26100 | 10.1 | 3.2% |
| `[39, 29, 39]` | 44109 | 15.6 | 5.5% |

Those measurements cover the solver only. The new renderer also samples a 3D
density grid, extracts a surface, and uploads the mesh and volume each frame.
Compression climbing with size is just the
deeper pool — more hydrostatic load for the same iteration count.

**Finer detail, same water.** Lower `fluid.spacing`, and lower
`fluid.smoothing_radius` with it — what matters is their *ratio*, which sets the
neighbour count and the whole calibration. Keep it near 2.0.

The catch is that this one does not stand alone. Peak speed comes from gravity
and the drop height, not from resolution, so shrinking the kernel means a
particle covers more of it per step. Once that crosses 1 kernel radius,
particles cross their neighbours before the lists are rebuilt and the fluid
detonates. The fix is `solver.substeps`, and the startup check computes the
Courant number and refuses the combination rather than letting you find out at
runtime:

```
error: config.toml: the fluid would move too far per solver step to stay stable
(Courant number 1.90, limit 0.95): at spacing 4 the kernel radius is 8.0 units,
but the fluid is expected to peak near 910 units/s. Set solver.substeps = 2
(currently 1), or raise fluid.spacing, or lower world.gravity
```

One trap worth naming, because it looks like an obvious optimisation: **do not
cut `solver.iterations` to pay for the extra substeps.** Jacobi iteration
carries pressure roughly one particle per pass, so a finer grid makes the pool
deeper *in particles* and needs more passes, not fewer.

## What this does not do

![the pool settled](docs/settled.png)

**An approximate single-phase wave tank.** Breaking develops from the particle
motion and reef interaction. A clean hollow slab is sensitive to swell steepness,
reef slope, water depth, and resolution; the model is not calibrated against
measured surf breaks. The prescribed height and direction are input targets,
not measured guarantees at the reef. Numerical dissipation and finite tank walls
still affect the arriving swell, especially at large angles. The damping beach
reduces reflections but is not a perfect open boundary. Entrained air and bubble
pressure are outside this solver. Thin lips need several particle layers to
survive reconstruction, so finer spacing costs substantially more.

**Approximate solid support in the wave scene.** The density constraint includes
an analytical estimate of kernel volume inside the reef and tank walls, using
local tangent planes. Reef collisions project along the bed normal so a steep
slope does not turn horizontal corrections into artificial upward jets. This
improves water-level retention but does not constitute an exact boundary-particle
model, especially at corners and rapidly changing bed slopes. The original dam
break retains its previous boundary behavior: its settled pool is roughly 10%
shallower than its rest volume implies.

## Layout

| file | |
|---|---|
| `src/sim.rs` | the solver — no rendering, no ECS in the hot path |
| `src/sim_gpu.rs`, `src/sim.wgsl` | the same solver as compute shaders, and its readouts |
| `src/gpu.rs` | the compute device, shared with the renderer, and pipeline helpers |
| `src/config.rs` | the `config.toml` format, its defaults, and validation |
| `src/wave.rs` | directional swell generation, dispersion, and shared reef geometry |
| `src/surface.rs` | parallel density sampling and 3D surface reconstruction |
| `src/surface_gpu.rs`, `src/surface.wgsl` | the same reconstruction on the GPU, written into the water mesh |
| `src/spray_gpu.rs`, `src/spray.wgsl` | spray and the particle diagnostic, written into a mesh on the GPU |
| `src/render.rs` | surface/volume uploads or GPU rebuilds, reef, spray, diagnostic view |
| `src/water.wgsl` | water absorption, reflections, light transmission, foam |
| `src/camera.rs` | orbit camera |
| `src/main.rs` | app wiring, input, window title readout |

The solver knows nothing about TOML: `Config` is the file format, and
`Config::fluid_params` converts it into the plain struct the solver takes. That
keeps sections like `[render]` out of the physics. `Config::default()` retains
the original dam-break defaults for compatibility; the shipped `config.toml`
selects the reef wave tank. Both configurations have physical regression tests.

The main water surface is one mesh/material. With the CPU solver, spray and the
particle diagnostic share a small sphere mesh and material so they can be
instanced; with the GPU solver they are one mesh of small octahedra written in
place, since a million entities is more than the ECS wants to carry. Past
131,072 particles the diagnostic view shows an even sample of them. The shaders
are embedded in the executable; no external art assets are needed.

## Tests

```
cargo test --release
```

The interesting checks are physical rather than mechanical, since a fluid solver
can be wrong in ways that still compile and still produce plausible motion:

- `rest_density_matches_the_seed_lattice` — the calibration agrees with the
  lattice it was derived from. This is the check that the 3D kernel constants
  and neighbour count actually made it across from 2D.
- `stays_finite_and_bounded` — no NaNs, nothing tunnels through a wall.
- `settles_without_gaining_energy` — ten seconds in, the fluid is nearly at rest
  instead of quietly accumulating energy.
- `settles_into_a_flat_pool` — the dam break runs out to every wall and levels
  off, sampled on both sides.
- `the_bulk_holds_its_rest_density` — measured geometrically, see above.
- `radial_impulse_pushes_out_and_pulls_in` — the mouse force, which is otherwise
  only reachable by hand.
- `reconstructed_water_is_closed_with_outward_normals` — watertight mesh edges,
  outward winding, unit normals, and a reasonable reconstructed volume.
- `dispersion_sets_wavelength_from_period_and_depth` — the prescribed period
  obeys the finite-depth dispersion relation and its deep-water limit.
- `wavemaker_obeys_height_period_and_direction` — proportional amplitude, correct
  period and direction, zero vertical motion at the bed, and an unforced reef.
- `incoming_swell_reaches_the_reef_and_resets_cleanly` — waves travel from a flat
  start to the shelf, exceed the motion in a zero-swell control, remain finite,
  respect the reef, preserve particle count, and replay deterministically.
- `slow_motion_preserves_the_physics` — identical particle states at the same
  simulated time at normal and quarter speed.
- `coherent_translation_does_not_generate_foam` — speed alone does not whiten water.
- `a_single_wave_plunges_over_the_reef` — `slab.toml` in a narrow flume starts
  whole and offshore, then overturns with air under the lip over the reef,
  before the beach.
- `a_single_wave_is_validated_on_its_own_terms` — the swell's period and
  wavelength limits do not apply to a single wave, but its own do.

The GPU has its own, each against the CPU as the reference. They skip, with a
note, on a machine with no GPU adapter:

- `neighbour_search_matches_the_cpu_exactly` — from identical positions, the GPU
  grid and neighbour lists equal the CPU's, order included.
- `one_step_matches_the_cpu` — from the same state, one step agrees in position,
  velocity, foam and spray on the dam break (with a mouse push), the swell flume
  mid-generation and the slab flume, and the GPU readouts agree with the CPU's.
- `the_slab_flume_matches_the_cpu_through_the_break` — a GPU step from the CPU's
  exact state agrees every half second through the break, and a free GPU run
  throws its lip within half a second of the CPU's (13.1 against 13.0 s).
- `gpu_runs_repeat_exactly` — bit-identical replays, including after a reset.
- `renumbering_keeps_every_particle_its_own` — a step taken after the particles
  are shuffled into grid order agrees, particle for particle, with one taken
  without.
- `crowded_neighbourhoods_widen_the_lists` — overfull neighbour lists grow, and
  the solver matches again once they have.
- `gpu_surface_matches_the_cpu` — voxel for voxel and triangle for triangle.
- `droplets_mark_the_spray` — droplets sit on the spray, face outwards, stretch
  with speed, and the diagnostic samples by stride.

For repeatable GPU screenshots (the app renders, saves, and exits):

```
FLUIDS_CAPTURE_PATH=/tmp/reef-swell.png FLUIDS_CAPTURE_AT=12 cargo run --release
```

`FLUIDS_CAPTURE_AT` is simulation time in seconds, not wall-clock time. Captures
disable automatic replay and allow the renderer to warm up at the requested time.

Set `FLUIDS_CAPTURE_UNTIL` as well and the path is a directory that receives a
numbered frame every `1 / FLUIDS_CAPTURE_FPS` simulated seconds (default 30),
from `FLUIDS_CAPTURE_AT` to `FLUIDS_CAPTURE_UNTIL`. The simulation holds at each
frame and the frames land on whole solver steps, so the sequence plays at the
simulation's own speed however slowly the machine renders it. The break in
`slab.toml`, at quarter speed:

```
FLUIDS_CAPTURE_PATH=/tmp/slab FLUIDS_CAPTURE_AT=12 FLUIDS_CAPTURE_UNTIL=14.5 \
  FLUIDS_CAPTURE_FPS=60 cargo run --release -- slab.toml
ffmpeg -framerate 15 -i /tmp/slab/frame-%05d.png -vf scale=1280:-2,format=yuv420p slab.mp4
```

The config has its own set, which mostly exist to keep it from failing quietly:
a typo'd field is rejected rather than ignored, a named config file that does
not exist is an error rather than a silent fallback to defaults, a block too big
for the tank reports the largest that fits, and `required_substeps_is_enough_to_pass`
checks that the number the error message tells you to set actually works at
every spacing — advice that is wrong is worse than no advice.

The ignored tests are diagnostics rather than assertions:

```
cargo test --release report  -- --ignored --nocapture          # trace + ms/step
cargo test --release sweep   -- --ignored --nocapture          # parameter sweep
cargo test --release scaling -- --ignored --nocapture          # cost vs particle count
cargo test --release speedup -- --ignored --nocapture          # CPU against GPU per step
cargo test --release sensitivity -- --ignored --nocapture      # GPU drift against the CPU's own
cargo test --release profile_phases -- --ignored --nocapture   # GPU time per solver pass
cargo test --release sustained -- --ignored --nocapture        # GPU step cost as the break develops
cargo test --release surface_cost -- --ignored --nocapture     # CPU against GPU surface rebuild
```

`scaling` produced the table above. `sweep` is how the defaults were chosen: it
reports peak speed, compression and bulk density across solver settings,
alongside the per-step cost, so the accuracy/speed trade is visible in one
table.

## Toolchain

`rust-toolchain.toml` pins 1.95, because Bevy 0.19 requires it. If you'd rather
use your default toolchain, `rustup update stable` and delete the file.
