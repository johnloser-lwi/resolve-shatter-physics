# Shatter Physics

An OpenFX plugin for DaVinci Resolve Studio that breaks the image into pieces and
throws them. It covers what After Effects' Shatter does -- a shatter pattern,
two forces with position, radius and strength, gravity with a direction, tumble,
randomness, viscosity and mass variance -- and adds two things AE does not have:
pieces taken straight from the alpha channel, and a real rigid-body simulation
with collisions.

Applied to a title or a logo with an alpha channel, it needs no roto and no
matte: the piece boundaries either come from a generated pattern clipped to the
alpha, or from the alpha itself, one connected region per piece.

## Requirements

DaVinci Resolve **Studio**. The free version refuses third-party OFX plugins.
Windows, Visual Studio 18 with a C++ toolset.

A CUDA toolkit is optional: without it the plugin still builds and runs entirely
on the CPU. With it, compositing runs on the GPU and frames stay resident on the
card, which is what makes realtime playback possible -- see *The GPU path* below.
CUDA support is declared to the host only when a device is actually present.

## Build

```powershell
.\build.ps1
```

Drives VS 18's own bundled CMake and Ninja from inside its developer
environment, because the system CMake predates VS 18. Produces
`build\bundle\ShatterPhysics.ofx.bundle` and `build\shattertest.exe`.

The OpenFX SDK lives in `external\openfx` and is not committed. If it is missing,
CMake fails with the exact clone command.

## Install

```powershell
.\install.ps1 -Build
```

or double-click `Install-ShatterPhysics.cmd`. The script probes whether
`C:\Program Files\Common Files\OFX\Plugins` is writable and only asks for
elevation if it has to, builds *before* elevating so `build\` is not left owned
by the admin account, and verifies the install with a SHA256 comparison.

**Resolve scans for OFX plugins only at launch, so restart it afterwards.** It
also holds a file lock on any plugin build it has already loaded; pass `-Force`
to have the script close it for you.

The effect appears at **OpenFX > Filters > Stylize > Shatter Physics**.

## Using it

Put the clip on its own track and apply the effect. Then:

1. Turn on **Show Pieces** first. It tints every detected piece a different
   colour, which tells you what is actually going to break apart before you
   worry about how it moves. Almost every disappointing result is a shape
   problem, not a physics problem.
2. Pick a **Pattern**. Glass for irregular shards, Bricks or Grid for regular
   ones, **Alpha Pieces** to let the artwork decide.
3. Set **Start (frames)**. This is clip-relative: 0 is the clip's first frame, so
   trimming or moving the clip carries the shatter with it.
4. Place **Force 1** and give it a radius and a strength.
5. Leave **Simulation** on Ballistic while you tune. Switch to Rigid Body when
   you want pieces to hit each other and land.

### Shape

| Control | What it does |
|---|---|
| Pattern | Glass (Voronoi), Bricks, Grid, Alpha Pieces, Custom Map, Manual Slices |
| Pieces | Target cell count for Glass. The real count comes out lower, because cells that miss the image are discarded |
| Irregularity | 0 is an even lattice, 1 is scattered like real broken glass |
| Rows / Row Offset | Brick courses and their stagger. 0.5 is a normal running bond |
| Map Levels | How finely the Custom Map input is quantised before regions are found |
| Alpha Threshold | How opaque a pixel must be to count as part of the image |
| Min Piece Size | Discards debris below this share of the frame's area |
| Bridge | Joins parts that nearly touch so they break away as one piece |
| Shape Seed | Moves the cuts without changing anything else |

**Alpha Pieces** uses no pattern at all. Every connected region of alpha becomes
one piece, so a title breaks into its own letters and a logo breaks into the
parts it was drawn from. Reach for **Bridge** when two nearly-touching parts
should travel together -- a script face whose strokes meet at hairlines, for
instance.

**Custom Map** reads a second input (General context only) and makes one piece
per region of flat colour. If nothing is connected it falls back to Glass.

### Cutting by hand

Turn on **Slice Tool** and drag across the viewer. Each drag adds one cut.

A cut separates whatever it crosses **all the way through**. One that stops
partway into a piece does not split it -- exactly like a real knife, and unlike
the infinite lines most "slice" implementations use, which would also cut every
other object that happened to lie along the same line. Drag the square handles at
either end to adjust a cut afterwards; **Clear Cuts** removes them all, and undo
brings them back.

Cuts apply **on top of whichever pattern is selected**, so the usual workflow is
to let Glass break the frame and then fix the two places it got wrong, rather
than choosing between automatic and manual. The **Manual Slices** pattern
generates nothing at all, leaving your cuts as the only thing that separates
anything -- that is the mode for full control.

The cuts live in the **Cuts** field as `x1,y1,x2,y2` in fractions of the frame,
so they can be read, edited by hand, and copied between clips. Up to 64 of them.

Slicing is lossless. The cut works by taking a hairline barrier out of the
connectivity test, and those pixels are then handed back to a neighbouring piece
rather than being left as background -- otherwise every cut would erase a
one-pixel crack through the artwork. There is a test for that, and another for
the case that actually bites: an eight-connected labeller steps diagonally, so a
diagonal cut one pixel wide leaks and fails to separate anything.

### Forces

Two independent forces, each with a shape, a position, a strength and a delay.
**Negative strength pulls inward** instead of throwing.

**Radial** blows outward from a point and fades with distance from it, bounded by
a **Radius**. This is AE's force.

**Directional** blows along an angle and fades with distance from the *line*,
bounded by a **Width** across it. Its reach *along* the line is unlimited, so it
behaves like wind rather than like a blast — a narrow band sweeping across the
frame moves only what it passes over and leaves the rest standing. Direction uses
the same convention as gravity: 180 is down, 90 is to the right.

**Delay** is how many frames after the start each force fires, and it is a
genuine second impulse rather than a blend. A delayed force arrives while the
pieces are already in the air and changes their trajectory — break the frame with
a radial burst, then sweep the debris sideways with a directional force eight
frames later. Both simulation modes handle it as a timeline of impulses, so the
maths stays closed-form in Ballistic and the solver applies each shove at substep
resolution in Rigid Body.

Both forces are draggable in the viewer. Set the on-screen-controls dropdown to
**Open FX Overlay**. A radial force draws as a crosshair with two circles — the
outer one is its actual reach, so everything outside it is untouched. A
directional force draws its centre line, the two band edges, and an arrow showing
which way it blows. Drag the centre handle to move it, the arrow tip to aim it,
and the handle on the band edge (or on the circle, for radial) to resize it. The
label carries the strength, says *push* or *pull*, and shows the delay, because
none of those are visible from the geometry alone. A disabled force is not drawn:
a handle that does nothing when dragged is worse than no handle.

**Force Mode** replaces AE's 3D force depth, which has no meaning in 2D:

- *All At Once* -- everything lets go on the start frame.
- *Inside Radius Only* -- only pieces a force actually covers ever move. The rest
  stay exactly where they were drawn, and act as obstacles for the pieces that
  are moving. This is the mode for punching a hole through something.
- *Radial Wave* -- the break spreads outward from the force at **Wave Speed**.
  This is the look AE gets from an expanding force sphere.

### Physics

Gravity, gravity direction (180 degrees is straight down), rotation speed,
randomness, viscosity and mass variance map onto AE's Physics group closely
enough that settings translate by eye.

**Simulation** picks between two quite different things:

- **Ballistic** evaluates each piece's arc in closed form at whatever frame is
  asked for. Instant to scrub, no memory, no warm-up. Pieces pass through each
  other and through the ground.
- **Rigid Body** runs an actual simulation: convex-hull collisions between
  pieces, a floor, optional walls, friction and restitution. It has to be
  stepped forward from the start frame, so the first render deep into a clip
  pauses briefly while the run catches up. After that, scrubbing anywhere in the
  clip is a lookup.

Both modes launch every piece with identical initial conditions, so switching
between them changes how pieces *interact*, not where they were thrown.

### Drawing obstacles

Rigid Body only. Turn on **Collider Tool**, pick a shape from **Draw**, and drag
across the viewer. Each drag adds one obstacle.

| Draw | One drag gives you |
|---|---|
| Box | opposite corners of an axis-aligned box |
| Circle | centre, then a point on the rim |
| Line | the two ends |

Existing obstacles keep whatever they were drawn as, so a scene can mix all
three. Every shape is stored as **two control points** plus, for a box, an angle
— which is what lets one pair of drag handles edit all of them.

A box carries a **rotation handle** on a short tether beyond its upper edge:
drag it to turn the box about its own centre. Tilt one past about 22 degrees and
it stops being a shelf and becomes a ramp — pieces landing on it slide off the
low end. (Below roughly the friction angle they will just sit there, which is
correct rather than broken; raise the tilt or lower **Friction**.) A circle needs
no rotation, and a line already has any orientation its two ends can express.

Every cut and every obstacle also has a small **red ×** just outside its top-right
corner while its tool is armed. Click it to delete that one. **Clear Cuts** and
**Clear Obstacles** remove all of them at once, and undo brings any of it back.

**Box and Circle are solid all the way through** — they are real convex bodies,
not outlines. Four lines around a box would leave an interior that anything
already inside simply stays in. A **Line** is a bare edge, which is what you want
for a ramp or a shelf.

An obstacle is solid from **both sides** — for a line the contact normal is
oriented toward whichever side the piece approached from, each substep, so the
same line can be landed on from above and bounced off from below without being
two objects. A line also **ends where you stopped drawing**: a piece sliding off
the end of a ramp keeps going rather than catching on an invisible extension.

Note that a circle sheds what lands off its centre — pieces slide down the sides
and away. That is correct, not a bug, and it is why the test suite checks that
nothing ever gets *inside* a circle rather than that anything comes to rest on
one.

Obstacles live in the **Obstacles** field as `shape,x1,y1,x2,y2,angle` (0 = line,
1 = box, 2 = circle), capped at 32 because every one is tested against every
moving piece every substep. Type an exact angle there if dragging the handle is
not precise enough. Changing any of it — the shape or the angle alone included —
rebuilds the cached simulation.

### Depth (3D Look)

Not a 3D environment — an illusion of one. The simulation stays completely flat:
forces, gravity, collisions and the solver all work in the plane, which is what
keeps the rigid-body mode tractable and the ballistic mode closed-form. What gets
added is a per-piece z that nothing collides against and only the compositor
reads.

That buys three of the four cues an eye actually uses for depth — perspective,
occlusion, and aerial haze. The fourth, binocular disparity, is unavailable on a
flat screen anyway. The remaining giveaway is that each piece is a flat cutout,
which is what **Edge Shading** and **Tumble** are for.

Everything here defaults to **0**. Turning it on is three sliders:

| Control | What it does |
|---|---|
| Depth Spread | Master switch. Throws pieces toward and away from camera as well as across frame, scaled by how hard each was actually thrown |
| Toward / Away | Biases everything one way regardless of the forces — recede or come at you |
| Perspective | How strongly depth changes size. 0 is orthographic, and depth still decides what covers what |
| Distance Fade | Darkens pieces as they recede |
| Tumble | Flips pieces end-over-end as well as spinning them flat, so they narrow to an edge and open out again |
| Edge Shading | Lights a band along every cut edge, bright toward the light and dark away, so pieces read as having thickness |
| Edge Width | How thick they look, as a percentage of frame height. Wide is masonry, narrow is glass |
| Light Angle | Same convention as Gravity Direction; 180 is from directly above |

A reasonable starting recipe: **Depth Spread 1.0, Perspective 0.9, Distance Fade
0.45, Tumble 1.1, Edge Shading 1.1**.

**Edge Shading is the one that does the most work**, and it works with everything
else at 0. Note that it only shows on both sides where the artwork has tone to
spare — on pure white, the lit side clips and you see only the shadow.

Occlusion comes free: the draw list is sorted furthest-first before it is binned,
so nearer pieces cover farther ones. Ties break on piece id, because before the
shatter starts every piece sits at z = 0 and an unstable order there flickers
wherever two pieces overlap by a pixel.

Depth is applied *after* the simulation, like the fade, so changing any of it
never invalidates a cached rigid-body run.

### Render

Fade Out (0 means pieces never fade), Shrink While Fading, Smooth Edges, Motion
Blur with a shutter angle, and Show Pieces.

## How it works

1. **Threshold** the alpha to decide which pixels are part of the image at all.
2. **Bridge**, optionally, with a separable dilation whose cost does not depend
   on the radius.
3. **Build a pattern** as one integer key per pixel -- a Voronoi cell id, a brick
   id, or a quantised map colour.
4. **Label** with an 8-connectivity two-pass union-find in which two touching
   pixels join only when their keys match. This is where the pattern and the
   alpha get intersected: it happens inside the labelling rather than as a
   separate clipping pass, so a cell split into two islands by a gap in the
   artwork correctly becomes two pieces.
5. **Despeckle** anything below Min Piece Size, then renumber so labels stay
   dense.
6. **Hull** each piece's boundary pixels into a convex polygon, but only when the
   rigid-body mode is going to use it.
7. **Pose** every piece: closed form, or one step of the solver per frame.
8. **Composite** by binning transformed piece bounds into a 32-pixel tile grid,
   then asking each destination pixel which pieces land on it.

The label image does double duty throughout: it is the segmentation result *and*
the render stencil, so drawing an arbitrarily shaped piece costs one integer
compare instead of a per-piece mask.

## Testing without Resolve

`src/core` contains no OFX headers, so the offline harness links the identical
code Resolve runs. Tuning a shatter through it costs a second; through Resolve it
costs a relaunch.

```powershell
.\build\shattertest.exe testdata\caps.png out.png --pieces
.\build\shattertest.exe testdata\caps.png strip.png --cells 200 --strip 8 --frames 30
.\build\shattertest.exe testdata\caps.png pile.png --rigid --at 55 --floor 0.95 --walls
.\build\shattertest.exe testdata\res_2160.png x.png --bench 30
```

`--strip N` writes a contact sheet across the shatter, which is the fastest way
to judge timing. `--bench N` times the render with no PNG encoding in the way --
worth knowing, because a 4K `stb_write_png` costs more than twenty frames of
compositing and will otherwise dominate any measurement you take.

```powershell
ctest --test-dir build --output-on-failure
```

## Measured

24-core CPU, all frames full-frame (tiling is off). These are the **CPU** path,
measured standalone with no host transfers -- which is exactly the caveat that
makes them misleading on their own; see *The GPU path*.

| Case | Per frame |
|---|---|
| 3840x2160, ballistic | 17.9 ms (56 fps) |
| 3840x2160, rigid body with collisions | 19.0 ms (53 fps) |
| 3840x2160, ballistic, motion blur 8 samples | 22.3 ms (45 fps) |
| 3840x2160, nothing to draw (floor cost) | 15.5 ms |
| 1920x540, 93 pieces | 1.2 ms |

The floor-cost row is the important one: at 4K the shard work is about 2.4 ms and
everything else is clearing and walking the frame. The compositor is bound by
frame area, not by piece count, so raising Pieces is close to free.

It is also the row that shows why these numbers do not settle the GPU question.
Reading and writing one 4K float frame is most of the budget before any work
happens at all, and inside Resolve a CPU plugin pays that same bandwidth *again*,
twice, just moving the frame off the card and back.

## The GPU path

There is a CUDA kernel, and the reason for it is not the one you would guess
from the table above.

Judged on compute alone it looks unnecessary: the CPU path clears 4K at 45-56
fps, and of its 17.9 ms only about 2.4 ms is shard work. A kernel appears to buy
two milliseconds.

That reasoning measures the wrong thing. **Resolve's pipeline is GPU-resident.**
A plugin that does not declare CUDA support is not merely a little slower -- it
forces the host to copy every frame down to system memory and back up again. At
4K float RGBA that is about 133 MB each way, roughly 11-22 ms per frame of PCIe
traffic depending on the slot, serialised against Resolve's own GPU work. None
of it appears in a standalone benchmark, and it is larger than the entire cost
of compositing the frame. That transfer, not the arithmetic, is what stops a CPU
plugin from playing back in realtime.

So the kernel exists to keep the frame on the card. What runs where:

- **On the GPU:** compositing, and the content fingerprint.
- **On the CPU:** segmentation and the physics. Labelling is inherently serial
  and is cached per content change, so it runs once per edit rather than once per
  frame; the solver is well under a millisecond.

The trick that keeps the CPU-side analysis from costing a download every frame is
`CudaHashSource`: the fingerprint is computed on the device and read back as
**eight bytes**, so a cache hit transfers essentially nothing and the full frame
comes down only when the content has actually changed. The label image goes up
once per geometry change, keyed on pointer identity, not once per frame.

`CudaKernel.cu` contains no maths of its own. It includes `core/Compositor.h`
and calls the same `ShadePixel` the CPU path calls, compiled as device code via
the `SHP_HD` macro. `tests/paritytest.cpp` renders the same frame both ways and
compares:

```
rest frame, AA off          worst delta 0.00e+00,  0/153600 pixels differ
rest frame, AA on           worst delta 0.00e+00,  0/153600 pixels differ
mid shatter, AA off         worst delta 3.31e-05,  0/153600 pixels differ
mid shatter, AA on          worst delta 3.05e-05,  0/153600 pixels differ
mid shatter, show pieces    worst delta 0.00e+00,  0/153600 pixels differ
mid shatter, motion blur    worst delta 1.05e-05,  0/153600 pixels differ
```

Rest frames come back bit-identical. The 3e-05 on moving frames is the two
compilers contracting multiply-adds differently, which is expected and
meaningless; a real disagreement -- a wrong stencil, a flipped axis, a missed
shard -- is orders of magnitude larger. The parity test skips itself, passing, on
a machine with no CUDA device.

`Custom Map` is not supported on the GPU path yet: the map would need its own
download, and reading a device pointer on the host is a crash rather than a
glitch. It falls back to the generated pattern.

## Notes for anyone changing this

**Tiling is off and must stay off.** Both the alpha segmentation and the cell
pattern are whole-image measurements. Handed a tile, the labeller would shatter
every piece crossing the boundary into two, and each tile would number its pieces
from 1 independently.

**`isIdentity` returns false unconditionally.** Answering "yes" lets the host
substitute the source frame and skip the effect -- and it caches that verdict.
Frames the host decided were pass-through before the shatter started stay
pass-through after it does. The cost is a copy on frames before the start; the
benefit is an effect that works when scrubbed, not only when played forward.

**All timing is in frames, never seconds, and nothing asks the host for a frame
rate.** Fusion does not publish `kOfxImageEffectPropFrameRate` on clips, and the
OFX Support library *throws* on a missing property. Thrown out of the render
action that becomes `kOfxStatErrMissingHostFeature` and fails every single frame,
so the effect appears to do nothing at all. Working in frames removes the
dependency rather than guarding it.

**Every parameter sets `eCacheInvalidateValueAll`.** Nothing here is keyframed --
parameters are constants the simulation interprets against the render time -- so
every parameter affects every frame. The OFX default only invalidates the range a
parameter's own keyframe covers, which gives a caching host no reason to discard
anything, and edits made while the playhead sits outside the shatter appear to do
nothing until a manual cache purge.

**Visibility lives in one `syncVisibility()`, called from the constructor and
`changedParam`, and never from render.** Splitting it across per-parameter
handlers is how a control ends up visible in one state and hidden in an
equivalent one depending on click order.

**Never write a parameter from `render`.** It runs on host worker threads outside
any edit block, and a host that dispatches the resulting `changedParam` back onto
one deadlocks against the analysis cache's mutex.

**`src/core` includes no OFX header.** That rule is the only thing keeping the
offline harness and the tests honest -- they link the identical code Resolve runs,
rather than a copy that has quietly drifted.

**Static CRT and static cudart.** A missing VCRUNTIME or cudart DLL does not
produce an error: the plugin simply never appears in Resolve's list.

**On the CUDA path the pixel pointers are DEVICE memory.** Nothing in `render`
may dereference them on the host, which is why the CUDA branch is taken before
any `ImageView` is built and why there is deliberately no fallback to the CPU
loop when it fails -- by that point the addresses are device memory and the CPU
cannot touch them. A failure leaves a black frame, which is a visible bug; a
host-side dereference is a crash that takes Resolve with it.

**`CudaKernel.cu` must contain no maths.** It calls the same `ShadePixel` the CPU
path calls, compiled as device code through `SHP_HD`. The moment shading logic
gets duplicated there, the two paths start drifting and the parity test becomes
the only thing standing between that and a bug nobody can see. Keep the shared
code in `core/Compositor.h`.

**`CMAKE_CUDA_ARCHITECTURES` is set before `enable_language(CUDA)`.** That call
installs a default, after which a `if(NOT DEFINED ...)` guard silently never
fires and the plugin ships PTX-only code the driver has to JIT on first use.

**The device hash is deliberately a different function from the host hash.**
FNV-1a is a sequential chain and reproducing its order across thousands of
threads would cost more than it saves, so the device version uses an
order-independent XOR combine. The two are never compared against each other: an
instance runs on one path or the other, and if a host switched mid-session the
worst case is one extra re-analysis.

**Geometry is authored in normalized space and scaled to pixels once.** A control
denominated in pixels means something different at every resolution, so a shatter
tuned in a half-res timeline would rearrange itself on delivery. There is a test
for this.

**The rigid-body solver must stay deterministic.** SimCache replays it from the
start frame whenever its key changes; if two runs of the same inputs diverged,
scrubbing backward would show a different shatter than scrubbing forward. That
means: single-threaded, fixed iteration order, no containers whose traversal
order depends on addresses. The broadphase is built as CSR in body order for
exactly this reason. Three tests cover it and they are not optional.

**The simulation cache key deliberately excludes the fade controls.** They are
applied after the poses come back, so nudging a fade must not throw away a run.

**Contact normals point from A toward B, and a static plane's normal is negated
to match.** Getting that sign wrong drives pieces *into* the floor instead of out
of it, which reads as pieces sinking through the ground -- and it passes every
test that only looks at piece counts.

**Motion blur samples are averaged, not composited over one another.** Over-
compositing N samples of alpha a/N gives 1-(1-a/N)^N, so a piece standing still
would render at about 63% opacity. There is a test that renders a resting piece
with blur on and demands it come back solid.

**The content hash rounds rather than truncates.** Truncating `alpha * 15` makes
bucket 15 reachable only by exactly 1.0, so the most common value in an alpha
channel sits alone on a boundary and the tiniest dip below opaque -- a re-encode,
a colour-managed round trip -- invalidates the analysis cache on every frame.

**A cut is a barrier in the connectivity test, not a half-plane.** The tempting
implementation is a per-pixel bitmask of which side of each cut a pixel falls on,
which is exact and free — but it makes every cut an infinite line, so slicing one
letter also slices whatever else lies along that line. Barriers cost a fill pass
and are worth it. The barrier radius is 0.75 px rather than 0.5 because an
eight-connected labeller steps diagonally: a thinner diagonal cut leaks, and it
leaks as a function of the angle, which is the worst kind of intermittent.

**Static convex obstacles go through the same `addPolyContacts` as shard-shard
collisions.** A box gets real reference-face clipping rather than an
approximation nobody remembers is there. The winding must give a POSITIVE
shoelace area, because outward normals are derived from it — reversed, every
normal points into the box and pieces fall straight through. Check that against
`PolygonArea` rather than by eye: core coordinates are top-down, so
"counter-clockwise" reads backwards on screen, and this was written the wrong way
round the first time.

**Anything added to `Collider` must also be added to `HashSimParams` and to the
copy loop in `World::reset`.** Both were missed when the shape field went in.
Forgetting the hash means changing a shape silently keeps the old simulation;
forgetting the copy means every obstacle becomes whatever the struct's default
is. Neither produces an error, and both look like a physics bug. Note that the
copy loop scales coordinates to pixels but must NOT scale the angle.

**In a vector distance transform, the distances being right is no evidence the
directions are.** `EdgeField` propagates an offset from each pixel to its nearest
edge, and the step added at each neighbour is the vector FROM that neighbour TO
this pixel. Four of the eight had the wrong sign at first: the offset *lengths*
were unaffected, so every distance assertion passed, while the stored normals
came out backwards wherever those neighbours won — lighting some bevels from the
wrong side. That reads as a lighting preference, not a bug. Test the direction
explicitly.

Related: the offset accumulates as steps taken *away* from a seed, so it already
points from the edge into the piece. It is the inward normal as stored and must
not be negated.

**A rotated box stores its corners unrotated**, and applies the angle only when
drawing or when building the collision hull. Two consequences, both of which
were got wrong first time and both of which now live in `core/BoxHandles.h`
where the test suite can reach them:

- The corner **handles** must be rotated too, or they sit off the outline they
  belong to.
- Resizing must pivot on the **opposite corner, captured at pen-down**. The
  obvious version un-rotates the pointer about the box's centre — but the centre
  is the midpoint of the two corners, one of which the drag is rewriting, so
  every event is measured against a pivot the previous event already moved. It
  compounds, and the handle becomes impossible to control. `testRotatedBoxHandles`
  replays fifty events at one pointer position and demands nothing drift.

**`RotateHandleDir` and `RotateHandleAngle` are exact inverses, and must stay
that way.** Placing the handle in one direction while reading the angle back
with a formula that assumes another makes the box jump the instant the handle is
grabbed — by 180 degrees if the placement is upside down, by twice the angle if
the sign is flipped. Both of those shipped, together, and the round-trip test
that catches them is one loop over a range of angles.

Note that a round trip alone is not sufficient: a sign flip round-trips
perfectly and still turns the box the wrong way, so there is a separate check
that dragging the handle to the right gives a quarter turn *clockwise*.

That geometry is in `core` rather than in the overlay specifically because the
overlay needs OFX headers and therefore cannot be tested. Anything in the
interact layer that is arithmetic rather than plumbing belongs on the other side
of that line.

**Overlay handle positions come from one function, `handlesFor`.** Drawing and
hit-testing both call it. Deriving them separately is how a handle ends up drawn
somewhere it cannot be grabbed — and the delete cross has to be hit-tested
*first*, because it is the smallest target and sits close enough to the shape
that any other handle would otherwise win on a small obstacle.

**The angle convention is converted in exactly one place.** Positive turns a box
clockwise on screen. Core space is top-down and canonical overlay space is
y-up, so `boxCorners` negates it; nothing else may.

**A drawn obstacle picks its normal from which side the piece is on, every
substep.** That is what makes one line solid from both directions. The
alternative -- baking a side in when it is drawn -- would mean a ramp you can
land on but fall up through, and the bug would only show on the pieces that
happened to arrive from underneath.

**Colliders are deliberately not part of `PhysicsParams`.** That struct is passed
by reference into `SHP_HD` functions, so giving it a `std::vector` member would
make a device-compiled header depend on host-only types. They travel alongside
instead, and they are hashed into the simulation key by hand.

**Forces are impulses with a time, not a launch velocity.** Folding a delayed
force into how a piece left would be simpler and would look completely wrong: the
whole point is that it acts on something already moving. `Ballistic.h` walks the
segments between impulses; the solver applies them in `releaseDue`. Detach timing
deliberately depends only on geometry — not on mass, not on the randomness draw —
so dragging Mass Variance does not shift when things let go.

**There is no continuous collision detection.** A piece thrown hard enough can
pass through a thin obstacle in one substep. The integrator clamps per-substep
travel to half a piece's own radius as a backstop, and Accuracy raises the
substep count. Both are cheaper than a proper swept test for what this effect
needs.
