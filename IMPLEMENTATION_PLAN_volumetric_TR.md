# Implementation Plan: Volumetric Raymarching per TR.md (doc/volumetric/TR.md)

> **STATUS: ✅ COMPLETE** — alle Phasen (A–D) umgesetzt und verifiziert:
> - B1–B7: `src/kernel/volume/` (density, phase, dither, integrate, volume_kernel) — kompiliert
> - C1–C4: `render_mode`-Dispatch (Default: volumetrisch), Zero-Copy-Guard, GUI-Switch — kompiliert
> - D1–D3: `cargo build` 0 Errors; Headless-GPU-Tests beider Kernel bestehen:
>   - `test_volumetric_launch`: 100% Pixel belegt, Ø-Lum 0.455, Max 0.865
>   - `test_kernel_launch` (Surface, Regression): 99.9%, Ø-Lum 0.157, Max 0.601
> - Nebenbefund: tote Dependency `massively` 0.97 (inkompatibel mit cubecl 0.11,
>   nirgends referenziert) aus Cargo.toml entfernt — Voraussetzung für jeden Build.

Goal: Transform the SDF surface raymarcher into a zero-copy volumetric path/ray
tracer (CubeCL), per every requirement in TR.md — while keeping the app runnable
(Surface mode stays reachable via a GUI toggle).

---

## 0. Reconnaissance facts (verified in code — do NOT re-read large files)

| Fact | Evidence |
|---|---|
| CubeCL version | `cubecl = 0.11.0`, features `["wgpu"]` (Cargo.toml) |
| Kernel dir | `src/kernel/`: `mod.rs`, `types.rs`, `objects.rs`, `scene.rs`, `fog.rs`, `main_kernel.rs` |
| Idiom f32 literal | `f32::new(3.0)` — never bare `3.0` in `#[cube]` code |
| Idiom u32→f32 cast | `f32::cast_from(u)` (TR's `f32::cast` does not exist here) |
| Idiom u32 literal | `u32::new(0)`, counter `i += u32::new(1)` |
| Idiom runtime loop | `loop { if i >= n { break; } ... i += u32::new(1); }` — TR's `for _ in 0..n` is NOT used in this codebase (main_kernel.rs:154 comment shows it was replaced) |
| Idiom exp | `(-x).exp()` (fog.rs); sqrt: `.sqrt()`; sin/cos/floor: `.sin() .cos() .floor()` all used |
| min/max/abs | `a.min(b)`, `a.max(b)`, `a.abs()` all used |
| Vec3 (kernel) | `#[cube]` struct in types.rs: fields x,y,z; methods `new add sub scale dot length normalize` |
| GpuVec3 | 16-byte aligned host struct (x,y,z,_pad), used inside CameraSettings/EnvSettings |
| Custom cube struct pattern | `#[derive(CubeType, Copy, Clone)] #[cube(derive(Copy, Clone))]` + struct-literal construction (SdfResult, types.rs:23) |
| Existing gyroid | `evaluate_dynamic_gyroid(p: Vec3, time: f32, size: f32) -> f32` in objects.rs:27 (sphere-bounded → finite cloud, perfect for volume) |
| Launch signature pattern | `raymarch_sdf_kernel(output: &mut Tensor<f32>, scene: &[f32], camera, env, arch, fold, render)` main_kernel.rs:18 |
| Pixel coords | `ABSOLUTE_POS_X / ABSOLUTE_POS_Y` (main_kernel.rs:31) — NOT `LAUNCH_ID_X` |
| Bounds guard | main_kernel wraps body (no early `return` used anywhere in kernels — grep proved it) |
| Output layout | **RGB, 3×f32 per pixel**: `(y*w+x)*3` (main_kernel.rs:432), `output_byte_size = w*h*3*4` (gpu.rs) |
| Tensor index type | `usize` (`output[pixel_index]`, `usize::new(3)`) |
| Cube dispatch | gpu.rs:456 `CubeCount::Static(w.div_ceil(16), h.div_ceil(4), 1)`, `CUBE_DIM_X=16, CUBE_DIM_Y=4` |
| Host structs | `ApplicationState` (app_state.rs) has `camera_launch() env_launch() arch_launch() fold_launch() render_launch(time,w,h)` and `scene_data()` |
| Frame driver | render_loop.rs:95 `write_state_buffers(&s)`; :105 `render_frame(&view, dw, dh, &s_render, time)` → gpu.rs `launch_kernel` → kernel launch |
| GUI | wgpui_kit sidebar with slider entities + subscriptions (src/gui/, built in main.rs:87) |
| Headless test | `src/bin/test_kernel_launch.rs` exists (pattern for a no-window kernel smoke test) |
| `position_to_direction_vector` | does NOT exist — must be created (TR Step 4 uses it) |

## Decisions / documented deviations from TR.md sketch code

- **D1 — Output layout stays RGB (3×f32)**, TR's RGBA×4 write is adapted to the
  existing blit contract (`shader.wgsl` + `output_byte_size`). Alpha is implicit
  1.0. TR's own code contains a typo (`.b` on Vec3) proving it is a sketch, not
  a contract; the host blit pipeline is out of TR scope. Keeps the app displayable.
- **D2 — Loops** use the codebase's `loop/break/counter` idiom instead of
  `for _ in 0..n` (not supported for runtime bounds in this CubeCL setup).
- **D3 — Camera** keeps the existing yaw/pitch basis + mouse controls from
  main_kernel.rs (TR's fixed `(uvx*aspect, uvy, -1)` camera would break input).
  TR's aspect-corrected UV mapping `((fx/w)*2-1)*aspect` is adopted.
- **D4 — Mode switch instead of deletion**: add `render_mode` (0=surface,
  1=volumetric, default 1) to ApplicationState + GUI toggle. Satisfies the TR
  transformation while keeping regression path. The volumetric kernel needs NO
  scene buffer (zero-copy) → scene upload skipped in volumetric mode.
- **D5 — Constants hardcoded per TR**: steps=32, max_dist=25, volume size=1.5,
  g=0.65, shadow_steps=4, shadow_step_size=1.5×step, absorption=0.8,
  density_scale=6.0, noise_amp=0.15, emission profile (3.5h, 1.4h², 0.3h⁴),
  early-out threshold 0.001, wind (0.2t, 0.5t, 0.1t), noise octaves 3.0/6.12 @
  0.7/0.3, jitter hash constants 12.9898/78.233/43758.5453.
- **D6 — `f32::exp`**: use method form `(-x).exp()` as in fog.rs.

---

## Micro-steps (each step = 1 small edit + `cargo check`; context-safe)

### Phase A — remaining recon (tiny, targeted; only what each step needs)
- **A1** `cat src/kernel/mod.rs` (45 lines) → know exact export style for B7.
- **A2** `sed -n '40,90p' src/kernel/main_kernel.rs` → see exact bounds-guard
  style + output write to copy patterns for B6.
- **A3** `ls src/gui/ && grep -n "render_mode\|Slider\|subscribe" src/gui/*.rs | head`
  → know where to add the toggle (needed only for C4).
- **A4** `head -60 src/bin/test_kernel_launch.rs` → copy pattern for D2.

### Phase B — new volumetric kernel module `src/kernel/volume/`
- **B1** Create `src/kernel/volume/mod.rs`: `pub mod density; pub mod phase;
  pub mod dither; pub mod integrate; pub mod volume_kernel;` + re-exports.
  → `cargo check` (module empty files compile).
- **B2** Create `src/kernel/volume/density.rs`:
  - `VolumeSample { density: f32, emission: Vec3 }` (cube struct pattern, types.rs:23 style)
  - `hash_3d(p: Vec3) -> f32` (sin-hash, fract via `x - x.floor()`)
  - `noise_3d(p: Vec3) -> f32` (8-corner trilinear Hermite, per TR Step 1)
  - `get_volumetric_data(p, time, size) -> VolumeSample`:
    gyroid via `super::super::objects::evaluate_dynamic_gyroid` + wind FBM
    (2 octaves) + `density = clip(-sdf*6, 0, 1)` + heat/emission profile.
  → `cargo check`.
- **B3** Create `src/kernel/volume/phase.rs`:
  - `position_to_direction_vector(p: GpuVec3) -> Vec3` (normalize, fixes missing TR method)
  - `henyey_greenstein(cos_theta: f32, g: f32) -> f32` (TR Step 2 formula)
  - `calculate_volume_shadow(sample_pos, light_dir, time, size, shadow_steps,
    shadow_step_size) -> f32` — loop/break idiom, absorption 0.8, `(-x).exp()`.
  → `cargo check`.
- **B4** Create `src/kernel/volume/dither.rs`: `noise_2d_time(x: u32, y: u32,
  time: f32) -> f32` (TR Step 3). → `cargo check`.
- **B5** Create `src/kernel/volume/integrate.rs`: `volumetric_raymarch_integrated(
  ray_origin, ray_dir, max_distance, steps, pixel_x, pixel_y, settings,
  env) -> Vec3` — TR Step 4 verbatim semantics:
  jitter start pos; precompute HG phase (g=0.65) once per ray; loop/break over
  steps; if density>0: shadow (4 steps), in-scatter `env.light_intensity *
  phase * light_color * T_light`, add emission; `opacity = ρ*Δx*env.fog_density`
  (Beer-Lambert recursive transmittance, κ_fog = EnvSettings.fog_density);
  accumulate; early break at T<0.001 (set T=0); composite `bg*T`.
  → `cargo check`.
- **B6** Create `src/kernel/volume/volume_kernel.rs`: `#[cube(launch)] pub fn
  render_volumetric_kernel(output: &mut Tensor<f32>, camera: CameraSettings,
  env: EnvSettings, arch: ArchParams, fold: FoldParams, settings:
  RenderSettings)` — TR Step 5 semantics: ABSOLUTE_POS_X/Y; bounds guard in
  codebase style; TR UV mapping × aspect; existing yaw/pitch basis (D3);
  max_dist 25, steps 32, size 1.5; write RGB×3 via usize indexing (D1).
  → `cargo check`.
- **B7** Export in `src/kernel/mod.rs` (A1 style). → `cargo check` + `cargo build`.

### Phase C — host integration
- **C1** `app_state.rs`: add `pub render_mode: u32` (default 1) + Default impl.
  → `cargo check`.
- **C2** `gpu.rs` `launch_kernel`: `if s.render_mode == 1 { volume
  kernel launch (output_arg, camera, env, arch, fold, render — NO scene arg) }
  else { existing launch }`. → `cargo check` + `cargo build`.
- **C3** `render_loop.rs`: guard `write_state_buffers` (scene upload) with
  `s.render_mode == 0` — zero-copy: volumetric mode uploads nothing per frame.
  → `cargo check`.
- **C4** GUI toggle (A3 findings): checkbox/slider `render_mode` 0/1 bound to
  ApplicationState (surface ↔ volumetric), default volumetric.
  → `cargo check` + `cargo build`.

### Phase D — verification
- **D1** `cargo build` clean (all targets).
- **D2** Headless smoke test: new `src/bin/test_volumetric_launch.rs` cloned
  from test_kernel_launch pattern (A4): init cubecl client, small buffer
  (e.g. 64×64), launch `render_volumetric_kernel`, read back buffer, assert:
  no panic, buffer non-uniform (some pixels ≠ 0, variance > 0 — proves density
  field + emission + jitter actually run on GPU). `cargo run --bin
  test_volumetric_launch` and report output.
- **D3** Faithfulness pass against TR.md checklist (§2.1–2.5, Steps 1–5):
  density kernel ✓, Beer-Lambert ✓, shadow ray ✓, HG ✓, temporal dither ✓,
  emission ✓, early termination ✓, launch bounds guard ✓, zero-copy (no
  Texture3D, no scene buffer in volumetric mode) ✓. Report deviations D1–D6.

## Risks / fallbacks
- If CubeCL rejects an op (e.g. struct return from `#[cube]` fn): fall back to
  two scalar returns or out-params, keeping TR math identical.
- If GUI wiring (C4) exceeds context: ship CLI/env-var toggle (`render_mode`
  already in ApplicationState) and document — kernel work unaffected.
- If headless test can't init wgpu in this environment: report `cargo build`
  + kernel-compile (WGSL codegen happens at launch; test_kernel_launch pattern
  is the precedent) and state the limitation honestly.
