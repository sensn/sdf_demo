---
name: sdf-kernel-architecture
description: Describes the complete CPU/GPU architecture of the SDF demo (CubeCL kernel, wgpu pipeline, ApplicationState packing, render loop) enabling agents to add new kernel features, host parameters, and wire them end-to-end.
---

# SDF Demo Kernel Architecture — Agent Reference

This skill documents the full **host↔GPU data flow**, **kernel module structure**, **tensor ABI**, and **render loop** so a coding agent can **add new shader features, host-side parameters, buffers, and GUI controls** correctly and safely.

---

## 1. Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MAIN THREAD (wgpui)                        │
│  SurfaceExample ── GuiState ── ApplicationState (Arc<Mutex<…>>)    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Arc<Mutex<ApplicationState>>
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RENDER THREAD (std::thread)                   │
│  run_render_loop()                                                 │
│   ├─ input_manager.update_camera_movement(&mut state, dt)         │
│   ├─ pipeline.write_state_buffers(&state)  ──► GPU buffers         │
│   ├─ pipeline.launch_kernel(…)              ──► CubeCL dispatch    │
│   └─ surface.present()                                             │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principle:** All mutable render parameters live in **one flat `ApplicationState` struct**. The render thread snapshots it under the mutex, writes GPU buffers, then releases the lock before the expensive GPU dispatch.

---

## 2. ApplicationState — The Single Source of Truth

`src/app_state.rs` — a flat `pub` struct with **every** tunable parameter.

### Categories
| Prefix | Domain | Example |
|---|---|---|
| `cam_*` | Camera position/yaw/pitch | `cam_x`, `cam_yaw` |
| `w_pressed`…`e_pressed` | Input flags | |
| `light_*`, `ambient_*`, `fog_*` | Environment (Tensor 4) | `key_light_x`, `fog_density` |
| `pillar_*`, `room_*`, `arch_*`, `decor_*` | Architecture (Tensor 5) | `pillar_dist`, `arch_radius` |
| `cell_size`, `fold_speed`, `pillar_spacing`, `pillar_height` | Fold (Tensor 6) | |
| `slot_*` (8 vectors) | Dynamic objects (Crystal/Gyroid/Torus) | `slot_types`, `slot_offsets_x`, `slot_roughness`… |
| `sdf_*` (14 vectors) | **NEW** SDF grid objects | `sdf_paths`, `sdf_center_x`, `sdf_metallic`… |
| `reflection_*`, `pillar_roughness` | Material/Reflection | |
| `mesh_scale_factor`, `bounding_sphere_radius_offset` | Global bake/settings | |

### Packing Methods (CPU → GPU Registers)
Each `*_launch()` method returns a **CubeLaunch** struct matching the kernel’s `paramsN` layout.

| Method | Kernel Params | Returns |
|---|---|---|
| `camera_launch()` | `params5` (camera) | `CameraSettingsLaunch<GpuFloat>` |
| `env_launch()` | `params4` (env/lights/fog/AO/shadow) | `EnvSettingsLaunch<GpuFloat>` |
| `arch_launch()` | `params?` (architecture) | `ArchParamsLaunch<GpuFloat>` |
| `fold_launch()` | `params?` (fold) | `FoldParamsLaunch<GpuFloat>` |
| `render_launch(time, w, h)` | `params3` (meta + modes + per-frame) | `RenderSettingsLaunch<GpuFloat>` |
| `scene_data()` | **Mega-Scene Tensor** (slots/materials/rotations) | `Vec<GpuFloat>` (1500 f32) |
| `sdf_objects_data()` | **SDF-Object Tensor** (N headers + N grids) | `Vec<GpuFloat>` (~1 M f32) |

**Alignment rules** (WebGPU std140 / CubeCL `CubeLaunch`):
- Every `vec3` → `GpuVec3 { x, y, z, _pad }` (16 bytes).
- Struct fields ordered **exactly** as in `kernel/types.rs` launch structs.
- `GpuFloat = f32` (centralized in `gpu.rs` for future `f16` switch).

---

## 3. GPU Pipeline (`src/gpu.rs`)

### Constants (tensor layout)
```rust
pub const MAX_SLOTS: usize = 100;
pub const SLOTS_COUNT: usize = MAX_SLOTS * 8;        // 800
pub const MATERIALS_COUNT: usize = MAX_SLOTS * 4;    // 400
pub const ROTATIONS_COUNT: usize = MAX_SLOTS * 3;    // 300
pub const TOTAL_BUFFER_SIZE: usize = 1500;           // scene buffer

pub const MAX_SDF_OBJECTS: usize = 4;
pub const SDF_HEADER_STRIDE: usize = 24;
pub const SDF_GRID_RES: usize = 64;
pub const SDF_GRID_COUNT: usize = 64*64*64;        // 262144
pub const SDF_OBJECTS_FLOATS: usize = MAX_SDF_OBJECTS*SDF_HEADER_STRIDE + MAX_SDF_OBJECTS*SDF_GRID_COUNT; // 1_048_672
```

### Handles
```rust
pub struct GpuPipeline {
    pub scene_handle: Handle,         // 1500 f32
    pub sdf_objects_handle: Handle,   // 1_048_672 f32
    pub output_handle: Handle,        // render target
    pub bind_group: Option<BindGroup>, // blit pipeline
    …
}
```

### Initialization
```rust
let scene_handle = client.empty(TOTAL_SCENE_BYTES);       // 6 KB
let sdf_objects_handle = client.empty(SDF_OBJECTS_BYTES); // ~4 MB
```

### Per-Frame Write (`write_state_buffers`)
```rust
fn write_state_buffers(&self, s: &ApplicationState) {
    let packed_scene = s.scene_data();        // 1500 f32
    queue.write_buffer(&self.scene_handle, 0, bytemuck::cast_slice(&packed_scene));

    let packed_objects = s.sdf_objects_data(); // 1 M f32
    queue.write_buffer(&self.sdf_objects_handle, 0, bytemuck::cast_slice(&packed_objects));
}
```

### Kernel Launch (`launch_kernel`)
```rust
let scene_arg = TensorArg::from_raw_parts(scene_handle, [TOTAL_SCENE_FLOATS], [1]);
let objects_arg = TensorArg::from_raw_parts(sdf_objects_handle, [SDF_OBJECTS_FLOATS], [1]);

raymarch_sdf_kernel::launch::<GpuFloat>(
    &client, cube_count, cube_dim,
    output_arg, scene_arg, objects_arg,
    s.camera_launch(), s.env_launch(), s.arch_launch(), s.fold_launch(),
    s.render_launch(time, dw, dh),
);
```
**Two tensor arguments** → kernel signature `raymarch_sdf_kernel(output, scene, objects, camera, env, arch, fold, render)`.

---

## 4. Kernel Module Structure (`src/kernel/`)

| File | Responsibility |
|---|---|
| `types.rs` | `Vec3/Vec2/SdfResult/GpuVec3`, `smin/smin_material`, **all Launch structs** (ABI) |
| `objects.rs` | Dynamic primitives: `evaluate_dynamic_crystal/gyroid/torus` |
| `scene.rs` | `scene_sdf` (slots + temple architecture), `scene_sdf_normal` |
| `statue_grid.rs` | **SDF-Object Tensor**: `sample_sdf_grid`, `scene_map_with_material`, `sdf_object_pbr`, `calculate_normal` |
| `fog.rs` | `integrate_volumetric_fog` |
| `reflection.rs` | `trace_reflection` (1-bounce) |
| `shadows.rs` | `calculate_soft_shadow` |
| `ao.rs` | `calculate_ao` |
| `pbr.rs` | Cook-Torrance (`fresnel_schlick`, `d_ggx`, `g_smith`, `cook_torrance_light`, `aces_tonemap`) |
| `main_kernel.rs` | `raymarch_sdf_kernel` — per-pixel entry point |
| `mod.rs` | Re-exports everything |

### Adding a New Kernel Feature (Checklist)
1. **Host parameter** → add field to `ApplicationState`.
2. **Packing** → extend the appropriate `*_launch()` method.
3. **Launch struct** → add field in `kernel/types.rs` (match order, pad `vec3`).
3b. If it’s a **buffer/texture** → add handle + constant in `gpu.rs`, write in `write_state_buffers`, add tensor arg in `launch_kernel`.
4. **Kernel side** → add parameter to the relevant function signature, thread through callers.
5. **Use in `main_kernel.rs`** or new helper module.
6. **GUI** → add slider entity + subscription + sync + tab row (see GUI skill).

---

## 5. Tensor ABI — The Contract

### Mega-Scene Tensor (`scene` tensor, 1500 f32)
```
[0..800)       SLOTS:       100 × [type, size, pos_x, pos_y, pos_z, r, g, b]      (stride 8)
[800..1200)    MATERIALS:   100 × [roughness, metallic, emissive, specular]         (stride 4)
[1200..1500)   ROTATIONS:   100 × [rot_x, rot_y, rot_z]                             (stride 3)
```
Accessed in `scene.rs` via hard-coded offsets (`SLOTS_COUNT`, `MATERIALS_COUNT`).

### SDF-Object Tensor (`objects` tensor, ~1 M f32)
```
[0..96)        HEADERS:  4 × 24 f32
    per header (stride 24):
      0  active (0/1)
      1  res   (64.0)
      2  size  (world extent)
      3  pad
      4  center_x  5 center_y  6 center_z  7 pad
      8  rot_x  9 rot_y  10 rot_z  11 pad
      12 r  13 g  14 b  15 roughness
      16 metallic  17 emissive  18 specular  19..23 pad
[96..96+262144)   GRID 0 (64³)
[96+262144..)     GRID 1 …
…
```
All grids have fixed resolution `SDF_GRID_RES = 64`.

### Launch Structs (kernel/types.rs) — Must Match Exactly
```rust
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct RenderSettings<F: Float> {
    pub time: F, pub width: u32, pub height: u32,
    pub blend_factor: F, pub active_slots: u32,
    pub shadow_mode: u32, pub enable_ao_mode: u32,
    pub enable_key: u32, pub enable_fill: u32, pub enable_rim: u32,
    pub reflection_max_steps: u32, pub reflection_roughness_gate: F,
    pub fresnel_strength_factor: F, pub pillar_roughness: F,
    pub bounding_sphere_radius_offset: F,
}
```
**Any field added/removed/reordered here must be mirrored in `ApplicationState::render_launch()`**.

---

## 6. Render Loop (`src/render_loop.rs`)

```rust
loop {
    surface.wait_for_present();                    // frame pacing
    let Some((view, (dw, dh))) = surface.back_view_with_size() else { sleep; continue };
    if resized { pipeline.resize(dw, dh); }

    let mut s = state.lock().unwrap();
    input_manager.update_camera_movement(&mut s, dt);
    pipeline.write_state_buffers(&s);              // upload both tensors
    drop(s);                                       // release lock early!

    let s_render = state.lock().unwrap();
    pipeline.render_frame(&view, dw, dh, &s_render, time);
    drop(s_render);

    surface.present();
    // FPS reporting…
}
```
- **Double lock**: first for input+upload, second (short) for render params.
- `pipeline.render_frame` calls `launch_kernel` + blit to surface.

---

## 7. Input (`src/input.rs`)

`InputManager` polls `evdev` (Linux) for keyboard/gamepad, updates `w_pressed`…`e_pressed` flags.  
`update_camera_movement(&mut ApplicationState, dt)` applies WASD/QE + yaw/pitch.

---

## 8. SDF Preprocessor (`src/sdf_preprocessor.rs`)

- `load_obj_to_sdf(path, res, mesh_scale_factor)` → `Vec<f32>` (64³).
- `save_sdf_cache / load_sdf_cache` — raw binary `.sdfcache` (magic `SDF3` + len + f32 data).
- `load_or_bake_sdf(path, scale)` — cache or bake.
- **NEW** `get_or_bake_grid(path, scale) -> Arc<Vec<f32>>` — process-wide `OnceLock<Mutex<HashMap<(String,u32), Arc<Vec<f32>>>>>` keyed by `(path, scale.to_bits())`. Multiple slots sharing the same OBJ reuse the same baked grid.

---

## 9. Adding a Brand-New Feature — End-to-End Example

> **Goal:** Add a “global color grade” (lift/gamma/gain) controlled by three sliders.

### Step 1 — Host Parameter
```rust
// app_state.rs
pub color_grade_lift: f32,
pub color_grade_gamma: f32,
pub color_grade_gain: f32,
```
Default in `Default::impl`: `1.0, 1.0, 1.0`.

### Step 2 — Packing
```rust
// app_state.rs in render_launch()
color_grade_lift: self.color_grade_lift as GpuFloat,
color_grade_gamma: self.color_grade_gamma as GpuFloat,
color_grade_gain: self.color_grade_gain as GpuFloat,
```

### Step 3 — Launch Struct
```rust
// kernel/types.rs
pub struct RenderSettings<F: Float> {
    // … existing fields …
    pub color_grade_lift: F,
    pub color_grade_gamma: F,
    pub color_grade_gain: F,
}
```

### Step 4 — Kernel Usage
```rust
// main_kernel.rs (inside pixel loop, after tonemap)
let lift = render.color_grade_lift;
let gamma = render.color_grade_gamma;
let gain = render.color_grade_gain;
let final_col = (col * gain + lift).powf(gamma); // simplified
output[…] = final_col;
```

### Step 5 — GUI
```rust
// gui/mod.rs GuiState fields + new() + sync_sliders()
// gui/tabs/env.rs (or new tab) add three slider_row calls
```

### Step 6 — Test
`cargo check` → `cargo run --bin test_kernel_launch` → run app, tweak sliders.

---

## 10. Common Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Shader validation error / misaligned buffer | Launch struct field order ≠ `ApplicationState::*_launch()` | Keep them in **exact same order**; pad `vec3` with `_pad`. |
| New slider does nothing | Missing `subscribe_slider` or `sync_one` | Add both in `GuiState::new()` and `sync_sliders()`. |
| Per-slot slider shows stale value | `sync_sliders` reads wrong index | Use `current_selected_slot` / `current_selected_sdf_object` every frame. |
| Kernel panics “index out of bounds” | Tensor size constant wrong | Update `gpu.rs` constants (`SDF_OBJECTS_FLOATS` etc.). |
| File dialog freezes UI | Forgot `cx.spawn(async …).detach()` | Always spawn background task for `rfd`. |
| `cargo check` OK but headless test fails | WGSL codegen error (e.g. unsupported op) | Run `cargo run --bin test_kernel_launch` to catch CubeCL compile errors. |

---

## 11. File Map for Quick Navigation

| Path | What to Touch For… |
|---|---|
| `src/app_state.rs` | New host parameters, packing methods |
| `src/gpu.rs` | New buffers, tensor constants, `write_state_buffers`, `launch_kernel` |
| `src/kernel/types.rs` | Launch structs (ABI), `GpuFloat` alias |
| `src/kernel/main_kernel.rs` | Main pixel loop, wiring helpers |
| `src/kernel/*.rs` | Feature modules (shadows, AO, fog, etc.) |
| `src/render_loop.rs` | Frame pacing, double-lock pattern |
| `src/input.rs` | Camera/keyboard handling |
| `src/sdf_preprocessor.rs` | Mesh→SDF baking, cache |
| `src/gui/mod.rs` | Slider entities, subscriptions, sync |
| `src/gui/tabs/*.rs` | Tab UI, slider rows |
| `src/gui/sidebar.rs` | TabBar, tab dispatch |
| `src/gui/menubar.rs` | Native menus, actions |
| `src/gui/widgets.rs` | Reusable row helpers |

---

## 12. Version / Precision Switch

`gpu.rs`:
```rust
pub type GpuFloat = f32;  // single place to swap to f16 later
```
All launch structs and packing use `GpuFloat`. **Never hard-code `f32` in kernel signatures** — use the generic `F: Float`.

---

**Follow this architecture and the GUI skill, and every new feature will integrate cleanly, compile, and run.**