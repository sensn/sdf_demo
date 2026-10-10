# REFACTOR_PARAMS_PLA — Proper GPU Parameter Passing

Refactor of the kernel parameter path in `src/kernel.rs`, `src/app_state.rs`, `src/gpu.rs`
according to the prescriptions in `doc/params/params1.md` … `params5.md`.

---

## 1. Goal

Replace the current parameter passing — **8 storage-buffer tensors + 14 loose scalars,
all read by index** — with **typed `#[derive(CubeLaunch)]` structs passed directly at
kernel launch**, so that:

- the WebGPU storage-buffer budget drops from 8/8 to **4/8** (params1),
- small parameters travel through the **info uniform** (cached by cubecl-wgpu's
  `MetadataInfoCache` → zero re-upload when unchanged, params4) instead of wasting
  storage-buffer slots,
- every parameter has a **name and a type** instead of a magic index (params2/params3),
- `vec3` parameters are **16-byte aligned** via `GpuVec3 { x, y, z, _pad }` (params5),
- flags/counts are passed as **`u32` (0/1)**, never as f32 (params3).

---

## 2. Current state (what is wrong)

Verified in `src/kernel.rs` / `src/gpu.rs` / `src/app_state.rs`:

| Problem | Evidence |
|---|---|
| **8 storage buffers** — exactly at the WebGPU limit | `GpuPipeline` allocates `output, meta, slots, materials, rotations, env, arch, fold` handles; all bound as `TensorArg` |
| 4 of them are **tiny parameter tensors** (1 / 16 / 12 / 4 f32) | `meta_data() -> [f32;1]`, `env_data() -> [f32;16]`, `arch_data() -> [f32;12]`, `fold_data() -> [f32;4]` — written via `queue.write_buffer` **every frame** |
| **14 loose scalars** at launch | `time, width, height, shadow_mode, cam_x, cam_y, cam_z, blend_factor, enable_ao_mode, cam_yaw, cam_pitch, enable_key, enable_fill, enable_rim` |
| **Index-based reads**, layout is an undocumented ABI | `env_settings[usize::new(0)]`, `arch_params[usize::new(0)]`, `fold_params[usize::new(0)]`, `meta[usize::new(0)]` |
| `fog_enabled` packed as **f32** into the env tensor | `env_data()[12] = self.fog_enabled as f32` — violates params3 (flags = `u32`) |
| **Rotations (Tensor 7) is dead data** | buffer allocated + written every frame (`rotations_data()`), but the tensor is **commented out** in kernel signature and launch (`//rotationss: &Tensor<f32>`) — the TODO from commit `3e4141c` |
| Slot count travels as **f32 in a tensor** | `meta_data() = [active_slots_count: f32]` |

---

## 3. Doc requirements → measures

| Doc | Requirement | Measure in this refactor |
|---|---|---|
| **params1** | WebGPU ≈ 8 storage buffers max → pack params into structs | Storage buffers **8 → 4** (`output`, `slots`, `materials`, `rotations`). Everything small moves into launch structs (info uniform). |
| **params2** | `#[derive(CubeLaunch)]` custom structs, passed **directly** at launch, no `TensorArg` | `CameraSettings`, `EnvSettings`, `ArchParams`, `FoldParams`, `RenderSettings` — kernel takes the structs, host passes `*Launch::new(...)` |
| **params3** | Mix `u32`/floats, 16-byte alignment padding, `bool` → `u32` flag (0/1) | Flags & counts as `u32` fields; cubecl packs scalar struct fields 16-byte-aligned into the info uniform (`padded_size`); `GpuVec3._pad: f32` |
| **params4** | Custom structs for (quasi-)static config; **tensors only for large dynamic data** | env/arch/fold/camera/render → structs; `slots` (800 f32), `materials` (400 f32), `rotations` (300 f32) stay tensors |
| **params5** | `GpuVec3` with `_pad: f32` for 16-byte alignment; camera settings struct | `GpuVec3 { x, y, z, _pad }` + `CameraSettings { position: GpuVec3, yaw, pitch }` |

---

## 4. New parameter ABI (defined in `src/kernel.rs`)

All structs `#[derive(CubeType, CubeLaunch)]`. Scalar fields become **runtime scalars**
in the info uniform (no kernel recompilation on value change; cubecl-wgpu caches the
uniform per content → params4's "constant cache" for free).

```rust
/// 16-Byte-Alignment für vec3-Parameter (params5): vec3 + Padding-Feld.
#[derive(CubeType, CubeLaunch)]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

/// Kamera-Parameter (params5: "camera settings struct").
#[derive(CubeType, CubeLaunch)]
pub struct CameraSettings {
    pub position: GpuVec3,
    pub yaw: f32,
    pub pitch: f32,
}

/// Umwelt-Parameter (ersetzt Tensor 4 / env_data()[16]).
#[derive(CubeType, CubeLaunch)]
pub struct EnvSettings {
    pub key_light_pos: GpuVec3,   // was env[0..2]
    pub light_intensity: f32,     // was env[3]
    pub key_color: GpuVec3,       // was env[4..6]
    pub ambient_strength: f32,    // was env[7]
    pub bg_color: GpuVec3,        // was env[8..10]
    pub fog_density: f32,         // was env[11]
    pub fog_enabled: u32,         // was env[12] (f32) → params3: u32-Flag 0/1
}

/// Tempel-Architektur (ersetzt Tensor 5 / arch_data()[12]).
#[derive(CubeType, CubeLaunch)]
pub struct ArchParams {
    pub pillar_dist: f32,    // was arch[0]
    pub pillar_thick: f32,   // was arch[1]
    pub room_height: f32,    // was arch[2]
    pub ceiling_thick: f32,  // was arch[3]
    pub arch_radius: f32,    // was arch[4]
    pub arch_height: f32,    // was arch[5]
    pub decor_freq: f32,     // was arch[6]
    pub decor_depth: f32,    // was arch[7]
    pub decor_thick: f32,    // was arch[8]
}

/// Unendliche Raumfaltung (ersetzt Tensor 6 / fold_data()[4]).
#[derive(CubeType, CubeLaunch)]
pub struct FoldParams {
    pub cell_size: f32,   // was fold[0]
    pub half_cell: f32,   // was fold[1] (host computed cell_size * 0.5)
    pub fold_speed: f32,  // was fold[2]
}

/// Frame-Parameter & Mode-Flags (ersetzt den Meta-Tensor + 12 loose Skalare).
#[derive(CubeType, CubeLaunch)]
pub struct RenderSettings {
    pub time: f32,
    pub width: u32,
    pub height: u32,
    pub blend_factor: f32,
    pub active_slots: u32,   // ersetzt meta[0] (f32) → params3: Count als u32
    pub shadow_mode: u32,
    pub enable_ao_mode: u32,
    pub enable_key: u32,
    pub enable_fill: u32,
    pub enable_rim: u32,
}
```

Alignment note (params3): cubecl packs scalar struct fields itself into 16-byte-aligned
info-uniform rows (`padded_size` in `cubecl-wgpu/src/compiler/wgsl/metadata.rs`), so the
flat structs need no manual `_pad` fields. `GpuVec3` keeps the explicit `_pad` because a
vec3 in a uniform layout must round up to 16 bytes (params5).

---

## 5. New kernel signature (`src/kernel.rs`)

```rust
#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<f32>,
    slots: &Tensor<f32>,       // 800 f32, Stride 8 — große dynamische Daten (params4)
    materials: &Tensor<f32>,   // 400 f32, Stride 4
    rotations: &Tensor<f32>,   // 300 f32, Stride 3 — 🟢 TENSOR 7 wieder aktiviert
    camera: CameraSettings,
    env: EnvSettings,
    arch: ArchParams,
    fold: FoldParams,
    render: RenderSettings,
)
```

Body changes (mechanical):
- `env_settings[usize::new(i)]` → `env.key_light_pos.x`, `env.light_intensity`,
  `env.key_color.r/g/b` (via `.x/.y/.z` of the `GpuVec3`s), `env.ambient_strength`,
  `env.bg_color.*`, `env.fog_density`, `env.fog_enabled`
- `arch_params[usize::new(i)]` → `arch.pillar_dist` … `arch.decor_thick`
- `fold_params[usize::new(i)]` → `fold.cell_size`, `fold.half_cell`, `fold.fold_speed`
- `meta[usize::new(0)]` → `f32::cast_from(render.active_slots)`
- `cam_x/cam_y/cam_z` → `camera.position.x/y/z`; `cam_yaw/cam_pitch` → `camera.yaw/pitch`
- `time/width/height/blend_factor/shadow_mode/enable_*` → `render.*`
- `fog_enabled` comparisons: `env.fog_enabled > u32::new(0)` instead of f32 compare

## 6. Helper signatures (`scene_sdf` and friends)

`env` is **not** needed in the SDF itself (only lighting/bg use it) — same as today.

```rust
#[cube]
pub fn scene_sdf(
    p: Vec3,
    time: f32,
    blend_factor: f32,
    active_slots: u32,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>,
    rotations: &Tensor<f32>,
    arch: ArchParams,
    fold: FoldParams,
) -> SdfResult
```

`scene_sdf_normal`, `calculate_ao`, `calculate_soft_shadow` get the same parameter list
(they only forward to `scene_sdf`).

**Rotations (Tensor 7)**: per slot `i` read `rot_x = rotations[i*3]`, `rot_y = rotations[i*3+1]`,
`rot_z = rotations[i*3+2]` and rotate the sample point with the **inverse Euler rotation**
before evaluating the primitive SDF (standard SDF transform). This closes the
`todo: fix obj rotations implementation` from commit `3e4141c` — the parameter path now
has a free storage-buffer slot for it.

---

## 7. Host side — `src/app_state.rs`

**Remove** the tensor builders: `meta_data`, `env_data`, `arch_data`, `fold_data`
(only `gpu.rs` called them — verified).

**Keep** the true dynamic-array builders: `slots_data`, `materials_data`, `rotations_data`.

**Add** struct builders (replace the removed ones 1:1):

```rust
use crate::kernel::{
    ArchParamsLaunch, CameraSettingsLaunch, EnvSettingsLaunch, FoldParamsLaunch,
    GpuVec3Launch, RenderSettingsLaunch,
};

pub fn camera_launch(&self) -> CameraSettingsLaunch { /* position, yaw, pitch */ }
pub fn env_launch(&self) -> EnvSettingsLaunch { /* light, colors, fog (u32-Flag) */ }
pub fn arch_launch(&self) -> ArchParamsLaunch { /* 9 Architektur-Parameter */ }
pub fn fold_launch(&self) -> FoldParamsLaunch { /* cell_size, half_cell, fold_speed */ }
pub fn render_launch(&self, time: f32, width: u32, height: u32) -> RenderSettingsLaunch
```

`active_slots_count: f32` stays in the state (GUI binds it); the builder casts to `u32`
(params3 type discipline at the ABI, zero GUI churn).

---

## 8. Host side — `src/gpu.rs`

- **Delete** handles `meta_handle`, `env_handle`, `arch_handle`, `fold_handle`
  (+ their `client.empty(...)` allocations, `ENV_BYTES`/`ARCH_BYTES`/`FOLD_BYTES` consts,
  and the four `TensorArg::from_raw_parts` blocks).
- `write_state_buffers` shrinks to `slots` / `materials` / `rotations` — the only buffers
  whose content actually changes with scene edits.
- Launch becomes:

```rust
crate::kernel::raymarch_sdf_kernel::launch(
    &self.client, cube_count, cube_dim,
    output_arg, slots_arg, materials_arg, rotations_arg,
    s.camera_launch(), s.env_launch(), s.arch_launch(), s.fold_launch(),
    s.render_launch(time, dw, dh),
);
```

---

## 9. Buffer budget

| Buffer | Before | After |
|---|---|---|
| output (RGB f32) | storage 0 | storage 0 |
| slots (800 f32) | storage 1 | storage 1 |
| materials (400 f32) | storage 2 | storage 2 |
| rotations (300 f32) | *(allocated, not bound)* | storage 3 ✅ |
| meta (1 f32) | storage 3 | **info uniform** |
| env (16 f32) | storage 4 | **info uniform** |
| arch (12 f32) | storage 5 | **info uniform** |
| fold (4 f32) | storage 6 | **info uniform** |
| 14 loose scalars | info uniform (uncached grouping) | **info uniform via structs** |
| **Storage buffers total** | **8 / 8** ⚠️ | **4 / 8** ✅ |

Per-frame `queue.write_buffer` calls for params: **4 → 0** (uniform is cached & managed
by cubecl; only slots/materials/rotations still written when the scene changes).

---

## 10. Migration steps

1. `src/kernel.rs`: add the six param structs (§4).
2. `src/kernel.rs`: rewrite `raymarch_sdf_kernel` signature + body (§5).
3. `src/kernel.rs`: rewrite `scene_sdf`, `scene_sdf_normal`, `calculate_ao`,
   `calculate_soft_shadow` (§6) + wire rotations into the slot evaluator.
4. `src/app_state.rs`: replace `meta_data/env_data/arch_data/fold_data` with the
   `*_launch()` builders (§7).
5. `src/gpu.rs`: drop the four param handles/consts/TensorArgs, new launch call (§8).
6. `cargo check` → fix → `cargo clippy` → done. (GUI untouched: it reads/writes
   `ApplicationState` fields, which are unchanged.)

## 11. Risks & fallbacks

| Risk | Fallback |
|---|---|
| Nested `CubeLaunch` struct (`GpuVec3` inside `CameraSettings`/`EnvSettings`) unsupported | Flatten to `pos_x/pos_y/pos_z` scalar fields (keep `_pad` semantics) |
| Custom `CubeType` args in plain `#[cube]` helpers (`arch: ArchParams`) unsupported | Pass `&ArchParams`, or extract scalars at call sites |
| Generated `*Launch` types not nameable outside `kernel.rs` | Build the structs inside `gpu.rs` instead of `app_state.rs` |
| Rotation math visual regression | Standard inverse-Euler transform; verify visually, gate behind existing slot data (all zeros by default → identity) |

## 12. Verification

- `cargo check` / `cargo clippy` clean.
- Launch still `CubeCount::Static(w/16, h/4, 1)`, `CubeDim(16,4,1)` — unchanged.
- Runtime smoke test: scene renders identically with default state (5 slots, fog off,
  rotations all zero → identity transform).
