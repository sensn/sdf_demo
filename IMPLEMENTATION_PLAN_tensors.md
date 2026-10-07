# Implementation Plan: Environmental Control Tensor + Data-Driven Architecture & Space-Folding

Blueprint sources:
- `doc/Environmental Control Tensor.md` (Tensor 4: `env_settings`)
- `doc/Data driven arch_params and infinte Modulo grid fold_params.md` (Tensor 5: `arch_params`, Tensor 6: `fold_params`)

Goal: Make every constant in the raymarching scene **data-driven** — streamed to the GPU via
persistent tensor handles + `client.write` — and control it interactively via **wgpui key
handling** (same `track_focus` pattern as the WASD fix) and **wgpui-kit GUI elements**
(Sliders/Switch in the sidebar).

---

## Tensor Layouts (final)

### Tensor 4 — `env_settings` (16 f32 = 4 × Vec4 registers)
| Index | Variable        | Default        | Function                          |
|-------|-----------------|----------------|-----------------------------------|
| 0..2  | key light pos   | (4.0, 7.0, -4.0)| Key-light position (Vec3)         |
| 3     | `l_intensity`   | 1.2            | Key-light intensity              |
| 4..6  | key RGB         | (1.0, 0.95, 0.85)| Key-light color                 |
| 7     | `a_strength`    | 0.25           | Ambient strength                 |
| 8..10 | bg/fog RGB      | (0.35, 0.45, 0.60)| Background & fog color         |
| 11    | `fog_density`   | 0.5            | Fog compression divisor          |
| 12    | `fog_enabled`   | 1.0            | Fog on/off (1.0 = on)            |
| 13..15| padding         | 0.0            | Alignment                        |

**Replaces** the kernel scalars `light_intensity: f32` and `ambient_strength: f32`
(both are REMOVED from the launch signature).

### Tensor 5 — `arch_params` (12 f32 = 3 × Vec4 registers)
| Index | Variable        | Default | Replaces hardcoded |
|-------|-----------------|---------|--------------------|
| 0     | `pillar_dist`   | 5.0     | `5.0`              |
| 1     | `pillar_thick`  | 0.6     | `0.6`              |
| 2     | `room_height`   | 3.0     | `3.0`              |
| 3     | `ceiling_thick` | 0.1     | `0.1`              |
| 4     | `arch_radius`   | 3.2     | `3.2`              |
| 5     | `arch_height`   | 1.0     | `1.0`              |
| 6     | `decor_freq`    | 2.0     | `2.0`              |
| 7     | `decor_depth`  | 0.03    | `0.03`             |
| 8     | `decor_thick`   | 0.01    | `0.01`             |
| 9..11 | padding         | 0.0     | —                  |

### Tensor 6 — `fold_params` (4 f32 = 1 × Vec4 register)
| Index | Variable    | Default | Replaces hardcoded |
|-------|-------------|---------|--------------------|
| 0     | `cell_size` | 10.0    | `10.0` (modulo grid)|
| 1     | `cell_half` | 5.0     | `5.0` (= cell_size * 0.5, keep in sync!) |
| 2     | `fold_speed`| 1.0     | time scale for object animation |
| 3     | padding     | 0.0     | —                  |

---

## Part 1 — Host side (`src/main.rs`)

### 1.1 `ApplicationState` extensions
Add three payload arrays + continuous-adjust key flags:

```rust
// Tensor 4: environment
pub env_payload: Vec<f32>,   // 16 floats, layout as above
// Tensor 5: architecture
pub arch_payload: Vec<f32>,  // 12 floats
// Tensor 6: space folding
pub fold_payload: Vec<f32>,  // 4 floats

// hold-to-adjust flags (set by on_key_down / cleared by on_key_up)
pub light_up: bool, pub light_down: bool,       // Q / E
pub ambient_up: bool, pub ambient_down: bool,  // F / R
pub fog_up: bool, pub fog_down: bool,          // O / L
pub temple_narrow: bool, pub temple_wide: bool,
pub space_shrink: bool, pub space_grow: bool,
pub fold_slower: bool, pub fold_faster: bool,
```

Remove the old scalar fields `light_intensity` / `ambient_strength` (now in `env_payload`).

### 1.2 Persistent tensor handles (render thread setup)
```rust
let env_handle      = client.create(cubecl::bytes::Bytes::from_elems(state.env_payload.clone()));
let arch_param_handle = client.create(cubecl::bytes::Bytes::from_elems(state.arch_payload.clone()));
let fold_param_handle = client.create(cubecl::bytes::Bytes::from_elems(state.fold_payload.clone()));
```

### 1.3 Streaming in the render loop
Each frame, apply held-key deltas to the payloads, then stream changed tensors:
```rust
if env_changed   { client.write(&env_handle,   Bytes::from_elems(s.env_payload.clone())); }
if arch_changed  { client.write(&arch_param_handle, Bytes::from_elems(s.arch_payload.clone())); }
if fold_changed  { client.write(&fold_param_handle, Bytes::from_elems(s.fold_payload.clone())); }
```
Clamps: intensity 0..3, ambient 0..1, fog_density 0..2, pillar_dist 2..15,
cell_size 4..30 (with `cell_half = cell_size * 0.5` kept in sync), fold_speed 0..5.

### 1.4 Key handling — wgpui (identical pattern to WASD fix)
Same focused-root-div `on_key_down` / `on_key_up` listeners that already work for WASD:

| Key(s)      | Action                                   | Mode   |
|-------------|------------------------------------------|--------|
| Q / E       | light intensity − / +                    | hold   |
| F / R       | ambient strength − / +                   | hold   |
| O / L       | fog density − / +                        | hold   |
| G           | toggle fog_enabled                       | tap (`!is_held`) |
| T / Z       | temple narrow / wide (pillar_dist)      | hold   |
| U / J       | room height − / +                        | hold   |
| C / V       | cell_size shrink / grow                  | hold   |
| N / M       | fold_speed − / +                         | hold   |

> NOTE: no Numpad-only bindings — on Linux/winit, numpad digits produce the same
> keystroke strings as top-row digits (verified in wgpui `winit_key_to_keystroke`),
> which would collide with the existing 1/2/3/4 toggles.

Key-up clears the flags; window-deactivation clears all flags (existing stuck-key guard).

### 1.5 GUI sidebar — wgpui-kit components
Extend the existing sidebar with a "Tensors" section:

- **Sliders** (wgpui-kit `Slider` bound to `Entity<SliderState>`):
  - Light Intensity (0..3, step 0.05)
  - Ambient (0..1, step 0.02)
  - Fog Density (0..2, step 0.05)
  - Pillar Distance (2..15, step 0.1)
  - Room Height (1..6, step 0.1)
  - Cell Size (4..30, step 0.1)
  - Fold Speed (0..5, step 0.1)
- **Switch** (wgpui-kit `Switch`): Fog on/off.

Wiring: create each `Entity<SliderState>` in the view constructor with
`SliderState::new().min(..).max(..).step(..).default_value(..)`; subscribe with
`cx.subscribe(&entity, |this, _, event: &SliderEvent, cx| ...)` (`SliderState` is an
`EventEmitter<SliderEvent>`; `SliderEvent::Change(v)` / `Release(v)`), and write the
value into `ApplicationState` (the render thread streams it). `Switch::new(id)
.checked(..).on_change(|checked, _, _| ...)` for the fog toggle.

Sliders/Switch steal focus on interaction → the existing per-frame
`window.focus(&self.focus_handle)` re-claim (from the WASD fix) keeps keyboard nav alive.

### 1.6 Launch call update
```rust
TensorArg::from_raw_parts(env_handle.clone(), vec![16], vec![]),
TensorArg::from_raw_parts(arch_param_handle.clone(), vec![12], vec![]),
TensorArg::from_raw_parts(fold_param_handle.clone(), vec![4], vec![]),
// scalars light_intensity / ambient_strength REMOVED
```

---

## Part 2 — GPU kernel (`src/kernel.rs`)

### 2.1 Signature cascade (thread the tensors through)
```
raymarch_sdf_kernel(output, meta, slots, materials,
                    env_settings,      // NEW (5th arg)
                    arch_params,       // NEW
                    fold_params,       // NEW
                    time, width, height, shadow_mode, cam_*, blend_factor,
                    enable_ao_mode, cam_yaw, cam_pitch,
                    enable_key, enable_fill, enable_rim)   // light_intensity/ambient_strength REMOVED
scene_sdf(p, time, blend_factor, meta, slots, materials, arch_params, fold_params)
scene_sdf_normal(...)        // + arch_params, fold_params → passes to scene_sdf
calculate_soft_shadow(...)    // + arch_params, fold_params → passes to scene_sdf
calculate_ao(...)             // + arch_params, fold_params → passes to scene_sdf
```
Per the blueprint: `env_settings` is NOT passed into `scene_sdf` — it is only used in
the main shading stage of `raymarch_sdf_kernel`.

### 2.2 Environment extraction (top of kernel, after bounds check)
```rust
let key_light_pos = Vec3::new(env_settings[0], env_settings[1], env_settings[2]);
let l_intensity   = env_settings[3];
let key_r = env_settings[4]; let key_g = env_settings[5]; let key_b = env_settings[6];
let a_strength    = env_settings[7];
let bg_r = env_settings[8]; let bg_g = env_settings[9]; let bg_b = env_settings[10];
let fog_density   = env_settings[11];
let fog_enabled   = env_settings[12];
```
All downstream uses of the old scalars switch to these.

### 2.3 Dynamic fog compositing (after lighting, before pixel pack)
```rust
let mut fog = f32::new(1.0);
if fog_enabled == f32::new(1.0) {
    fog = (f32::new(40.0) - hit_dist) / (f32::new(40.0) - f32::new(15.0) * fog_density);
    if fog > f32::new(1.0) { fog = f32::new(1.0); }
    if fog < f32::new(0.0) { fog = f32::new(0.0); }
}
final_x = final_x * fog + bg_r * (f32::new(1.0) - fog);  // same for y/z
```

### 2.4 `scene_sdf` — space folding (Tensor 6)
```rust
let cell_size  = fold_params[0];
let half_cell  = fold_params[1];
let fold_speed = fold_params[2];
let animated_time = time * fold_speed;
```
- Replace every hardcoded `10.0` / `5.0` modulo-grid constant with `cell_size` / `half_cell`
  (both the object-slot grid AND the architecture grid).
- Pass `animated_time` into `evaluate_dynamic_crystal/gyroid/torus` instead of raw `time`.

### 2.5 `scene_sdf` — temple architecture (Tensor 5)
```rust
let p_dist   = arch_params[0]; let p_thick  = arch_params[1];
let r_height = arch_params[2]; let c_thick  = arch_params[3];
let a_radius = arch_params[4]; let a_height = arch_params[5];
let d_freq   = arch_params[6]; let d_depth  = arch_params[7];
let d_thick  = arch_params[8];
```
Replace the hardcoded literals in the pillar/room/arch/decor SDF math with these registers.

---

## Part 3 — Build & verify

1. `cargo check` — fix compile errors.
2. `cargo run --release` — verify:
   - WASD still works (regression check on the focus fix)
   - Q/E/F/R/O/L morph lighting live; G toggles fog
   - T/Z/U/J morph the temple; C/V/N/M morph the infinite grid
   - Sidebar sliders/switch mirror the same values (two-way: keys move sliders on next frame)
//3. Clear shader cache if artifacts appear: `rm -rf ~/.cache/cubecl*`.

## Execution order
1. `src/kernel.rs` — signatures + env/fog + arch/fold (Part 2)
2. `src/main.rs` — state payloads, tensor handles, streaming, launch args (Part 1.1–1.3, 1.6)
3. `src/main.rs` — key handling (Part 1.4)
4. `src/main.rs` — GUI sidebar (Part 1.5)
5. Build + verify (Part 3)
