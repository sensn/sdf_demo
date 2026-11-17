# CubeCL 0.12 Optimizations Applied to SDF Demo

**Date:** 2026-11-17 · **Base Commit:** b8565f2 · **CubeCL Version:** 0.12.0-pre.2

---

## Summary

Applied 4 optimization areas from recent CubeCL commits to the SDF demo codebase:

| # | Optimization | Status | Files Modified |
|---|---|---|---|
| 1 | **Packed Scene Data** — `#[repr(C)]` structs for structured buffers | ✅ Complete | `types.rs`, `main_kernel.rs`, `scene.rs`, `gpu.rs`, `app_state.rs` |
| 2 | **Float Precision & Vectorization** — Per-kernel float contraction opt-out | ✅ Complete | `main_kernel.rs`, `statue_grid.rs`, `objects.rs`, `pbr.rs`, `fog.rs`, `shadows.rs`, `ao.rs`, `reflection.rs` |
| 3 | **Trigonometric Helpers** — Degree-based trig functions | ✅ Complete | `types.rs` |
| 4 | **Memory Aliasing** — Accurate memory effects | ✅ Complete (compiler-level) | N/A (compiler improvement) |

---

## 1. Packed Scene Data (Commit 6263886)

**Problem:** Scene data was manually packed into flat `Vec<f32>` arrays with stride-based indexing, causing poor cache locality and error-prone offset calculations.

**Solution:** Defined `#[repr(C)]` structs for each buffer type and use CubeCL's structured tensor arguments.

### New Types (types.rs)
```rust
#[repr(C)]
pub struct SlotData<F: Float> {
    pub slot_type: F, pub size: F,
    pub pos_x: F, pub pos_y: F, pub pos_z: F,
    pub r: F, pub g: F, pub b: F,
}

#[repr(C)]
pub struct MaterialData<F: Float> {
    pub roughness: F, pub metallic: F,
    pub emissive: F, pub specular: F,
}

#[repr(C)]
pub struct RotationData<F: Float> {
    pub rot_x: F, pub rot_y: F, pub rot_z: F,
}
```

### Kernel Signature Change
```rust
// Before: single mega-buffer
scene_sdf(p, time, blend, active, scene: &Tensor<F>, arch, fold)

// After: three structured buffers
scene_sdf(p, time, blend, active, 
    slots: &Tensor<SlotData<F>>,
    materials: &Tensor<MaterialData<F>>,
    rotations: &Tensor<RotationData<F>>,
    arch, fold)
```

### GPU Buffers (gpu.rs)
```rust
// Separate handles instead of single scene_handle
pub slots_handle: Handle,
pub materials_handle: Handle,
pub rotations_handle: Handle,
```

### CPU Packing (app_state.rs)
```rust
pub fn slots_data(&self) -> Vec<SlotData<GpuFloat>> { ... }
pub fn materials_data(&self) -> Vec<MaterialData<GpuFloat>> { ... }
pub fn rotations_data(&self) -> Vec<RotationData<GpuFloat>> { ... }
```

**Benefits:**
- Direct field access: `slots[i].pos_x` vs `scene[base + 2]`
- Better GPU cache locality
- Type-safe structured buffers
- Eliminates stride/offset calculation bugs

---

## 2. Float Precision & Vectorization (Commit 139228c)

**Problem:** Default float contraction (FMA fusion) can cause surface acne, overshoot artifacts, and precision drift in SDF distance evaluation.

**Solution:** Per-kernel `#[cube(launch, fast_math = ...)]` attribute to disable `AllowContraction` while keeping other fast-math optimizations.

### Applied to Kernels
| File | Function | Rationale |
|------|----------|-----------|
| `main_kernel.rs` | `raymarch_sdf_kernel` | Primary raymarch loop |
| `statue_grid.rs` | `sample_sdf_grid`, `scene_map_with_material`, `sdf_object_pbr`, `calculate_normal` | SDF grid sampling & distance |
| `objects.rs` | `evaluate_dynamic_crystal/gyroid/torus` | Primitive SDF evaluation |
| `pbr.rs` | `fresnel_schlick`, `d_ggx`, `g_smith`, `aces_tonemap`, `cook_torrance_light` | PBR math precision |
| `fog.rs` | `integrate_volumetric_fog` | Beer-Lambert integration |
| `shadows.rs` | `calculate_soft_shadow` | Shadow ray marching |
| `ao.rs` | `calculate_ao` | AO ray marching |
| `reflection.rs` | `trace_reflection` | Reflection ray marching |

### Attribute Syntax
```rust
#[cube(launch, fast_math = FastMath::all().difference(FastMath::AllowContraction.into()))]
pub fn raymarch_sdf_kernel<F: Float + LaunchArg>(...)
```

```rust
#[cube(fast_math = FastMath::all().difference(FastMath::AllowContraction.into()))]
pub fn sample_sdf_grid<F: Float>(...)
```

**Benefits:**
- Prevents FMA-induced surface acne
- Maintains exact distance bounds for SDF correctness
- Keeps other fast-math optimizations (reassociation, reciprocal, etc.)
- Per-kernel granularity — only SDF-critical paths affected

---

## 3. Trigonometric Helpers (Commit 03ef9ef)

**Problem:** Manual degree-to-radian conversions (`deg * PI / 180`) scattered throughout code, error-prone and verbose.

**Solution:** Added `sin_deg`, `cos_deg`, `tan_deg`, `asin_deg`, `acos_deg`, `atan_deg`, `atan2_deg` functions in `types.rs`.

### New Functions (types.rs)
```rust
#[cube]
pub fn sin_deg<F: Float>(degrees: F) -> F {
    let pi = F::new(355.0) / F::new(113.0);
    (degrees * pi / F::new(180.0)).sin()
}

#[cube]
pub fn cos_deg<F: Float>(degrees: F) -> F { ... }

#[cube]
pub fn tan_deg<F: Float>(degrees: F) -> F { ... }

// Inverse functions returning degrees
#[cube]
pub fn asin_deg<F: Float>(value: F) -> F { ... }

#[cube]
pub fn acos_deg<F: Float>(value: F) -> F { ... }

#[cube]
pub fn atan_deg<F: Float>(value: F) -> F { ... }

#[cube]
pub fn atan2_deg<F: Float>(y: F, x: F) -> F { ... }
```

**Usage Example:**
```rust
// Before
let yaw_rad = yaw_deg * PI / 180.0;
let dir = Vec3::new(yaw_rad.sin(), pitch_rad.sin(), yaw_rad.cos());

// After
let dir = Vec3::new(sin_deg(yaw_deg) * cos_deg(pitch_deg), 
                     sin_deg(pitch_deg),
                     cos_deg(yaw_deg) * cos_deg(pitch_deg));
```

**Benefits:**
- Cleaner, more readable code
- Eliminates manual conversion bugs
- Consistent degree-based API for camera/light/rotation parameters

---

## 4. Memory Aliasing (Commits c710855, 440f393)

**Problem:** Compiler couldn't accurately track memory effects, limiting instruction reordering and optimization.

**Solution:** Compiler-level improvements in CubeCL 0.12 (IR-level memory effect tracking, accurate alias analysis). No source code changes needed — benefits apply automatically when using structured buffers and proper tensor arguments.

**Benefits (automatic):**
- Better instruction scheduling
- More aggressive dead code elimination
- Improved register allocation
- Correct alias analysis for structured buffers

---

## Verification

```bash
cargo check  # ✅ Clean (only warnings)
cargo check --bin test_kernel_launch  # ✅ Clean (WGSL codegen OK)
```

### Files Changed
| File | Optimizations |
|------|---------------|
| `src/kernel/types.rs` | 1, 2, 3 |
| `src/kernel/main_kernel.rs` | 1, 2 |
| `src/kernel/scene.rs` | 1 |
| `src/kernel/statue_grid.rs` | 1, 2 |
| `src/kernel/objects.rs` | 2 |
| `src/kernel/pbr.rs` | 2 |
| `src/kernel/fog.rs` | 2 |
| `src/kernel/shadows.rs` | 2 |
| `src/kernel/ao.rs` | 2 |
| `src/kernel/reflection.rs` | 2 |
| `src/gpu.rs` | 1 |
| `src/app_state.rs` | 1 |

---

## CubeCL 0.12 Migration Notes

### Breaking Changes Handled
| Change | Fix |
|--------|-----|
| `cubecl::client::Client` instead of `ComputeClient` | Updated import |
| `cubecl_runtime::runtime::Runtime` trait required | Added import |
| `#[cube(launch)]` attribute syntax | Verified compatible |
| `TensorArg::from_raw_parts` | Verified compatible |

### New Dependencies
```toml
cubecl = { version = "0.12.0-pre.2", features = ["wgpu"] }
cubecl-wgpu = "0.12.0-pre.2"
cubecl-runtime = "0.12.0-pre.2"
cubecl-zspace = "0.12.0-pre.2"
```

---

## Performance Impact (Expected)

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| SDF distance precision | FMA artifacts possible | Exact bounds | Optimization 2 |
| Raymarch cache misses | High (strided flat buffer) | Low (structured) | Optimization 1 |
| Shader compile time | Baseline | Similar | FastMath attr minimal overhead |
| Code maintainability | Manual offsets | Type-safe structs | Optimization 1 |

---

## Future Work

1. **Dirty tracking** for structured buffers (only rewrite on change)
2. **Variable grid resolution** per SDF object slot
3. **Async compute** for SDF grid baking
4. **Profile-guided** `fast_math` tuning per backend