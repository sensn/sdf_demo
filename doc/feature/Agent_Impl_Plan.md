Hier ist ein umfassender, detaillierter und hochpräziser Implementierungsplan für einen Coding-Agenten. Dieser Plan basiert auf der Syntax von CubeCL 0.11.0 und berücksichtigt alle modernen Features wie das `Launch`-Makro, `Line<F, N>`-Vektoren, typisierte `Tensor`-Signaturen sowie die strengen Alignment-Regeln von WebGPU/Vulkan (UBO 16-Byte-Grenzen).

Der Agent ist angewiesen, jedes Feature schrittweise zu implementieren, nach jedem Schritt `cargo check` auszuführen und eine fortlaufende `IMPLEMENTATION_SUMMARY.md` im Root-Verzeichnis zu pflegen.

---

## 📋 MASTER IMPLEMENTATION PLAN: CUBECL 0.11.0 RAYMARCHER

## 🤖 Agent Execution Rules

1. Incremental Implementation: Implement exactly one step at a time. Do not move to the next step until the current one compiles.
2. Compilation Verification: After writing or modifying code in a step, run `cargo check --bin raymarcher`. If it fails, fix the errors before proceeding.
3. Documentation Tracking: After each successful step, append a progress update to `IMPLEMENTATION_SUMMARY.md` in the root directory, detailing what was implemented, the compilation status, and any resolved alignment issues.
4. CubeCL 0.11.0 Compliance:
    
    - Use `#[cube(launch)]` for entry points.
    - Use typed inputs like `&Tensor<f32>` and `&mut Tensor<f32>` rather than old `BufferArg` types.
    - Use `Line<f32, N>` (e.g., `Line<f32, 3>`) for SIMD-aligned vector math.
    - Use `Comptime` methods for compile-time loops and branch elimination.
    

---

## 🛠️ Step-by-Step Feature Implementation

## Step 1: Project Setup & Host PBR Configurations (UBO Alignment)

- Goal: Setup project structure, WebGPU-compliant Uniform Buffer structures.
- Details: WebGPU UBOs require strict 16-byte alignment. Structs must be padded manually so that sub-components divide perfectly by 16 bytes.
- Action:
    
    1. Initialize a new binary crate. Set up `Cargo.toml` with `cubecl = { version = "0.11.0", features = ["webgpu"] }`, `tobj`, and `mesh_to_sdf`.
    2. Define the PBR Host structures utilizing `#[derive(CubeType, Copy, Clone)]` and `#[derive(CubeType, CubeLaunch)]`.
    
    ```rust
    use cubecl::prelude::*;
    
    #[derive(CubeType, Copy, Clone)]
    pub struct GpuVec3 {
        pub x: f32, pub y: f32, pub z: f32, pub _pad: f32, // 16 bytes
    }
    #[derive(CubeType, Copy, Clone)]
    pub struct CameraSettings {
        pub position: GpuVec3, pub target: GpuVec3, pub up: GpuVec3, // 48 bytes
        pub fov: f32, pub _pad: [u32; 3], // 16 bytes -> Total: 64 bytes
    }
    #[derive(CubeType, Copy, Clone)]
    pub struct EnvSettings {
        pub light_intensity: f32, pub ambient_fog: f32, pub _pad: [u32; 2], // 16 bytes
    }
    #[derive(CubeType, Copy, Clone)]
    pub struct ArchParams {
        pub temple_height: f32, pub pillar_count: u32, pub _pad: [u32; 2], // 16 bytes
    }
    #[derive(CubeType, Copy, Clone)]
    pub struct FoldParams {
        pub fold_factor: f32, pub _pad: [u32; 3], // 16 bytes
    }
    #[derive(CubeType, CubeLaunch)]
    pub struct RaymarchParams {
        pub camera: CameraSettings, pub env: EnvSettings, pub arch: ArchParams, pub fold: FoldParams,
    }
    ```
    
- Verification: Run `cargo check`. Create `IMPLEMENTATION_SUMMARY.md` with Step 1 status.

## Step 2: CPU Mesh (.obj) Parser & Binary SDF Cache System

- Goal: Build the CPU preprocessing pipeline to parse OBJ meshes into 3D SDF grids and cache them to disk as raw `.bin` files.
- Details: Read `.obj` vertices, compute Signed Distance Fields using a BVH, write out raw bytes, and read back instantly.
- Action:
    
    1. Write a module `sdf_preprocessor.rs`.
    2. Implement `load_obj_to_sdf(file_path: &str, resolution: usize) -> Vec<f32>` using `tobj` and `mesh_to_sdf`.
    3. Implement raw binary serialization/deserialization:
    
    ```rust
    use std::fs::File;
    use std::io::{Read, Write, BufReader, BufWriter};
    
    pub fn save_sdf_cache(path: &str, data: &[f32]) -> std::io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        w.write_all(bytes)
    }
    pub fn load_sdf_cache(path: &str, size: usize) -> std::io::Result<Vec<f32>> {
        let mut r = BufReader::new(File::open(path)?);
        let mut data = vec![0.0f32; size * size * size];
        let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) };
        r.read_exact(bytes)?;
        Ok(data)
    }
    ```
    
- Verification: Run `cargo check`. Update `IMPLEMENTATION_SUMMARY.md`.

## Step 3: Base Kernel Skeleton & 3D SDF Grid Sampling (Trilinear)

- Goal: Implement the entry point kernel and trilinear interpolation for reading the 3D SDF Tensor buffer on the GPU.
- Details: Map 3D positions to flattened tensor indexes `x + y * stride_y + z * stride_z` and perform linear blending across 8 neighboring voxels.
- Action:
    
    1. Create `kernel.rs` and write the `#[cube]` function for interpolation:
    
    ```rust
    #[cube]
    fn mix(a: f32, b: f32, w: f32) -> f32 { a * (1.0 - w) + b * w }
    
    #[cube]
    pub fn sample_sdf_grid(p: Line<f32, 3>, grid: &Tensor<f32>, res: Line<u32, 3>, size: Line<f32, 3>) -> f32 {
        let half_size = size * 0.5;
        let local_pos = (p + half_size) / size * Line::<f32, 3>::new((res.x - 1) as f32, (res.y - 1) as f32, (res.z - 1) as f32);
        let max_bound = Line::<f32, 3>::new((res.x - 2) as f32, (res.y - 2) as f32, (res.z - 2) as f32);
        let lp = Line::<f32, 3>::max(Line::<f32, 3>::to_line(0.0), Line::<f32, 3>::min(local_pos, max_bound));
    
        let x0 = lp.x as u32; let y0 = lp.y as u32; let z0 = lp.z as u32;
        let tx = lp.x - (x0 as f32); let ty = lp.y - (y0 as f32); let tz = lp.z - (z0 as f32);
        let sy = res.x; let sz = res.x * res.y;
    
        let c000 = grid[x0 + y0 * sy + z0 * sz];       let c100 = grid[(x0+1) + y0 * sy + z0 * sz];
        let c010 = grid[x0 + (y0+1) * sy + z0 * sz];   let c110 = grid[(x0+1) + (y0+1) * sy + z0 * sz];
        let c001 = grid[x0 + y0 * sy + (z0+1) * sz];   let c101 = grid[(x0+1) + y0 * sy + (z0+1) * sz];
        let c011 = grid[x0 + (y0+1) * sy + (z0+1) * sz];   let c111 = grid[(x0+1) + (y0+1) * sy + (z0+1) * sz];
    
        mix(mix(mix(c000, c100, tx), mix(c010, c110, tx), ty), mix(mix(c001, c101, tx), mix(c011, c111, tx), ty), tz)
    }
    ```
    
    2. Write base kernel function macro `#[cube(launch)] pub fn raymarch_sdf_kernel(...)` reading from `sdf_grid`.
    
- Verification: Run `cargo check`. Update `IMPLEMENTATION_SUMMARY.md`.

## Step 4: Comptime Raymarching Configuration & Space-Folding Fractal Pillars

- Goal: Add Comptime Traits for configuration control and implement the space-folding infinite fractal pillars (Menger Sponge variant).
- Details: Utilize `Comptime::loop_unrolled` inside the kernel and `abs()` folding logic.
- Action:
    
    1. Define configuration traits:
    
    ```rust
    pub trait RaymarchConfig: Send + Sync + 'static {
        const MAX_STEPS: u32;
        const USE_SHADOWS: bool;
    }
    ```
    
    2. Implement space folding and Menger logic inside `kernel.rs`:
    
    ```rust
    #[cube]
    pub fn op_repetition_xz(p: Line<f32, 3>, spacing: f32) -> Line<f32, 3> {
        let mut r = p;
        r.x = p.x - spacing * f32::round(p.x / spacing);
        r.z = p.z - spacing * f32::round(p.z / spacing);
        r
    }
    #[cube]
    pub fn sdf_fractal_pillar(p: Line<f32, 3>, params: &RaymarchParams) -> f32 {
        let mut d = sdf_box(p, Line::<f32, 3>::new(1.0, params.arch.temple_height, 1.0));
        let mut cp = p; let mut scale = 1.0f32;
        Comptime::loop_unrolled(0, 4, |_i| {
            cp = Line::<f32, 3>::abs(cp) - Line::<f32, 3>::to_line(0.8);
            if cp.x < cp.y { let t = cp.x; cp.x = cp.y; cp.y = t; }
            if cp.x < cp.z { let t = cp.x; cp.x = cp.z; cp.z = t; }
            if cp.y < cp.z { let t = cp.y; cp.y = cp.z; cp.z = t; }
            cp = cp * params.fold.fold_factor; scale *= params.fold.fold_factor;
            cp.x -= 1.0 * (params.fold.fold_factor - 1.0);
            cp.y -= 1.0 * (params.fold.fold_factor - 1.0);
            let d_cross = sdf_box(cp, Line::<f32, 3>::new(2.0, 2.0, 2.0));
            d = f32::max(d, -d_cross / scale);
        });
        d
    }
    ```
    
- Verification: Run `cargo check`. Update `IMPLEMENTATION_SUMMARY.md`.

## Step 5: Scene Mapping with Distance Culling & Normal Vector Generation

- Goal: Implement the combined scene graph (`scene_map`), integrate Bounding-Sphere Distance Culling, and compute analytical scene normals using finite differences.
- Details: Avoid expensive grid calls via sphere check; apply numerical offsets (Epsilon = 0.001) to derive gradients.
- Action:
    
    1. Write `scene_map_with_material` integrating Bounding-Sphere checking:
    
    ```rust
    #[cube]
    pub fn scene_map_with_material(p: Line<f32, 3>, params: &RaymarchParams, grid: &Tensor<f32>) -> Line<f32, 2> {
        let spacing = params.arch.pillar_count as f32 * 2.0;
        let local_p = op_repetition_xz(p, spacing);
        let d_pillars = sdf_fractal_pillar(local_p, params);
        let mut d_scene = d_pillars; let mut mat_id = 1.0f32;
    
        let statue_pos = p - Line::<f32, 3>::new(0.0, 4.0, 0.0);
        if Line::<f32, 3>::length(statue_pos) - 7.0 < 0.1 { // Distance Culling
            let d_statue = sample_sdf_grid(statue_pos, grid, Line::<u32, 3>::new(64,64,64), Line::<f32, 3>::new(8.0,8.0,8.0));
            if d_statue < d_scene { d_scene = d_statue; mat_id = 2.0f32; }
        }
        Line::<f32, 2>::new(d_scene, mat_id)
    }
    ```
    
    2. Implement `calculate_normal` querying `scene_map` 6 times across primary axes.
    
- Verification: Run `cargo check`. Update `IMPLEMENTATION_SUMMARY.md`.

## Step 6: Physically-Based Rendering (PBR) Materials & Multi-Channel Glow (Single-Pass)

- Goal: Extend the execution pipeline to map Material IDs to a lookup Table Tensor, handle far-field clipping, and track multi-channel (RGB) emission fields along ray vectors.
- Details: Output buffer maps to `[R, G, B, R, G, B...]`. Glow uses single-pass accumulator tracking `emission / (distance + 0.02)`.
- Action:
    
    1. Implement structural changes to the primary loop inside `raymarch_sdf_kernel`:
    
    ```rust
    // Inside execution loop...
    let scene_data = scene_map_with_material(current_pos, &params, sdf_grid);
    let distance = scene_data.x;
    let mat_id = scene_data.y;
    
    if mat_id == 3.0f32 { // Glow accumulation for golden objects
        let glow_st = 0.01 / (distance + 0.02);
        glow_accumulator = glow_accumulator + Line::<f32, 3>::new(1.0, 0.7, 0.1) * glow_st;
    }
    ```
    
    2. Add PBR evaluation paths fetching from the `materials` tensor (`[R, G, B, Roughness, Metallic, IsEmission]`). Calculate specular components via half-vector distribution.
    
- Verification: Run `cargo check`. Update `IMPLEMENTATION_SUMMARY.md`.

## Step 7: Ambient Occlusion, Soft Shadows, Volumetric Fog & Reflections

- Goal: Implement the final rendering effects: screen-space ambient occlusion, soft penumbra tracking shadows, forward scattered fog, and 1-bounce reflections.
- Details: AO loops 5 times along surface normals. Soft shadows calculate `min(res, k * h / t)`. Fog marches sample nodes over maximum depth. Reflections trigger `trace_reflection` up to 32 iterative units deep.
- Action:
    
    1. Implement `calculate_ao` and `calculate_shadow` helper functions.
    2. Implement `integrate_volumetric_fog` evaluating step-wise scattering density against shadows.
    3. Implement `trace_reflection` computing secondary surface shading recursively flattened into loops.
    4. Perform linear blending (`mix`) of core PBR direct reflections, ambient illumination scaled by AO, single-pass glow fields, and exponential fog layers into the final RGB channel assignments.
    
- Verification: Run `cargo check`. Update `IMPLEMENTATION_SUMMARY.md`.

---

