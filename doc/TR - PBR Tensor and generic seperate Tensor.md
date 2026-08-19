## Technical Report: Invariant 3-Tensor PBR and Parameter Streaming Framework in CubeCL 0.11

This specification outlines the architecture, implementation history, and future extension mechanics of our high-performance, real-time Signed Distance Field (SDF) Raymarching Engine using CubeCL 0.11.0-pre.2. By moving from a single unified buffer configuration to a decoupled, multi-tensor streaming model, the engine achieves absolute structural invariance, ensuring optimal memory cache coherence, preventing implicit padding errors, and bypassing JIT cache invalidation bugs.

---

## 1. Die erweiterte Daten-Topologie (Das 3-Tensor-Modell)

To support Physically Based Rendering (PBR) properties alongside geometry data without inducing micro-stutters or CPU-side allocation churn, the streaming pipeline is strictly partitioned into three independent memory blocks.

```unset
 [Tensor 1: meta_handle]      ---> Shape: [1]   -> [ live_active_slots_count ]
 [Tensor 2: slots_handle]     ---> Shape: [800] -> [ Typ, Size, X, Y, Z, R, G, B ] per Slot (Stride: 8)
 [Tensor 3: materials_handle] ---> Shape: [400] -> [ Roughness, Metallic, Emissive, Specular ] per Slot (Stride: 4)
```

## Mathematical Stride Layout and SIMD Alignment Rules

## Tensor 1: Global Loop Metadata (`meta: &Tensor<f32>`)

- Shape: `[1]` (Scalar value containing `slot_types.len() as f32`).
- Function: Informs the device execution mask of the precise dynamic loop ceiling, preventing warp lanes from scanning uninitialized slots.

## Tensor 2: Geometry & Albedo Matrix (`slots: &Tensor<f32>`)

- Shape: Fixed maximum allocation layout of `[800]` elements ($100 \text{ Slots} \times 8 \text{ fields}$).
- Stride Factor: $8 \times 4 \text{ Bytes} = 32 \text{ Bytes}$. This exactly matches two contiguous 16-byte hardware vectors (`Vec4`).
- Memory Offset per Element $i$: $\text{Offset}(i) = i \times 8$.

## Tensor 3: Physical Surface Attributes (`materials: &Tensor<f32>`)

- Shape: Fixed maximum allocation layout of `[400]` elements ($100 \text{ Slots} \times 4 \text{ fields}$).
- Stride Factor: $4 \times 4 \text{ Bytes} = 16 \text{ Bytes}$. This forms a perfect, solitary GPU hardware register vector (`Vec4`), preventing unaligned cross-boundary read operations.
- Memory Offset per Element $i$: $\text{Offset}(i) = i \times 4$.

```unset
Slots Layout Matrix (Stride 8):
Index: [ 0  |  1   | 2 | 3 | 4 | 5 | 6 | 7  ][ 8  |  9   | ... ]
Fields: [Typ | Size | X | Y | Z | R | G | B  ][Typ | Size | ... ]

Materials Layout Matrix (Stride 4):
Index: [    0    |    1     |    2     |    3     ][    4    | ... ]
Fields: [Roughness| Metallic | Emissive | Specular ][Roughness| ... ]
```

---

## 2. Step-by-Step Guide: Integrating the Material Tensor

This roadmap logs every modification point required across both host and device scopes to successfully append the material tensor layer, bypassing compiler type-inference and move-ownership traps.

## Step 1: Host State Vector and Buffer Initialization (`src/main.rs`)

To manage individual slot rendering properties dynamically via keyboard streaming, tracking vectors must be instantiated on the host and mirrored into a static allocation block.

```rust
// 1. Instantiation of CPU structural tracking vectors
let mut slot_roughness = vec![0.2f32, 0.5f32, 0.1f32, 0.5f32, 0.5f32]; 
let mut slot_metallic  = vec![1.0f32, 0.0f32, 0.8f32, 0.0f32, 0.0f32]; 
let mut slot_emissive  = vec![0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32];
let mut slot_specular  = vec![1.0f32, 0.5f32, 1.0f32, 0.5f32, 0.5f32];

const MAX_SLOTS: usize = 100;
const TOTAL_MATERIAL_FLOATS: usize = MAX_SLOTS * 4; // Flat 400 floats capacity

// 2. Interleave mapping into flat memory arrays
let mut initial_materials = vec![0.0f32; TOTAL_MATERIAL_FLOATS];
for i in 0..slot_types.len() {
    let base = i * 4;
    initial_materials[base]     = slot_roughness[i];
    initial_materials[base + 1] = slot_metallic[i];
    initial_materials[base + 2] = slot_emissive[i];
    initial_materials[base + 3] = slot_specular[i];
}

// 3. Persistent allocation assignment
let materials_handle = client.create(cubecl::bytes::Bytes::from_elems(initial_materials));
```

## Step 2: The Loop Invariant Re-write Phase (`src/main.rs`)

Inside the window polling closure (`Event::AboutToWait`), the streaming sequence must pass fixed sizes matching `TOTAL_MATERIAL_FLOATS` over `client.write`.

```rust
if config_dirty {
    let current_len = slot_types.len() as f32;
    client.write(&meta_handle, cubecl::bytes::Bytes::from_elems(vec![current_len]));

    // Update geometry tensor layout...
    
    // Serializing physical material changes downstream into existing allocation slice
    let mut dynamic_materials = vec![0.0f32; TOTAL_MATERIAL_FLOATS];
    for i in 0..slot_types.len() {
        if i >= MAX_SLOTS { break; }
        let base = i * 4;
        dynamic_materials[base]     = slot_roughness[i];
        dynamic_materials[base + 1] = slot_metallic[i];
        dynamic_materials[base + 2] = slot_emissive[i];
        dynamic_materials[base + 3] = slot_specular[i];
    }
    client.write(&materials_handle, cubecl::bytes::Bytes::from_elems(dynamic_materials));
    
    config_dirty = false;
}
```

## Step 3: Synchronous Expansion of Dynamic Allocation Events

To ensure no indexing errors occur during dynamic runtime adjustments (e.g., spawning Slot 6), all property arrays must grow identically via synchronized operations.

```rust
// KeyCode::Digit6 Spawning matching blocks
slot_types.push(1.0f32);
slot_sizes.push(1.0f32);
// ... geometry offsets ...

// Append material properties simultaneously to enforce vector alignment parity
slot_roughness.push(0.3f32);
slot_metallic.push(0.0f32);
slot_emissive.push(0.0f32);
slot_specular.push(0.5f32);
```

## Step 4: Host Pipeline Launch Metric Configuration

The host launch sequence maps types by extracting dimensions from raw `Vec<usize>` structures using explicit parameters, bypassing implicit macro evaluations.

```rust
let meta_shape: Vec<usize> = vec![1];
let meta_strides: Vec<usize> = Vec::<usize>::new();

let slots_shape: Vec<usize> = vec![TOTAL_SLOT_FLOATS];
let slots_strides: Vec<usize> = Vec::<usize>::new();

let materials_shape: Vec<usize> = vec![TOTAL_MATERIAL_FLOATS];
let materials_strides: Vec<usize> = Vec::<usize>::new();

unsafe {
    kernel::raymarch_sdf_kernel::launch(
        &client,
        CubeCount::Static((width + 15) / 16, (height + 3) / 4, 1),
        cube_dim,
        TensorArg::from_raw_parts(output_handle.clone(), shape.into(), strides.into()),
        TensorArg::from_raw_parts(meta_handle.clone(), meta_shape.into(), meta_strides.into()),
        TensorArg::from_raw_parts(slots_handle.clone(), slots_shape.into(), slots_strides.into()),
        // Injection of the material tensor parameter layer
        TensorArg::from_raw_parts(materials_handle.clone(), materials_shape.into(), materials_strides.into()),
        // ... remainder scalar arguments
```

## Step 5: Device Function Header Refactoring (`src/kernel.rs`)

On the device side, macro processing targets require strict parameter bindings. Every intermediate component execution block receives the reference down the call stack.

```rust
#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<u32>,
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>, // Param 4 mapping matching host setup
    time: f32, // ... scalars follow
)

// Cascade reference passing inside the trace loop:
let step_res = scene_sdf(p, t_val, b_factor, meta, slots, materials);
```

## Step 6: GPU-Side Memory Alignment Mapping (`scene_sdf`)

Inside the distance evaluation function, the geometry data array index increments by steps of 8, while the newly introduced material parameter lookup steps by 4.

```rust
let base_idx = i * usize::new(8);
let obj_type = u32::cast_from(slots[base_idx]);

if obj_type > u32::new(0) {
    // Spatial calculations...
    
    // Dedicated Material Array Lookup - Shifted independently via stride of 4
    let mat_idx   = i * usize::new(4);
    let obj_rough = materials[mat_idx];
    let obj_metal = materials[mat_idx + usize::new(1)];
    let obj_emiss = materials[mat_idx + usize::new(2)];
    let obj_spec  = materials[mat_idx + usize::new(3)];
```

---

## 3. Advanced Compilation Fixes: Overcoming Macro Expansion Traps

During development, several compiler errors exposed the unique type-handling characteristics of CubeCL 0.11. This section documents how those roadblocks were systematically resolved.

## I. The Custom Struct Assignment Trap (`E0599` / `E0271`)

- The Error: Using the equals operator to pass custom structure evaluations inside branch trees (`core_system = smin_material(...)`) triggered compiler faults. The code block failed because `SdfResultExpand` does not implement the primitive assignment traits utilized during macro expansion.
- The Fix: The structure assignment was dismantled into discrete, scalar `f32` field operations:
    
    ```rust
    let blended_step = smin_material(core_system.clone(), obj_result, blend_factor);
    core_system.d = blended_step.d;
    core_system.r = blended_step.r;
    // Repeat for all structural fields...
    ```
    

## II. The Structural Ternary `if-else` Violation

- The Error: Generating conditional selection nodes via shorthand inline expressions (`let final_res = if core.d < arch.d { core } else { arch };`) crashed during translation to the execution targets (WGSL/SPIR-V). CubeCL only permits raw primitives inside inline ternary structures.
- The Fix: Conditional paths were flatted into scalar updates, followed by fresh structure composition at the function's terminal return point:
    
    ```rust
    let mut final_d = f32::new(1000.0);
    if check_core.d < check_arch.d { 
        final_d = check_core.d; 
    } else { 
        final_d = check_arch.d; 
    }
    SdfResult { d: final_d, /* ... fields */ }
    ```
    

## III. The Subgroup Closure Ownership Move (`E0382` / `E0507`)

- The Error: Passing complex types across processing closures inside loop constraints caused compiler panic. The expansion engine translates `loop {}` components into individual code closures (`FnMut`). Referencing out-of-scope tracking attributes inside the loop consumed their ownership state on iteration step 2.
- The Fix: Explicit object duplication traits were enabled via specific device-side derive declarations, allowing parameters to be passed safely into the execution loops:
    
    ```rust
    #[derive(CubeType, Copy, Clone)]
    #[cube(derive(Copy, Clone))] // Forces CloneExpand deployment
    pub struct SdfResult { ... }
    
    // Inside Loop Body:
    let blended_step = smin_material(core_system.clone(), obj_result, blend_factor);
    ```
    

---

## 4. Operational Demonstrations & Use-Case Paradigm Analysis

With the material streaming framework in place, users can perform fine-grained parameter updates on the fly. Adjusting object parameters directly influences lighting calculations and scene integration.

## Material Mutation Scenario

- The Chrome Transformation: Selecting the central Crystal object and dialing down its `roughness` parameter to `0.05` while scaling up its `metallic` parameter to `1.0` instantly transforms its appearance. It transitions from a simple geometric block into mirror-polished chrome, catching and reflecting sharp light highlights across the scene.
- The Matte Sandstone Juxtaposition: Concurrently, an adjacent Gyroid object can be configured with a `roughness` value of `0.90` and a `metallic` coefficient of `0.0`. This isolates its shading logic entirely, rendering it as a porous, diffuse surface that smoothly diffuses light across its geometric bounds.

## Technical Applications Summary

## 1. Volumetric Translucency Mapping

By modifying the unused `specular` channel into an absolute density variable, medical imaging interfaces can render continuous volumetric CT density files. Bone tissues use dense parameters, whereas soft vessels are assigned highly transparent values. This allows real-time interactive clipping plane shifts over PCIe without rebuilding pipeline paths.

## 2. Non-Destructive CSG Inversion Analysis

By mapping the structural properties to coordinate calculation steps, object nodes can dynamically flip roles from additive primitives to subtractive volumes (`Type 2.0` = `ssub`). This enables real-time physical simulation tools to dynamically cut away interior sections of complex engine blocks under test without requiring heavy geometric modifications.

---

## 5. Blueprint: Procedural Guide for Future Core Extensions

This blueprint provides developers with a step-by-step methodology for extending the kernel architecture with additional multi-tensor control blocks.

## Scenario A: Adding an Object Parameter Tensor (Rotation/State Tracking)

To introduce a dedicated object parameter block (e.g., adding continuous rotation matrices or physics vectors), use a standard 4-Float Uniform Stride Layout to preserve optimal VRAM alignment.

```unset
 [rotation_handle] ---> Stride 4 Layout: [ Pitch, Yaw, Roll, ScalePadding ]
```

1. Host Matrix Construction (`src/main.rs`):
    
    ```rust
    const TOTAL_ROTATION_FLOATS: usize = MAX_SLOTS * 4;
    let mut initial_rotations = vec![0.0f32; TOTAL_ROTATION_FLOATS];
    // Fill rotation states sequentially for active objects...
    let rotation_handle = client.create(cubecl::bytes::Bytes::from_elems(initial_rotations));
    ```
    
2. Streaming Execution Sequence Update (`src/main.rs`):  
    Inside the dirty flag evaluation branch, bind the rotation updates to mirror data over the bus using `client.write`:
    
    ```rust
    client.write(&rotation_handle, cubecl::bytes::Bytes::from_elems(dynamic_rotations));
    ```
    
3. Pipeline Launch Extension:  
    Append the structural description arrays and inject the raw data components directly into the execution launch list:
    
    ```rust
    let rot_shape: Vec<usize> = vec![TOTAL_ROTATION_FLOATS];
    let rot_strides: Vec<usize> = Vec::<usize>::new();
    // Inside kernel launch:
    TensorArg::from_raw_parts(rotation_handle.clone(), rot_shape.into(), rot_strides.into()),
    ```
    
4. Device Code Processing Integration (`src/kernel.rs`):  
    Update the primary processing function signature and use the stride mapping inside the search iteration loop:
    
    ```rust
    #[cube(launch)]
    pub fn raymarch_sdf_kernel(
        // ... existing tensor parameters ...
        rotation_matrix: &Tensor<f32>,
    ) {
        // Trace logic...
        let rot_idx = i * usize::new(4);
        let pitch = rotation_matrix[rot_rot_idx];
        let yaw   = rotation_matrix[rot_idx + usize::new(1)];
        let roll  = rotation_matrix[rot_idx + usize::new(2)];
    
        // Pass parameters down to space transformation logic blocks
    }
    ```
    

## Scenario B: Adding Global Environmental Controllers (Lights/Atmosphere)

Global parameters (such as light positions, fog options, and ambient coefficients) do not scale with the object slot count. To handle these variables, implement a dedicated Environment Control Tensor structured as a single aligned array.

```unset
 [env_handle] ---> Fixed Stride 16 Matrix Layout:
                   [ LightX, LightY, LightZ, Intensity ]  <- Register Block 0 (Vec4)
                   [ LightR, LightG, LightB, Ambient   ]  <- Register Block 1 (Vec4)
                   [ FogR,   FogG,   FogB,   FogDensity]  <- Register Block 2 (Vec4)
                   [ FogOn,  Padding,Padding,Padding   ]  <- Register Block 3 (Vec4)
```

1. Host Matrix Setup (`src/main.rs`):  
    Define a flat array matching the 16-element footprint to keep memory access aligned:
    
    ```rust
    let mut env_payload = vec![0.0f32; 16];
    env_payload[0] = 4.0;  env_payload[1] = 7.0;  env_payload[2] = -4.0; env_payload[3] = 1.5; // Light Pos + Power
    env_payload[4] = 1.0;  env_payload[5] = 0.95; env_payload[6] = 0.85; env_payload[7] = 0.2; // Light Color + Ambient
    env_payload[8] = 0.35; env_payload[9] = 0.45; env_payload[10]= 0.60; env_payload[11]= 0.05;// Fog Color + Density
    env_payload[12]= 1.0;  // Fog Switch On (1.0 = On, 0.0 = Off)
    
    let env_handle = client.create(cubecl::bytes::Bytes::from_elems(env_payload));
    ```
    
2. Asynchronous Modification Streaming:  
    Whenever a global state parameter is modified via user input (e.g., dialing down light intensity with `Q` or toggling fog effects), update the corresponding matrix index on the host and flush the data to the GPU:
    
    ```rust
    env_payload[3] = light_intensity; // Update intensity element
    client.write(&env_handle, cubecl::bytes::Bytes::from_elems(env_payload.clone()));
    ```
    
3. Device Extraction and Integration Mechanics:  
    Bind the environmental parameters to the execution launcher. Inside the primary shader block, extract the properties using explicit constants to drive the lighting equations:
    
    ```rust
    #[cube(launch)]
    pub fn raymarch_sdf_kernel(
        // ... existing tensor parameters ...
        env_settings: &Tensor<f32>,
    ) {
        // Read Light Position & Parameters from Unified Register 0
        let key_light_pos = Vec3::new(env_settings[usize::new(0)], env_settings[usize::new(1)], env_settings[usize::new(2)]);
        let l_intensity   = env_settings[usize::new(3)];
    
        // Read Light & Ambient Color Attributes from Unified Register 1
        let key_r = env_settings[usize::new(4)];
        let key_g = env_settings[usize::new(5)];
        let key_b = env_settings[usize::new(6)];
        let a_strength = env_settings[usize::new(7)];
    
        // Read Volumetric Fog configurations from Unified Register 2 & 3
        let fog_density = env_settings[usize::new(11)];
        let fog_enabled = env_settings[usize::new(12)];
    
        // Shading Logic Pass...
        if fog_enabled == f32::new(1.0) {
            // Apply exponential volumetric fog calculations using fog_density
        }
    }
    ```
    

By following this architectural blueprint, future modifications can expand the framework smoothly. Developers can introduce new scene parameters while maintaining optimal performance, keeping memory accesses fully aligned, and ensuring compatibility with the CubeCL 0.11 compilation model.