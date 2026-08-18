## Architecture Specification & Technical Report: Real-Time Dynamic Raymarching Engine## 1. Architectural Overview & Design Philosophy
This document specifies the architecture of a high-performance, real-time Signed Distance Field (SDF) Raymarching Engine implemented in Rust utilizing CubeCL 0.11.0-pre.2.
Traditional GPU compute architectures for raymarching typically rely on static code generation or uniform parameter updates bound via explicit hardware bindings. In contrast, this engine implements an Invariant Buffer Layout with Dynamic Payload Streaming architecture. It is designed to allow arbitrary runtime object spawning, real-time mutations, and interactive scene graphs without triggering GPU memory re-allocations or JIT shader recompilations.

+--------------------------------------------------------------------------+

| HOST CPU (Rust Context)                                                  |
|                                                                          |
|  [Parallel Vectors] ----> [Interleaved Matrix] ---> [Static VRAM Allocation]
|  - slot_types             (Array of Structures)     - Width: 401 Floats  
|  - slot_sizes             (1 Header + 100 * 5)     - Lifecycle: Persistent

|  - slot_offsets_x,y,z                                                    |
+--------------------------------------------------------------------+-----+
                                                                     |
                                                       client.write  | (Zero-Cost Stream)
                                                                     v
+--------------------------------------------------------------------+-----+

| DEVICE GPU (CubeCL Compute Core)                                         |
|                                                                          |
|  [Global Memory Cache] <-------------------------------------------------+

|   `-- config: &Tensor<f32>` (401 contiguous floats)                      |
|                                                                          |
|  [Uniform Thread Scan]                                                   |
|   `-- u32::cast_from(config[usize::new(0)])` -> Global Dynamic Limit     |
|                                                                          |
|  [Spatial Ray Intersection Loop]                                         |
|   `-- Shared Matrix Repetition Execution Path                           |
+--------------------------------------------------------------------------+

## Core Architecture Trade-offs
To achieve a completely fluid development and production framework ("Space-Lab"), the architecture intentionally balances structural flexibility against ultimate hardware optimization limits:

   1. AOT Structural Invariance: All GPU metadata structures (shape, strides, and dispatch sizes) are completely frozen at startup.
   2. Payload-Driven Customization: All scene mutations are treated as state updates streamed over the PCIe bus into pre-allocated memory slices, avoiding runtime API layout mutations.
   3. Cohesive State Management: Object types, physical bounds, and spatial displacements are bundled sequentially to ensure structural consistency across execution threads.

------------------------------
## 2. Memory Topology & Serialization Strategy
The backend forces a strict AoS (Array of Structures) Interleaved Matrix to track spatial geometry allocations. Global memory buffers represent uniform multi-dimensional states flattened entirely into a sequential 1D GPU memory segment.
## Matrix Footprint Formulation
The memory topology allocates space for a static maximum ceiling of 100 independent object nodes (MAX_SLOTS = 100), ensuring that physical buffer boundaries never change regardless of the current live scene complexity.

                  Index 0: Global Active Frame Length Header
                                  |
                                  v
 raw_config Matrix = [ active_slots_count, 
                       Type_0, Size_0, OffsetX_0, OffsetY_0, OffsetZ_0,  <-- Slot 0 (5 Floats)
                       Type_1, Size_1, OffsetX_1, OffsetY_1, OffsetZ_1,  <-- Slot 1 (5 Floats)
                       ...
                       Type_n, Size_n, OffsetX_n, OffsetY_n, OffsetZ_n ] <-- Slot 99 (5 Floats)

## Mathematical Stride Calculation
The discrete buffer boundary size is structurally invariant and calculated as follows:
$$\text{CONFIG\_BUFFER\_FLOATS} = 1 + (\text{MAX\_SLOTS} \times 5) = 401 \text{ floats}$$ 

* Index 0 (Metadata Header): Contains active_slots_count (represented as f32). This tells the shader compiler how many sequential chunks must be physically scanned during the spatial evaluation cycle.
* Payload Chunk Stride: Each logical slot contains exactly 5 floating-point components. The byte layout offset for any given arbitrary slot identifier ($i$) inside the array tracking is calculated via:
$$\text{BaseOffset}(i) = 1 + (i \times 5)$$ 

------------------------------
## 3. GPU Branching, Divergence, and Execution Analysis
Integrating a Domain Repetition Framework (Infinite Modulo Grids) with a dynamic loop tracking schema introduces complex wave-front execution properties on modern AMD (RDNA) and Nvidia (RTX) architectures.
## Uniform Control Scans
Because the root config tensor memory is visually identical across all concurrent fragment evaluation spaces within a given viewport render pass, the outer loop initialization benefits heavily from Uniform Branching. Every SIMD lane within a Warp/Wavefront accesses the identical memory payload and counts up concurrently, maintaining a unified execution pointer.
## Spatial Branch Divergence
Because cells are instantiated infinitely via structural modulo operations:
$$\text{GridPosition} = P - \text{CellSize} \times \lfloor\frac{P + \text{HalfCell}}{\text{CellSize}}\rfloor$$ 
Threads tracking adjacent pixels cross physical module boundaries in the virtual space. If Thread $A$ evaluates an active structural node (e.g., a Gyroid, Type 2.0) while adjacent Thread $B$ within the same Warp maps to a vacant or alternative geometric node (e.g., a Crystal, Type 1.0), Execution Masking occurs:

   1. The hardware compute unit temporarily serializes both branch targets.
   2. Inactive lanes are masked off during execution block $A$, then swapped for execution block $B$.
   3. Performance Impact: In regions containing high-frequency scene mixtures, execution throughput scales with the sum of the processed branch costs rather than the maximum path cost.

------------------------------
## 4. CubeCL 0.11 Implementation Summary: Mechanics & Syntax Rules
Upgrading from CubeCL 0.10 to 0.11 enforces advanced safety rules, eliminating implicit type inference overhead to maximize compile-time optimizations.
## I. Pure Device Primitive Constraints
Raw floating-point or unsigned integer suffix constructions (e.g., 0.0f32, 1u32) are invalid inside a #[cube] expansion layer. Variables must be initialized using modern device-native frontend type constructors.

// ❌ Invalid CubeCL 0.11 Construction (Triggers Compiler Trait Panic)let half_cell = cell_size * 0.5f32; 
// ✅ Correct Type Construction for 0.11 GPU Contextlet half_cell = cell_size * f32::new(0.5);

## II. Rigid Explicit Type Casting
Implicit arithmetic type conversion or native Rust keywords (val as f32) are completely rejected by the compiler expansion framework. Conversion calls require explicit casting functions mapped directly to target primitive identifiers.

// ❌ Invalid Castlet uv_x = x as f32;
// ✅ Valid Casting Contextlet uv_x = f32::cast_from(x);

## III. Restricted Vector & Struct Lifecycles within Closures
In CubeCL 0.11, loop statements expand internally into localized execution closures (FnMut). Because advanced user-defined macro representations (such as Vec3Expand) do not natively implement an implicit, unchecked copy pipeline across sub-blocks, accessing a structural parameter within a persistent loop structure consumes its reference, triggering an ownership error (E0507).
Variables must be explicitly instantiated or copied via primitive component extraction inside the iteration boundary.

// ❌ Relies on Implicit Copy (Triggers Use of Moved Value inside Loop)let p = ro.add(rd.scale(t));
// ✅ Explicit Component Construction to Bypass Closure Constraintslet current_p = Vec3::new(p.x, p.y, p.z);

## IV. Typed Invariant Tensor Indexing
Tensor collections no longer accept arbitrary integer indexing patterns (u32). Every device memory address reference requires a strict usize descriptor, ensuring compliance with native 64-bit hardware indexing features.

// ❌ Invalid Indexing Patternlet data = config[u32::new(0)];
// ✅ Valid Indexing Specificationlet data = config[usize::new(0)];

## V. Decoupled Spatial Coordinate Construction
Thread blocks are defined using isolated multidimensional coordinate initializers. Thread assignments cannot be bound using explicit instance mutations combined with standard runtime contexts.

// ❌ Overwrites 3D layout bounds with 1D allocationslet mut cube_dim = CubeDim::new(&client, 16); 
// ✅ Correct Multidimensional Pre-Allocation APIlet cube_dim = CubeDim::new_3d(16, 4, 1);

------------------------------
## 5. Performance Blueprint & Reference Implementation
This blueprint provides an optimal reference for implementing the complete dynamic payload architecture. It highlights correct memory streaming mechanics and explicit hardware loop wrapping across modules.
## Custom Component Declaration (src/kernel.rs)

use cubecl::prelude::*;

#[derive(Copy, Clone, CubeType)]pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }
}

#[cube]pub fn scene_sdf(p: Vec3, time: f32, blend_factor: f32, config: &Tensor<f32>) -> f32 {
    let cell_size = f32::new(10.0);
    let half_cell = cell_size * f32::new(0.5);
    
    // Domain Repetition Framework Math
    let grid_p_x = p.x - cell_size * ((p.x + half_cell) / cell_size).floor();
    let grid_p_y = p.y - cell_size * ((p.y + half_cell) / cell_size).floor(); 
    let grid_p_z = p.z - cell_size * ((p.z + half_cell) / cell_size).floor();
    
    let local_p = Vec3::new(grid_p_x, grid_p_y, grid_p_z);
    let mut core_system = f32::new(1000.0);

    // Dynamic Execution Scan Ceiling Bound to Array Length
    let active_slots = usize::cast_from(config[usize::new(0)]);

    let mut i = usize::new(0);
    loop {
        if i >= usize::new(100) || i >= active_slots {
            break; 
        }

        // Stride offset computation based on 5-Float specification
        let base_idx = usize::new(1) + i * usize::new(5);
        
        let obj_type  = u32::cast_from(config[base_idx]);
        let obj_size  = config[base_idx + usize::new(1)];
        let offset_x  = config[base_idx + usize::new(2)];
        let offset_y  = config[base_idx + usize::new(3)];
        let offset_z  = config[base_idx + usize::new(4)];
        
        // Component-isolated structural creation to satisfy macro boundaries
        let p_slot = Vec3::new(local_p.x - offset_x, local_p.y - offset_y, local_p.z - offset_z);

        if obj_type == u32::new(1) {
            core_system = core_system.min(p_slot.x.abs() + p_slot.y.abs() + p_slot.z.abs() - obj_size);
        }
        // Additional Type blocks go here...

        i += usize::new(1);
    }
    
    core_system
}

## High-Frequency State Streaming Loop (src/main.rs)

// =========================================================================// RENDER LOOP CYCLE - Executed Every Frame // =========================================================================if config_dirty {
    // Instantiate a localized payload array pre-sized to match the static hardware buffer
    let mut dynamic_raw_config = vec![0.0f32; CONFIG_BUFFER_FLOATS]; // Pre-calculated as 401 Elements
    
    // Inject structural headers and payload strings natively
    dynamic_raw_config[0] = active_slots_count;
    
    for i in 0..slot_types.len() {
        if i >= MAX_SLOTS { break; }
        let base = 1 + i * 5;
        dynamic_raw_config[base]     = slot_types[i];
        dynamic_raw_config[base + 1] = slot_sizes[i];
        dynamic_raw_config[base + 2] = slot_offsets_x[i];
        dynamic_raw_config[base + 3] = slot_offsets_y[i];
        dynamic_raw_config[base + 4] = slot_offsets_z[i];
    }
    
    // Zero-Cost Direct Memory Update Pathway
    let config_bytes = cubecl::bytes::Bytes::from_elems(dynamic_raw_config);
    
    // Modifies existing allocated buffer space directly without reallocation or cache invalidation
    client.write(&config_handle, config_bytes);
    
    config_dirty = false;
}
// Structurally sound 3D grid layout passlet cube_dim = CubeDim::new_3d(16, 4, 1);
unsafe {
    kernel::raymarch_sdf_kernel::launch(
        &client,
        CubeCount::Static((width + 15) / 16, (height + 3) / 4, 1),
        cube_dim,
        TensorArg::from_raw_parts(output_handle.clone(), shape, strides),
        TensorArg::from_raw_parts(config_handle.clone(), vec![total_elements].into(), Vec::<usize>::new().into()),
        time, width, height, // Direct raw scalar passing continues safely...
    );
}

------------------------------
## 6. Execution Guidelines for Intelligent Agents
When extending this codebase or auto-generating new geometry structures, adhere strictly to the following execution constraints:

   1. Do not modify the tracking strides: If you add custom attributes (e.g., color values, rotation parameters), update the stride calculation globally across both host and device blocks (e.g., from 5 to 8). Never leave trailing unmapped indices.
   2. Enforce variable cloning within loops: Any variable declared outside a loop {} context inside a #[cube] function must be component-cloned or manually rebuilt using field primitive extracts before entering loop math operations.
   3. Isolate client.create calls: Ensure that client.create is exclusively called during initial repository startup. All real-time, interactive frame manipulations must be driven by client.write targeting the invariant config_handle buffer reference.

Would you like to analyze or extend the primitive math logic for advanced CSG operations (such as smooth boolean blends) under the new 0.11 syntax constraints?

