//! `kernel/statue_grid.rs` — SDF-Grid-Sampling: Statue (trilinear), scene_map_with_material, calculate_normal.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;
use crate::gpu::{SLOTS_COUNT, MATERIALS_COUNT, ROTATIONS_COUNT};
//use cubecl::frontend::Tensor; // 💡 UNBEDINGT NÖTIG für &Tensor<F>
use super::types::{ArchParams, GpuVec3, Vec2 as OtherVec2, Vec3};
// ============================================================================
// 🟢 PLAN-IMPLEMENTATION (Agent_Impl_Plan.md, Steps 3–7)
// ============================================================================
/// Trilineares Sampling des 3D-SDF-Grids (Step 3).
///
/// `p` liegt im Welt-Raum; das Grid spannt `[-size/2, size/2]^3` auf und
/// ist in ZYX-Reihenfolge geflacht (index = x + res*(y + res*z)).
/// 8 Voxel werden nach den fraktionalen Anteilen gemischt.

#[cube]
pub fn sample_sdf_grid<F: Float>(
    p: Vec3<F>, 
    scene: &Tensor<F>, // 💡 KORREKTUR: Von 'grid: &[F]' auf 'scene: &Tensor<F>' umgestellt!
    res: u32, 
    size: F
) -> F {
    let res_f = F::cast_from(res);
    let res_us = usize::cast_from(res);
    let half = size * F::new(0.5);

    // In Grid-Koordinaten [0, res-1] mappen.
    let local_x = (p.x + half) / size * (res_f - F::new(1.0));
    let local_y = (p.y + half) / size * (res_f - F::new(1.0));
    let local_z = (p.z + half) / size * (res_f - F::new(1.0));
    
    // Auf [0, res-2] klemmen, damit x0+1/y0+1/z0+1 valide Indizes sind.
    let max_bound = res_f - F::new(2.0);
    let lp_x = local_x.max(F::new(0.0)).min(max_bound);
    let lp_y = local_y.max(F::new(0.0)).min(max_bound);
    let lp_z = local_z.max(F::new(0.0)).min(max_bound);
    
    // Basis-Voxel + fraktionale Gewichte.
    let x0 = u32::cast_from(lp_x);
    let y0 = u32::cast_from(lp_y);
    let z0 = u32::cast_from(lp_z);
    
    let tx = lp_x - F::cast_from(x0);
    let ty = lp_y - F::cast_from(y0);
    let tz = lp_z - F::cast_from(z0);
    
    let x0_us = usize::cast_from(x0);
    let y0_us = usize::cast_from(y0);
    let z0_us = usize::cast_from(z0);
    
     // 💡 Das SDF-Grid startet im gepackten Buffer genau nach den Rotations-Daten!
    let grid_start = usize::cast_from(SLOTS_COUNT + MATERIALS_COUNT + ROTATIONS_COUNT); // 1500

    // 8 Ecken des Voxels (ZYX-Flattening) inkl. `grid_start` & korrektem Z-Klammerung-Bugfix!
    let c000 = scene[grid_start + x0_us + res_us * (y0_us + res_us * z0_us)];
    let c100 = scene[grid_start + (x0_us + usize::new(1)) + res_us * (y0_us + res_us * z0_us)];
    let c010 = scene[grid_start + x0_us + res_us * ((y0_us + usize::new(1)) + res_us * z0_us)];
    let c110 = scene[grid_start + (x0_us + usize::new(1)) + res_us * ((y0_us + usize::new(1)) + res_us * z0_us)];
    
    let c001 = scene[grid_start + x0_us + res_us * (y0_us + res_us * (z0_us + usize::new(1)))];
    let c101 = scene[grid_start + (x0_us + usize::new(1)) + res_us * (y0_us + res_us * (z0_us + usize::new(1)))];
    let c011 = scene[grid_start + x0_us + res_us * ((y0_us + usize::new(1)) + res_us * (z0_us + usize::new(1)))];
    let c111 = scene[grid_start + (x0_us + usize::new(1)) + res_us * ((y0_us + usize::new(1)) + res_us * (z0_us + usize::new(1)))];
    
    // Trilineare Mischung entlang X, dann Y, dann Z.
    let x00 = c000 * (F::new(1.0) - tx) + c100 * tx;
    let x10 = c010 * (F::new(1.0) - tx) + c110 * tx;
    let x01 = c001 * (F::new(1.0) - tx) + c101 * tx;
    let x11 = c011 * (F::new(1.0) - tx) + c111 * tx;
    
    let y0v = x00 * (F::new(1.0) - ty) + x10 * ty;
    let y1v = x01 * (F::new(1.0) - ty) + x11 * ty;
    
    y0v * (F::new(1.0) - tz) + y1v * tz
}

#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct Vec2<F: Float> {
    pub d: F,
    pub mat_id: F,
}

#[cube]
pub fn scene_map_with_material<F: Float>(
    p: Vec3<F>,
    spacing: F,
    pillar_height: F,
    scene: &Tensor<F>, // 💡 KORREKTUR: Von 'grid' zu 'scene' umbenannt!
    grid_res: u32,
    grid_size: F,
    use_sphere_bound: bool,
    statue_center: GpuVec3<F>,
    bounding_sphere_radius_offset: F,
) -> Vec2<F> {
    let mut d_scene = F::new(1000.0);
    let mut mat_id = F::new(2.0);
    
    let statue_center_v = Vec3::<F>::new(
        statue_center.x,
        statue_center.y,
        statue_center.z,
    );
    let statue_pos = p.sub(statue_center_v);
    let sphere_d = statue_pos.length() - grid_size * F::new(0.5);
    
    if sphere_d < bounding_sphere_radius_offset {
        let d_statue = sample_sdf_grid::<F>(statue_pos, scene, grid_res, grid_size);
        if d_statue < d_scene {
            d_scene = d_statue;
        }
    } else if use_sphere_bound {
        d_scene = sphere_d;
        mat_id = F::new(0.0);
    }
    
    Vec2::<F> { d: d_scene, mat_id }
}

/// Normale via zentrale Differenzen über `scene_map_with_material`
/// (Step 5): 6 SDF-Auswertungen entlang der Hauptachsen, ε = 0.001.
/// Normale via zentrale Differenzen über `scene_map_with_material`
/// (Step 5): 6 SDF-Auswertungen entlang der Hauptachsen, ε = 0.001.
#[cube]
pub fn calculate_normal<F: Float>(
    p: Vec3<F>,
    spacing: F,
    pillar_height: F,
    scene: &Tensor<F>, // 💡 KORREKTUR: Von 'grid' zu 'scene' umbenannt!
    grid_res: u32,
    grid_size: F,
    statue_center: GpuVec3<F>,
    bounding_sphere_radius_offset: F,
) -> Vec3<F> {
    let eps = F::new(0.001);
    
    // Normalen brauchen die reale Distanz (kein Sphere-Bound — sonst
    // gäbe es an der unsichtbaren Kugelgrenze falsche Normalen).
    let d = scene_map_with_material::<F>(
        p.clone(), spacing, pillar_height, scene, grid_res, grid_size, false, 
        statue_center.clone(), bounding_sphere_radius_offset
    ).d;
    
    let nx = scene_map_with_material::<F>(
        Vec3::<F>::new(p.x + eps, p.y, p.z),
        spacing,
        pillar_height,
        scene,
        grid_res,
        grid_size,
        false,
        statue_center.clone(),
        bounding_sphere_radius_offset,
    )
    .d - d;
    
    let ny = scene_map_with_material::<F>(
        Vec3::<F>::new(p.x, p.y + eps, p.z),
        spacing,
        pillar_height,
        scene,
        grid_res,
        grid_size,
        false,
        statue_center.clone(),
        bounding_sphere_radius_offset,
    )
    .d - d;
    
    let nz = scene_map_with_material::<F>(
        Vec3::<F>::new(p.x, p.y, p.z + eps),
        spacing,
        pillar_height,
        scene,
        grid_res,
        grid_size,
        false,
        statue_center,
        bounding_sphere_radius_offset,
    )
    .d - d;
    
    Vec3::<F>::new(nx, ny, nz).normalize()
}
