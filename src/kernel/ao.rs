//! `kernel/ao.rs` — Ambient Occlusion: calculate_ao.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::scene::scene_sdf;
use super::statue_grid::scene_map_with_material;
use super::types::{ArchParams, FoldParams, GpuVec3, Vec3};
#[cube]
pub fn calculate_ao(
    p: Vec3,
    normal: Vec3,
    time: f32,
    blend_factor: f32,
    active_slots: u32,
    slots: &[f32],
    materials: &[f32],
    rotations: &[f32],
    arch: ArchParams,
    fold: FoldParams,
    // 🟢 PLAN STEP 7: Pfeiler/Statue tragen ebenfalls zur Okklusion bei.
    pillar_spacing: f32,
    pillar_height: f32,
    sdf_grid: &[f32],
    grid_res: u32,
    grid_size: f32,
    statue_center: GpuVec3,
    bounding_sphere_radius_offset: f32,
    ao_iterations: u32,
) -> f32 {
    let mut occ = f32::new(0.0);
    let mut sca = f32::new(1.0);

    // 🟢 OPTIMIERUNG 2: Schleife komplett entrollen
    // Schaltet jeglichen Branch-Overhead für die AO-Berechnung aus
    // 🟢 PARAM-GUI: Iterationszahl datengetrieben (1..10, default 5).
    let ao_iter_f = f32::cast_from(ao_iterations).max(f32::new(1.0));
    let mut i = f32::new(0.0);
    loop {
        if i >= ao_iter_f {
            break;
        }
        i += f32::new(1.0);

        let current_p = p.clone();
        let current_normal = normal.clone();

        // i als f32 berechnen (1-basiert wie das alte `for i in 1..5`)
        let hr = (i + f32::new(1.0)) * f32::new(0.15);
        let ao_pos = current_p.add(current_normal.scale(hr));

        let sdf_res = scene_sdf(ao_pos, time, blend_factor, active_slots, slots, materials,
          rotations, arch, fold);
        let mut dd = sdf_res.d;
        // 🟢 PLAN STEP 7: Statue trägt zur Okklusion bei (kein Sphere-Bound).
        let plan_res = scene_map_with_material(
            ao_pos.clone(), pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size, false, statue_center, bounding_sphere_radius_offset,
        );
        if plan_res.d < dd {
            dd = plan_res.d;
        }

        occ += (hr - dd) * sca;
        sca *= f32::new(0.90);
    }

    (f32::new(1.0) - (occ * f32::new(0.5))).max(f32::new(0.3))
}
