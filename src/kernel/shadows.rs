//! `kernel/shadows.rs` — Weiche Schatten: calculate_soft_shadow.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::scene::scene_sdf;
use super::statue_grid::scene_map_with_material;
use super::types::{ArchParams, FoldParams, GpuVec3, Vec3};
#[cube]
pub fn calculate_soft_shadow(
    ro: Vec3,
    rd: Vec3,
    time: f32,
    blend_factor: f32,
    active_slots: u32,
    slots: &[f32],
    materials: &[f32],
    rotations: &[f32],
    arch: ArchParams,
    fold: FoldParams,
    // 🟢 PLAN STEP 7: Pfeiler/Statue werfen ebenfalls Schatten.
    pillar_spacing: f32,
    pillar_height: f32,
    sdf_grid: &[f32],
    grid_res: u32,
    grid_size: f32,
    statue_center: GpuVec3,
    bounding_sphere_radius_offset: f32,
    shadow_softness_k: f32,
) -> f32 {
    let mut res = f32::new(1.0);
    let mut t = f32::new(0.04);
    let t_max = f32::new(25.0);

    let mut step = u32::new(0);
    loop {
        // 📉 OPTIMIERUNG 1: Testweise auf 24 Schritte reduzieren, falls 32 noch zu langsam ist
        if step >= u32::new(24) { 
            break;
        }

        let current_ro = ro.clone();
        let current_rd = rd.clone();
        let p = current_ro.add(current_rd.scale(t));
        
        let sdf_res = scene_sdf(p, time, blend_factor, active_slots, slots, materials,
          rotations, arch, fold);
        let mut h = sdf_res.d;
        // 🟢 PLAN STEP 7: Statue blockiert den Schattenstrahl ebenfalls
        // (KEIN Sphere-Bound — sonst Phantom-Schatten-Ring um die Statue).
        let plan_res = scene_map_with_material(
            p.clone(), pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size, false, statue_center, bounding_sphere_radius_offset,
        );
        if plan_res.d < h {
            h = plan_res.d;
        }

        if h < f32::new(0.001) {
            res = f32::new(0.0);
            break;
        }
        
        res = res.min(shadow_softness_k * h / t);
        
        // 🟢 MASSIVER FPS-BOOST: Early Exit im Kernschatten
        // Wenn es bereits fast komplett schwarz ist, rechenintensiven Strahl stoppen!
        if res < f32::new(0.02) {
            res = f32::new(0.0);
            break;
        }

        // Schrittweite dynamisch anpassen
        t += h.max(f32::new(0.04));
        if t > t_max {
            break;
        }

        step += u32::new(1);
    }

    res.max(f32::new(0.2))
}


