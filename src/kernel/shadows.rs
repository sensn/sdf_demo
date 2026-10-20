//! `kernel/shadows.rs` — Weiche Schatten: calculate_soft_shadow.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::scene::scene_sdf;
use super::statue_grid::scene_map_with_material;
use super::types::{ArchParams, FoldParams, GpuVec3, Vec3};

#[cube]
pub fn calculate_soft_shadow<F: Float>(
    ro: Vec3<F>,
    rd: Vec3<F>,
    time: F,
    blend_factor: F,
    active_slots: u32,
    scene: &Tensor<F>, // 💡 KORREKTUR: Jetzt die direkte Tensor-Referenz statt der Slices
    arch: ArchParams<F>,
    fold: FoldParams<F>,
    // 🟢 PLAN STEP 7: Pfeiler/Statue werfen ebenfalls Schatten.
    pillar_spacing: F,
    pillar_height: F,
    grid_res: u32,
    grid_size: F,
    statue_center: GpuVec3<F>,
    bounding_sphere_radius_offset: F,
    shadow_softness_k: F,
) -> F {
    let mut res = F::new(1.0);
    let mut t = F::new(0.04);
    let t_max = F::new(25.0);

    let mut step = u32::new(0);
    loop {
        // 📉 OPTIMIERUNG 1: Testweise auf 24 Schritte reduzieren, falls 32 noch zu langsam ist
        if step >= u32::new(24) { 
            break;
        }

        let current_ro = ro.clone();
        let current_rd = rd.clone();
        let p = current_ro.add(current_rd.scale(t));
        
        // 💡 KORREKTUR: `scene` anstelle der alten Slices weiterreichen + Turbofisch ergänzt
        let sdf_res = scene_sdf::<F>(
            p.clone(), time, blend_factor, active_slots, scene,
            arch.clone(), fold.clone()
        );
        let mut h = sdf_res.d;
        
        // 🟢 PLAN STEP 7: Statue blockiert den Schattenstrahl ebenfalls
        // (KEIN Sphere-Bound — sonst Phantom-Schatten-Ring um die Statue).
        // 💡 KORREKTUR: `scene` anstelle des alten `sdf_grid` Slices weiterreichen + Turbofisch ergänzt
        let plan_res = scene_map_with_material::<F>(
            p, pillar_spacing, pillar_height, scene, grid_res, grid_size, false, 
            statue_center.clone(), bounding_sphere_radius_offset,
        );
        if plan_res.d < h {
            h = plan_res.d;
        }

        if h < F::new(0.001) {
            res = F::new(0.0);
            break;
        }
        
        res = res.min(shadow_softness_k * h / t);
        
        // 🟢 MASSIVER FPS-BOOST: Early Exit im Kernschatten
        // Wenn es bereits fast komplett schwarz ist, rechenintensiven Strahl stoppen!
        if res < F::new(0.02) {
            res = F::new(0.0);
            break;
        }

        // Schrittweite dynamisch anpassen
        t += h.max(F::new(0.04));
        if t > t_max {
            break;
        }

        step += u32::new(1);
    }

    res.max(F::new(0.2))
}
