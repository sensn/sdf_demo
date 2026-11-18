//! `kernel/shadows.rs` — Weiche Schatten: calculate_soft_shadow.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::scene::scene_sdf;
use super::statue_grid::scene_map_with_material;
use super::types::{ArchParams, FoldParams, Vec3};

#[cube]
pub fn calculate_soft_shadow<F: Float>(
    ro: Vec3<F>,
    rd: Vec3<F>,
    time: F,
    blend_factor: F,
    active_slots: u32,
    slots: &Tensor<F>,           // 🟢 OPTIMIZATION 1: Flat f32 buffer for SlotData (100 * 8 = 800 f32)
    materials: &Tensor<F>,       // 🟢 OPTIMIZATION 1: Flat f32 buffer for MaterialData (100 * 4 = 400 f32)
    rotations: &Tensor<F>,       // 🟢 OPTIMIZATION 1: Flat f32 buffer for RotationData (100 * 3 = 300 f32)
    arch: ArchParams<F>,
    fold: FoldParams<F>,
    // 🟢 SDF-OBJEKT-TENSOR: Grid-Slots werfen ebenfalls Schatten.
    objects: &Tensor<F>,
    bound_offset: F,
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
        
        // 💡 KORREKTUR: `slots/materials/rotations` statt der alten Slices weiterreichen
        let sdf_res = scene_sdf::<F>(
            p.clone(), time, blend_factor, active_slots,
            slots, materials, rotations,
            arch.clone(), fold.clone()
        );
        let mut h = sdf_res.d;
        
        // 🟢 SDF-OBJEKT-TENSOR: Grid-Slots blockieren den Schattenstrahl
        // ebenfalls (KEIN Sphere-Bound — sonst Phantom-Schatten-Ring).
        let plan_res = scene_map_with_material::<F>(
            p, objects, bound_offset, false,
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
