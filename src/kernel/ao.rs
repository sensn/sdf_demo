//! `kernel/ao.rs` — Ambient Occlusion: calculate_ao.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::scene::scene_sdf;
use super::statue_grid::scene_map_with_material;
use super::types::{ArchParams, FoldParams, Vec3};

#[cube]
pub fn calculate_ao<F: Float>(
    p: Vec3<F>,
    normal: Vec3<F>,
    time: F,
    blend_factor: F,
    active_slots: u32,
    slots: &Tensor<F>,           // 🟢 OPTIMIZATION 1: Flat f32 buffer for SlotData (100 * 8 = 800 f32)
    materials: &Tensor<F>,       // 🟢 OPTIMIZATION 1: Flat f32 buffer for MaterialData (100 * 4 = 400 f32)
    rotations: &Tensor<F>,       // 🟢 OPTIMIZATION 1: Flat f32 buffer for RotationData (100 * 3 = 300 f32)
    arch: ArchParams<F>,
    fold: FoldParams<F>,
    // 🟢 SDF-OBJEKT-TENSOR: Grid-Slots tragen ebenfalls zur Okklusion bei.
    objects: &Tensor<F>,
    bound_offset: F,
    ao_iterations: u32,
) -> F {
    let mut occ = F::new(0.0);
    let mut sca = F::new(1.0);

    // 🟢 OPTIMIERUNG 2: Schleife komplett entrollen
    // Schaltet jeglichen Branch-Overhead für die AO-Berechnung aus
    // 🟢 PARAM-GUI: Iterationszahl datengetrieben (1..10, default 5).
    let ao_iter_f = F::cast_from(ao_iterations).max(F::new(1.0));
    let mut i = F::new(0.0);
    loop {
        if i >= ao_iter_f {
            break;
        }
        i += F::new(1.0);

        let current_p = p.clone();
        let current_normal = normal.clone();

        // i als f32/F berechnen (1-basiert wie das alte `for i in 1..5`)
        let hr = (i + F::new(1.0)) * F::new(0.15);
        let ao_pos = current_p.add(current_normal.scale(hr));

        // 💡 KORREKTUR: `slots/materials/rotations` statt der alten Slices weiterreichen
        let sdf_res = scene_sdf::<F>(
            ao_pos.clone(), time, blend_factor, active_slots,
            slots, materials, rotations,
            arch.clone(), fold.clone()
        );
        let mut dd = sdf_res.d;
        
        // 🟢 SDF-OBJEKT-TENSOR: Grid-Slots tragen zur Okklusion bei
        // (kein Sphere-Bound).
        let plan_res = scene_map_with_material::<F>(
            ao_pos, objects, bound_offset, false,
        );
        if plan_res.d < dd {
            dd = plan_res.d;
        }

        occ += (hr - dd) * sca;
        sca *= F::new(0.90);
    }

    (F::new(1.0) - (occ * F::new(0.5))).max(F::new(0.3))
}
