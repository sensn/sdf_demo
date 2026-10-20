//! `kernel/ao.rs` — Ambient Occlusion: calculate_ao.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::scene::scene_sdf;
use super::statue_grid::scene_map_with_material;
use super::types::{ArchParams, FoldParams, GpuVec3, Vec3};

#[cube]
pub fn calculate_ao<F: Float>(
    p: Vec3<F>,
    normal: Vec3<F>,
    time: F,
    blend_factor: F,
    active_slots: u32,
    scene: &Tensor<F>, // 💡 KORREKTUR: Jetzt die direkte Tensor-Referenz statt der Slices
    arch: ArchParams<F>,
    fold: FoldParams<F>,
    // 🟢 PLAN STEP 7: Pfeiler/Statue tragen ebenfalls zur Okklusion bei.
    pillar_spacing: F,
    pillar_height: F,
    grid_res: u32,
    grid_size: F,
    statue_center: GpuVec3<F>,
    bounding_sphere_radius_offset: F,
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

        // 💡 KORREKTUR: `scene` anstelle der alten Slices weiterreichen
        let sdf_res = scene_sdf::<F>(
            ao_pos.clone(), time, blend_factor, active_slots, scene,
            arch.clone(), fold.clone()
        );
        let mut dd = sdf_res.d;
        
        // 🟢 PLAN STEP 7: Statue trägt zur Okklusion bei (kein Sphere-Bound).
        // 💡 KORREKTUR: `scene` anstelle des alten `sdf_grid` Slices weiterreichen
        let plan_res = scene_map_with_material::<F>(
            ao_pos, pillar_spacing, pillar_height, scene, grid_res, grid_size, false, 
            statue_center.clone(), bounding_sphere_radius_offset,
        );
        if plan_res.d < dd {
            dd = plan_res.d;
        }

        occ += (hr - dd) * sca;
        sca *= F::new(0.90);
    }

    (F::new(1.0) - (occ * F::new(0.5))).max(F::new(0.3))
}
