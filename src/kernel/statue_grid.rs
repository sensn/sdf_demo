//! `kernel/statue_grid.rs` — SDF-Objekt-Tensor: N Grid-Slots (Trilinear-Sampling,
//! scene_map_with_material, sdf_object_pbr, calculate_normal).
//! (Refactor: Die Statue war fix im Mega-Buffer an Index 1500 angehängt;
//! jetzt N Slots mit Header + Grid-Daten in einem separaten Tensor.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;
use crate::gpu_iced::{MAX_SDF_OBJECTS, SDF_HEADER_STRIDE};
use super::types::{SdfResult, Vec3};

// ============================================================================
// 🟢 SDF-OBJEKT-TENSOR ABI (doc/feature/SDF_OBJECT_TENSOR.md):
//
// Tensor-Layout: [ Header Slot 0 (Stride 24) ][ Header 1 ]…[ Header N-1 ]
//                [ Grid 0 (64³) ][ Grid 1 ]…[ Grid N-1 ]
//
// Header (Stride 24, f32):
//   0 active  1 res  2 size  3 pad
//   4 center_x 5 center_y 6 center_z 7 pad
//   8 rot_x 9 rot_y 10 rot_z 11 pad
//   12 r 13 g 14 b 15 roughness
//   16 metallic 17 emissive 18 specular 19..23 pad
// ============================================================================

/// Einstiegsoffset des Grids von Slot `i` im Objekt-Tensor.
/// Die Grids liegen direkt hinter den N Headern.
#[cube]
pub fn sdf_grid_start() -> usize {
    usize::cast_from(MAX_SDF_OBJECTS) * usize::cast_from(SDF_HEADER_STRIDE)
}

/// Header-Basis-Offset von Slot `i` (=`i * SDF_HEADER_STRIDE`).
#[cube]
pub fn sdf_header_start<F: Float>(slot: usize) -> usize {
    slot * usize::cast_from(SDF_HEADER_STRIDE)
}

// ============================================================================
// Trilineares Sampling eines einzelnen Grid-Slots (ehem. Step 3).
//
// `p` liegt im lokalen Grid-Raum (bereits um center verschoben und invers
// rotiert); das Grid spannt `[-size/2, size/2]^3` auf und ist in ZYX-
// Reihenfolge geflacht (index = x + res*(y + res*z)). 8 Voxel werden nach
// den fraktionalen Anteilen gemischt.
// ============================================================================
#[cube]
pub fn sample_sdf_grid<F: Float>(
    p: Vec3<F>,
    objects: &Tensor<F>, // 🟢 SDF-Objekt-Tensor statt Mega-Scene-Tensor
    grid_start: usize,   // Slot-Offset im Tensor (nicht mehr hart codiert)
    res: u32,
    size: F,
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

    // 8 Ecken des Voxels (ZYX-Flattening) inkl. Slot-Offset & Z-Klammer-Bugfix.
    let c000 = objects[grid_start + x0_us + res_us * (y0_us + res_us * z0_us)];
    let c100 = objects[grid_start + (x0_us + usize::new(1)) + res_us * (y0_us + res_us * z0_us)];
    let c010 = objects[grid_start + x0_us + res_us * ((y0_us + usize::new(1)) + res_us * z0_us)];
    let c110 = objects[grid_start + (x0_us + usize::new(1)) + res_us * ((y0_us + usize::new(1)) + res_us * z0_us)];

    let c001 = objects[grid_start + x0_us + res_us * (y0_us + res_us * (z0_us + usize::new(1)))];
    let c101 = objects[grid_start + (x0_us + usize::new(1)) + res_us * (y0_us + res_us * (z0_us + usize::new(1)))];
    let c011 = objects[grid_start + x0_us + res_us * ((y0_us + usize::new(1)) + res_us * (z0_us + usize::new(1)))];
    let c111 = objects[grid_start + (x0_us + usize::new(1)) + res_us * ((y0_us + usize::new(1)) + res_us * (z0_us + usize::new(1)))];

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

/// 🟢 OPTIMIZATION (Hot-Loop): Minimale Sphere-Bound-Distanz über ALLE
/// Grid-Slots — **ohne** das Grid anzufassen (nur 24f Header pro Slot statt
/// 64³ Voxel). Gibt einen Wert > 0 zurück, wenn kein Slot näher als seine
/// Kugel sein kann.
///
/// Nutzung im March: `if objects_min_bound_distance::<F>(p, objects) < d { … }`
/// — der Marsch kann so ganze Strahlsegmente überspringen, ohne in den
/// ~4 MB Grid-Speicher zu greifen.
#[cube(fast_math = FastMath::all().difference(FastMath::AllowContraction.into()))]
pub fn objects_min_bound_distance<F: Float>(
    p: Vec3<F>,
    objects: &Tensor<F>,
    bound_offset: F,
) -> F {
    let mut min_d = F::new(1000.0);
    let n = usize::cast_from(MAX_SDF_OBJECTS);
    let mut slot = usize::new(0);
    loop {
        if slot >= n {
            break;
        }
        let hdr = slot * usize::cast_from(SDF_HEADER_STRIDE);
        // Inaktiver Slot → komplett überspringen.
        let active = objects[hdr] > F::new(0.5);
        if active {
            let size = objects[hdr + usize::new(2)];
            let cx = objects[hdr + usize::new(4)];
            let cy = objects[hdr + usize::new(5)];
            let cz = objects[hdr + usize::new(6)];

            let dx = p.x - cx;
            let dy = p.y - cy;
            let dz = p.z - cz;
            let sphere_d = (dx * dx + dy * dy + dz * dz).sqrt() - size * F::new(0.5) - bound_offset;

            // Sphere-Distanz ist immer eine *untere* Schranke der echten
            // Grid-Distanz (die Kugel umschließt das Grid vollständig).
            if sphere_d < min_d {
                min_d = sphere_d;
            }
        }
        slot += usize::new(1);
    }
    min_d
}

/// Material des Grid-Treffers aus dem Slot-Header: `mat_id = 2 + slot`.
/// mat_id 0 = Bounding-Shell (kein echter Treffer) → PBR-Neutralwerte.
#[cube]
pub fn sdf_object_pbr<F: Float>(objects: &Tensor<F>, mat_id: F) -> SdfResult<F> {
    let mut r = F::new(0.55);
    let mut g = F::new(0.52);
    let mut b = F::new(0.50);
    let mut rough = F::new(0.8);
    let mut metal = F::new(0.0);
    let mut emiss = F::new(0.0);
    let mut spec = F::new(0.5);

    // Slot-Index aus mat_id (2.0 + i); außerhalb [2, 2+N) → neutral.
    let slot_f = mat_id - F::new(2.0);
    let slot_ok = (slot_f >= F::new(0.0)) & (slot_f < F::cast_from(MAX_SDF_OBJECTS as f32));
    if slot_ok {
        let slot_us = usize::cast_from(u32::cast_from(slot_f.floor()));
        let hdr = slot_us * usize::cast_from(SDF_HEADER_STRIDE);
        let hdr_r = hdr + usize::new(12);
        r = objects[hdr_r];
        g = objects[hdr_r + usize::new(1)];
        b = objects[hdr_r + usize::new(2)];
        rough = objects[hdr_r + usize::new(3)];
        metal = objects[hdr_r + usize::new(4)];
        emiss = objects[hdr_r + usize::new(5)];
        spec = objects[hdr_r + usize::new(6)];
    }

    SdfResult::<F> {
        d: F::new(0.0),
        r,
        g,
        b,
        roughness: rough,
        metallic: metal,
        emissive: emiss,
        specular: spec,
    }
}

/// Kombinierte Grid-Map über ALLE SDF-Objekt-Slots (ehem. nur die eine Statue).
///
/// Liefert (d, mat_id): mat_id = 2 + Slot-Index bei Grid-Treffer,
/// 0.0 bei reiner Bounding-Shell-Distanz (use_sphere_bound).
/// Je Slot: Sphere-Bound-Culling (grobe Distanz zur Slot-Kugel), inverse
/// Euler-Rotation (Z→Y→X rückwärts, wie bei den dynamischen Slots) und
/// Trilinear-Sampling des 64³-Grids.
#[cube]
pub fn scene_map_with_material<F: Float>(
    p: Vec3<F>,
    objects: &Tensor<F>, // 🟢 SDF-Objekt-Tensor
    bound_offset: F,
    use_sphere_bound: bool,
) -> Vec2<F> {
    let mut d_scene = F::new(1000.0);
    let mut mat_id = F::new(0.0);

    let n = usize::cast_from(MAX_SDF_OBJECTS);
    let mut slot = usize::new(0);
    loop {
        if slot >= n {
            break;
        }
        let hdr = slot * usize::cast_from(SDF_HEADER_STRIDE);

        // Inaktiver Slot → komplett überspringen (d bleibt 1000).
        let active = objects[hdr] > F::new(0.5);
        if active {
            let res = u32::cast_from(objects[hdr + usize::new(1)]);
            let size = objects[hdr + usize::new(2)];
            let cx = objects[hdr + usize::new(4)];
            let cy = objects[hdr + usize::new(5)];
            let cz = objects[hdr + usize::new(6)];
            let rx = objects[hdr + usize::new(8)];
            let ry = objects[hdr + usize::new(9)];
            let rz = objects[hdr + usize::new(10)];

            let center = Vec3::<F>::new(cx, cy, cz);
            let local = p.sub(center);

            // Sphere-Bound-Culling: grobe Distanz zur Slot-Kugel.
            let sphere_d = local.length() - size * F::new(0.5);
            if sphere_d < bound_offset {
                // Inverse Euler-Rotation (Z → Y → X rückwärts) in den
                // lokalen Grid-Raum — identisch zu den dynamischen Slots.
                let cz_c = rz.cos();
                let sz_c = rz.sin();
                let rx1 = local.x * cz_c + local.y * sz_c;
                let ry1 = -local.x * sz_c + local.y * cz_c;

                let cy_c = ry.cos();
                let sy_c = ry.sin();
                let rx2 = rx1 * cy_c - local.z * sy_c;
                let rz2 = rx1 * sy_c + local.z * cy_c;

                let cx_c = rx.cos();
                let sx_c = rx.sin();
                let ry3 = ry1 * cx_c + rz2 * sx_c;
                let rz3 = -ry1 * sx_c + rz2 * cx_c;

                let rotated = Vec3::<F>::new(rx2, ry3, rz3);

                let g_start = sdf_grid_start() + slot * usize::cast_from(crate::gpu_iced::SDF_GRID_COUNT);
                let d_obj = sample_sdf_grid::<F>(rotated, objects, g_start, res, size);
                if d_obj < d_scene {
                    d_scene = d_obj;
                    mat_id = F::new(2.0) + F::cast_from(slot as u32);
                }
            } else if use_sphere_bound {
                // Bounding-Shell als grobe Distanz (nur Primärmarch).
                if sphere_d < d_scene {
                    d_scene = sphere_d;
                    mat_id = F::new(0.0);
                }
            }
        }
        slot += usize::new(1);
    }

    Vec2::<F> { d: d_scene, mat_id }
}

/// Normale via zentrale Differenzen über `scene_map_with_material`
/// (6 SDF-Auswertungen entlang der Hauptachsen, ε = 0.001).
#[cube]
pub fn calculate_normal<F: Float>(
    p: Vec3<F>,
    objects: &Tensor<F>,
    bound_offset: F,
) -> Vec3<F> {
    let eps = F::new(0.001);

    // Normalen brauchen die reale Distanz (kein Sphere-Bound — sonst
    // gäbe es an der unsichtbaren Kugelgrenze falsche Normalen).
    let d = scene_map_with_material::<F>(p.clone(), objects, bound_offset, false).d;

    let nx = scene_map_with_material::<F>(
        Vec3::<F>::new(p.x + eps, p.y, p.z), objects, bound_offset, false,
    ).d - d;

    let ny = scene_map_with_material::<F>(
        Vec3::<F>::new(p.x, p.y + eps, p.z), objects, bound_offset, false,
    ).d - d;

    let nz = scene_map_with_material::<F>(
        Vec3::<F>::new(p.x, p.y, p.z + eps), objects, bound_offset, false,
    ).d - d;

    Vec3::<F>::new(nx, ny, nz).normalize()
}
