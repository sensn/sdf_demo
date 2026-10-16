//! `kernel/scene.rs` — Szenen-Map: scene_sdf (Slots + Arch) + scene_sdf_normal.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::objects::{evaluate_dynamic_crystal, evaluate_dynamic_gyroid, evaluate_dynamic_torus};
use super::types::{ArchParams, FoldParams, SdfResult, Vec3};
#[cube]
pub fn scene_sdf(
    p: Vec3,
    time: f32,
    blend_factor: f32,
    active_slots: u32,
    slots: &[f32],
    materials: &[f32], // 🟢 Dritter Tensor empfangen
    rotations: &[f32], // 🟢 TENSOR 7: Objekt-Rotation (Stride 3)
    arch: ArchParams,        // 🟢 TENSOR 5 → Struct: Tempel-Architektur
    fold: FoldParams,        // 🟢 TENSOR 6 → Struct: Unendliche Raumfaltung
) -> SdfResult {
    // 🟢 TENSOR 6: Modulo-Grid-Konstanten dynamisch aus dem Faltungs-Register
    let cell_size = fold.cell_size;
    let half_cell = fold.half_cell;

    let grid_p_x = p.x - cell_size * ((p.x + half_cell) / cell_size).floor();
    let grid_p_y = p.y - cell_size * ((p.y + half_cell) / cell_size).floor();
    let grid_p_z = p.z - cell_size * ((p.z + half_cell) / cell_size).floor();

    let local_p = Vec3::new(grid_p_x, grid_p_y, grid_p_z);

    // Wir tracken die reine, weich verschmolzene Distanz separat
    let mut min_dist = f32::new(1000.0);

    // Akkumulatoren für die gewichtete Farbmischung UND PBR-Materialien
    let mut sum_r = f32::new(0.0);
    let mut sum_g = f32::new(0.0);
    let mut sum_b = f32::new(0.0);
    //🟢 New PBR Akkulumatoren
    let mut sum_rough = f32::new(0.0);
    let mut sum_metal = f32::new(0.0);
    let mut sum_emiss = f32::new(0.0);
    let mut sum_spec = f32::new(0.0);
    let mut sum_w = f32::new(0.0); // Gesamtgewichtung

    let active_slots = usize::cast_from(active_slots);

    let mut i = usize::new(0);
    loop {
        if i >= usize::new(100) || i >= active_slots {
            break;
        }

        let base_idx = i * usize::new(8); // Starr 8er Stride für Geometrie
        let obj_type = u32::cast_from(slots[base_idx]);

        if obj_type > u32::new(0) {
            let obj_size = slots[base_idx + usize::new(1)];
            let offset_x = slots[base_idx + usize::new(2)];
            let offset_y = slots[base_idx + usize::new(3)];
            let offset_z = slots[base_idx + usize::new(4)];
            let obj_r = slots[base_idx + usize::new(5)];
            let obj_g = slots[base_idx + usize::new(6)];
            let obj_b = slots[base_idx + usize::new(7)];

            // 🟢 Auslesen aus dem dritten Tensor (Materialien nutzen Stride 4!)
            let mat_idx = i * usize::new(4);
            let obj_rough = materials[mat_idx];
            let obj_metal = materials[mat_idx + usize::new(1)];
            let obj_emiss = materials[mat_idx + usize::new(2)];
            let obj_spec = materials[mat_idx + usize::new(3)];
            // TENSOR 7: Rotation auslesen (Stride 3: rotX, rotY, rotZ)
            let rot_idx = i * usize::new(3);
            let rot_x = rotations[rot_idx];
            let rot_y = rotations[rot_idx + usize::new(1)];
            let rot_z = rotations[rot_idx + usize::new(2)];

            // 1. Die relative Position des Objekts berechnen (Translation)
            let shifted_p_x = p.x - offset_x;
            let shifted_p_y = p.y - offset_y;
            let shifted_p_z = p.z - offset_z;

            // =========================================================================
            // TENSOR 7: Inverse Euler-Rotation des Objekt-Raums (Z -> Y -> X rueckwaerts)
            // Standard-SDF-Transform: Punkt in den lokalen Objektraum drehen.
            // Alle Rotationen 0 -> Identitaet (kein visueller Unterschied zu vorher).
            // =========================================================================
            // Inverse Rotation um Z
            let cz = rot_z.cos();
            let sz = rot_z.sin();
            let rx1 = shifted_p_x * cz + shifted_p_y * sz;
            let ry1 = -shifted_p_x * sz + shifted_p_y * cz;
            // Inverse Rotation um Y
            let cy = rot_y.cos();
            let sy = rot_y.sin();
            let rx2 = rx1 * cy - shifted_p_z * sy;
            let rz2 = rx1 * sy + shifted_p_z * cy;
            // Inverse Rotation um X
            let cx = rot_x.cos();
            let sx = rot_x.sin();
            let ry3 = ry1 * cx + rz2 * sx;
            let rz3 = -ry1 * sx + rz2 * cx;
            let rotated_p = Vec3::new(rx2, ry3, rz3);

            // =========================================================================
            // UNENDLICHES RAUMFALTUNGS-GRID (auf rotierten lokalen Punkt)
            // =========================================================================
            let grid_p_x = rotated_p.x - cell_size * ((rotated_p.x + half_cell) / cell_size).floor();
            let grid_p_y = rotated_p.y - cell_size * ((rotated_p.y + half_cell) / cell_size).floor();
            let grid_p_z = rotated_p.z - cell_size * ((rotated_p.z + half_cell) / cell_size).floor();
            let p_slot = Vec3::new(grid_p_x, grid_p_y, grid_p_z);

            // Ab hier laufen deine Auswertungen wie gewohnt weiter:
            // 

            let mut d_obj = f32::new(1000.0);
            if obj_type == u32::new(1) {
                d_obj = evaluate_dynamic_crystal(p_slot, time, obj_size);
            }
            if obj_type == u32::new(2) {
                d_obj = evaluate_dynamic_gyroid(p_slot, time, obj_size);
            }
            if obj_type == u32::new(3) {
                d_obj = evaluate_dynamic_torus(p_slot, time, obj_size);
            }

            if min_dist > f32::new(999.0) {
                min_dist = d_obj;
            } else {
                let h = (blend_factor - (min_dist - d_obj).abs()).max(f32::new(0.0)) / blend_factor;
                min_dist = min_dist.min(d_obj) - h * h * blend_factor * f32::new(0.25);
            }

            // Unzerstörbarer, gewichteter Akkumulator für physikalische Eigenschaften
            //let w = f32::new(1.0) / (d_obj.max(f32::new(0.001))).powf(f32::new(2.0));  //original
            let dist_max = d_obj.max(f32::new(0.001)); let w = f32::new(1.0) / (dist_max * dist_max); //optimized
            
            sum_r += obj_r * w;
            sum_g += obj_g * w;
            sum_b += obj_b * w;
            sum_rough += obj_rough * w;
            sum_metal += obj_metal * w;
            sum_emiss += obj_emiss * w;
            sum_spec += obj_spec * w;
            sum_w += w;
        }
        i += usize::new(1);
    }

    let mut core_r = f32::new(1.0);
    let mut core_g = f32::new(1.0);
    let mut core_b = f32::new(1.0);
    let mut core_rough = f32::new(0.5);
    let mut core_metal = f32::new(0.0);
    let mut core_emiss = f32::new(0.0);
    let mut core_spec = f32::new(0.5);

    if sum_w > f32::new(0.0) {
        core_r = sum_r / sum_w;
        core_g = sum_g / sum_w;
        core_b = sum_b / sum_w;
        core_rough = sum_rough / sum_w;
        core_metal = sum_metal / sum_w;
        core_emiss = sum_emiss / sum_w;
        core_spec = sum_spec / sum_w;
    }
    let core_system = SdfResult {
        d: min_dist,
        r: core_r,
        g: core_g,
        b: core_b,
        roughness: core_rough,
        metallic: core_metal,
        emissive: core_emiss,
        specular: core_spec,
    };

    // =========================================================================
    // 🟢 TENSOR 5: ARCHITEKTUR-GENERIERUNG (Datengetrieben statt starr)
    // =========================================================================
    let pillar_dist = arch.pillar_dist;
    let pillar_thick = arch.pillar_thick;
    let room_height = arch.room_height;
    let ceiling_thick = arch.ceiling_thick;
    let arch_radius = arch.arch_radius;
    let arch_height = arch.arch_height;
    let decor_freq = arch.decor_freq;
    let decor_depth = arch.decor_depth;
    let decor_thick = arch.decor_thick;

    let pillar_x = (local_p.x.abs() - pillar_dist).abs() - pillar_thick;
    let pillar_z = (local_p.z.abs() - pillar_dist).abs() - pillar_thick;
    let corner_pillars = pillar_x.max(pillar_z);

    let room_floor = local_p.y + room_height;
    let room_ceiling = room_height - local_p.y;
    let floor_and_ceiling = room_floor.min(room_ceiling) - ceiling_thick;

    let arch_z =
        (local_p.x * local_p.x + (local_p.y - arch_height) * (local_p.y - arch_height)).sqrt()
            - arch_radius;
    let arch_x =
        (local_p.z * local_p.z + (local_p.y - arch_height) * (local_p.y - arch_height)).sqrt()
            - arch_radius;
    let wall_arches = arch_z.min(arch_x);

    let mut arch_d = corner_pillars.min(floor_and_ceiling);
    arch_d = arch_d.max(-wall_arches);

    let pillar_holes = ((local_p.x * decor_freq).sin().abs()
        + (local_p.y * decor_freq).cos().abs()
        + (local_p.z * decor_freq).sin().abs())
        * decor_depth;
    arch_d = arch_d.max(-(pillar_holes - decor_thick));

    // (Starre Raumarchitektur erhält Standard-PBR-Werte: Mattstein = roughness 0.8, metallic 0.0)
    let architecture = SdfResult {
        d: arch_d,
        r: f32::new(0.7),
        g: f32::new(0.7),
        b: f32::new(0.7),
        roughness: f32::new(0.8),
        metallic: f32::new(0.0),
        emissive: f32::new(0.0),
        specular: f32::new(0.2),
    };

    // =========================================================================
    // FINALE REINE SCHNITTAUSWERTUNG (Muss zwingend auf primitiver Ebene erfolgen!)
    // =========================================================================
    let mut final_d = f32::new(1000.0);
    let mut final_r = f32::new(1.0);
    let mut final_g = f32::new(1.0);
    let mut final_b = f32::new(1.0);

    // 🟢 NEU: Primitive Register für die PBR-Materialien anlegen
    let mut final_rough = f32::new(0.5);
    let mut final_metal = f32::new(0.0);
    let mut final_emiss = f32::new(0.0);
    let mut final_spec = f32::new(0.5);

    let check_core = core_system.clone();
    let check_arch = architecture.clone();

    if check_core.d < check_arch.d {
        final_d = check_core.d;
        final_r = check_core.r;
        final_g = check_core.g;
        final_b = check_core.b;
        // PBR-Zuweisung vom Core-System
        final_rough = check_core.roughness;
        final_metal = check_core.metallic;
        final_emiss = check_core.emissive;
        final_spec = check_core.specular;
    } else {
        final_d = check_arch.d;
        final_r = check_arch.r;
        final_g = check_arch.g;
        final_b = check_arch.b;
        // PBR-Zuweisung von der starren Raum-Architektur
        final_rough = check_arch.roughness;
        final_metal = check_arch.metallic;
        final_emiss = check_arch.emissive;
        final_spec = check_arch.specular;
    }

    // Baue das SdfResult erst ganz am Ende beim return frisch zusammen
    SdfResult {
        d: final_d,
        r: final_r,
        g: final_g,
        b: final_b,
        roughness: final_rough,
        metallic: final_metal,
        emissive: final_emiss,
        specular: final_spec,
    }
}

#[cube]
pub fn scene_sdf_normal(
    p: Vec3,
    time: f32,
    blend_factor: f32,
    active_slots: u32,
    slots: &[f32],
    materials: &[f32],
    rotations: &[f32],
    arch: ArchParams,
    fold: FoldParams,
) -> Vec3 {
    let eps = f32::new(0.002);
    let d_res = scene_sdf(p.clone(), time, blend_factor, active_slots, slots, materials,
      rotations, arch, fold);
    let d = d_res.d;
    let p_x = Vec3::new(p.x + eps, p.y, p.z);
    let p_y = Vec3::new(p.x, p.y + eps, p.z);
    let p_z = Vec3::new(p.x, p.y, p.z + eps);
    let nx = scene_sdf(p_x, time, blend_factor, active_slots, slots, materials,
      rotations, arch, fold).d - d;
    let ny = scene_sdf(p_y, time, blend_factor, active_slots, slots, materials,
      rotations, arch, fold).d - d;
    let nz = scene_sdf(p_z, time, blend_factor, active_slots, slots, materials,
      rotations, arch, fold).d - d;
    Vec3::new(nx, ny, nz).normalize()
}
