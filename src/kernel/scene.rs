//! `kernel/scene.rs` — Szenen-Map: scene_sdf (Slots + Arch) + scene_sdf_normal.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::objects::{evaluate_dynamic_crystal, evaluate_dynamic_gyroid, evaluate_dynamic_torus};
use super::types::{ArchParams, FoldParams, SdfResult, Vec3};

// In src/kernel/scene.rs oben bei den Imports hinzufügen:
use crate::gpu::{SLOTS_COUNT, MATERIALS_COUNT};

#[cube]
pub fn scene_sdf<F: Float>(
    p: Vec3<F>,
    time: F,
    blend_factor: F,
    active_slots: u32,
    scene: &Tensor<F>, // 💡 KORREKTUR: Akzeptiert den verpackten Mega-Buffer
    arch: ArchParams<F>,
    fold: FoldParams<F>,
) -> SdfResult<F> {
    // 🟢 TENSOR 6: Modulo-Grid-Konstanten dynamisch aus dem Faltungs-Register
    let cell_size = fold.cell_size;
    let half_cell = fold.half_cell;

    let grid_p_x = p.x - cell_size * ((p.x + half_cell) / cell_size).floor();
    let grid_p_y = p.y - cell_size * ((p.y + half_cell) / cell_size).floor();
    let grid_p_z = p.z - cell_size * ((p.z + half_cell) / cell_size).floor();

    let local_p = Vec3::<F>::new(grid_p_x, grid_p_y, grid_p_z);

    let mut min_dist = F::new(1000.0);

    let mut sum_r = F::new(0.0);
    let mut sum_g = F::new(0.0);
    let mut sum_b = F::new(0.0);
    let mut sum_rough = F::new(0.0);
    let mut sum_metal = F::new(0.0);
    let mut sum_emiss = F::new(0.0);
    let mut sum_spec = F::new(0.0);
    let mut sum_w = F::new(0.0);

    // 💡 Die fixen Offsets deines Mega-Buffers als GPU-Usize-Konstanten
       // 💡 KORREKTUR: Nutze cast_from für Host-Konstanten innerhalb des #[cube]-Makros
    let slots_start = usize::new(0);
    let materials_start = usize::cast_from(SLOTS_COUNT);                   // 800
    let rotations_start = usize::cast_from(SLOTS_COUNT + MATERIALS_COUNT); // 1200


    let active_slots = usize::cast_from(active_slots);

    let mut i = usize::new(0);
    loop {
        if i >= usize::new(100) || i >= active_slots {
            break;
        }

              let base_idx = i * usize::new(8);
        // 💡 KORREKTUR: Auslesen aus dem Tensor mit manuellem Offset auf GPU-Usize-Ebene
        let obj_type = u32::cast_from(scene[slots_start + base_idx]);

        if obj_type > u32::new(0) {
            let obj_size: F = scene[slots_start + base_idx + usize::new(1)];
            let offset_x: F = scene[slots_start + base_idx + usize::new(2)];
            let offset_y: F = scene[slots_start + base_idx + usize::new(3)];
            let offset_z: F = scene[slots_start + base_idx + usize::new(4)];
            let obj_r: F = scene[slots_start + base_idx + usize::new(5)];
            let obj_g: F = scene[slots_start + base_idx + usize::new(6)];
            let obj_b: F = scene[slots_start + base_idx + usize::new(7)];

            let mat_idx = i * usize::new(4);
            let obj_rough: F = scene[materials_start + mat_idx];
            let obj_metal: F = scene[materials_start + mat_idx + usize::new(1)];
            let obj_emiss: F = scene[materials_start + mat_idx + usize::new(2)];
            let obj_spec: F = scene[materials_start + mat_idx + usize::new(3)];
            
            let rot_idx = i * usize::new(3);
            let rot_x: F = scene[rotations_start + rot_idx];
            let rot_y: F = scene[rotations_start + rot_idx + usize::new(1)];
            let rot_z: F = scene[rotations_start + rot_idx + usize::new(2)];


            // ... (Restlicher Schleifencode bleibt unberührt!)


            // 1. Die relative Position des Objekts berechnen (Translation)
            let shifted_p_x: F = p.x - offset_x;
            let shifted_p_y: F = p.y - offset_y;
            let shifted_p_z: F = p.z - offset_z;
//=========================================================================
            // TENSOR 7: Inverse Euler-Rotation des Objekt-Raums (Z -> Y -> X rueckwaerts)
            // Standard-SDF-Transform: Punkt in den lokalen Objektraum drehen.
            // Alle Rotationen 0 -> Identitaet (kein visueller Unterschied zu vorher).
            // =========================================================================
                    // Inverse Rotation um Z
            let cz: F = rot_z.cos();
            let sz: F = rot_z.sin();
            let rx1: F = shifted_p_x * cz + shifted_p_y * sz;
            let ry1: F = -shifted_p_x * sz + shifted_p_y * cz;
            
            // Inverse Rotation um Y
            let cy: F = rot_y.cos();
            let sy: F = rot_y.sin();
            let rx2: F = rx1 * cy - shifted_p_z * sy;
            let rz2: F = rx1 * sy + shifted_p_z * cy;
            
            // Inverse Rotation um X
            let cx: F = rot_x.cos();
            let sx: F = rot_x.sin();
            let ry3: F = ry1 * cx + rz2 * sx;
            let rz3: F = -ry1 * sx + rz2 * cx;
            
            // 💡 KORREKTUR: Expliziter Turbofisch, um Inferenzabbruch zu verhindern
            let rotated_p = Vec3::<F>::new(rx2, ry3, rz3);

            // =========================================================================
            // UNENDLICHES RAUMFALTUNGS-GRID (auf rotierten lokalen Punkt)
            // =========================================================================
            let grid_p_x: F = rotated_p.x - cell_size * ((rotated_p.x + half_cell) / cell_size).floor();
            let grid_p_y: F = rotated_p.y - cell_size * ((rotated_p.y + half_cell) / cell_size).floor();
            let grid_p_z: F = rotated_p.z - cell_size * ((rotated_p.z + half_cell) / cell_size).floor();
            
            // 💡 KORREKTUR: Expliziter Turbofisch beim p_slot Konstruktor
            let p_slot = Vec3::<F>::new(grid_p_x, grid_p_y, grid_p_z);

            // Ab hier laufen deine Auswertungen wie gewohnt weiter:
            let mut d_obj: F = F::new(1000.0);
            if obj_type == u32::new(1) {
                d_obj = evaluate_dynamic_crystal::<F>(p_slot.clone(), time, obj_size);
            }
            if obj_type == u32::new(2) {
                d_obj = evaluate_dynamic_gyroid::<F>(p_slot.clone(), time, obj_size);
            }
            if obj_type == u32::new(3) {
                d_obj = evaluate_dynamic_torus::<F>(p_slot.clone(), time, obj_size);
            }

            if min_dist > F::new(999.0) {
                min_dist = d_obj;
            } else {
                let h = (blend_factor - (min_dist - d_obj).abs()).max(F::new(0.0)) / blend_factor;
                min_dist = min_dist.min(d_obj) - h * h * blend_factor * F::new(0.25);
            }

            // Unzerstörbarer, gewichteter Akkumulator für physikalische Eigenschaften
            let dist_max = d_obj.max(F::new(0.001)); 
            let w = F::new(1.0) / (dist_max * dist_max); // optimized
            
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


       let mut core_r = F::new(1.0);
    let mut core_g = F::new(1.0);
    let mut core_b = F::new(1.0);
    let mut core_rough = F::new(0.5);
    let mut core_metal = F::new(0.0);
    let mut core_emiss = F::new(0.0);
    let mut core_spec = F::new(0.5);

    if sum_w > F::new(0.0) {
        core_r = sum_r / sum_w;
        core_g = sum_g / sum_w;
        core_b = sum_b / sum_w;
        core_rough = sum_rough / sum_w;
        core_metal = sum_metal / sum_w;
        core_emiss = sum_emiss / sum_w;
        core_spec = sum_spec / sum_w;
    }
    let core_system = SdfResult::<F> {
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
    let architecture = SdfResult::<F> {
        d: arch_d,
        r: F::new(0.7),
        g: F::new(0.7),
        b: F::new(0.7),
        roughness: F::new(0.8),
        metallic: F::new(0.0),
        emissive: F::new(0.0),
        specular: F::new(0.2),
    };


    // =========================================================================
    // FINALE REINE SCHNITTAUSWERTUNG (Muss zwingend auf primitiver Ebene erfolgen!)
    // =========================================================================
        let mut final_d = F::new(1000.0);
    let mut final_r = F::new(1.0);
    let mut final_g = F::new(1.0);
    let mut final_b = F::new(1.0);

    // 🟢 NEU: Primitive Register für die PBR-Materialien anlegen
    let mut final_rough = F::new(0.5);
    let mut final_metal = F::new(0.0);
    let mut final_emiss = F::new(0.0);
    let mut final_spec = F::new(0.5);

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
    SdfResult::<F> {
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
pub fn scene_sdf_normal<F: Float>(
    p: Vec3<F>,
    time: F,
    blend_factor: F,
    active_slots: u32,
    scene: &Tensor<F>, // 💡 KORREKTUR: Auch hier Tensor anstelle der Slices
    arch: ArchParams<F>,
    fold: FoldParams<F>,
) -> Vec3<F> {
    let eps = F::new(0.002);
    let d_res = scene_sdf::<F>(
        p.clone(), time, blend_factor, active_slots, scene, arch.clone(), fold.clone()
    );
    let d = d_res.d;
    
    let p_x = Vec3::<F>::new(p.x + eps, p.y, p.z);
    let p_y = Vec3::<F>::new(p.x, p.y + eps, p.z);
    let p_z = Vec3::<F>::new(p.x, p.y, p.z + eps);
    
    let nx = scene_sdf::<F>(p_x, time, blend_factor, active_slots, scene, arch.clone(), fold.clone()).d - d;
    let ny = scene_sdf::<F>(p_y, time, blend_factor, active_slots, scene, arch.clone(), fold.clone()).d - d;
    let nz = scene_sdf::<F>(p_z, time, blend_factor, active_slots, scene, arch, fold).d - d;
    
    Vec3::<F>::new(nx, ny, nz).normalize()
}

