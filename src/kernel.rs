#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

// Hole die Basis-Konstanten aus dem GPU-Modul
use crate::gpu::{MAX_SLOTS, SLOTS_COUNT, MATERIALS_COUNT, ROTATIONS_COUNT};
// 🟢 PLAN STEP 3: SDF-Grid-Konstanten (gpu.rs + sdf_preprocessor.rs)
use crate::gpu::{SDF_GRID_COUNT, SDF_GRID_RES};


#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct SdfResult {
    pub d: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    // 🟢 NEU: Physikalische Oberflächen-Eigenschaften
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: f32,
    pub specular: f32,
}

#[cube]
pub fn smin_material(a: SdfResult, b: SdfResult, k: f32) -> SdfResult {
    let h = (k - (a.d - b.d).abs()).max(f32::new(0.0)) / k;

    // Berechne die interpolierte Distanz
    let mixed_d = a.d.min(b.d) - h * h * k * f32::new(0.25);

    // Lineare Farbmischung basierend auf der Nähe zum jeweiligen Objekt
    // Wenn h nah an 1 ist, befinden wir uns in der Übergangszone
    let mix_factor = (f32::new(0.5) + f32::new(0.5) * (b.d - a.d) / k)
        .min(f32::new(1.0))
        .max(f32::new(0.0));

    let mixed_r = a.r + mix_factor * (b.r - a.r);
    let mixed_g = a.g + mix_factor * (b.g - a.g);
    let mixed_b = a.b + mix_factor * (b.b - a.b);
    // 🟢 NEU: Weiches Überblenden der PBR-Materialien in der Mischzone!
    let mixed_rough = a.roughness + mix_factor * (b.roughness - a.roughness);
    let mixed_metal = a.metallic + mix_factor * (b.metallic - a.metallic);
    let mixed_emiss = a.emissive + mix_factor * (b.emissive - a.emissive);
    let mixed_spec = a.specular + mix_factor * (b.specular - a.specular);

    SdfResult {
        d: mixed_d,
        r: mixed_r,
        g: mixed_g,
        b: mixed_b,
        roughness: mixed_rough,
        metallic: mixed_metal,
        emissive: mixed_emiss,
        specular: mixed_spec,
    }
}

// 1. Zuerst die normalen Rust-Derives
#[derive(CubeType, Copy, Clone)]
// HIER: Damit wird das Trait CloneExpand automatisch für Vec3Expand generiert!
#[cube(derive(Copy, Clone))]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[cube]
impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }
    pub fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
    pub fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
    pub fn scale(self, factor: f32) -> Vec3 {
        Vec3::new(self.x * factor, self.y * factor, self.z * factor)
    }
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn length(self) -> f32 {
        f32::sqrt(self.dot(self))
    }
    pub fn normalize(self) -> Vec3 {
        let len = self.length();
        let mut x = self.x;
        let mut y = self.y;
        let mut z = self.z;
        if len > 0.0 {
            let factor = 1.0 / len;
            x = self.x * factor;
            y = self.y * factor;
            z = self.z * factor;
        }
        Vec3::new(x, y, z)
    }
}

#[cube]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(f32::new(0.0)) / k;
    a.min(b) - h * h * k * f32::new(0.25)
}

#[cube]
pub fn evaluate_dynamic_crystal(p: Vec3, time: f32, size: f32) -> f32 {
    let rot_speed = time * f32::new(0.4);
    let cos_r = rot_speed.cos();
    let sin_r = rot_speed.sin();
    let crystal_x = p.x * cos_r - p.z * sin_r;
    let crystal_z = p.x * sin_r + p.z * cos_r;

    let base_crystal =
        (crystal_x.abs() + p.y.abs() + crystal_z.abs() - size) * f32::new(0.57735027);
    let waves = (crystal_x * f32::new(6.0)).sin()
        * (p.y * f32::new(6.0)).sin()
        * (crystal_z * f32::new(6.0)).sin()
        * f32::new(0.04);
    base_crystal + waves
}

#[cube]
pub fn evaluate_dynamic_gyroid(p: Vec3, time: f32, size: f32) -> f32 {
    // Fix: Variable auslagern vor der Verwendung, um CubeType-Fehler zu vermeiden
    let center_y = (time * f32::new(3.5)).sin() * f32::new(0.2);
    let sphere_center = Vec3::new(f32::new(0.0), center_y, f32::new(0.0));
    let base_sphere = p.sub(sphere_center).length() - size;

    let scale = f32::new(6.0);
    let gyroid = ((p.x * scale).sin() * (p.y * scale).cos()
        + (p.x * scale).cos() * (p.z * scale).sin()
        + (p.y * scale).sin() * (p.z * scale).cos())
        / scale;

    base_sphere.max(gyroid * f32::new(0.5))
}

#[cube]
pub fn evaluate_dynamic_torus(p: Vec3, time: f32, size: f32) -> f32 {
    let rot_speed_x = time * f32::new(0.6);
    let rot_speed_y = time * f32::new(0.3);

    let cos_x = rot_speed_x.cos();
    let sin_x = rot_speed_x.sin();
    let cos_y = rot_speed_y.cos();
    let sin_y = rot_speed_y.sin();

    let ry_x = p.x * cos_y - p.z * sin_y;
    let ry_z = p.x * sin_y + p.z * cos_y;
    let rx_y = p.y * cos_x - ry_z * sin_x;
    let rx_z = p.y * sin_x + ry_z * cos_x;

    let r_major = size;
    let r_minor = size * f32::new(0.15);

    let q_x = (ry_x * ry_x + rx_z * rx_z).sqrt() - r_major;
    (q_x * q_x + rx_y * rx_y).sqrt() - r_minor
}


// ============================================================================
// PARAMETER-STRUCTS (doc/params/params1-5.md)
// Ersetzen die Index-basierten Parameter-Tensoren (meta/env/arch/fold) und
// die 14 losen Launch-Skalare durch typisierte CubeLaunch-Structs.
// Skalar-Felder landen im Info-Uniform (16-Byte-Alignment via cubecl
// padded_size), NICHT in Storage-Buffern -> Budget 8 -> 4 (params1).
// Flags/Counts als u32 (0/1), nie f32 (params3).
// ============================================================================

/// 16-Byte-Alignment fuer vec3-Parameter (params5): vec3 + explizites Pad.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

/// Kamera-Parameter (params5: "camera settings struct").
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct CameraSettings {
    pub position: GpuVec3,
    pub yaw: f32,
    pub pitch: f32,
}

/// Umwelt-Parameter (ersetzt Tensor 4 / env_data()[16]).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct EnvSettings {
    pub key_light_pos: GpuVec3, // war env[0..2]
    pub light_intensity: f32,   // war env[3]
    pub key_color: GpuVec3,     // war env[4..6]
    pub ambient_strength: f32,  // war env[7]
    pub bg_color: GpuVec3,      // war env[8..10]
    pub fog_density: f32,       // war env[11]
    pub fog_enabled: u32,       // war env[12] (f32) -> params3: u32-Flag 0/1
    // 🟢 PARAM-GUI: Atmosphäre & Lighting
    pub volumetric_fog_steps: u32, // Nebel-Integrationsschritte (4..64, default 16)
    pub fog_density_falloff: f32,  // Höhenprofil-Falloff (0.01..0.5, default 0.08)
    pub ao_iterations: u32,        // AO-Sampling-Iterationen (1..10, default 5)
    pub shadow_softness_k: f32,     // Weichheits-Konstante (1.0..64.0, default 8.0)
}

/// Tempel-Architektur (ersetzt Tensor 5 / arch_data()[12]).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct ArchParams {
    pub pillar_dist: f32,   // war arch[0]
    pub pillar_thick: f32,  // war arch[1]
    pub room_height: f32,   // war arch[2]
    pub ceiling_thick: f32, // war arch[3]
    pub arch_radius: f32,   // war arch[4]
    pub arch_height: f32,   // war arch[5]
    pub decor_freq: f32,    // war arch[6]
    pub decor_depth: f32,   // war arch[7]
    pub decor_thick: f32,   // war arch[8]
}

/// Unendliche Raumfaltung (ersetzt Tensor 6 / fold_data()[4]).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct FoldParams {
    pub cell_size: f32,  // war fold[0]
    pub half_cell: f32,  // war fold[1]
    pub fold_speed: f32, // war fold[2]
    // 🟢 PARAM-GUI: Infinite Repetition (Pfeiler-Faltung)
    pub pillar_spacing: f32, // XZ-Wiederholungs-Abstand (2.0..20.0, default 6.0)
    pub pillar_height: f32,   // Pfeilerhöhe (1.0..10.0, default 4.0)
}

/// Frame-Parameter & Mode-Flags (ersetzt Meta-Tensor + 12 lose Skalare).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct RenderSettings {
    pub time: f32,
    pub width: u32,
    pub height: u32,
    pub blend_factor: f32,
    pub active_slots: u32, // ersetzt meta[0] (f32) -> params3: Count als u32
    pub shadow_mode: u32,
    pub enable_ao_mode: u32,
    pub enable_key: u32,
    pub enable_fill: u32,
    pub enable_rim: u32,
    // 🟢 PARAM-GUI: Materialien & Reflexionen
    pub reflection_max_steps: u32,   // Sekundär-Schritte (8..64, default 32)
    pub reflection_roughness_gate: f32, // Aktivierungsschwelle (0.05..1.0, default 0.25)
    pub fresnel_strength_factor: f32,  // Reflexions-Stärke (0.0..1.0, default 0.35)
    pub statue_metallic: f32,          // Statue PBR (ID 2), default 0.8
    pub pillar_roughness: f32,         // Pfeiler PBR (ID 1), default 0.8
    // 🟢 PARAM-GUI: Objekt / Statue-Konfiguration
    pub sdf_grid_size: f32,             // Grid-Ausdehnung (1.0..16.0, default 4.0)
    pub statue_center: GpuVec3,         // Statue-Zentrum (default [0, -1.4, 0])
    pub bounding_sphere_radius_offset: f32, // Proximity-Schranke (default 0.1)
}

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
// ============================================================================
// 🟢 PLAN-IMPLEMENTATION (Agent_Impl_Plan.md, Steps 3–7)
// ============================================================================
/// Trilineares Sampling des 3D-SDF-Grids (Step 3).
///
/// `p` liegt im Welt-Raum; das Grid spannt `[-size/2, size/2]^3` auf und
/// ist in ZYX-Reihenfolge geflacht (index = x + res*(y + res*z)).
/// 8 Voxel werden nach den fraktionalen Anteilen gemischt.
#[cube]
pub fn sample_sdf_grid(p: Vec3, grid: &[f32], res: u32, size: f32) -> f32 {
    let res_f = f32::cast_from(res);
    let res_us = usize::cast_from(res);
    let half = size * f32::new(0.5);
    // In Grid-Koordinaten [0, res-1] mappen.
    let local_x = (p.x + half) / size * (res_f - f32::new(1.0));
    let local_y = (p.y + half) / size * (res_f - f32::new(1.0));
    let local_z = (p.z + half) / size * (res_f - f32::new(1.0));
    // Auf [0, res-2] klemmen, damit x0+1/y0+1/z0+1 valide Indizes sind.
    let max_bound = res_f - f32::new(2.0);
    let lp_x = local_x.max(f32::new(0.0)).min(max_bound);
    let lp_y = local_y.max(f32::new(0.0)).min(max_bound);
    let lp_z = local_z.max(f32::new(0.0)).min(max_bound);
    // Basis-Voxel + fraktionale Gewichte.
    let x0 = u32::cast_from(lp_x);
    let y0 = u32::cast_from(lp_y);
    let z0 = u32::cast_from(lp_z);
    let tx = lp_x - f32::cast_from(x0);
    let ty = lp_y - f32::cast_from(y0);
    let tz = lp_z - f32::cast_from(z0);
    let x0_us = usize::cast_from(x0);
    let y0_us = usize::cast_from(y0);
    let z0_us = usize::cast_from(z0);
    // 8 Ecken des Voxels (ZYX-Flattening).
    let c000 = grid[x0_us + res_us * (y0_us + res_us * z0_us)];
    let c100 = grid[(x0_us + usize::new(1)) + res_us * (y0_us + res_us * z0_us)];
    let c010 = grid[x0_us + res_us * ((y0_us + usize::new(1)) + res_us * z0_us)];
    let c110 = grid[(x0_us + usize::new(1)) + res_us * ((y0_us + usize::new(1)) + res_us * z0_us)];
    let c001 = grid[x0_us + res_us * (y0_us + res_us * (z0_us + usize::new(1)))];
    let c101 = grid[(x0_us + usize::new(1)) + res_us * (y0_us + res_us * (z0_us + usize::new(1)))];
    let c011 = grid[x0_us + res_us * ((y0_us + usize::new(1)) + res_us * (z0_us + usize::new(1)))];
    let c111 = grid[(x0_us + usize::new(1)) + res_us * ((y0_us + usize::new(1)) + res_us * (z0_us + usize::new(1)))];
    // Trilineare Mischung entlang X, dann Y, dann Z.
    let x00 = c000 * (f32::new(1.0) - tx) + c100 * tx;
    let x10 = c010 * (f32::new(1.0) - tx) + c110 * tx;
    let x01 = c001 * (f32::new(1.0) - tx) + c101 * tx;
    let x11 = c011 * (f32::new(1.0) - tx) + c111 * tx;
    let y0v = x00 * (f32::new(1.0) - ty) + x10 * ty;
    let y1v = x01 * (f32::new(1.0) - ty) + x11 * ty;
    y0v * (f32::new(1.0) - tz) + y1v * tz
}
// HINWEIS: `op_repetition_xz` + `sdf_fractal_pillar` (Plan Step 4)
// wurden entfernt — sie kollidierten mit der Arch-Zellen-Faltung aus
// `scene_sdf` (Phantom-Pfeiler in Nachbarräumen, kaputte Schatten).
// Die Architektur generiert ihre eigenen Pfeiler datengetrieben
// (ArchParams: pillar_dist/thick, room_height, ...).
/// Szenen-Map mit Material-ID (Step 5): das Statuen-SDF-Grid mit
/// Bounding-Sphere-Distance-Culling für das teure Grid-Sampling.
/// Rückgabe: `Vec2 { d, mat_id }` (mat_id: 2=Statue).
///
/// ⚠️ Die Pfeiler des Plans wurden ENTFERNT: sie kollidierten mit der
/// bestehenden Arch-Zelle (anderes Folding, andere Höhe) und erzeugten
/// Phantom-Pfeiler in Nachbarräumen + kaputte Schatten. Die Architektur
/// hat ihre eigenen, korrekt gefalteten Pfeiler (siehe `scene_sdf`).
#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct Vec2 {
    pub d: f32,
    pub mat_id: f32,
}
#[cube]
pub fn scene_map_with_material(
    p: Vec3,
    spacing: f32,
    pillar_height: f32,
    grid: &[f32],
    grid_res: u32,
    grid_size: f32,
    use_sphere_bound: bool,
    statue_center: GpuVec3,
    bounding_sphere_radius_offset: f32,
) -> Vec2 {
    let mut d_scene = f32::new(1000.0);
    let mut mat_id = f32::new(2.0);
    // Statuen-Grid: Bounding-Sphere-Culling — nur samplen, wenn der
    // Punkt nahe an der Statuen-Kugel liegt (Radius = grid_size/2).
    // WICHTIG für den Raymarch: außerhalb der Kugel ist die exakte
    // Distanz zur Kugel eine gültige untere Schranke (Lipschitz-
    // Bedingung) — der Marcher kann nicht durch die Statue tunneln.
    // `use_sphere_bound`: Primär-March JA (sicheres Stepping), Schatten/
    // AO NEIN (sonst wirft die unsichtbare Kugel einen Phantom-Schatten
    // — dunkler Ring um die Statue).
    let statue_center_v = Vec3::new(
        statue_center.x,
        statue_center.y,
        statue_center.z,
    );
    let statue_pos = p.sub(statue_center_v);
    let sphere_d = statue_pos.length() - grid_size * f32::new(0.5);
    if sphere_d < bounding_sphere_radius_offset {
        let d_statue = sample_sdf_grid(statue_pos, grid, grid_res, grid_size);
        if d_statue < d_scene {
            d_scene = d_statue;
        }
    } else if use_sphere_bound {
        // Außerhalb: Distanz zur Bounding-Kugel als sichere Schranke
        // (nur für den Primär-March — Schatten/AO sehen das nicht!).
        // mat_id bleibt 0 — die Kugel ist KEIN Objekt, nur eine
        // Schritt-Schranke (kein Material, keine Normalen von hier).
        d_scene = sphere_d;
        mat_id = f32::new(0.0);
    }
    Vec2 { d: d_scene, mat_id }
}
/// Normale via zentrale Differenzen über `scene_map_with_material`
/// (Step 5): 6 SDF-Auswertungen entlang der Hauptachsen, ε = 0.001.
#[cube]
pub fn calculate_normal(
    p: Vec3,
    spacing: f32,
    pillar_height: f32,
    grid: &[f32],
    grid_res: u32,
    grid_size: f32,
    statue_center: GpuVec3,
    bounding_sphere_radius_offset: f32,
) -> Vec3 {
    let eps = f32::new(0.001);
    // Normalen brauchen die reale Distanz (kein Sphere-Bound — sonst
    // gäbe es an der unsichtbaren Kugelgrenze falsche Normalen).
    let d = scene_map_with_material(p.clone(), spacing, pillar_height, grid, grid_res, grid_size, false, statue_center, bounding_sphere_radius_offset)
        .d;
    let nx = scene_map_with_material(
        Vec3::new(p.x + eps, p.y, p.z),
        spacing,
        pillar_height,
        grid,
        grid_res,
        grid_size,
        false,
        statue_center,
        bounding_sphere_radius_offset,
    )
    .d - d;
    let ny = scene_map_with_material(
        Vec3::new(p.x, p.y + eps, p.z),
        spacing,
        pillar_height,
        grid,
        grid_res,
        grid_size,
        false,
        statue_center,
        bounding_sphere_radius_offset,
    )
    .d - d;
    let nz = scene_map_with_material(
        Vec3::new(p.x, p.y, p.z + eps),
        spacing,
        pillar_height,
        grid,
        grid_res,
        grid_size,
        false,
        statue_center,
        bounding_sphere_radius_offset,
    )
    .d - d;
    Vec3::new(nx, ny, nz).normalize()
}
/// Volumetrischer Nebel (Step 7): schrittweise Integration der
/// Streu-Dichte entlang des Primärstrahls bis zur Treffer-Distanz.
/// Liefert Transmissions-Faktor [0,1] (1 = keine Absorption).
#[cube]
pub fn integrate_volumetric_fog(
    ro: Vec3,
    rd: Vec3,
    hit_dist: f32,
    max_dist: f32,
    fog_density: f32,
    steps: u32,
    fog_density_falloff: f32,
) -> f32 {
    let march_dist = hit_dist.min(max_dist);
    
    // Gültigkeitsprüfung ohne 'return' verarbeitet
    let is_valid = fog_density > f32::new(0.0) && march_dist > f32::new(0.0);

    let mut optical_depth = f32::new(0.0);

    if is_valid {
        let n = f32::cast_from(steps).max(f32::new(1.0));
        let step_len = march_dist / n;
        let mut i = u32::new(0);

        loop {
            if i >= steps {
                break;
            }

            // Sample-Knoten in Strahl-Mitte der Segmente
            let t = (f32::cast_from(i) + f32::new(0.5)) * step_len;
            let pos = ro.add(rd.scale(t));

            // Höhenprofil
            let height_falloff = (f32::new(1.0) - pos.y * fog_density_falloff).max(f32::new(0.1));
            let density = fog_density * height_falloff;

            // Optische Tiefe aufsummieren
            optical_depth += density;

            // Früher Ausstieg aus der Schleife (break ist auf der GPU meist erlaubt)
            if optical_depth * step_len > f32::new(4.605) {
                optical_depth = f32::new(4.605) / step_len;
                break;
            }

            i += u32::new(1);
        }

        // Exponentiation am Ende, falls der Pfad gültig war
        (-optical_depth * step_len).exp()
    } else {
        // Falls ungültig (kein Nebel oder Distanz 0), ist die Transmittanz 1.0
        f32::new(1.0)
    }
}


/// 1-Bounce-Reflexion (Step 7): sekundärer Raymarch vom Trefferpunkt
/// entlang der reflektierten Richtung, max. 32 Schritte. Liefert die
/// reflektierte Szenen-Farbe (Hintergrund-Fallback bei Miss).
/// 🟢 MARSCHIERT DIE KOMPLETTE SZENE (Slots + Arch + Statue) — nur so
/// spiegeln glatte Flächen wirklich die Umgebung (Mirror-Effekt).
#[cube]
pub fn trace_reflection(
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
    sdf_grid: &[f32],
    grid_res: u32,
    grid_size: f32,
    bg_color: Vec3,
    max_dist: f32,
    reflection_max_steps: u32,
    statue_center: GpuVec3,
    bounding_sphere_radius_offset: f32,
) -> Vec3 {
    let mut t = f32::new(0.02);
    let mut hit_dist = max_dist;
    let mut i = u32::new(0);
    // Material-Register des Reflektions-Treffers.
    let mut hit_r = bg_color.x;
    let mut hit_g = bg_color.y;
    let mut hit_b = bg_color.z;
    loop {
        if i >= reflection_max_steps {
            break;
        }
        let p = ro.add(rd.scale(t));
        // Komplette Szene: Slots + Arch (scene_sdf) …
        let sdf_res = scene_sdf(p.clone(), time, blend_factor, active_slots, slots, materials,
          rotations, arch, fold);
        let mut d = sdf_res.d;
        let mut r = sdf_res.r;
        let mut g = sdf_res.g;
        let mut b = sdf_res.b;
        // … plus Statue (ohne Sphere-Bound — sonst Phantom-Reflexionen).
        let plan_res = scene_map_with_material(
            p.clone(), f32::new(0.0), f32::new(0.0), sdf_grid, grid_res, grid_size, false, statue_center, bounding_sphere_radius_offset,
        );
        if plan_res.d < d {
            d = plan_res.d;
            // Statue: gold-bronze.
            r = f32::new(0.85);
            g = f32::new(0.7);
            b = f32::new(0.3);
        }
        if d < f32::new(0.001) {
            hit_dist = t;
            hit_r = r;
            hit_g = g;
            hit_b = b;
            break;
        }
        t += d;
        if t > max_dist {
            break;
        }
        i += u32::new(1);
    }
    // Ergebnis: Albedo des getroffenen Objekts mit einfachem N·L-
    // Shading (Licht-Richtung wie im Haupt-Renderer).
    let mut out_r = bg_color.x;
    let mut out_g = bg_color.y;
    let mut out_b = bg_color.z;
    if hit_dist < max_dist {
        let p = ro.add(rd.scale(hit_dist));
        let mut n_x = f32::new(0.0);
        let mut n_y = f32::new(1.0);
        let mut n_z = f32::new(0.0);
        // Normale über die Map, die den Treffer geliefert hat.
        let sdf_n = scene_sdf_normal(p.clone(), time, blend_factor, active_slots, slots, materials,
          rotations, arch, fold);
        let plan_n = calculate_normal(p.clone(), f32::new(0.0), f32::new(0.0), sdf_grid, grid_res, grid_size, statue_center, bounding_sphere_radius_offset);
        // Wähle per Distanz-Vergleich (keine if-else-Expr für Structs).
        let sdf_d = scene_sdf(p.clone(), time, blend_factor, active_slots, slots, materials,
          rotations, arch, fold).d;
        let plan_d = scene_map_with_material(
            p.clone(), f32::new(0.0), f32::new(0.0), sdf_grid, grid_res, grid_size, false, statue_center, bounding_sphere_radius_offset,
        ).d;
        let use_plan = plan_d < sdf_d;
        n_x = select(use_plan, plan_n.x, sdf_n.x);
        n_y = select(use_plan, plan_n.y, sdf_n.y);
        n_z = select(use_plan, plan_n.z, sdf_n.z);
        let light_dir = Vec3::new(
            f32::new(0.5),
            f32::new(0.8),
            f32::new(0.3),
        )
        .normalize();
        let ndl = Vec3::new(n_x, n_y, n_z).dot(light_dir).max(f32::new(0.0));
        out_r = hit_r * ndl;
        out_g = hit_g * ndl;
        out_b = hit_b * ndl;
    }
    Vec3::new(out_r, out_g, out_b)
}
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

// =========================================================================
// 🟢 PBR: COOK-TORRANCE-HILFSFUNKTIONEN (siehe PBR_Optimisation_plan.md)
// =========================================================================

/// Schlick-Fresnel: Reflexionsgrad bei Einfallswinkel cos_theta.
#[cube]
pub fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let f = (f32::new(1.0) - cos_theta).powf(f32::new(5.0));
    f0 + (f32::new(1.0) - f0) * f
}

/// GGX-Normalverteilung: wie viel Energie geht in den Halbwinkel.
#[cube]
pub fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - f32::new(1.0)) + f32::new(1.0);
    // PI als Bruch 355/113 (clippy: keine approximierten Konstanten)
    let pi = f32::new(355.0) / f32::new(113.0);
    (a2 / (pi * d * d)).max(f32::new(0.0))
}

/// Smith-Schlick-Geometrie: Maskierung/Schattierung der Mikrofacetten.
#[cube]
pub fn g_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let k = (alpha + f32::new(1.0)) * (alpha + f32::new(1.0)) / f32::new(8.0);
    let gv = n_dot_v / (n_dot_v * (f32::new(1.0) - k) + k);
    let gl = n_dot_l / (n_dot_l * (f32::new(1.0) - k) + k);
    gv * gl
}

/// 🟢 ACES-Narkowicz-Tone-Mapping: HDR-Werte > 1.0 werden natürlich
/// komprimiert statt hart geclampt (behebt das Emissive-Weiß-Clipping).
#[cube]
pub fn aces_tonemap(x: f32) -> f32 {
    let a = f32::new(2.51);
    let b = f32::new(0.03);
    let c = f32::new(2.43);
    let d = f32::new(0.59);
    let e = f32::new(0.14);
    (x * (a * x + b)) / (x * (c * x + d) + e)
}

/// 🟢 Ein komplettes Cook-Torrance-Licht für einen Punkt.
/// Gibt den Beitrag (diffuse + specular) eines Lichts zurück.
/// n_dot_l <= 0 (Licht hinter der Fläche) ergibt automatisch 0, da alle
/// Terme mit n_dot_l bzw. max(0,·) skaliert sind — kein early return nötig.
#[cube]
pub fn cook_torrance_light(
    albedo: Vec3,
    normal: Vec3,
    light_dir: Vec3, // zum Licht gerichtet, normalisiert
    view_dir: Vec3,  // zur Kamera gerichtet, normalisiert
    light_color: Vec3,
    intensity: f32,
    roughness: f32,
    metallic: f32,
    specular: f32,
) -> Vec3 {
    let eps = f32::new(0.0001);
    // PI als Bruch (clippy: keine approximierten Konstanten-Literale)
    let pi = f32::new(355.0) / f32::new(113.0);
    let n_dot_l = normal.dot(light_dir.clone()).max(f32::new(0.0));
    let n_dot_v = normal.dot(view_dir.clone()).max(eps);
    let h = light_dir.add(view_dir).normalize();
    let n_dot_h = normal.dot(h.clone()).max(f32::new(0.0));
    let l_dot_h = light_dir.dot(h).max(f32::new(0.0));

    // F0: Dielektrikum ~0.04 (durch specular-Slider skalierbar), Metall = Albedo
    let f0_scalar = f32::new(0.04) * (f32::new(1.0) + specular * f32::new(3.0));
    let f0 = Vec3::new(
        f0_scalar + (albedo.x - f0_scalar) * metallic,
        f0_scalar + (albedo.y - f0_scalar) * metallic,
        f0_scalar + (albedo.z - f0_scalar) * metallic,
    );

    let alpha = roughness * roughness;
    let d = d_ggx(n_dot_h, alpha);
    let g = g_smith(n_dot_v, n_dot_l, alpha);
    let f = fresnel_schlick(l_dot_h, f0.x); // Skalar-Fresnel (visuell ausreichend)

    // Specular-BRDF: D*G*F / (4 * NdotL * NdotV)
    let denom = (n_dot_v * n_dot_l * f32::new(4.0)).max(eps);
    let spec_brdf = d * g * f / denom;

    // Diffuse-Rest: (1-F) * (1-metallic) — Metalle haben keine diffuse Farbe
    let kd = (f32::new(1.0) - f) * (f32::new(1.0) - metallic);

    // Lambert / PI
    let diff_r = kd * albedo.x / pi * n_dot_l;
    let diff_g = kd * albedo.y / pi * n_dot_l;
    let diff_b = kd * albedo.z / pi * n_dot_l;

    // Specular ist farbig (F0-farbig für Metalle, weiß für Dielektrika)
    let spec_r = spec_brdf * f0.x;
    let spec_g = spec_brdf * f0.y;
    let spec_b = spec_brdf * f0.z;

    Vec3::new(
        (diff_r + spec_r) * light_color.x * intensity,
        (diff_g + spec_g) * light_color.y * intensity,
        (diff_b + spec_b) * light_color.z * intensity,
    )
}

#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<f32>, // 🟢 OPTIMIERT: Jetzt f32 statt u32!
    scene: &[f32], // 📦 Der einzige, gepackte Mega-Buffer  // 🟢 TENSOR 7: Objekt-Rotation (Stride 3)
    camera: CameraSettings,   // 🟢 params5: Kamera-Struct (16-Byte-Alignment)
    env: EnvSettings,         // 🟢 TENSOR 4 → Struct: Umwelt-Parameter
    arch: ArchParams,         // 🟢 TENSOR 5 → Struct: Tempel-Architektur
    fold: FoldParams,         // 🟢 TENSOR 6 → Struct: Unendliche Raumfaltung
    render: RenderSettings,   // 🟢 Meta-Tensor + 12 Skalare → Struct
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    let w_val = render.width;
    let h_val = render.height;
    let t_val = render.time;
    let b_factor = render.blend_factor;


//
   // --- Konstanten für das automatische Slicing ---
      // --- Konstanten für das automatische Slicing (Jetzt sauber als usize) ---
  
    // --- 🟢 KORRIGIERT: Nutzt jetzt die exakten Namen aus dem gpu-Import (_COUNT) ---
    const SLOTS_START: usize = 0;
    const MATERIALS_START: usize = SLOTS_COUNT;                      // 800
    const ROTATIONS_START: usize = SLOTS_COUNT + MATERIALS_COUNT;    // 1200
    const TOTAL_SIZE: usize = ROTATIONS_START + ROTATIONS_COUNT;     // 1500

    // Slices für die alten Funktionen herausschneiden
    let slots = &scene[SLOTS_START..MATERIALS_START];
    let materials = &scene[MATERIALS_START..ROTATIONS_START];
    let rotations = &scene[ROTATIONS_START..TOTAL_SIZE];
    // 🟢 PLAN STEP 3: SDF-Statuen-Grid-Slice aus dem Mega-Buffer.
    // Layout: [slots|materials|rotations|sdf_grid(64³)].
    const SDF_GRID_START: usize = TOTAL_SIZE;                        // 1500
    const SDF_GRID_END: usize = SDF_GRID_START + SDF_GRID_COUNT;     // 263644
    let sdf_grid = &scene[SDF_GRID_START..SDF_GRID_END];
    // Grid-Parameter (müssen zu sdf_preprocessor.rs passen).
    let grid_res = u32::cast_from(SDF_GRID_RES);
    // 🟢 PARAM-GUI: Grid-Größe datengetrieben (default 4.0).
    let grid_size = render.sdf_grid_size;
    // Plan-Parameter: Pfeiler-Faltung + Statuen-Platzierung.
    // 🟢 PARAM-GUI: aus FoldParams (datengetrieben statt hardcoded).
    let pillar_spacing = fold.pillar_spacing;
    let pillar_height = fold.pillar_height;
    // 🟢 PARAM-GUI: Statue-Zentrum + Bounding-Offset aus RenderSettings.
    let statue_center = render.statue_center;
    let bounding_sphere_radius_offset = render.bounding_sphere_radius_offset;
//
    if x < w_val && y < h_val {
    
    
        // =========================================================================
        // 🟢 PARAMETER-STRUCTS: Umwelt direkt aus EnvSettings (params2/params4)
        // =========================================================================
        // Block 0: Licht-Vektor und Intensität
        let key_light_pos = Vec3::new(
            env.key_light_pos.x,
            env.key_light_pos.y,
            env.key_light_pos.z,
        );
        let l_intensity = env.light_intensity;

        // Block 1: Licht-Farben und globale Umgebung
        let key_r = env.key_color.x;
        let key_g = env.key_color.y;
        let key_b = env.key_color.z;
        let a_strength = env.ambient_strength;

        // Block 2: Hintergrund-Farbe und Nebel-Dichte
        let bg_r = env.bg_color.x;
        let bg_g = env.bg_color.y;
        let bg_b = env.bg_color.z;
        let fog_density = env.fog_density;

        // Block 3: Schalter (params3: u32-Flag 0/1)
        let fog_enabled = env.fog_enabled;

        let w_f = f32::cast_from(w_val);
        let h_f = f32::cast_from(h_val);

        let uv_x = (f32::cast_from(x) - (w_f / f32::new(2.0))) / h_f;
        let uv_y = ((h_f / f32::new(2.0)) - f32::cast_from(y)) / h_f;

        let ro = Vec3::new(camera.position.x, camera.position.y, camera.position.z);
        let rd = Vec3::new(uv_x, uv_y, f32::new(1.2));
        let cos_y = camera.yaw.cos();
        let sin_y = camera.yaw.sin();
        let cos_p = camera.pitch.cos();
        let sin_p = camera.pitch.sin();

        let rd_y1 = rd.y * cos_p - rd.z * sin_p;
        let rd_z1 = rd.y * sin_p + rd.z * cos_p;

        let final_rd = Vec3::new(
            rd.x * cos_y + rd_z1 * sin_y,
            rd_y1,
            -rd.x * sin_y + rd_z1 * cos_y,
        )
        .normalize();

        let mut t = f32::new(0.1);
        let mut hit_dist = f32::new(100.0);
        // 🟢 PLAN STEP 5: Welches Objekt hat den Strahl getroffen?
        // false = Slot-Objekt (scene_sdf), true = Plan-Objekt (Pfeiler/Statue).
        let mut hit_is_plan_obj = false;

        let mut hit_d = f32::new(1000.0);
        let mut hit_r = f32::new(0.0);
        let mut hit_g = f32::new(0.0);
        let mut hit_b = f32::new(0.0);

        let mut hit_rough = f32::new(0.5);
        let mut hit_metal = f32::new(0.0);
        let mut hit_emiss = f32::new(0.0);
        let mut hit_spec = f32::new(0.5);

        let mut ray_step = u32::new(0);
        // 🟢 PLAN STEP 6: Glow-Akkumulator — sammelt Emissive-Anteile
        // entlang des Strahls: glow += 0.01/(d+0.02) × Emissive-Farbe.
        let mut glow_r = f32::new(0.0);
        let mut glow_g = f32::new(0.0);
        let mut glow_b = f32::new(0.0);
        
        //for ray_step in 0..100 {
        loop {
            if ray_step >= u32::new(100) {
                break;
            }

            let current_ro = ro.clone();
            let current_rd = final_rd.clone();
            let p = current_ro.add(current_rd.scale(t));

            let step_res = scene_sdf(p, t_val, b_factor, render.active_slots, slots, materials,
              rotations, arch, fold);
            hit_d = step_res.d;
            hit_r = step_res.r;
            hit_g = step_res.g;
            hit_b = step_res.b;
            hit_rough = step_res.roughness;
            hit_metal = step_res.metallic;
            hit_emiss = step_res.emissive;
            hit_spec = step_res.specular;
            // 🟢 FIX (Bugfix-Runde 3): Flag JEDEM Schritt zurücksetzen —
            // sonst bleibt es hängen, wenn der Sphere-Bound mal gewinnt,
            // und der Boden neben der Statue bekommt Statuen-Normalen
            // (≈0-Gradient → schwarzer Fleck).
            hit_is_plan_obj = false;
            // 🟢 PLAN STEP 5: Statue in die Primär-Map mischen.
            // mat_id 2 = Statue (Gold-Bronze). Sphere-Bound aktiv —
            // der Primär-March braucht die sichere Schritt-Schranke.
            let plan_res = scene_map_with_material(
                p.clone(), pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size, true,
                statue_center, bounding_sphere_radius_offset,
            );
            if plan_res.d < hit_d {
                hit_d = plan_res.d;
                // Nur als Statuen-Treffer zählen, wenn die Statue selbst
                // (nicht nur die unsichtbare Kugel) die nächste Fläche ist.
                hit_is_plan_obj = plan_res.mat_id > f32::new(1.5);
                let is_statue = plan_res.mat_id > f32::new(1.5);
                hit_r = select(is_statue, f32::new(0.85), f32::new(0.55));
                hit_g = select(is_statue, f32::new(0.7), f32::new(0.52));
                hit_b = select(is_statue, f32::new(0.3), f32::new(0.5));
                hit_rough = select(is_statue, f32::new(0.35), render.pillar_roughness);
                hit_metal = select(is_statue, render.statue_metallic, f32::new(0.0));
                hit_emiss = f32::new(0.0);
                hit_spec = f32::new(0.5);
            }
            // 🟢 PLAN STEP 6: Glow-Akkumulation — nur wenn der Strahl
            // nah an einer Emissive-Fläche vorbeistreift.
            let glow_kernel = f32::new(0.01) / (hit_d + f32::new(0.02));
            glow_r += glow_kernel * step_res.emissive * step_res.r;
            glow_g += glow_kernel * step_res.emissive * step_res.g;
            glow_b += glow_kernel * step_res.emissive * step_res.b;

            if hit_d < f32::new(0.001) {
                hit_dist = t;
                break;
            }
            t += hit_d;

            if t > f32::new(40.0) {
                break;
            }
            ray_step += u32::new(1);
        }

        // 🟢 TENSOR 4: Hintergrund-Farbe dynamisch aus dem Umwelt-Register
        let bg_r = bg_r - uv_y * f32::new(0.10);
        let bg_g = bg_g - uv_y * f32::new(0.12);
        let bg_b = bg_b - uv_y * f32::new(0.15);

        let mut final_color_x = bg_r;
        let mut final_color_y = bg_g;
        let mut final_color_z = bg_b;

        if hit_dist < f32::new(40.0) {
            let p = ro.add(final_rd.scale(hit_dist));
            // 🟢 PLAN STEP 5: Normale über die Map, die den Treffer
            // geliefert hat — Slot-Objekte behalten ihre alte Normalen-
            // Funktion, Pfeiler/Statue nutzen die kombinierte Plan-Map.
            // (if-else-Expressions sind für Vec3 nicht unterstützt →
            // Statement-Form mit skalaren Akkumulatoren.)
            let mut n_x = f32::new(0.0);
            let mut n_y = f32::new(0.0);
            let mut n_z = f32::new(0.0);
            if hit_is_plan_obj {
                let n = calculate_normal(
                    p.clone(), pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size,
                    statue_center, bounding_sphere_radius_offset,
                );
                n_x = n.x;
                n_y = n.y;
                n_z = n.z;
            } else {
                let n = scene_sdf_normal(p.clone(), t_val, b_factor, render.active_slots, slots, materials,
                  rotations, arch, fold);
                n_x = n.x;
                n_y = n.y;
                n_z = n.z;
            }
            let normal = Vec3::new(n_x, n_y, n_z);

            // 🟢 TENSOR 4: Key-Licht-Position dynamisch aus dem Umwelt-Register
            let fill_light_pos = Vec3::new(f32::new(-5.0), f32::new(3.0), f32::new(-3.0));
            let rim_light_pos = Vec3::new(f32::new(0.0), f32::new(6.0), f32::new(5.0));

            let key_dir = key_light_pos.sub(p.clone()).normalize();
            let fill_dir = fill_light_pos.sub(p.clone()).normalize();
            let rim_dir = rim_light_pos.sub(p.clone()).normalize();

            let view_dir = final_rd.scale(f32::new(-1.0)).normalize();

            // 🟢 Schatten nur aufs Key-Licht (wie bisher)
            let mut key_visibility = f32::new(1.0);
            if render.shadow_mode == u32::new(1) {
                let offset_p = p.add(normal.scale(f32::new(0.02)));
                let shadow_factor = calculate_soft_shadow(
                    offset_p, key_dir.clone(), t_val, b_factor, render.active_slots, slots, materials,
                     rotations, arch, fold,
                     pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size,
                     statue_center, bounding_sphere_radius_offset, env.shadow_softness_k,
                );
                key_visibility = shadow_factor;
            }

            let mut ao_factor = f32::new(1.0);
            if render.enable_ao_mode == u32::new(1) {
                ao_factor = calculate_ao(p, normal.clone(), t_val, b_factor, render.active_slots, slots, materials,
                 rotations, arch, fold,
                 pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size,
                 statue_center, bounding_sphere_radius_offset, env.ao_iterations);
            }

            // 🟢 TENSOR 4: Licht-Farben (Key dynamisch, Fill/Rim fix)
            let key_color = Vec3::new(key_r, key_g, key_b);
            let fill_color = Vec3::new(f32::new(0.25), f32::new(0.40), f32::new(0.60));
            let rim_color = Vec3::new(f32::new(0.50), f32::new(0.70), f32::new(1.00));

            let key_on = render.enable_key != u32::new(0);
            let fill_on = render.enable_fill != u32::new(0);
            let rim_on = render.enable_rim != u32::new(0);

            let albedo = Vec3::new(hit_r, hit_g, hit_b);

            // =========================================================================
            // 🟢 COOK-TORRANCE PBR (siehe PBR_Optimisation_plan.md):
            // - F0 = mix(0.04*(1+3*specular), albedo, metallic)
            // - GGX-Verteilung, Smith-Geometrie, Schlick-Fresnel
            // - kd = (1-F)*(1-metallic): Metalle haben keine diffuse Farbe
            // =========================================================================
            let mut light_r = f32::new(0.0);
            let mut light_g = f32::new(0.0);
            let mut light_b = f32::new(0.0);

            if key_on {
                let contrib = cook_torrance_light(
                    albedo.clone(), normal.clone(), key_dir.clone(), view_dir.clone(),
                    key_color, l_intensity, hit_rough, hit_metal, hit_spec,
                );
                light_r += contrib.x * key_visibility;
                light_g += contrib.y * key_visibility;
                light_b += contrib.z * key_visibility;
            }
            if fill_on {
                let contrib = cook_torrance_light(
                    albedo.clone(), normal.clone(), fill_dir.clone(), view_dir.clone(),
                    fill_color, f32::new(1.0), hit_rough, hit_metal, hit_spec,
                );
                light_r += contrib.x;
                light_g += contrib.y;
                light_b += contrib.z;
            }
            if rim_on {
                let contrib = cook_torrance_light(
                    albedo.clone(), normal.clone(), rim_dir.clone(), view_dir.clone(),
                    rim_color, f32::new(1.0), hit_rough, hit_metal, hit_spec,
                );
                light_r += contrib.x;
                light_g += contrib.y;
                light_b += contrib.z;
            }

            // 🟢 Ambient: konstant — Metalle bekommen ihr Ambient über die
            // Spiegel-Farbe F0 im Cook-Torrance-Specular, nicht über Grau
            let ambient_r = a_strength * hit_r;
            let ambient_g = a_strength * hit_g;
            let ambient_b = a_strength * hit_b;

            // 🟢 Emissive: farbig statt weiß (Slot-Farbe × Emissive-Skalar)
            let emissive_r = hit_emiss * hit_r;
            let emissive_g = hit_emiss * hit_g;
            let emissive_b = hit_emiss * hit_b;

            // Finaler Farb-Kanal-Zusammenbau inklusive AO & Emission
            final_color_x = (light_r + ambient_r) * ao_factor + emissive_r;
            final_color_y = (light_g + ambient_g) * ao_factor + emissive_g;
            final_color_z = (light_b + ambient_b) * ao_factor + emissive_b;
            // 🟢 PLAN STEP 7: 1-Bounce-Reflexion — nur auf glatten,
            // nicht-metallischen Flächen sinnvoll (Roughness-Gate).
            if hit_rough < render.reflection_roughness_gate {
                let refl_dir = final_rd.sub(normal.scale(f32::new(2.0) * final_rd.dot(normal.clone()))).normalize();
                let refl_color = trace_reflection(
                    p.add(normal.scale(f32::new(0.02))),
                    refl_dir,
                    t_val,
                    b_factor,
                    render.active_slots,
                    slots,
                    materials,
                    rotations,
                    arch,
                    fold,
                    sdf_grid,
                    grid_res,
                    grid_size,
                    Vec3::new(bg_r, bg_g, bg_b),
                    f32::new(40.0),
                    render.reflection_max_steps,
                    statue_center,
                    bounding_sphere_radius_offset,
                );
                // Fresnel-artige Mischung: glatte Flächen spiegeln stärker.
                let refl_strength = (f32::new(1.0) - hit_rough) * render.fresnel_strength_factor;
                final_color_x += refl_color.x * refl_strength;
                final_color_y += refl_color.y * refl_strength;
                final_color_z += refl_color.z * refl_strength;
            }
        }
        // 🟢 PLAN STEP 6: Glow-Anteil addieren (unabhängig vom Treffer —
        // Leuchten sichtbar auch wenn der Strahl die Quelle verfehlt).
        final_color_x += glow_r;
        final_color_y += glow_g;
        final_color_z += glow_b;
        // =====================================================================
        // 🟢 PLAN STEP 7: VOLUMETRISCHER NEBEL (Beer-Lambert-Integration)
        // Ersetzt die alte lineare Distanz-Kompression: schrittweise
        // Transmittanz entlang des Strahls mit Höhen-Profil.
        // =====================================================================
        let mut fog = f32::new(1.0);
        if fog_enabled > u32::new(0) {
            fog = integrate_volumetric_fog(
                ro.clone(),
                final_rd.clone(),
                hit_dist,
                f32::new(40.0),
                fog_density,
                env.volumetric_fog_steps,
                env.fog_density_falloff,
            );
        }

        final_color_x = final_color_x * fog + bg_r * (f32::new(1.0) - fog);
        final_color_y = final_color_y * fog + bg_g * (f32::new(1.0) - fog);
        final_color_z = final_color_z * fog + bg_b * (f32::new(1.0) - fog);

        // 🟢 ACES-TONE-MAPPING: HDR-Werte (>1.0 durch Emissive, hohe
        // Intensitäten, Glanzlichter) werden natürlich komprimiert statt
        // vom 8-Bit-Framebuffer hart geclampt. Behebt das
        // „Emissive > 1.0 = totes Weiß"-Problem.
        final_color_x = aces_tonemap(final_color_x);
        final_color_y = aces_tonemap(final_color_y);
        final_color_z = aces_tonemap(final_color_z);

        // 🟢 NEU: Direkte, lineare Zuweisung im f32 VRAM-Puffer
        // 1. Hole die Thread-Koordinaten direkt als usize
        let x_idx = usize::cast_from(x);
        let y_idx = usize::cast_from(y);
        let w_idx = usize::cast_from(w_val);

        // 2. Berechne den Index direkt in einem Rutsch als usize (Kein u32-Zwischenschritt!)
        let pixel_index = (y_idx * w_idx + x_idx) * usize::new(3);

        // 3. Direktes Schreiben ohne Typprobleme
        output[pixel_index] = final_color_x;
        output[pixel_index + usize::new(1)] = final_color_y;
        output[pixel_index + usize::new(2)] = final_color_z;
    }
}
