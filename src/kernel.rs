#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use cubecl::prelude::*;

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

#[cube]
pub fn scene_sdf(
    p: Vec3,
    time: f32,
    blend_factor: f32,
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>, // 🟢 Dritter Tensor empfangen
   //rotationss: &Tensor<f32>, // 🟢 TENSOR 7: Objekt-Rotation (Stride 3)
    //    env_settings: &Tensor<f32>, // 🟢 Hier eingefügt
    arch_params: &Tensor<f32>, // 🟢 TENSOR 5: Tempel-Architektur
    fold_params: &Tensor<f32>, // 🟢 TENSOR 6: Unendliche Raumfaltung
) -> SdfResult {
    // 🟢 TENSOR 6: Modulo-Grid-Konstanten dynamisch aus dem Faltungs-Register
    let cell_size = fold_params[usize::new(0)];
    let half_cell = fold_params[usize::new(1)];

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

    let active_slots = usize::cast_from(meta[usize::new(0)]);

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
      /*
                  // 🟢 TENSOR 7: Rotation auslesen (Stride 3: rotX, rotY, rotZ)
            let rot_idx = i * usize::new(3);
            let rot_x = //rotationss[rot_idx];
            let rot_y = //rotationss[rot_idx + usize::new(1)];
            let rot_z = //rotationss[rot_idx + usize::new(2)];
            let shifted_p_x = p.x - offset_x;
            let shifted_p_y = p.y - offset_y;
            let shifted_p_z = p.z - offset_z;
            // 🟢 TENSOR 7: Euler-Rotation des Objekt-Raums (X → Y → Z angewandt)
            // Rotation um X
            let cx = rot_x.cos();
            let sx = rot_x.sin();
            let ry1 = shifted_p_y * cx - shifted_p_z * sx;
            let rz1 = shifted_p_y * sx + shifted_p_z * cx;
            // Rotation um Y
            let cy = rot_y.cos();
            let sy = rot_y.sin();
            let rx2 = shifted_p_x * cy + rz1 * sy;
            let rz2 = -shifted_p_x * sy + rz1 * cy;
            // Rotation um Z
            let cz = rot_z.cos();
            let sz = rot_z.sin();
            let rx3 = rx2 * cz - ry1 * sz;
            let ry3 = rx2 * sz + ry1 * cz;
            let rotated_p = Vec3::new(rx3, ry3, rz2);
*/
        /*
                let grid_p_x =
                rotated_p.x - cell_size * ((rotated_p.x + half_cell) / cell_size).floor();
            let grid_p_y =
                rotated_p.y - cell_size * ((rotated_p.y + half_cell) / cell_size).floor();
            let grid_p_z =
                rotated_p.z - cell_size * ((rotated_p.z + half_cell) / cell_size).floor();
            let p_slot = Vec3::new(grid_p_x, grid_p_y, grid_p_z);

*/
            // 1. Die relative Position des Objekts berechnen (Translation)
            let shifted_p_x = p.x - offset_x;
            let shifted_p_y = p.y - offset_y;
            let shifted_p_z = p.z - offset_z;

            // =========================================================================
            // 🟢 UNENDLICHES RAUMFALTUNGS-GRID (Zurück auf unrotiertes shifted_p geleitet)
            // =========================================================================
            // Da 'rotated_p' durch den Puffer-Rückbau pausiert ist, füttern wir das Modulo-Grid
            // direkt mit 'shifted_p_x/y/z'. Das verhindert Compiler-Fehler bei 'p_slot'.
            let grid_p_x = shifted_p_x - cell_size * ((shifted_p_x + half_cell) / cell_size).floor();
            let grid_p_y = shifted_p_y - cell_size * ((shifted_p_y + half_cell) / cell_size).floor();
            let grid_p_z = shifted_p_z - cell_size * ((shifted_p_z + half_cell) / cell_size).floor();
            
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
    let pillar_dist = arch_params[usize::new(0)];
    let pillar_thick = arch_params[usize::new(1)];
    let room_height = arch_params[usize::new(2)];
    let ceiling_thick = arch_params[usize::new(3)];
    let arch_radius = arch_params[usize::new(4)];
    let arch_height = arch_params[usize::new(5)];
    let decor_freq = arch_params[usize::new(6)];
    let decor_depth = arch_params[usize::new(7)];
    let decor_thick = arch_params[usize::new(8)];

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
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>,
  //  //rotationss: &Tensor<f32>,
      //  env_settings: &Tensor<f32>, // 🟢 NEU: Muss hier rein, um das Verschieben der Pointer zu stoppen!
    arch_params: &Tensor<f32>,
    fold_params: &Tensor<f32>,
) -> Vec3 {
    let eps = f32::new(0.002);
     
   // let d_res = scene_sdf(p.clone(), time, blend_factor, meta, slots, materials, //rotationss, env_settings, arch_params, fold_params).d;

    let d_res = scene_sdf(p.clone(), time, blend_factor, meta, slots, materials,
     //rotationss,
      arch_params, fold_params);
    let d = d_res.d;
    let p_x = Vec3::new(p.x + eps, p.y, p.z);
    let p_y = Vec3::new(p.x, p.y + eps, p.z);
    let p_z = Vec3::new(p.x, p.y, p.z + eps);
    let nx = scene_sdf(p_x, time, blend_factor, meta, slots, materials,
     //rotationss,
      arch_params, fold_params).d - d;
    let ny = scene_sdf(p_y, time, blend_factor, meta, slots, materials,
     //rotationss,
      arch_params, fold_params).d - d;
    let nz = scene_sdf(p_z, time, blend_factor, meta, slots, materials,
     //rotationss,
      arch_params, fold_params).d - d;
    Vec3::new(nx, ny, nz).normalize()
}

#[cube]
pub fn calculate_soft_shadow(
    ro: Vec3,
    rd: Vec3,
    time: f32,
    blend_factor: f32,
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>,
   //rotationss: &Tensor<f32>,
    arch_params: &Tensor<f32>,
    fold_params: &Tensor<f32>,
) -> f32 {
    let mut res = f32::new(1.0);
    let mut t = f32::new(0.04);
    let t_max = f32::new(25.0);

    let mut step = u32::new(0);
    loop {
        if step >= u32::new(32) {
            break;
        }

        let current_ro = ro.clone();
        let current_rd = rd.clone();
        let p = current_ro.add(current_rd.scale(t));
        // FIX: add PBR_MATERIALS Tensor-ref
        let sdf_res = scene_sdf(p, time, blend_factor, meta, slots, materials,
        //rotationss,
          arch_params, fold_params);
        let h = sdf_res.d;

        if h < f32::new(0.001) {
            res = f32::new(0.0);
            break;
        }
        res = res.min(f32::new(8.0) * h / t);
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
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>,
   //rotationss: &Tensor<f32>,
    arch_params: &Tensor<f32>,
    fold_params: &Tensor<f32>,
) -> f32 {
    let mut occ = f32::new(0.0);
    let mut sca = f32::new(1.0);

    let mut i = u32::new(1);
    loop {
        if i >= u32::new(5) {
            break;
        }

        let current_p = p.clone();
        let current_normal = normal.clone();

        let hr = f32::cast_from(i) * f32::new(0.15);
        let ao_pos = current_p.add(current_normal.scale(hr));

        // 🟢 FIX: Nutzen der neuen 2-Tensor-Signatur (meta, slots) statt config
        let sdf_res = scene_sdf(ao_pos, time, blend_factor, meta, slots, materials,
         //rotationss,
          arch_params, fold_params);
        let dd = sdf_res.d;

        occ += (hr - dd) * sca;
        sca *= f32::new(0.90);

        i += u32::new(1);
    }

    // 🟢 FIX: Rückgabewert vervollständigt
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
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>,
   //rotationss: &Tensor<f32>, // 🟢 TENSOR 7: Objekt-Rotation
    env_settings: &Tensor<f32>, // 🟢 TENSOR 4: Umwelt-Parameter (Licht, Ambient, Nebel)
    arch_params: &Tensor<f32>,  // 🟢 TENSOR 5: Tempel-Architektur
    fold_params: &Tensor<f32>,  // 🟢 TENSOR 6: Unendliche Raumfaltung
    time: f32,
    width: u32,
    height: u32,
    shadow_mode: u32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    blend_factor: f32,
    enable_ao_mode: u32,
    cam_yaw: f32,
    cam_pitch: f32,
    enable_key: u32,
    enable_fill: u32,
    enable_rim: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    let w_val = width;
    let h_val = height;
    let t_val = time;
    let b_factor = blend_factor;

    if x < w_val && y < h_val {
    
    
        // =========================================================================
        // 🟢 TENSOR 4: DYNAMISCHE EXTRAKTION DER UMWELT-REGISTER
        // =========================================================================
        // Block 0: Licht-Vektor und Intensität
        let key_light_pos = Vec3::new(
            env_settings[usize::new(0)],
            env_settings[usize::new(1)],
            env_settings[usize::new(2)],
        );
        let l_intensity = env_settings[usize::new(3)];

        // Block 1: Licht-Farben und globale Umgebung
        let key_r = env_settings[usize::new(4)];
        let key_g = env_settings[usize::new(5)];
        let key_b = env_settings[usize::new(6)];
        let a_strength = env_settings[usize::new(7)];

        // Block 2: Hintergrund-Farbe und Nebel-Dichte
        let bg_r = env_settings[usize::new(8)];
        let bg_g = env_settings[usize::new(9)];
        let bg_b = env_settings[usize::new(10)];
        let fog_density = env_settings[usize::new(11)];

        // Block 3: Schalter
        let fog_enabled = env_settings[usize::new(12)];

        let w_f = f32::cast_from(w_val);
        let h_f = f32::cast_from(h_val);

        let uv_x = (f32::cast_from(x) - (w_f / f32::new(2.0))) / h_f;
        let uv_y = ((h_f / f32::new(2.0)) - f32::cast_from(y)) / h_f;

        let ro = Vec3::new(cam_x, cam_y, cam_z);
        let rd = Vec3::new(uv_x, uv_y, f32::new(1.2));

        let cos_y = cam_yaw.cos();
        let sin_y = cam_yaw.sin();
        let cos_p = cam_pitch.cos();
        let sin_p = cam_pitch.sin();

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

        let mut hit_d = f32::new(1000.0);
        let mut hit_r = f32::new(0.0);
        let mut hit_g = f32::new(0.0);
        let mut hit_b = f32::new(0.0);

        let mut hit_rough = f32::new(0.5);
        let mut hit_metal = f32::new(0.0);
        let mut hit_emiss = f32::new(0.0);
        let mut hit_spec = f32::new(0.5);

        let mut ray_step = u32::new(0);
        
        //for ray_step in 0..100 {
        loop {
            if ray_step >= u32::new(100) {
                break;
            }

            let current_ro = ro.clone();
            let current_rd = final_rd.clone();
            let p = current_ro.add(current_rd.scale(t));

            let step_res = scene_sdf(p, t_val, b_factor, meta, slots, materials,
             //rotationss,
              arch_params, fold_params);

            hit_d = step_res.d;
            hit_r = step_res.r;
            hit_g = step_res.g;
            hit_b = step_res.b;
            hit_rough = step_res.roughness;
            hit_metal = step_res.metallic;
            hit_emiss = step_res.emissive;
            hit_spec = step_res.specular;

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
            let normal = scene_sdf_normal(p.clone(), t_val, b_factor, meta, slots, materials,
             //rotationss,
              arch_params, fold_params);

            // 🟢 TENSOR 4: Key-Licht-Position dynamisch aus dem Umwelt-Register
            let fill_light_pos = Vec3::new(f32::new(-5.0), f32::new(3.0), f32::new(-3.0));
            let rim_light_pos = Vec3::new(f32::new(0.0), f32::new(6.0), f32::new(5.0));

            let key_dir = key_light_pos.sub(p.clone()).normalize();
            let fill_dir = fill_light_pos.sub(p.clone()).normalize();
            let rim_dir = rim_light_pos.sub(p.clone()).normalize();

            let view_dir = final_rd.scale(f32::new(-1.0)).normalize();

            // 🟢 Schatten nur aufs Key-Licht (wie bisher)
            let mut key_visibility = f32::new(1.0);
            if shadow_mode == u32::new(1) {
                let offset_p = p.add(normal.scale(f32::new(0.02)));
                let shadow_factor = calculate_soft_shadow(
                    offset_p, key_dir.clone(), t_val, b_factor, meta, slots, materials,
                    //rotationss,
                     arch_params, fold_params,
                );
                key_visibility = shadow_factor;
            }

            let mut ao_factor = f32::new(1.0);
            if enable_ao_mode == u32::new(1) {
                ao_factor = calculate_ao(p, normal.clone(), t_val, b_factor, meta, slots, materials, 
                //rotationss,
                 arch_params, fold_params);
            }

            // 🟢 TENSOR 4: Licht-Farben (Key dynamisch, Fill/Rim fix)
            let key_color = Vec3::new(key_r, key_g, key_b);
            let fill_color = Vec3::new(f32::new(0.25), f32::new(0.40), f32::new(0.60));
            let rim_color = Vec3::new(f32::new(0.50), f32::new(0.70), f32::new(1.00));

            let key_on = enable_key != u32::new(0);
            let fill_on = enable_fill != u32::new(0);
            let rim_on = enable_rim != u32::new(0);

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
        }
        // =====================================================================
        // 🟢 TENSOR 4: DYNAMISCHE NEBEL-KOMPOSITION (Puffer-gesteuert)
        // =====================================================================
        let mut fog = f32::new(1.0);
        if fog_enabled > f32::new(0.5) {
            // Nutzt fog_density dynamisch als Teiler für die Nebel-Kompression
            fog = (f32::new(40.0) - hit_dist) / (f32::new(40.0) - (f32::new(15.0) * fog_density));
            if fog > f32::new(1.0) {
                fog = f32::new(1.0);
            }
            if fog < f32::new(0.0) {
                fog = f32::new(0.0);
            }
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
