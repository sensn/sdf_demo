#![allow(warnings)]

use cubecl::prelude::*;
use cubecl::frontend::CubeType;

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
    let mix_factor = (f32::new(0.5) + f32::new(0.5) * (b.d - a.d) / k).min(f32::new(1.0)).max(f32::new(0.0));
    
    let mixed_r = a.r + mix_factor * (b.r - a.r);
    let mixed_g = a.g + mix_factor * (b.g - a.g);
    let mixed_b = a.b + mix_factor * (b.b - a.b);
     // 🟢 NEU: Weiches Überblenden der PBR-Materialien in der Mischzone!
    let mixed_rough = a.roughness + mix_factor * (b.roughness - a.roughness);
    let mixed_metal = a.metallic + mix_factor * (b.metallic - a.metallic);
    let mixed_emiss = a.emissive + mix_factor * (b.emissive - a.emissive);
    let mixed_spec  = a.specular + mix_factor * (b.specular - a.specular);
    
    SdfResult { 
        d: mixed_d, 
        r: mixed_r, 
        g: mixed_g, 
        b: mixed_b,
        roughness: mixed_rough,
        metallic: mixed_metal,
        emissive: mixed_emiss,
        specular: mixed_spec
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
    
    let base_crystal = (crystal_x.abs() + p.y.abs() + crystal_z.abs() - size) * f32::new(0.57735027);
    let waves = (crystal_x * f32::new(6.0)).sin() * (p.y * f32::new(6.0)).sin() * (crystal_z * f32::new(6.0)).sin() * f32::new(0.04);
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
                + (p.y * scale).sin() * (p.z * scale).cos()) / scale;
                
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
    materials: &Tensor<f32> // 🟢 Dritter Tensor empfangen
) -> SdfResult {
    let cell_size = f32::new(10.0);
    let half_cell = cell_size * f32::new(0.5);
    
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
    let mut sum_rough = f32::new(0.0); let mut sum_metal = f32::new(0.0);
    let mut sum_emiss = f32::new(0.0); let mut sum_spec  = f32::new(0.0);
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
            let obj_size  = slots[base_idx + usize::new(1)];
            let offset_x  = slots[base_idx + usize::new(2)];
            let offset_y  = slots[base_idx + usize::new(3)]; 
            let offset_z  = slots[base_idx + usize::new(4)];
            let obj_r     = slots[base_idx + usize::new(5)]; 
            let obj_g     = slots[base_idx + usize::new(6)]; 
            let obj_b     = slots[base_idx + usize::new(7)]; 
            
            // 🟢 Auslesen aus dem dritten Tensor (Materialien nutzen Stride 4!)
            let mat_idx  = i * usize::new(4);
            let obj_rough = materials[mat_idx];
            let obj_metal = materials[mat_idx + usize::new(1)];
            let obj_emiss = materials[mat_idx + usize::new(2)];
            let obj_spec  = materials[mat_idx + usize::new(3)];

            let shifted_p_x = p.x - offset_x;
            let shifted_p_y = p.y - offset_y;
            let shifted_p_z = p.z - offset_z;

            let grid_p_x = shifted_p_x - cell_size * ((shifted_p_x + half_cell) / cell_size).floor();
            let grid_p_y = shifted_p_y - cell_size * ((shifted_p_y + half_cell) / cell_size).floor(); 
            let grid_p_z = shifted_p_z - cell_size * ((shifted_p_z + half_cell) / cell_size).floor();
            let p_slot = Vec3::new(grid_p_x, grid_p_y, grid_p_z);

            let mut d_obj = f32::new(1000.0);
            if obj_type == u32::new(1) { d_obj = evaluate_dynamic_crystal(p_slot, time, obj_size); }
            if obj_type == u32::new(2) { d_obj = evaluate_dynamic_gyroid(p_slot, time, obj_size); }
            if obj_type == u32::new(3) { d_obj = evaluate_dynamic_torus(p_slot, time, obj_size); }

            if min_dist > f32::new(999.0) {
                min_dist = d_obj;
            } else {
                let h = (blend_factor - (min_dist - d_obj).abs()).max(f32::new(0.0)) / blend_factor;
                min_dist = min_dist.min(d_obj) - h * h * blend_factor * f32::new(0.25);
            }

            // Unzerstörbarer, gewichteter Akkumulator für physikalische Eigenschaften
            let w = f32::new(1.0) / (d_obj.max(f32::new(0.001))).powf(f32::new(2.0));
            sum_r     += obj_r * w;     sum_g     += obj_g * w;     sum_b     += obj_b * w;
            sum_rough += obj_rough * w; sum_metal += obj_metal * w;
            sum_emiss += obj_emiss * w; sum_spec  += obj_spec * w;
            sum_w     += w;
        }
        i += usize::new(1);
    }

    let mut core_r = f32::new(1.0); let mut core_g = f32::new(1.0); let mut core_b = f32::new(1.0);
    let mut core_rough = f32::new(0.5); let mut core_metal = f32::new(0.0);
    let mut core_emiss = f32::new(0.0); let mut core_spec  = f32::new(0.5);

    if sum_w > f32::new(0.0) {
        core_r = sum_r / sum_w; core_g = sum_g / sum_w; core_b = sum_b / sum_w;
        core_rough = sum_rough / sum_w; core_metal = sum_metal / sum_w;
        core_emiss = sum_emiss / sum_w; core_spec = sum_spec / sum_w;
    }
    let core_system = SdfResult { 
        d: min_dist, r: core_r, g: core_g, b: core_b,
        roughness: core_rough, metallic: core_metal, emissive: core_emiss, specular: core_spec
    };

    // =========================================================================
    // ARCHITEKTUR-GENERIERUNG (Unverändert starr, Farbe: Steingrau 0.7)
    // =========================================================================
    let pillar_x = (local_p.x.abs() - f32::new(5.0)).abs() - f32::new(0.6);
    let pillar_z = (local_p.z.abs() - f32::new(5.0)).abs() - f32::new(0.6);
    let corner_pillars = pillar_x.max(pillar_z);
    
    let room_floor = local_p.y + f32::new(3.0); 
    let room_ceiling = f32::new(3.0) - local_p.y; 
    let floor_and_ceiling = room_floor.min(room_ceiling) - f32::new(0.1);

    let arch_radius = f32::new(3.2); 
    let arch_z = (local_p.x * local_p.x + (local_p.y - f32::new(1.0)) * (local_p.y - f32::new(1.0))).sqrt() - arch_radius;
    let arch_x = (local_p.z * local_p.z + (local_p.y - f32::new(1.0)) * (local_p.y - f32::new(1.0))).sqrt() - arch_radius;
    let wall_arches = arch_z.min(arch_x);

    let mut arch_d = corner_pillars.min(floor_and_ceiling);
    arch_d = arch_d.max(-wall_arches);

    let s1_decor = f32::new(2.0);
    let pillar_holes = ( (local_p.x * s1_decor).sin().abs() + (local_p.y * s1_decor).cos().abs() + (local_p.z * s1_decor).sin().abs() ) * f32::new(0.03);
    arch_d = arch_d.max(-(pillar_holes - f32::new(0.01)));

    // (Starre Raumarchitektur erhält Standard-PBR-Werte: Mattstein = roughness 0.8, metallic 0.0)
    let architecture = SdfResult { 
        d: arch_d, r: f32::new(0.7), g: f32::new(0.7), b: f32::new(0.7),
        roughness: f32::new(0.8), metallic: f32::new(0.0), emissive: f32::new(0.0), specular: f32::new(0.2)
    };

        // =========================================================================
    // FINALE REINE SCHNITTAUSWERTUNG (Muss zwingend auf primitiver Ebene erfolgen!)
    // =========================================================================
    let mut final_d     = f32::new(1000.0);
    let mut final_r     = f32::new(1.0);
    let mut final_g     = f32::new(1.0);
    let mut final_b     = f32::new(1.0);
    
    // 🟢 NEU: Primitive Register für die PBR-Materialien anlegen
    let mut final_rough = f32::new(0.5);
    let mut final_metal = f32::new(0.0);
    let mut final_emiss = f32::new(0.0);
    let mut final_spec  = f32::new(0.5);

    let check_core = core_system.clone();
    let check_arch = architecture.clone();

    if check_core.d < check_arch.d { 
        final_d     = check_core.d;
        final_r     = check_core.r;
        final_g     = check_core.g;
        final_b     = check_core.b;
        // PBR-Zuweisung vom Core-System
        final_rough = check_core.roughness;
        final_metal = check_core.metallic;
        final_emiss = check_core.emissive;
        final_spec  = check_core.specular;
    } else { 
        final_d     = check_arch.d;
        final_r     = check_arch.r;
        final_g     = check_arch.g;
        final_b     = check_arch.b;
        // PBR-Zuweisung von der starren Raum-Architektur
        final_rough = check_arch.roughness;
        final_metal = check_arch.metallic;
        final_emiss = check_arch.emissive;
        final_spec  = check_arch.specular;
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
        specular: final_spec
    }
}

#[cube]
pub fn scene_sdf_normal(p: Vec3, time: f32, blend_factor: f32, meta: &Tensor<f32>, slots: &Tensor<f32>, materials: &Tensor<f32>) -> Vec3 {
    let eps = f32::new(0.002);
    let d_res = scene_sdf(p.clone(), time, blend_factor, meta, slots, materials);
    let d = d_res.d;
    let p_x = Vec3::new(p.x + eps, p.y, p.z);
    let p_y = Vec3::new(p.x, p.y + eps, p.z);
    let p_z = Vec3::new(p.x, p.y, p.z + eps);
    let nx = scene_sdf(p_x, time, blend_factor, meta, slots, materials).d - d;
    let ny = scene_sdf(p_y, time, blend_factor, meta, slots, materials).d - d;
    let nz = scene_sdf(p_z, time, blend_factor, meta, slots, materials).d - d;
    Vec3::new(nx, ny, nz).normalize()
}

#[cube]
pub fn calculate_soft_shadow(ro: Vec3, rd: Vec3, time: f32, blend_factor: f32, meta: &Tensor<f32>, slots: &Tensor<f32>, materials: &Tensor<f32>) -> f32 {
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
        let sdf_res = scene_sdf(p, time, blend_factor, meta, slots, materials);
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
pub fn calculate_ao(p: Vec3, normal: Vec3, time: f32, blend_factor: f32, meta: &Tensor<f32>, slots: &Tensor<f32>, materials: &Tensor<f32>) -> f32 {
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
        let sdf_res = scene_sdf(ao_pos, time, blend_factor, meta, slots, materials);
        let dd = sdf_res.d;
        
        occ += (hr - dd) * sca;
        sca *= f32::new(0.90);

        i += u32::new(1);
    }
    
    // 🟢 FIX: Rückgabewert vervollständigt
    (f32::new(1.0) - (occ * f32::new(0.5))).max(f32::new(0.3))
}


#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<u32>,
    meta: &Tensor<f32>,   // 🟢 REPARIERT: Eindeutiger Metadaten-Tensor
    slots: &Tensor<f32>,  // 🟢 REPARIERT: Ausgerichteter Geometrie-Tensor
    materials: &Tensor<f32>, // 🟢 FIX 1: Hier fehlte der PBR-Material-Tensor im Funktionskopf!
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
    light_intensity: f32, 
    ambient_strength: f32,
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
            -rd.x * sin_y + rd_z1 * cos_y
        ).normalize();

        let mut t = f32::new(0.1);
        let mut hit_dist = f32::new(100.0);
        
        let mut hit_d = f32::new(1000.0);
        let mut hit_r = f32::new(0.0);
        let mut hit_g = f32::new(0.0);
        let mut hit_b = f32::new(0.0);
        
        // 🟢 FIX: Diese primitiven Variablen registrieren und aus der Schleife retten!
        let mut hit_rough = f32::new(0.5);
        let mut hit_metal = f32::new(0.0);
        let mut hit_emiss = f32::new(0.0);
        let mut hit_spec  = f32::new(0.5);
        
        let mut ray_step = u32::new(0);
        loop {
            if ray_step >= u32::new(100) {
                break;
            }
            
            let current_ro = ro.clone();
            let current_rd = final_rd.clone();
            let p = current_ro.add(current_rd.scale(t));
            
            let step_res = scene_sdf(p, t_val, b_factor, meta, slots, materials);
            
            hit_d     = step_res.d;
            hit_r     = step_res.r;
            hit_g     = step_res.g;
            hit_b     = step_res.b;
            // 🟢 Zuweisung der neuen PBR-Kanäle auf primitiver Ebene innerhalb der Schleife
            hit_rough = step_res.roughness;
            hit_metal = step_res.metallic;
            hit_emiss = step_res.emissive;
            hit_spec  = step_res.specular;
            
            if hit_d < f32::new(0.001) {
                hit_dist = t;
                break;
            }
            t += hit_d;
            
            if t > f32::new(40.0) { 
                break; 
            }
            ray_step += u32::new(1);
        } // Loop-Ende

        let bg_r = f32::new(0.35) - uv_y * f32::new(0.10);
        let bg_g = f32::new(0.45) - uv_y * f32::new(0.12);
        let bg_b = f32::new(0.60) - uv_y * f32::new(0.15);

        let mut final_color_x = bg_r;
        let mut final_color_y = bg_g;
        let mut final_color_z = bg_b;

        if hit_dist < f32::new(40.0) {
            let p = ro.add(final_rd.scale(hit_dist));
            // 🟢 REPARIERT: Übergabe der zwei getrennten Tensoren an die Normalenberechnung
            //let normal = scene_sdf_normal(p.clone(), t_val, b_factor, meta, slots);
            // 🟢 FIX 3: Materialien an die Normalenberechnung durchreichen
            let normal = scene_sdf_normal(p.clone(), t_val, b_factor, meta, slots, materials);
            let key_light_pos  = Vec3::new(f32::new(4.0), f32::new(7.0), f32::new(-4.0));
            let fill_light_pos = Vec3::new(f32::new(-5.0), f32::new(3.0), f32::new(-3.0));
            let rim_light_pos  = Vec3::new(f32::new(0.0), f32::new(6.0), f32::new(5.0)); 
            
            let key_dir  = key_light_pos.sub(p.clone()).normalize();
            let fill_dir = fill_light_pos.sub(p.clone()).normalize();
            let rim_dir  = rim_light_pos.sub(p.clone()).normalize();
            
            let mut diff_key  = normal.dot(key_dir).max(f32::new(0.0));
            let diff_fill = normal.dot(fill_dir).max(f32::new(0.0));
            let diff_rim  = normal.dot(rim_dir).max(f32::new(0.0));
            
            // 🟢 PBR SPEBULAR BLENDUNG:
            // Berechne die Blickrichtung des Auges/Kamera (View Direction)
            let view_dir = final_rd.scale(f32::new(-1.0)).normalize();
            
            // Halbwertsvektor für Blinn-Phong Specular (Key Light)
            let half_vec = key_dir.clone().add(view_dir).normalize();
            let spec_angle = normal.dot(half_vec).max(f32::new(0.0));
            
            // Schärfe des Highlights wird umgekehrt proportional zur Rauheit berechnet!
            // Ein glattes Objekt (roughness 0.1) bekommt einen extrem scharfen, gleißenden Reflex Exponent 128.
                        // ✅ BEHOBEN: Nutzt die geretteten primitiven Variablen aus der Schleife
            let spec_power = f32::new(1.0) / hit_rough.max(f32::new(0.01));
            let specular_highlight = spec_angle.powf(spec_power * f32::new(15.0)) * hit_spec;

           
            //---SHADOW
            if shadow_mode == u32::new(1) {
                let offset_p = p.add(normal.scale(f32::new(0.02)));
                // 🟢 REPARIERT: Übergabe der zwei getrennten Tensoren an die Schattenberechnung
               // let shadow_factor = calculate_soft_shadow(offset_p, key_dir, t_val, b_factor, meta, slots);
               // 🟢 FIX 4: Materialien an Weichschatten durchreichen
                let shadow_factor = calculate_soft_shadow(offset_p, key_dir, t_val, b_factor, meta, slots, materials);
                
                diff_key *= shadow_factor;
            }
            
            let mut ao_factor = f32::new(1.0);
            if enable_ao_mode == u32::new(1) {
                // 🟢 REPARIERT: Übergabe der zwei getrennten Tensoren an Ambient Occlusion
                //ao_factor = calculate_ao(p, normal, t_val, b_factor, meta, slots);
                // 🟢 FIX 5: Materialien an Ambient Occlusion durchreichen
                ao_factor = calculate_ao(p, normal, t_val, b_factor, meta, slots, materials);
           
            }
            
            let mut key_r = f32::new(1.00); let mut key_g = f32::new(0.95); let mut key_b = f32::new(0.85);
            let mut fill_r = f32::new(0.25); let mut fill_g = f32::new(0.40); let mut fill_b = f32::new(0.60);
            let mut rim_r = f32::new(0.50); let mut rim_g = f32::new(0.70); let mut rim_b = f32::new(1.00);
            
            if enable_key == u32::new(0)  { key_r = f32::new(0.0); key_g = f32::new(0.0); key_b = f32::new(0.0); }
            if enable_fill == u32::new(0) { fill_r = f32::new(0.0); fill_g = f32::new(0.0); fill_b = f32::new(0.0); }
            if enable_rim == u32::new(0)  { rim_r = f32::new(0.0); rim_g = f32::new(0.0); rim_b = f32::new(0.0); }
            
                       // =========================================================================
            // DYNAMISCHE ALBEDO- & MATERIAL-ÜBERTRAGUNG (Integrierte PBR-Lichtberechnung)
            // =========================================================================
            let mat_r = hit_r;
            let mat_g = hit_g;
            let mat_b = hit_b;
            
            let l_intensity = light_intensity;
            
            // 🟢 PBR SPECULAR: Berechne den Blickrichtungsvektor (View Direction)
            let view_dir = final_rd.scale(f32::new(-1.0)).normalize();
            
            // Halbwertsvektor für das Blinn-Phong Glanzlicht des Key-Lights
            let half_vec = key_dir.clone().add(view_dir).normalize();
            let spec_angle = normal.dot(half_vec).max(f32::new(0.0));
            
            // Die Schärfe des Highlights wird umgekehrt proportional zur Rauheit skaliert
            let spec_power = f32::new(1.0) / hit_rough.max(f32::new(0.01));
            let specular_highlight = spec_angle.powf(spec_power * f32::new(15.0)) * hit_spec;
            
            // Integration aller Lichtquellen inklusive des neuen Glanzlichts und Eigenleuchtens (Emissive)
            let lit_r = (key_r * diff_key * l_intensity) + (fill_r * diff_fill * f32::new(0.7)) + (rim_r * diff_rim.powf(f32::new(3.0)) * f32::new(1.5)) + specular_highlight + hit_emiss;
            let lit_g = (key_g * diff_key * l_intensity) + (fill_g * diff_fill * f32::new(0.7)) + (rim_g * diff_rim.powf(f32::new(3.0)) * f32::new(1.5)) + specular_highlight + hit_emiss;
            let lit_b = (key_b * diff_key * l_intensity) + (fill_b * diff_fill * f32::new(0.7)) + (rim_b * diff_rim.powf(f32::new(3.0)) * f32::new(1.5)) + specular_highlight + hit_emiss;
            
            let a_strength = ambient_strength;
            let final_r = mat_r * (a_strength * ao_factor + lit_r * ao_factor);
            let final_g = mat_g * (a_strength * ao_factor + lit_g * ao_factor);
            let final_b = mat_b * (a_strength * ao_factor + lit_b * ao_factor);
            
            let mut fog = (f32::new(40.0) - hit_dist) / (f32::new(40.0) - f32::new(15.0));
            if fog > f32::new(1.0) { fog = f32::new(1.0); }
            if fog < f32::new(0.0) { fog = f32::new(0.0); }
            
            final_color_x = final_r * fog + bg_r * (f32::new(1.0) - fog);
            final_color_y = final_g * fog + bg_g * (f32::new(1.0) - fog);
            final_color_z = final_b * fog + bg_b * (f32::new(1.0) - fog);
        }

        let r_u32 = u32::cast_from(final_color_x.min(f32::new(1.0)).max(f32::new(0.0)) * f32::new(255.0));
        let g_u32 = u32::cast_from(final_color_y.min(f32::new(1.0)).max(f32::new(0.0)) * f32::new(255.0));
        let b_u32 = u32::cast_from(final_color_z.min(f32::new(1.0)).max(f32::new(0.0)) * f32::new(255.0));
        let packed_pixel = (r_u32 << u32::new(16)) | (g_u32 << u32::new(8)) | b_u32;
        
        let x_usize = usize::cast_from(x);
        let y_usize = usize::cast_from(y);
        let width_usize = usize::cast_from(width);
        let pixel_index = y_usize * width_usize + x_usize;
        
        output[pixel_index] = packed_pixel;
    }
}


