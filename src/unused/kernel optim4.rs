use cubecl::prelude::*;

#[derive(CubeType, Copy, Clone)]
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
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = f32::max(k - f32::abs(a - b), 0.0) / k;
    a.min(b) - h * h * k * 0.25
}

#[cube]
fn evaluate_dynamic_crystal(p: Vec3, time: f32, size: f32) -> f32 {
    let rot_speed = time * 0.4;
    let cos_r = f32::cos(rot_speed);
    let sin_r = f32::sin(rot_speed);
    let crystal_x = p.x * cos_r - p.z * sin_r;
    let crystal_z = p.x * sin_r + p.z * cos_r;
    
    let base_crystal = (f32::abs(crystal_x) + f32::abs(p.y) + f32::abs(crystal_z) - size) * 0.57735027f32;
    let waves = f32::sin(crystal_x * 6.0) * f32::sin(p.y * 6.0) * f32::sin(crystal_z * 6.0) * 0.04;
    base_crystal + waves
}

#[cube]
fn evaluate_dynamic_gyroid(p: Vec3, time: f32, size: f32) -> f32 {
    let sphere_center = Vec3::new(0.0, f32::sin(time * 3.5) * 0.2, 0.0);
    let base_sphere = p.sub(sphere_center).length() - size;
    
    let scale = 6.0f32;
    let gyroid = (f32::sin(p.x * scale) * f32::cos(p.y * scale) 
                + f32::cos(p.x * scale) * f32::sin(p.z * scale) 
                + f32::sin(p.y * scale) * f32::cos(p.z * scale)) / scale;
                
    f32::max(base_sphere, gyroid * 0.5)
}

#[cube]
fn evaluate_dynamic_torus(p: Vec3, time: f32, size: f32) -> f32 {
    let rot_speed_x = time * 0.6;
    let rot_speed_y = time * 0.3;
    
    let cos_x = f32::cos(rot_speed_x);
    let sin_x = f32::sin(rot_speed_x);
    let cos_y = f32::cos(rot_speed_y);
    let sin_y = f32::sin(rot_speed_y);
    
    let ry_x = p.x * cos_y - p.z * sin_y;
    let ry_z = p.x * sin_y + p.z * cos_y;
    let rx_y = p.y * cos_x - ry_z * sin_x;
    let rx_z = p.y * sin_x + ry_z * cos_x;
    
    let r_major = size;
    let r_minor = size * 0.15;
    
    let q_x = f32::sqrt(ry_x * ry_x + rx_z * rx_z) - r_major;
    f32::sqrt(q_x * q_x + rx_y * rx_y) - r_minor
}

#[cube]
pub fn scene_sdf(p: Vec3, time: f32, blend_factor: f32, config: &Tensor<f32>) -> f32 {
    let cell_size = 10.0f32;
    let half_cell = cell_size * 0.5;
    
    let grid_p_x = p.x - cell_size * f32::floor((p.x + half_cell) / cell_size);
    let grid_p_y = p.y - cell_size * f32::floor((p.y + half_cell) / cell_size); 
    let grid_p_z = p.z - cell_size * f32::floor((p.z + half_cell) / cell_size);
    
    let local_p = Vec3::new(grid_p_x, grid_p_y, grid_p_z);
    let mut core_system = 1000.0f32;

    let active_slots = config[0usize] as usize;

    for i in 0..100 {
        if i >= active_slots {
            break; 
        }

        let base_idx = 1usize + i * 4usize;
        let obj_type  = config[base_idx] as u32;
        let obj_size  = config[base_idx + 1usize];
        let offset_x  = config[base_idx + 2usize];
        let offset_z  = config[base_idx + 3usize];

        let p_slot = Vec3::new(local_p.x - offset_x, local_p.y, local_p.z - offset_z);

        let d_crystal = evaluate_dynamic_crystal(p_slot, time, obj_size);
        let d_gyroid  = evaluate_dynamic_gyroid(p_slot, time, obj_size);
        let d_torus   = evaluate_dynamic_torus(p_slot, time, obj_size);

        // Masken ermitteln
        let m_crystal = (obj_type == 1) as u32 as f32;
        let m_gyroid  = (obj_type == 2) as u32 as f32;
        let m_torus   = (obj_type == 3) as u32 as f32;
        
        // Ist es überhaupt ein gültiges Objekt? (1, 2 oder 3)
        let is_valid = (obj_type >= 1 && obj_type <= 3) as u32 as f32;

        // Berechne die reine Geometrie-SDF für diesen Slot
        let raw_object_d = (d_crystal * m_crystal) 
                         + (d_gyroid * m_gyroid) 
                         + (d_torus * m_torus);

        // WICHTIGER MATHEMATISCHER FIX: 
        // Wir führen das smin NUR mit dem echten Objekt aus.
        let merged_system = smin(core_system, raw_object_d, blend_factor);

        // Wenn der Slot gültig ist, übernehmen wir das fusionierte smin-Ergebnis.
        // Wenn nicht (Luft/ungültig), behalten wir das alte core_system unverändert bei!
        core_system = (merged_system * is_valid) + (core_system * (1.0f32 - is_valid));
    }

    let pillar_x = f32::abs(f32::abs(local_p.x) - 5.0) - 0.6;
    let pillar_z = f32::abs(f32::abs(local_p.z) - 5.0) - 0.6;
    let corner_pillars = f32::max(pillar_x, pillar_z);
    
    let room_floor = local_p.y + 3.0; 
    let room_ceiling = 3.0 - local_p.y; 
    let floor_and_ceiling = f32::min(room_floor, room_ceiling) - 0.1;

    let arch_radius = 3.2f32; 
    let arch_z = f32::sqrt(local_p.x * local_p.x + (local_p.y - 1.0) * (local_p.y - 1.0)) - arch_radius;
    let arch_x = f32::sqrt(local_p.z * local_p.z + (local_p.y - 1.0) * (local_p.y - 1.0)) - arch_radius;
    let wall_arches = f32::min(arch_z, arch_x);

    let mut architecture = f32::min(corner_pillars, floor_and_ceiling);
    architecture = f32::max(architecture, -wall_arches);

    let s1_decor = 2.0f32;
    let pillar_holes = (f32::abs(f32::sin(local_p.x * s1_decor)) + f32::abs(f32::cos(local_p.y * s1_decor)) + f32::abs(f32::sin(local_p.z * s1_decor))) * 0.03;
    architecture = f32::max(architecture, -(pillar_holes - 0.01));

    let combined_core = smin(core_system, architecture, blend_factor);
    
    let mut final_res = f32::min(core_system, architecture);
    if blend_factor > 10.0 {
        final_res = smin(combined_core, architecture, 0.7) - 0.1;
    }
    
    final_res
}


#[cube]
fn scene_sdf_normal(p: Vec3, time: f32, blend_factor: f32, config: &Tensor<f32>) -> Vec3 {
    let eps = 0.002f32;
    let d = scene_sdf(p, time, blend_factor, config);
    let nx = scene_sdf(Vec3::new(p.x + eps, p.y, p.z), time, blend_factor, config) - d;
    let ny = scene_sdf(Vec3::new(p.x, p.y + eps, p.z), time, blend_factor, config) - d;
    let nz = scene_sdf(Vec3::new(p.x, p.y, p.z + eps), time, blend_factor, config) - d;
    Vec3::new(nx, ny, nz).normalize()
}

#[cube]
fn calculate_soft_shadow(ro: Vec3, rd: Vec3, time: f32, blend_factor: f32, config: &Tensor<f32>) -> f32 {
    let mut res = 1.0f32;
    let mut t = 0.04f32; 
    let t_max = 25.0f32;
    
    for _ in 0..32 {
        let p = ro.add(rd.scale(t));
        let h = scene_sdf(p, time, blend_factor, config);
        if h < 0.001f32 {
            res = 0.0f32;
            break;
        }
        res = f32::min(res, 8.0f32 * h / t);
        t += f32::max(h, 0.04f32);
        if t > t_max {
            break;
        }
    }
    f32::max(res, 0.2f32)
}

#[cube]
fn calculate_ao(p: Vec3, normal: Vec3, time: f32, blend_factor: f32, config: &Tensor<f32>) -> f32 {
    let mut occ = 0.0f32;
    let mut sca = 1.0f32;
    
    for i in 1..5 {
        let hr = (i as f32) * 0.15f32;
        let ao_pos = p.add(normal.scale(hr));
        let dd = scene_sdf(ao_pos, time, blend_factor, config);
        occ += (hr - dd) * sca;
        sca *= 0.90f32;
    }
    f32::max(1.0f32 - occ * 0.5f32, 0.3f32)
}

#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<u32>,
    config: &Tensor<f32>, 
    time: &f32,
    width: &u32,
    height: &u32,
    shadow_mode: &u32,
    cam_x: &f32,
    cam_y: &f32,
    cam_z: &f32,
    blend_factor: &f32,
    enable_ao_mode: &u32,
    cam_yaw: &f32,
    cam_pitch: &f32,
    light_intensity: &f32, 
    ambient_strength: &f32,
    enable_key: &u32,   
    enable_fill: &u32,  
    enable_rim: &u32,   
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    let w_val = *width;
    let h_val = *height;
    let t_val = *time;
    let b_factor = *blend_factor;

    if x < w_val && y < h_val {
        let w_f = w_val as f32;
        let h_f = h_val as f32;
        let uv_x = (x as f32 - (w_f / 2.0)) / h_f;
        let uv_y = ((h_f / 2.0) - y as f32) / h_f;

        let ro = Vec3::new(*cam_x, *cam_y, *cam_z);
        let rd = Vec3::new(uv_x, uv_y, 1.2f32); 

        let cos_y = f32::cos(*cam_yaw);
        let sin_y = f32::sin(*cam_yaw);
        let cos_p = f32::cos(*cam_pitch);
        let sin_p = f32::sin(*cam_pitch);

        let rd_y1 = rd.y * cos_p - rd.z * sin_p;
        let rd_z1 = rd.y * sin_p + rd.z * cos_p;
        
        let final_rd = Vec3::new(
            rd.x * cos_y + rd_z1 * sin_y,
            rd_y1,
            -rd.x * sin_y + rd_z1 * cos_y
        ).normalize();

        let mut t = 0.1f32;
        let mut hit_dist = 100.0f32;
        
        for _ in 0..100 {
            let p = ro.add(final_rd.scale(t));
            let d = scene_sdf(p, t_val, b_factor, config);
            if d < 0.001f32 {
                hit_dist = t;
                break;
            }
            t += d;
            if t > 40.0f32 { break; }
        }

        let bg_r = 0.35f32 - uv_y * 0.10f32;
        let bg_g = 0.45f32 - uv_y * 0.12f32;
        let bg_b = 0.60f32 - uv_y * 0.15f32;

        let mut final_color_x = bg_r;
        let mut final_color_y = bg_g;
        let mut final_color_z = bg_b;

        if hit_dist < 40.0f32 {
            let p = ro.add(final_rd.scale(hit_dist));
            let normal = scene_sdf_normal(p, t_val, b_factor, config);
            
            let key_light_pos  = Vec3::new(4.0, 7.0, -4.0);
            let fill_light_pos = Vec3::new(-5.0, 3.0, -3.0);
            let rim_light_pos  = Vec3::new(0.0, 6.0, 5.0); 
            
            let key_dir  = key_light_pos.sub(p).normalize();
            let fill_dir = fill_light_pos.sub(p).normalize();
            let rim_dir  = rim_light_pos.sub(p).normalize();
            
// GEFIXT: Keine Verwendung von f32::max als Typen-Konstrukt.// In CubeCL ruft man die mathematischen Funktionen über die instanziierten Typen-Methoden auf!
let mut diff_key  = normal.dot(key_dir).max(0.0f32);let diff_fill = normal.dot(fill_dir).max(0.0f32);let diff_rim  = normal.dot(rim_dir).max(0.0f32);if *shadow_mode == 1u32 {let shadow_factor = calculate_soft_shadow(p.add(normal.scale(0.02f32)), key_dir, t_val, b_factor, config);diff_key *= shadow_factor;}let mut ao_factor = 1.0f32;if *enable_ao_mode == 1u32 {ao_factor = calculate_ao(p, normal, t_val, b_factor, config);}let mut key_r = 1.00f32; let mut key_g = 0.95f32; let mut key_b = 0.85f32;let mut fill_r = 0.25f32; let mut fill_g = 0.40f32; let mut fill_b = 0.60f32;let mut rim_r = 0.50f32; let mut rim_g = 0.70f32; let mut rim_b = 1.00f32;if *enable_key == 0u32  { key_r = 0.0; key_g = 0.0; key_b = 0.0; }if *enable_fill == 0u32 { fill_r = 0.0; fill_g = 0.0; fill_b = 0.0; }if *enable_rim == 0u32  { rim_r = 0.0; rim_g = 0.0; rim_b = 0.0; }let mat_r = 0.85f32;let mat_g = 0.82f32;let mat_b = 0.78f32;let l_intensity = *light_intensity;// GEFIXT: f32::powf(...) durch den instanziierten .powf() Aufruf ersetzt, um den Typenfehler im Makro aufzulösen!
let lit_r = (key_r * diff_key * l_intensity) + (fill_r * diff_fill * 0.7f32) + (rim_r * diff_rim.powf(3.0f32) * 1.5f32);let lit_g = (key_g * diff_key * l_intensity) + (fill_g * diff_fill * 0.7f32) + (rim_g * diff_rim.powf(3.0f32) * 1.5f32);let lit_b = (key_b * diff_key * l_intensity) + (fill_b * diff_fill * 0.7f32) + (rim_b * diff_rim.powf(3.0f32) * 1.5f32);let a_strength = *ambient_strength;let final_r = mat_r * (a_strength * ao_factor + lit_r * ao_factor);let final_g = mat_g * (a_strength * ao_factor + lit_g * ao_factor);let final_b = mat_b * (a_strength * ao_factor + lit_b * ao_factor);let mut fog = (40.0f32 - hit_dist) / (40.0f32 - 15.0f32);if fog > 1.0f32 { fog = 1.0f32; }if fog < 0.0f32 { fog = 0.0f32; }final_color_x = final_r * fog + bg_r * (1.0f32 - fog);final_color_y = final_g * fog + bg_g * (1.0f32 - fog);final_color_z = final_b * fog + bg_b * (1.0f32 - fog);}// GEFIXT: .max() und .min() direkt als Methoden auf den Werten aufrufen statt über den CPU-Pfad f32::max
let r_u32 = (final_color_x.min(1.0).max(0.0) * 255.0) as u32;
        let g_u32 = (final_color_y.min(1.0).max(0.0) * 255.0) as u32;
        let b_u32 = (final_color_z.min(1.0).max(0.0) * 255.0) as u32;
        let packed_pixel = (r_u32 << 16) | (g_u32 << 8) | b_u32;
        
        let x_usize = usize::cast_from(x);
        let y_usize = usize::cast_from(y);
        let width_usize = usize::cast_from(*width);
        let pixel_index = y_usize * width_usize + x_usize;
        output[pixel_index] = packed_pixel;
    }
}