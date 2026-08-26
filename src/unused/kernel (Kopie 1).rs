use cubecl::prelude::*;
use crate::shadows;
use crate::ambient_occlusion;

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

// HINZUGEFÜGT: Mathematische SDF für den rotierenden Torus
#[cube]
fn evaluate_dynamic_torus(p: Vec3, time: f32, size: f32) -> f32 {
    let rot_speed_x = time * 0.6;
    let rot_speed_y = time * 0.3;
    
    let cos_x = f32::cos(rot_speed_x);
    let sin_x = f32::sin(rot_speed_x);
    let cos_y = f32::cos(rot_speed_y);
    let sin_y = f32::sin(rot_speed_y);
    
    // Doppelachsen-Rotation
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
    let half_cell = cell_size * 0.5f32;
    
    let grid_p_x = p.x - cell_size * f32::floor((p.x + half_cell) / cell_size);
    let grid_p_y = p.y - cell_size * f32::floor((p.y + half_cell) / cell_size); 
    let grid_p_z = p.z - cell_size * f32::floor((p.z + half_cell) / cell_size);
    
    let local_p = Vec3::new(grid_p_x, grid_p_y, grid_p_z);
    let mut core_system = 1000.0f32;

    // KORREKTUR: Typenreiner Index-Zugriff für CubeCL 0.10
    let active_slots = config[0usize] as u32;

    // Zero-Abstraction Dynamic Trick: Schleife läuft bis zu einem sicheren Maximum
    for i in 0..100 {
        if (i as u32) >= active_slots {
            break; 
        }

        // KORREKTUR: Indizes konsequent über native mathematische Shader-Typen berechnen
        let base_idx = 1usize + i * 4usize;
        
        let obj_type  = config[base_idx] as u32;
        let obj_size  = config[base_idx + 1usize];
        let offset_x  = config[base_idx + 2usize];
        let offset_z  = config[base_idx + 3usize];

        // Lokalen Raum für den dynamischen Slot berechnen
        let p_slot = Vec3::new(local_p.x - offset_x, local_p.y, local_p.z - offset_z);

        // ZURÜCK ZUM ORIGINAL: Echte Verzweigungen überlassen das Prädizieren dem AMD-Treiber!
        if obj_type == 1u32 {
            core_system = smin(core_system, evaluate_dynamic_crystal(p_slot, time, obj_size), blend_factor);
        }
        if obj_type == 2u32 {
            core_system = smin(core_system, evaluate_dynamic_gyroid(p_slot, time, obj_size), blend_factor);
        }
        if obj_type == 3u32 {
            core_system = smin(core_system, evaluate_dynamic_torus(p_slot, time, obj_size), blend_factor);
        }
    }

    let pillar_x = f32::abs(f32::abs(local_p.x) - 5.0f32) - 0.6f32;
    let pillar_z = f32::abs(f32::abs(local_p.z) - 5.0f32) - 0.6f32;
    let corner_pillars = f32::max(pillar_x, pillar_z);
    
    let room_floor = local_p.y + 3.0f32; 
    let room_ceiling = 3.0f32 - local_p.y; 
    let floor_and_ceiling = f32::min(room_floor, room_ceiling) - 0.1f32;

    let arch_radius = 3.2f32; 
    let arch_z = f32::sqrt(local_p.x * local_p.x + (local_p.y - 1.0f32) * (local_p.y - 1.0f32)) - arch_radius;
    let arch_x = f32::sqrt(local_p.z * local_p.z + (local_p.y - 1.0f32) * (local_p.y - 1.0f32)) - arch_radius;
    let wall_arches = f32::min(arch_z, arch_x);

    let mut architecture = f32::min(corner_pillars, floor_and_ceiling);
    architecture = f32::max(architecture, -wall_arches);

    let s1_decor = 2.0f32;
    let pillar_holes = (f32::abs(f32::sin(local_p.x * s1_decor)) + f32::abs(f32::cos(local_p.y * s1_decor)) + f32::abs(f32::sin(local_p.z * s1_decor))) * 0.03f32;
    architecture = f32::max(architecture, -(pillar_holes - 0.01f32));

    let combined_core = smin(core_system, architecture, blend_factor);
    
    let mut final_res = f32::min(core_system, architecture);
    if blend_factor > 10.0f32 {
        final_res = smin(combined_core, architecture, 0.7f32) - 0.1f32;
    }
    
    final_res
}

#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<u32>,
    config: &Tensor<f32>, 
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

    if x < width && y < height {
        let w_f = width as f32;
        let h_f = height as f32;
        let uv_x = (x as f32 - (w_f / 2.0)) / h_f;
        let uv_y = ((h_f / 2.0) - y as f32) / h_f;

        let ro = Vec3::new(cam_x, cam_y, cam_z);
        let rd = Vec3::new(uv_x, uv_y, 1.2);

        let cos_y = f32::cos(cam_yaw);
        let sin_y = f32::sin(cam_yaw);
        let cos_p = f32::cos(cam_pitch);
        let sin_p = f32::sin(cam_pitch);

        let rd_y1 = rd.y * cos_p - rd.z * sin_p;
        let rd_z1 = rd.y * sin_p + rd.z * cos_p;
        
        let final_rd = Vec3::new(
            rd.x * cos_y + rd_z1 * sin_y,
            rd_y1,
            -rd.x * sin_y + rd_z1 * cos_y
        ).normalize();

        let mut t = 0.1;
        let mut hit_dist = 100.0;
        
        for _ in 0..100 {
            let p = ro.add(final_rd.scale(t));
            let d = scene_sdf(p, time, blend_factor, config);
            if d < 0.001 {
                hit_dist = t;
                break;
            }
            t += d;
            if t > 40.0 { break; }
        }

        let mut final_color_x = 0.005;
        let mut final_color_y = 0.01;
        let mut final_color_z = 0.025;

        if hit_dist < 40.0 {
            let p = ro.add(final_rd.scale(hit_dist));
            let eps = 0.002;

            let nx = scene_sdf(Vec3::new(p.x + eps, p.y, p.z), time, blend_factor, config) - scene_sdf(Vec3::new(p.x - eps, p.y, p.z), time, blend_factor, config);
            let ny = scene_sdf(Vec3::new(p.x, p.y + eps, p.z), time, blend_factor, config) - scene_sdf(Vec3::new(p.x, p.y - eps, p.z), time, blend_factor, config);
            let nz = scene_sdf(Vec3::new(p.x, p.y, p.z + eps), time, blend_factor, config) - scene_sdf(Vec3::new(p.x, p.y, p.z - eps), time, blend_factor, config);
            let normal = Vec3::new(nx, ny, nz).normalize();

            let cell_size = 10.0f32;
            let half_cell = cell_size * 0.5;
            let local_hit_x = p.x - cell_size * f32::floor((p.x + half_cell) / cell_size);
            let local_hit_y = p.y - cell_size * f32::floor((p.y + half_cell) / cell_size);
            let local_hit_z = p.z - cell_size * f32::floor((p.z + half_cell) / cell_size);
            let local_hit_pos = Vec3::new(local_hit_x, local_hit_y, local_hit_z);

            let key_pos = Vec3::new(2.0, 2.5, -1.5);
            let key_vec = key_pos.sub(local_hit_pos);
            let key_dir = key_vec.normalize();
            let key_attenuation = 1.0 / (1.0 + 0.04 * key_vec.dot(key_vec));
            
            let mut shadow = 1.0;
            if shadow_mode == 1 {
                shadow = shadows::calculate_hard_shadow(p, key_dir, time, blend_factor, config);
            } else if shadow_mode == 2 {
                shadow = shadows::calculate_soft_shadow(p, key_dir, time, 14.0, blend_factor, config);
            }
            let key_diffuse = f32::max(normal.dot(key_dir), 0.0) * 0.8 * shadow * key_attenuation * (enable_key as f32);

            let fill_pos = Vec3::new(-2.5, 0.0, -1.0);
            let fill_vec = fill_pos.sub(local_hit_pos);
            let fill_dir = fill_vec.normalize();
            let fill_attenuation = 1.0 / (1.0 + 0.05 * fill_vec.dot(fill_vec));
            let fill_diffuse = f32::max(normal.dot(fill_dir), 0.0) * 0.35 * fill_attenuation * (enable_fill as f32);

            let rim_pos = Vec3::new(0.0, 2.0, 2.5);
            let rim_vec = rim_pos.sub(local_hit_pos);
            let rim_dir = rim_vec.normalize();
            let rim_attenuation = 1.0 / (1.0 + 0.03 * rim_vec.dot(rim_vec));
            let rim_dot = f32::max(normal.dot(rim_dir), 0.0);
            let rim_diffuse = rim_dot * rim_dot * rim_dot * rim_dot * 0.5 * rim_attenuation * (enable_rim as f32);

            let total_lighting_term = (key_diffuse + fill_diffuse + rim_diffuse) * light_intensity + ambient_strength;

            let mut ao = 1.0;
            if enable_ao_mode == 1 {
                ao = ambient_occlusion::calculate_ao(p, normal, time, blend_factor, config);
            }

            let final_shading = total_lighting_term * ao;

            let mut obj_color_x = 0.20; 
            let mut obj_color_y = 0.90;
            let mut obj_color_z = 0.75;

            let dist_from_center = f32::sqrt(local_hit_x * local_hit_x + local_hit_z * local_hit_z);
            if dist_from_center > 2.5 || f32::abs(local_hit_y) > 2.7 {
                obj_color_x = 0.95; 
                obj_color_y = 0.85;
                obj_color_z = 0.45;
            }
            
            let color_scaled = Vec3::new(obj_color_x, obj_color_y, obj_color_z).scale(final_shading);
            final_color_x = color_scaled.x;
            final_color_y = color_scaled.y;
            final_color_z = color_scaled.z;
        }
        
        let r_u32 = (final_color_x.min(1.0).max(0.0) * 255.0) as u32;
        let g_u32 = (final_color_y.min(1.0).max(0.0) * 255.0) as u32;
        let b_u32 = (final_color_z.min(1.0).max(0.0) * 255.0) as u32;
        let packed_pixel = (r_u32 << 16) | (g_u32 << 8) | b_u32;
        
        let x_usize = usize::cast_from(x);
        let y_usize = usize::cast_from(y);
        let width_usize = usize::cast_from(width);
        let pixel_index = y_usize * width_usize + x_usize;
        output[pixel_index] = packed_pixel;
    }
}
