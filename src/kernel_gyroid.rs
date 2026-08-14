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
pub fn scene_sdf(p: Vec3, time: f32, blend_factor: f32) -> f32 {
    // Unendliche Klonierung im Raum
    let cell_size = 5.0f32;
    let half_cell = cell_size * 0.5;
    
    let grid_p_x = p.x - cell_size * f32::floor((p.x + half_cell) / cell_size);
    let grid_p_y = p.y; 
    let grid_p_z = p.z - cell_size * f32::floor((p.z + half_cell) / cell_size);
    
    let local_p = Vec3::new(grid_p_x, grid_p_y, grid_p_z);

    // =========================================================================
    // OBJEKT 1: Organische Gyroid-Kugel (Perfekt für Schatten)
    // =========================================================================
    let sphere_center = Vec3::new(0.0, f32::sin(time * 0.5) * 0.3, 0.0);
    let base_sphere = local_p.sub(sphere_center).length() - 0.9;
    
    // Die mathematische Gyroid-Formel erzeugt fließende, tunnelartige Wellen im Raum.
    // Keine scharfen Noppenkanten -> Keine Schattenstreifen mehr!
    let scale = 6.0f32;
    let gyroid = (f32::sin(local_p.x * scale) * f32::cos(local_p.y * scale) 
                + f32::cos(local_p.x * scale) * f32::sin(local_p.z * scale) 
                + f32::sin(local_p.y * scale) * f32::cos(local_p.z * scale)) / scale;
                
    // Per CSG Intersection (max) stanzen wir den Gyroid aus der Kugel heraus
    let organic_object = f32::max(base_sphere, gyroid * 0.6);

    // =========================================================================
    // OBJEKT 2: Ein glatter, pulsierender Torus
    // =========================================================================
    let scale_factor = 1.0 + f32::sin(time) * 0.1;
    let horizontal_len = f32::sqrt(local_p.x * local_p.x + local_p.z * local_p.z) - (1.3 * scale_factor);
    let smooth_torus = f32::sqrt(horizontal_len * horizontal_len + local_p.y * local_p.y) - 0.15;

    // =========================================================================
    // OBJEKT 3: Die Bodenplattform (Box)
    // =========================================================================
    let box_p = local_p.sub(Vec3::new(0.0, -1.6, 0.0));
    let d_x = f32::abs(box_p.x) - 1.2;
    let d_y = f32::abs(box_p.y) - 0.1;
    let d_z = f32::abs(box_p.z) - 1.2;
    let box_outside = Vec3::new(f32::max(d_x, 0.0), f32::max(d_y, 0.0), f32::max(d_z, 0.0)).length();
    let box_inside = f32::min(f32::max(d_x, f32::max(d_y, d_z)), 0.0);
    let box_sdf = box_outside + box_inside;

    // =========================================================================
    // ZUSAMMENFÜHRUNG
    // =========================================================================
    let mut current_scene = smin(organic_object, smooth_torus, blend_factor);
    current_scene = f32::min(current_scene, box_sdf);
    
    // Level 2 Effekt morpht die Szene zu einer fließenden organischen Masse
    if blend_factor > 10.0 {
        current_scene = smin(organic_object, smooth_torus, 0.8).min(box_sdf) - 0.1;
    }
    
    current_scene
}

#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<u32>,
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
        
        for _ in 0..80 {
            let p = ro.add(final_rd.scale(t));
            let d = scene_sdf(p, time, blend_factor);
            if d < 0.001 {
                hit_dist = t;
                break;
            }
            t += d;
            if t > 20.0 { break; }
        }

        let mut final_color_x = 0.01;
        let mut final_color_y = 0.01;
        let mut final_color_z = 0.05;

        if hit_dist < 20.0 {
            let p = ro.add(final_rd.scale(hit_dist));
            let eps = 0.001;

            let nx = scene_sdf(Vec3::new(p.x + eps, p.y, p.z), time, blend_factor) - scene_sdf(Vec3::new(p.x - eps, p.y, p.z), time, blend_factor);
            let ny = scene_sdf(Vec3::new(p.x, p.y + eps, p.z), time, blend_factor) - scene_sdf(Vec3::new(p.x, p.y - eps, p.z), time, blend_factor);
            let nz = scene_sdf(Vec3::new(p.x, p.y, p.z + eps), time, blend_factor) - scene_sdf(Vec3::new(p.x, p.y, p.z - eps), time, blend_factor);
            let normal = Vec3::new(nx, ny, nz).normalize();

            let light_dir = Vec3::new(1.5, 2.0, -1.0).normalize();
            let diffuse = f32::max(normal.dot(light_dir), 0.0) + 0.1;

            let mut shadow = 1.0;
            if shadow_mode == 1 {
                shadow = shadows::calculate_hard_shadow(p, light_dir, time, blend_factor);
            } else if shadow_mode == 2 {
                shadow = shadows::calculate_soft_shadow(p, light_dir, time, 16.0, blend_factor);
            }

            let mut ao = 1.0;
            if enable_ao_mode == 1 {
                ao = ambient_occlusion::calculate_ao(p, normal, time, blend_factor);
            }

            let color_scaled = Vec3::new(0.4, 0.7, 1.0).scale(diffuse * shadow * ao);
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
