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
    // ----------------=========================================================
    // UNENDLICHER KOSMOS (Großzügiges Space-Repeating)
    // ----------------=========================================================
    // Wir erhöhen den Abstand der geklonten Zellen massiv auf 12.0 Einheiten.
    // Das verhindert das Gefühl, in einer Wand zu stehen, und öffnet den Raum.
    let cell_size = 12.0f32;
    let half_cell = cell_size * 0.5;
    
    let grid_p_x = p.x - cell_size * f32::floor((p.x + half_cell) / cell_size);
    let grid_p_y = p.y - cell_size * f32::floor((p.y + half_cell) / cell_size); 
    let grid_p_z = p.z - cell_size * f32::floor((p.z + half_cell) / cell_size);
    
    let local_p = Vec3::new(grid_p_x, grid_p_y, grid_p_z);

    // =========================================================================
    // OBJEKT 1: Der Kristall der Weisheit (Mandel-Oktaeder)
    // =========================================================================
    // Ein mathematisch exakter Oktaeder (Doppelpyramide), der im Raum rotiert
    let rot_speed = time * 0.5;
    let cos_r = f32::cos(rot_speed);
    let sin_r = f32::sin(rot_speed);
    
    // Rotation um die Y-Achse für den Kristall
    let crystal_x = local_p.x * cos_r - local_p.z * sin_r;
    let crystal_z = local_p.x * sin_r + local_p.z * cos_r;
    
    // Oktaeder-Distanzfeld-Formel (Betragssumme der Achsen)
    let base_crystal = (f32::abs(crystal_x) + f32::abs(local_p.y) + f32::abs(crystal_z) - 1.2) * 0.57735027f32;
    
    // Kristall-Strukturierung: Mathematische Wellen stanzen feine Spalten in den Diamanten
    let waves = f32::sin(crystal_x * 8.0) * f32::sin(local_p.y * 8.0) * f32::sin(crystal_z * 8.0) * 0.05;
    let final_crystal = base_crystal + waves;

    // =========================================================================
    // OBJEKT 2: Die Ringe des Wissens (Pulsierendes Saturn-System)
    // =========================================================================
    // Ein filigraner Torus, der sich entgegengesetzt um den Kristall neigt
    let ring_speed = time * -0.7;
    let cos_b = f32::cos(ring_speed);
    let sin_b = f32::sin(ring_speed);
    
    // Vertikale Neigungskonvertierung (Rotation um die X-Achse)
    let ring_y = local_p.y * cos_b - local_p.z * sin_b;
    let ring_z = local_p.y * sin_b + local_p.z * cos_b;
    
    let horizontal_len = f32::sqrt(local_p.x * local_p.x + ring_z * ring_z) - 2.0;
    let saturn_ring = f32::sqrt(horizontal_len * horizontal_len + ring_y * ring_y) - 0.08;

    // =========================================================================
    // FUSION (Schnittstellen der Geometrie)
    // =========================================================================
    // Der Kristall und sein planetares Ringsystem verschmelzen weich miteinander
    let mut current_scene = smin(final_crystal, saturn_ring, blend_factor);
    
    // Level 2 Effekt (Taste 2): Die Kristalle dehnen sich fraktal aus
    if blend_factor > 10.0 {
        current_scene = smin(final_crystal, saturn_ring, 0.9) - 0.15;
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
        
        // Erhöhte Reichweite (40.0), damit man die unendlichen Klone am Horizont sieht
        for _ in 0..90 {
            let p = ro.add(final_rd.scale(t));
            let d = scene_sdf(p, time, blend_factor);
            if d < 0.001 {
                hit_dist = t;
                break;
            }
            t += d;
            if t > 40.0 { break; }
        }

        // Tiefes Weltraum-Schwarz als Hintergrund
        let mut final_color_x = 0.002;
        let mut final_color_y = 0.005;
        let mut final_color_z = 0.01;

        if hit_dist < 40.0 {
            let p = ro.add(final_rd.scale(hit_dist));
            let eps = 0.001;

            let nx = scene_sdf(Vec3::new(p.x + eps, p.y, p.z), time, blend_factor) - scene_sdf(Vec3::new(p.x - eps, p.y, p.z), time, blend_factor);
            let ny = scene_sdf(Vec3::new(p.x, p.y + eps, p.z), time, blend_factor) - scene_sdf(Vec3::new(p.x, p.y - eps, p.z), time, blend_factor);
            let nz = scene_sdf(Vec3::new(p.x, p.y, p.z + eps), time, blend_factor) - scene_sdf(Vec3::new(p.x, p.y, p.z - eps), time, blend_factor);
            let normal = Vec3::new(nx, ny, nz).normalize();

            // Scheinwerfer-Licht von oben rechts für spektakuläre Schattenkanten
            let light_dir = Vec3::new(2.0, 2.5, -1.5).normalize();
            let diffuse = f32::max(normal.dot(light_dir), 0.0) + 0.08;

            let mut shadow = 1.0;
            if shadow_mode == 1 {
                shadow = shadows::calculate_hard_shadow(p, light_dir, time, blend_factor);
            } else if shadow_mode == 2 {
                shadow = shadows::calculate_soft_shadow(p, light_dir, time, 20.0, blend_factor);
            }

            let mut ao = 1.0;
            if enable_ao_mode == 1 {
                ao = ambient_occlusion::calculate_ao(p, normal, time, blend_factor);
            }

            // NEU: Leuchtendes Smaragd-Cyan/Türkis der transzendenten Erkenntnis
            let color_scaled = Vec3::new(0.20, 0.90, 0.75).scale(diffuse * shadow * ao);
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
