use cubecl::prelude::*;
use crate::kernel::{Vec3, scene_sdf};

/// Lokaler harter Schatten für das Baukastensystem
#[cube]
pub fn calculate_hard_shadow(
    p: Vec3, 
    light_dir: Vec3, 
    time: f32, 
    blend_factor: f32,
    config: &Tensor<f32>
) -> f32 {
    let cell_size = 10.0f32; 
    let half_cell = cell_size * 0.5;
    let local_x = p.x - cell_size * f32::floor((p.x + half_cell) / cell_size);
    let local_y = p.y - cell_size * f32::floor((p.y + half_cell) / cell_size);
    let local_z = p.z - cell_size * f32::floor((p.z + half_cell) / cell_size);
    let local_p = Vec3::new(local_x, local_y, local_z);
    
    let max_t = local_p.length();
    let mut t = 0.15; 
    let mut shadow_factor = 1.0;

    for _ in 0..25 {
        let current_pos = p.add(light_dir.scale(t));
        let d = scene_sdf(current_pos, time, blend_factor, config);
        
        if d < 0.001 {
            shadow_factor = 0.25; 
            break;
        }
        
        t += d;
        if t >= max_t {
            break; 
        }
    }

    shadow_factor
}

/// Lokaler weicher Schatten für das Baukastensystem
#[cube]
pub fn calculate_soft_shadow(
    p: Vec3, 
    light_dir: Vec3, 
    time: f32, 
    k: f32, 
    blend_factor: f32,
    config: &Tensor<f32>
) -> f32 {
    let cell_size = 10.0f32;
    let half_cell = cell_size * 0.5;
    let local_x = p.x - cell_size * f32::floor((p.x + half_cell) / cell_size);
    let local_y = p.y - cell_size * f32::floor((p.y + half_cell) / cell_size);
    let local_z = p.z - cell_size * f32::floor((p.z + half_cell) / cell_size);
    let local_p = Vec3::new(local_x, local_y, local_z);

    let max_t = local_p.length();
    let mut t = 0.15;
    let mut res = 1.0;

    for _ in 0..25 {
        let current_pos = p.add(light_dir.scale(t));
        let d = scene_sdf(current_pos, time, blend_factor, config);
        
        if d < 0.001 {
            res = 0.0;
            break;
        }
        
        res = f32::min(res, k * d / t);
        
        t += d;
        if t >= max_t {
            break;
        }
    }

    res.min(1.0).max(0.25)
}
