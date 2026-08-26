use cubecl::prelude::*;
use cubecl::frontend::CubeType;
// Importiere das originale Struct UND die CubeCL-Erweiterung
//use crate::kernel::{Vec3, Vec3Expand, scene_sdf, scene_sdf_expand};
use crate::kernel::{Vec3, Vec3Expand, scene_sdf}; 
// Sollte der Compiler im Hintergrund immer noch nach der Expansion suchen,
// holt dieser Wildcard-Import sie automatisch und fehlerfrei in den Scope:
use crate::kernel::*; 
/// Lokaler harter Schatten für das Baukastensystem
#[cube]
pub fn calculate_hard_shadow(
    p: Vec3, 
    light_dir: Vec3, 
    time: f32, 
    blend_factor: f32,
    config: &Tensor<f32>
) -> f32 {
    let cell_size = f32::new(10.0); 
    let half_cell = cell_size * f32::new(0.5);
    
    let local_x = p.x - cell_size * ((p.x + half_cell) / cell_size).floor();
    let local_y = p.y - cell_size * ((p.y + half_cell) / cell_size).floor();
    let local_z = p.z - cell_size * ((p.z + half_cell) / cell_size).floor();
    let local_p = Vec3::new(local_x, local_y, local_z);
    
    let max_t = local_p.length();
    let mut t = f32::new(0.15); 
    let mut shadow_factor = f32::new(1.0);

    let mut step = u32::new(0);
    loop {
        if step >= u32::new(25) {
            break;
        }

        // FIX: Lokale Kopien für die Schleifen-Closure erstellen
        let current_p = p.clone();
        let current_dir = light_dir.clone();

        let current_pos = current_p.add(current_dir.scale(t));
        let d = scene_sdf(current_pos, time, blend_factor, config);
        
        if d < f32::new(0.001) {
            shadow_factor = f32::new(0.25); 
            break;
        }
        
        t += d;
        if t >= max_t {
            break; 
        }

        step += u32::new(1);
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
    let cell_size = f32::new(10.0);
    let half_cell = cell_size * f32::new(0.5);
    
    let local_x = p.x - cell_size * ((p.x + half_cell) / cell_size).floor();
    let local_y = p.y - cell_size * ((p.y + half_cell) / cell_size).floor();
    let local_z = p.z - cell_size * ((p.z + half_cell) / cell_size).floor();
    let local_p = Vec3::new(local_x, local_y, local_z);

    let max_t = local_p.length();
    let mut t = f32::new(0.15);
    let mut res = f32::new(1.0);

    let mut step = u32::new(0);
    loop {
        if step >= u32::new(25) {
            break;
        }

        // FIX: Lokale Kopien für die Schleifen-Closure erstellen
        let current_p = p.clone();
        let current_dir = light_dir.clone();

        let current_pos = current_p.add(current_dir.scale(t));
        let d = scene_sdf(current_pos, time, blend_factor, config);
        
        if d < f32::new(0.001) {
            res = f32::new(0.0);
            break;
        }
        
        res = res.min(k * d / t);
        
        t += d;
        if t >= max_t {
            break;
        }

        step += u32::new(1);
    }

    res.min(f32::new(1.0)).max(f32::new(0.25))
}
