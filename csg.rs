use cubecl::prelude::*;

/// Glatte Verschmelzung (Smooth Minimum) zweier Distanzfelder
#[cube]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    // h = max(k - abs(a - b), 0.0) / k;
    let h = (k - (a - b).abs()).max(f32::new(0.0)) / k;
    
    // a.min(b) - h * h * k * 0.25;
    a.min(b) - h * h * k * f32::new(0.25)
}

/// Glatte Subtraktion (Smooth Subtraction) - Zieht B von A ab mit weicher Kante
#[cube]
pub fn ssub(a: f32, b: f32, k: f32) -> f32 {
    // h = max(k - abs(a + b), 0.0) / k;
    let h = (k - (a + b).abs()).max(f32::new(0.0)) / k;
    
    // a.max(-b) + h * h * k * 0.25;
    a.max(-b) + h * h * k * f32::new(0.25)
}

/// Glatte Schnittmenge (Smooth Intersection) - Kombiniert die Schnittmenge mit weicher Kante
#[cube]
pub fn sinter(a: f32, b: f32, k: f32) -> f32 {
    // h = max(k - abs(a - b), 0.0) / k;
    let h = (k - (a - b).abs()).max(f32::new(0.0)) / k;
    
    // a.max(b) + h * h * k * 0.25;
    a.max(b) + h * h * k * f32::new(0.25)
}

////
/*
use cubecl::prelude::*;
use crate::kernel::Vec3;
use crate::csg::{smin, ssub, sinter}; // Re-Export der CSG-Funktionen

#[cube]
pub fn scene_sdf(p: Vec3, time: f32, blend_factor: f32, config: &Tensor<f32>) -> f32 {
    let cell_size = f32::new(10.0);
    let half_cell = cell_size * f32::new(0.5);
    
    let grid_p_x = p.x - cell_size * ((p.x + half_cell) / cell_size).floor();
    let grid_p_y = p.y - cell_size * ((p.y + half_cell) / cell_size).floor(); 
    let grid_p_z = p.z - cell_size * ((p.z + half_cell) / cell_size).floor();
    
    let local_p = Vec3::new(grid_p_x, grid_p_y, grid_p_z);
    
    // Start-Untergrund der Architektur
    let mut final_scene = f32::new(1000.0); 

    let active_slots = usize::cast_from(config[usize::new(0)]);

    let mut i = usize::new(0);
    loop {
        if i >= usize::new(100) || i >= active_slots {
            break; 
        }

        let base_idx = usize::new(1) + i * usize::new(5);
        let obj_type  = u32::cast_from(config[base_idx]);
        let obj_size  = config[base_idx + usize::new(1)];
        let offset_x  = config[base_idx + usize::new(2)];
        let offset_y  = config[base_idx + usize::new(3)];
        let offset_z  = config[base_idx + usize::new(4)];
        
        let p_slot = Vec3::new(local_p.x - offset_x, local_p.y - offset_y, local_p.z - offset_z);

        // Instanziierung der Geometrie
        let d_obj = p_slot.x.abs() + p_slot.y.abs() + p_slot.z.abs() - obj_size;

        // CSG-Kombinations-Logik basierend auf dem Typ
        if obj_type == u32::new(1) {
            // Typ 1: Glatte Verschmelzung in die Gesamtszene (Addition)
            final_scene = smin(final_scene, d_obj, blend_factor);
        }
        if obj_type == u32::new(2) {
            // Typ 2: Glatte Subtraktion (Das Objekt stanzt ein Loch in die Szene)
            final_scene = ssub(final_scene, d_obj, blend_factor);
        }
        if obj_type == u32::new(3) {
            // Typ 3: Glatte Schnittmenge (Nur die Überschneidung bleibt sichtbar)
            final_scene = sinter(final_scene, d_obj, blend_factor);
        }

        i += usize::new(1);
    }
    
    final_scene
}
*/
