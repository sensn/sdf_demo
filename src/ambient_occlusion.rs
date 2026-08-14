use cubecl::prelude::*;
use crate::kernel::{Vec3, scene_sdf};

/// Berechnet die Umgebungsverdeckung (Ambient Occlusion) für das Baukastensystem
#[cube]
pub fn calculate_ao(
    p: Vec3, 
    normal: Vec3, 
    time: f32, 
    blend_factor: f32,
    config: &Tensor<f32>
) -> f32 {
    let mut occ = 0.0;
    let mut sca = 1.0;
    
    for i in 0..5 {
        let hr = 0.01 + 0.12 * (i as f32) / 4.0;
        let ao_pos = p.add(normal.scale(hr));
        let d = scene_sdf(ao_pos, time, blend_factor, config);
        
        occ += (hr - d) * sca;
        sca *= 0.95;
    }
    
    let ao_factor = 1.0 - (occ * 4.0).min(1.0).max(0.0);
    ao_factor
}
