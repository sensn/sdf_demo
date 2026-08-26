use cubecl::prelude::*;
use cubecl::frontend::CubeType;
// Auch hier müssen Struct, Funktionen und deren Expansions bekannt sein
//use crate::kernel::{Vec3, Vec3Expand, scene_sdf, scene_sdf_expand};
use crate::kernel::{Vec3, Vec3Expand, scene_sdf}; 
// Sollte der Compiler im Hintergrund immer noch nach der Expansion suchen,
// holt dieser Wildcard-Import sie automatisch und fehlerfrei in den Scope:
use crate::kernel::*; 
/// Berechnet die Umgebungsverdeckung (Ambient Occlusion) für das Baukastensystem
#[cube]
pub fn calculate_ao(
    p: Vec3, 
    normal: Vec3, 
    time: f32, 
    blend_factor: f32,
    config: &Tensor<f32>
) -> f32 {
    let mut occ = f32::new(0.0);
    let mut sca = f32::new(1.0);
    
    // Konforme Schleifenstruktur für das CubeCL 0.11 GPU-Frontend
    let mut i = u32::new(0);
    loop {
        if i >= u32::new(5) { 
            break; 
        }
        
        // FIX: Klone für die Closure innerhalb der Schleife erstellen
        let current_p = p.clone();
        let current_normal = normal.clone();
        
        // Sauberer Type-Cast via f32::cast_from
        let hr = f32::new(0.01) + f32::new(0.12) * f32::cast_from(i) / f32::new(4.0);
        let ao_pos = current_p.add(current_normal.scale(hr));
        let d = scene_sdf(ao_pos, time, blend_factor, config);
        
        occ += (hr - d) * sca;
        sca *= f32::new(0.95);
        
        i += u32::new(1);
    }

    let ao_factor = f32::new(1.0) - (occ * f32::new(4.0)).min(f32::new(1.0)).max(f32::new(0.0));
    ao_factor
}
