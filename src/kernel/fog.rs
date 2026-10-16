//! `kernel/fog.rs` — Volumetrischer Nebel: Beer-Lambert-Integration.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)

#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::types::Vec3;
/// Volumetrischer Nebel (Step 7): schrittweise Integration der
/// Streu-Dichte entlang des Primärstrahls bis zur Treffer-Distanz.
/// Liefert Transmissions-Faktor [0,1] (1 = keine Absorption).
#[cube]
pub fn integrate_volumetric_fog(
    ro: Vec3,
    rd: Vec3,
    hit_dist: f32,
    max_dist: f32,
    fog_density: f32,
    steps: u32,
    fog_density_falloff: f32,
) -> f32 {
    let march_dist = hit_dist.min(max_dist);
    
    // Gültigkeitsprüfung ohne 'return' verarbeitet
    let is_valid = fog_density > f32::new(0.0) && march_dist > f32::new(0.0);

    let mut optical_depth = f32::new(0.0);

    if is_valid {
        let n = f32::cast_from(steps).max(f32::new(1.0));
        let step_len = march_dist / n;
        let mut i = u32::new(0);

        loop {
            if i >= steps {
                break;
            }

            // Sample-Knoten in Strahl-Mitte der Segmente
            let t = (f32::cast_from(i) + f32::new(0.5)) * step_len;
            let pos = ro.add(rd.scale(t));

            // Höhenprofil
            let height_falloff = (f32::new(1.0) - pos.y * fog_density_falloff).max(f32::new(0.1));
            let density = fog_density * height_falloff;

            // Optische Tiefe aufsummieren
            optical_depth += density;

            // Früher Ausstieg aus der Schleife (break ist auf der GPU meist erlaubt)
            if optical_depth * step_len > f32::new(4.605) {
                optical_depth = f32::new(4.605) / step_len;
                break;
            }

            i += u32::new(1);
        }

        // Exponentiation am Ende, falls der Pfad gültig war
        (-optical_depth * step_len).exp()
    } else {
        // Falls ungültig (kein Nebel oder Distanz 0), ist die Transmittanz 1.0
        f32::new(1.0)
    }
}
