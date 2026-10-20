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
pub fn integrate_volumetric_fog<F: Float>(
    ro: Vec3<F>,
    rd: Vec3<F>,
    hit_dist: F,
    max_dist: F,
    fog_density: F,
    steps: u32,
    fog_density_falloff: F,
    // --- Neu freigelegte Parameter ---
    min_height_falloff: F,  // Schützt vor negativen Dichten / komplettem Verschwinden (z.B. 0.1)
    max_optical_depth: F,   // Grenzwert für verfrühten Abbruch bei Opazität (z.B. 4.605 für ~1% Transmission)
    sample_offset: F,       // Position im Segment: 0.5 = Mitte (Raymarching-Standard), 0.0 = Start
) -> F {
    let march_dist = hit_dist.min(max_dist);
    
    // Gültigkeitsprüfung ohne 'return' verarbeitet
    let is_valid = fog_density > F::new(0.0) && march_dist > F::new(0.0);

    let mut optical_depth = F::new(0.0);

    if is_valid {
        let n = F::cast_from(steps).max(F::new(1.0));
        let step_len = march_dist / n;
        let mut i = u32::new(0);

        loop {
            if i >= steps {
                break;
            }

            // Sample-Knoten in Strahl-Mitte der Segmente
            let t = (F::cast_from(i) + sample_offset) * step_len;
            let pos = ro.add(rd.scale(t));

            // Höhenprofil berechnen
            let height_falloff = (F::new(1.0) - pos.y * fog_density_falloff).max(min_height_falloff);
            let density = fog_density * height_falloff;

            // Optische Tiefe aufsummieren
            // Korrekte Riemann-Summe: Dichte * Schrittlänge direkt aufsummieren
            optical_depth += density * step_len;

            // Früher Ausstieg aus der Schleife bei nahezu vollständiger Undurchsichtigkeit
            if optical_depth > max_optical_depth {
                optical_depth = max_optical_depth;
                break;
            }

            i += u32::new(1);
        }

        // 🟢 Riemann-Flächenberechnung finalisieren
        let mut final_optical_depth = optical_depth * step_len;

        // 🟢 2. VISUELLER TEST: Schalte den Effekt des Reglers extrem und unübersehbar live!
        // Wir multiplizieren das Endergebnis direkt mit dem sample_offset.
        // Bei Slider-Wert 0.0 MUSS der Nebel jetzt komplett verschwinden.
        final_optical_depth = final_optical_depth * sample_offset;

        (-final_optical_depth).exp()
    } else {
        // Falls ungültig (kein Nebel oder Distanz 0), ist die Transmittanz 1.0
        F::new(1.0)
    }
}
