//! **TR.md Step 2 — The Phase Scattering and Shadowing Core.**
//!
//! Anisotrope Vorwärts-/Rückwärtsstreuung über die Henyey-Greenstein-Phasen-
//! funktion (TR §2.4) plus Beer-Lambert-Einzelstreuungs-Schattenstrahl
//! (TR §2.3): Sekundär-Raymarch geringerer Auflösung Richtung Lichtquelle.

use cubecl::prelude::*;

use crate::kernel::types::{GpuVec3, Vec3};
use crate::kernel::volume::density::get_volumetric_data;

/// Formuliert die Wahrscheinlichkeitsverteilungskurve für Vorwärts-/
/// Rückwärts-Partikel-Anisotropie (TR §2.4):
/// P(cosθ, g) = (1/(4π)) · (1−g²) / (1+g²−2g·cosθ)^1.5
#[cube]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let pi = f32::new(3.14159265);
    let numerator = f32::new(1.0) - g2;
    let denom_inner = f32::new(1.0) + g2 - f32::new(2.0) * g * cos_theta;
    let denominator = f32::new(4.0) * pi * denom_inner * denom_inner.sqrt();
    numerator / denominator
}

/// Richtungsvektor von der Lichtquelle zum Weltpunkt (normalisiert) —
/// ersetzt die im TR referenzierte (nicht existierende) Methode.
#[cube]
pub fn position_to_direction_vector(pos: GpuVec3) -> Vec3 {
    Vec3::new(pos.x, pos.y, pos.z).normalize()
}

/// Sekundäre Raymarch-Operation, die die optische Tiefe Richtung Lichtknoten
/// analysiert (TR §2.3):
/// T_light(p) = exp(−Σ_j ρ(p + j·Δs·L) · Δs · κ_absorb)
#[cube]
pub fn calculate_volume_shadow(
    sample_pos: Vec3,
    light_dir: Vec3,
    time: f32,
    size: f32,
    shadow_steps: u32,
    shadow_step_size: f32,
) -> f32 {
    // Optische Tiefe akkumulieren (Riemann-Summe) — Vec3 wird pro Schritt neu
    // konstruiert (Codebase-Idiom: kein Re-Assign von Cube-Structs, vgl. fog.rs)
    let mut accumulated_shadow_density = f32::new(0.0);

    // Loop/Break/Counter-Idiom des Codebase (statt `for _ in 0..n`)
    let mut s = u32::new(0);
    loop {
        if s >= shadow_steps {
            break;
        }

        // p_i + j·Δs·L  (TR §2.3) — Schrittweite entlang der Schattenrichtung
        let t = shadow_step_size * f32::cast_from(s + u32::new(1));
        let sample_pos_j = sample_pos.add(light_dir.scale(t));
        let sample = get_volumetric_data(sample_pos_j, time, size);
        accumulated_shadow_density = accumulated_shadow_density + sample.density;

        s += u32::new(1);
    }

    let absorption_coefficient = f32::new(0.8);
    (-accumulated_shadow_density * shadow_step_size * absorption_coefficient).exp()
}
