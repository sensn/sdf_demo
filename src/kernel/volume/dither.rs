//! **TR.md Step 3 — High-Frequency Spatial Dithering Operator.**
//!
//! Schneller, register-isolierter Sinus-Fraktional-PRNG für Unified-Shader-Core-
//! ALUs: pixellleckte Raumkoordinaten, gemischt mit globaler Frame-Zeit, um
//! gleichmäßige Banding-Artefakte („Slicing") in hochfrequentes Blau-/Weiß-
//! Rauschen zu verwandeln (TR §2.5) — löst sich per TAA bzw. hoher Framerate
//! sensorischer Persistenz auf, ohne die Ausführungskosten zu skalieren.

use cubecl::prelude::*;

/// Bewertet pixellleckte Raumkoordinaten, gemischt mit globaler Frame-Zeit,
/// für hochfrequentes Rauschen (TR §2.5: O_jittered = O + D·(Δx·R(pixel,t))).
#[cube]
pub fn noise_2d_time(x: u32, y: u32, time: f32) -> f32 {
    let fx = f32::cast_from(x);
    let fy = f32::cast_from(y);
    let dot_product = fx * f32::new(12.9898) + fy * f32::new(78.233) + time * f32::new(43758.5453);
    let sine_val = dot_product.sin();
    sine_val * f32::new(43758.5453) - (sine_val * f32::new(43758.5453)).floor()
}
