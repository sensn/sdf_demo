//! **TR.md Step 1 — The Zero-Copy 3D Noise and Core Density Field Generator.**
//!
//! Hochfrequentes fraktales Rauschen (FBM) wird vollständig in Registern über
//! trilineare Hermite-Interpolation erzeugt — ohne Texture3D-Sampler und ohne
//! Voxel-Grids (Zero-Copy-Philosophie des TR).
//!
//! Dichte-Mapping (TR §2.1): ρ(p) = clip(-d · σ_scale, 0, 1) — negative
//! Distanzen (Inneres des mathematischen Solids) ergeben hohe Dichte,
//! positive Distanzen zerfallen rapide zu ρ = 0.

use cubecl::prelude::*;

use crate::kernel::types::Vec3;
use crate::kernel::objects::evaluate_dynamic_gyroid;

/// 🌌 Volumen-Stichprobe: Dichte + HDR-Emission (Blackbody-Kern) — rein in
/// Registern berechnet, ohne Texturzugriff.
#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct VolumeSample {
    pub density: f32,
    pub emission: Vec3,
}

/// Berechnet einen register-isolierten 3D-Raum-Hash (sin-Fraktional-PRNG).
#[cube]
pub fn hash_3d(p: Vec3) -> f32 {
    let dot_product = p.x * f32::new(127.1)
        + p.y * f32::new(311.7)
        + p.z * f32::new(74.7);
    let sine_val = dot_product.sin();
    sine_val * f32::new(43758.5453) - (sine_val * f32::new(43758.5453)).floor()
}

/// Erzeugt eine glatte trilineare Wert-Rauschschicht (8-Ecken-Hermite-Lerp).
#[cube]
pub fn noise_3d(p: Vec3) -> f32 {
    let i = Vec3::new(p.x.floor(), p.y.floor(), p.z.floor());
    let f = Vec3::new(p.x - p.x.floor(), p.y - p.y.floor(), p.z - p.z.floor());

    // Kubische Hermite-Smoothstep-Kurven (u = f·f·(3 − 2f))
    let u = Vec3::new(
        f.x * f.x * (f32::new(3.0) - f32::new(2.0) * f.x),
        f.y * f.y * (f32::new(3.0) - f32::new(2.0) * f.y),
        f.z * f.z * (f32::new(3.0) - f32::new(2.0) * f.z),
    );

    let i000 = hash_3d(i);
    let i100 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(0.0), f32::new(0.0))));
    let i010 = hash_3d(i.add(Vec3::new(f32::new(0.0), f32::new(1.0), f32::new(0.0))));
    let i110 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(1.0), f32::new(0.0))));
    let i001 = hash_3d(i.add(Vec3::new(f32::new(0.0), f32::new(0.0), f32::new(1.0))));
    let i101 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(0.0), f32::new(1.0))));
    let i011 = hash_3d(i.add(Vec3::new(f32::new(0.0), f32::new(1.0), f32::new(1.0))));
    let i111 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(1.0), f32::new(1.0))));

    let x00 = i000 + u.x * (i100 - i000);
    let x10 = i010 + u.x * (i110 - i010);
    let x01 = i001 + u.x * (i101 - i001);
    let x11 = i011 + u.x * (i111 - i011);

    let y0 = x00 + u.y * (x10 - x00);
    let y1 = x01 + u.y * (x11 - x01);

    y0 + u.z * (y1 - y0)
}

/// Bewertet implizite Gyroid-Gleichungen, moduliert mit mehr-octavigem FBM-
/// Fluid-Rauschen — der Kern-Dichtefeld-Generator des Volumens.
#[cube]
pub fn get_volumetric_data(p: Vec3, time: f32, size: f32) -> VolumeSample {
    // Basis-Geometrie: dynamischer Gyroid (kugel-bounded → endliche Wolke)
    let base_gyroid = evaluate_dynamic_gyroid(p, time, size);

    // Extrapoliere strukturelles Wind-Geschwindigkeitsfeld über die Laufzeit
    let wind = Vec3::new(time * f32::new(0.2), time * f32::new(0.5), time * f32::new(0.1));
    let noise_pos = p.add(wind);

    // Multi-octavige fraktale Brownian-Bewegung (2 Oktaven, 0.7/0.3)
    let noise_octave1 = noise_3d(noise_pos.scale(f32::new(3.0)));
    let noise_octave2 = noise_3d(noise_pos.scale(f32::new(6.12)));
    let combined_noise = noise_octave1 * f32::new(0.7) + noise_octave2 * f32::new(0.3);

    // Verforme perfekte geometrische Grenzfelder zu turbulenten Dampfwolken
    let modulated_sdf = base_gyroid + combined_noise * f32::new(0.15);

    // TR §2.1: ρ = clip(−d · σ_scale, 0, 1) mit σ_scale = 6.0
    let density_scale = f32::new(6.0);
    let density = (-modulated_sdf * density_scale)
        .min(f32::new(1.0))
        .max(f32::new(0.0));

    // Interne Verbrennungs-Thermik (Emissionskern)
    let heat = (-base_gyroid * f32::new(4.0) + combined_noise * f32::new(2.0))
        .min(f32::new(1.0))
        .max(f32::new(0.0));

    // HDR-strahlendes thermisches Emissionsprofil (Blackbody-Simulation)
    let glow_color = Vec3::new(
        heat * f32::new(3.5),                       // Über-/gesättigte Strahlungsenergie
        heat * heat * f32::new(1.4),                // Mittlerer spektraler Grün-Anteil
        heat * heat * heat * heat * f32::new(0.3),  // Kern-lokalisiertes thermisches Blau
    );

    VolumeSample {
        density,
        emission: glow_color,
    }
}
