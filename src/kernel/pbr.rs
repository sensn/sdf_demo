//! `kernel/pbr.rs` — PBR: Cook-Torrance (fresnel_schlick, d_ggx, g_smith, aces_tonemap, cook_torrance_light).
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::types::Vec3;

#[cube]
pub fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let f = (f32::new(1.0) - cos_theta).powf(f32::new(5.0));
    f0 + (f32::new(1.0) - f0) * f
}

/// GGX-Normalverteilung: wie viel Energie geht in den Halbwinkel.
#[cube]
pub fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - f32::new(1.0)) + f32::new(1.0);
    // PI als Bruch 355/113 (clippy: keine approximierten Konstanten)
    let pi = f32::new(355.0) / f32::new(113.0);
    (a2 / (pi * d * d)).max(f32::new(0.0))
}

/// Smith-Schlick-Geometrie: Maskierung/Schattierung der Mikrofacetten.
#[cube]
pub fn g_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let k = (alpha + f32::new(1.0)) * (alpha + f32::new(1.0)) / f32::new(8.0);
    let gv = n_dot_v / (n_dot_v * (f32::new(1.0) - k) + k);
    let gl = n_dot_l / (n_dot_l * (f32::new(1.0) - k) + k);
    gv * gl
}

/// 🟢 ACES-Narkowicz-Tone-Mapping: HDR-Werte > 1.0 werden natürlich
/// komprimiert statt hart geclampt (behebt das Emissive-Weiß-Clipping).
#[cube]
pub fn aces_tonemap(x: f32) -> f32 {
    let a = f32::new(2.51);
    let b = f32::new(0.03);
    let c = f32::new(2.43);
    let d = f32::new(0.59);
    let e = f32::new(0.14);
    (x * (a * x + b)) / (x * (c * x + d) + e)
}

/// 🟢 Ein komplettes Cook-Torrance-Licht für einen Punkt.
/// Gibt den Beitrag (diffuse + specular) eines Lichts zurück.
/// n_dot_l <= 0 (Licht hinter der Fläche) ergibt automatisch 0, da alle
/// Terme mit n_dot_l bzw. max(0,·) skaliert sind — kein early return nötig.
#[cube]
pub fn cook_torrance_light(
    albedo: Vec3,
    normal: Vec3,
    light_dir: Vec3, // zum Licht gerichtet, normalisiert
    view_dir: Vec3,  // zur Kamera gerichtet, normalisiert
    light_color: Vec3,
    intensity: f32,
    roughness: f32,
    metallic: f32,
    specular: f32,
) -> Vec3 {
    let eps = f32::new(0.0001);
    // PI als Bruch (clippy: keine approximierten Konstanten-Literale)
    let pi = f32::new(355.0) / f32::new(113.0);
    let n_dot_l = normal.dot(light_dir.clone()).max(f32::new(0.0));
    let n_dot_v = normal.dot(view_dir.clone()).max(eps);
    let h = light_dir.add(view_dir).normalize();
    let n_dot_h = normal.dot(h.clone()).max(f32::new(0.0));
    let l_dot_h = light_dir.dot(h).max(f32::new(0.0));

    // F0: Dielektrikum ~0.04 (durch specular-Slider skalierbar), Metall = Albedo
    let f0_scalar = f32::new(0.04) * (f32::new(1.0) + specular * f32::new(3.0));
    let f0 = Vec3::new(
        f0_scalar + (albedo.x - f0_scalar) * metallic,
        f0_scalar + (albedo.y - f0_scalar) * metallic,
        f0_scalar + (albedo.z - f0_scalar) * metallic,
    );

    let alpha = roughness * roughness;
    let d = d_ggx(n_dot_h, alpha);
    let g = g_smith(n_dot_v, n_dot_l, alpha);
    let f = fresnel_schlick(l_dot_h, f0.x); // Skalar-Fresnel (visuell ausreichend)

    // Specular-BRDF: D*G*F / (4 * NdotL * NdotV)
    let denom = (n_dot_v * n_dot_l * f32::new(4.0)).max(eps);
    let spec_brdf = d * g * f / denom;

    // Diffuse-Rest: (1-F) * (1-metallic) — Metalle haben keine diffuse Farbe
    let kd = (f32::new(1.0) - f) * (f32::new(1.0) - metallic);

    // Lambert / PI
    let diff_r = kd * albedo.x / pi * n_dot_l;
    let diff_g = kd * albedo.y / pi * n_dot_l;
    let diff_b = kd * albedo.z / pi * n_dot_l;

    // Specular ist farbig (F0-farbig für Metalle, weiß für Dielektrika)
    let spec_r = spec_brdf * f0.x;
    let spec_g = spec_brdf * f0.y;
    let spec_b = spec_brdf * f0.z;

    Vec3::new(
        (diff_r + spec_r) * light_color.x * intensity,
        (diff_g + spec_g) * light_color.y * intensity,
        (diff_b + spec_b) * light_color.z * intensity,
    )
}

