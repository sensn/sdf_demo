//! `kernel/pbr.rs` — PBR: Cook-Torrance (fresnel_schlick, d_ggx, g_smith, aces_tonemap, cook_torrance_light).
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::types::Vec3;

#[cube]
pub fn fresnel_schlick<F: Float>(cos_theta: F, f0: F) -> F {
    let exponent: F = F::new(5.0); // 💡 KORREKTUR: Typ explizit erzwingen
    let f = (F::new(1.0) - cos_theta).powf(exponent);
    f0 + (F::new(1.0) - f0) * f
}


/// GGX-Normalverteilung: wie viel Energie geht in den Halbwinkel.
#[cube]
pub fn d_ggx<F: Float>(n_dot_h: F, alpha: F) -> F {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - F::new(1.0)) + F::new(1.0);
    // PI als Bruch 355/113 (clippy: keine approximierten Konstanten)
    let pi = F::new(355.0) / F::new(113.0);
    (a2 / (pi * d * d)).max(F::new(0.0))
}

/// Smith-Schlick-Geometrie: Maskierung/Schattierung der Mikrofacetten.
#[cube]
pub fn g_smith<F: Float>(n_dot_v: F, n_dot_l: F, alpha: F) -> F {
    let k = (alpha + F::new(1.0)) * (alpha + F::new(1.0)) / F::new(8.0);
    let gv = n_dot_v / (n_dot_v * (F::new(1.0) - k) + k);
    let gl = n_dot_l / (n_dot_l * (F::new(1.0) - k) + k);
    gv * gl
}

/// 🟢 ACES-Narkowicz-Tone-Mapping: HDR-Werte > 1.0 werden natürlich
/// komprimiert statt hart geclampt (behebt das Emissive-Weiß-Clipping).
#[cube]
pub fn aces_tonemap<F: Float>(x: F) -> F {
    let a = F::new(2.51);
    let b = F::new(0.03);
    let c = F::new(2.43);
    let d = F::new(0.59);
    let e = F::new(0.14);
    (x * (a * x + b)) / (x * (c * x + d) + e)
}

/// 🟢 Ein komplettes Cook-Torrance-Licht für einen Punkt.
/// Gibt den Beitrag (diffuse + specular) eines Lichts zurück.
/// n_dot_l <= 0 (Licht hinter der Fläche) ergibt automatisch 0, da alle
/// Terme mit n_dot_l bzw. max(0,·) skaliert sind — kein early return nötig.
#[cube]
pub fn cook_torrance_light<F: Float>(
    albedo: Vec3<F>,
    normal: Vec3<F>,
    light_dir: Vec3<F>, // zum Licht gerichtet, normalisiert
    view_dir: Vec3<F>,  // zur Kamera gerichtet, normalisiert
    light_color: Vec3<F>,
    intensity: F,
    roughness: F,
    metallic: F,
    specular: F,
) -> Vec3<F> {
    let eps: F = F::new(0.0001);
    // PI als Bruch (clippy: keine approximierten Konstanten-Literale)
    let pi: F = F::new(355.0) / F::new(113.0);
    let n_dot_l = normal.dot(light_dir.clone()).max(F::new(0.0));
    let n_dot_v = normal.dot(view_dir.clone()).max(eps);
    let h = light_dir.add(view_dir).normalize();
    let n_dot_h = normal.dot(h.clone()).max(F::new(0.0));
    let l_dot_h = light_dir.dot(h).max(F::new(0.0));

    // F0: Dielektrikum ~0.04 (durch specular-Slider skalierbar), Metall = Albedo
    let f0_scalar = F::new(0.04) * (F::new(1.0) + specular * F::new(3.0));
    
    // 💡 INFERENZ-RETTUNG STEP 1: Explizites Auslesen der Albedo-Komponenten als Typ F
    let albedo_x: F = albedo.x;
    let albedo_y: F = albedo.y;
    let albedo_z: F = albedo.z;

    let f0 = Vec3::<F>::new(
        f0_scalar + (albedo_x - f0_scalar) * metallic,
        f0_scalar + (albedo_y - f0_scalar) * metallic,
        f0_scalar + (albedo_z - f0_scalar) * metallic,
    );

        let alpha = roughness * roughness;
    // 💡 KORREKTUR: Expliziter Turbofisch <F> bei den Sub-Funktionsaufrufen!
    let d = d_ggx::<F>(n_dot_h, alpha);
    let g = g_smith::<F>(n_dot_v, n_dot_l, alpha);
    
    let f0_x: F = f0.x;
    let f0_y: F = f0.y;
    let f0_z: F = f0.z;
    // 💡 KORREKTUR: Expliziter Turbofisch <F> beim Fresnel-Aufruf!
    let f = fresnel_schlick::<F>(l_dot_h, f0_x); 
 // Skalar-Fresnel (visuell ausreichend)

    // Specular-BRDF: D*G*F / (4 * NdotL * NdotV)
    let denom = (n_dot_v * n_dot_l * F::new(4.0)).max(eps);
    let spec_brdf = d * g * f / denom;

    // Diffuse-Rest: (1-F) * (1-metallic) — Metalle haben keine diffuse Farbe
    let kd = (F::new(1.0) - f) * (F::new(1.0) - metallic);

    // Lambert / PI
    let diff_r = kd * albedo_x / pi * n_dot_l;
    let diff_g = kd * albedo_y / pi * n_dot_l;
    let diff_b = kd * albedo_z / pi * n_dot_l;

    // Specular ist farbig (F0-farbig für Metalle, weiß für Dielektrika)
    let spec_r = spec_brdf * f0_x;
    let spec_g = spec_brdf * f0_y;
    let spec_b = spec_brdf * f0_z;

    // 💡 INFERENZ-RETTUNG STEP 3: Explizites Auslesen der Lichtfarben als Typ F
    let lc_x: F = light_color.x;
    let lc_y: F = light_color.y;
    let lc_z: F = light_color.z;

    Vec3::<F>::new(
        (diff_r + spec_r) * lc_x * intensity,
        (diff_g + spec_g) * lc_y * intensity,
        (diff_b + spec_b) * lc_z * intensity,
    )
}
