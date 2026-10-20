//! **TR.md Step 4 — The Volumetric Integration Core Loop.**
//!
//! Bringt die gesamte Pipeline zusammen: Schrittweise Integration entlang des
//! Kamera-Sichtstrahls, analytische Transmittanz-Verfolgung (Beer-Lambert,
//! TR §2.2), Phasen-Streuprofile, immersive Lichtquellen-Injektion und
//! Early-Loop-Branching-Optimierungen (Early Ray Termination, TR §4.2).
//!
//! Codebase-Idiom: Farb-Akkumulation über skalare f32-Register
//! (vgl. `hit_r/g/b` im Surface-Kernel) — Cube-Structs werden nicht
//! re-assignt (kein `Assign` für `Vec3Expand` in dieser CubeCL-Version).

use cubecl::prelude::*;

use crate::kernel::types::{EnvSettings, RenderSettings, Vec3};
use crate::kernel::volume::density::get_volumetric_data;
use crate::kernel::volume::dither::noise_2d_time;
use crate::kernel::volume::phase::{
    calculate_volume_shadow, henyey_greenstein, position_to_direction_vector,
};

/// Volumetrischer Kern-Integrations-Loop (TR Step 4).
///
/// - Jitter des Startpunkts um Δx·R(pixel, t) gegen Banding (TR §2.5)
/// - Rekursive Transmittanz: T_{i+1} = T_i · (1 − α_i),
///   α_i = ρ·Δx·κ_fog (TR §2.2)
/// - Schattenstrahl + Henyey-Greenstein-In-Scattering (TR §2.3/§2.4)
/// - Early Ray Termination bei T < 0.001 (TR §4.2)
#[cube]
pub fn volumetric_raymarch_integrated(
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_distance: f32,
    steps: u32,
    pixel_x: u32,
    pixel_y: u32,
    settings: RenderSettings,
    env: EnvSettings,
) -> Vec3 {
    let step_size = max_distance / f32::cast_from(steps).max(f32::new(1.0));

    // Initialisiere hochfrequenten räumlich-zeitlichen Dithering-Offset
    let jitter = noise_2d_time(pixel_x, pixel_y, settings.time);
    let start_offset = ray_dir.scale(step_size * jitter);
    let jittered_origin = ray_origin.add(start_offset);
    let mut step_idx = u32::new(0);

    // Skalare HDR-Farb-Akkumulatoren (Codebase-Idiom, statt Vec3-Mutation)
    let mut acc_r = f32::new(0.0);
    let mut acc_g = f32::new(0.0);
    let mut acc_b = f32::new(0.0);
    let mut transmittance = f32::new(1.0);

    let light_direction = position_to_direction_vector(env.key_light_pos);
    let light_color = Vec3::new(env.key_color.x, env.key_color.y, env.key_color.z);

    // Vorberechnung des anisotropen Streuwinkels pro Sichtstrahl (g = 0.65)
    let cos_theta = ray_dir.dot(light_direction);
    let anisotropy_g = f32::new(0.65); // Positiv: stilisierter Front-Lit-Halo
    let phase_value = henyey_greenstein(cos_theta, anisotropy_g);

    let volume_size = f32::new(1.5);

    loop {
        if step_idx >= steps {
            break;
        }

        // Aktueller Abtastpunkt: O_jittered + i·Δx·D
        let ray_t = step_size * f32::cast_from(step_idx);
        let current_pos = jittered_origin.add(ray_dir.scale(ray_t));

        let sample = get_volumetric_data(current_pos, settings.time, volume_size);

        if sample.density > f32::new(0.0) {
            // Schatten-Tiefe über sekundäre Schatten-Schritte berechnen
            let shadow_steps = u32::new(4);
            let shadow_step_size = step_size * f32::new(1.5);
            let light_transmission = calculate_volume_shadow(
                current_pos,
                light_direction,
                settings.time,
                volume_size,
                shadow_steps,
                shadow_step_size,
            );

            // Globale Einzel-In-Scattering-Phasenmechanik
            let scattered_light_intensity = env.light_intensity * phase_value;
            let external_light = light_color.scale(light_transmission * scattered_light_intensity);

            // Externe In-Scattering mit interner Selbst-Beleuchtung vereinen
            let total_light = external_light.add(sample.emission);

            // Kontinuierliche Opazitäts-Integration: α = ρ·Δx·κ_fog
            let opacity = sample.density * step_size * env.fog_density;
            let weight = opacity * transmittance;

            acc_r = acc_r + total_light.x * weight;
            acc_g = acc_g + total_light.y * weight;
            acc_b = acc_b + total_light.z * weight;
            transmittance = transmittance * (f32::new(1.0) - opacity);

            // 🟢 Early Ray Termination (TR §4.2): Sobald die Transmittanz unter
            // die Sichtbarkeitsschwelle fällt, wird die Ausführung abgebrochen
            if transmittance < f32::new(0.001) {
                transmittance = f32::new(0.0);
                break;
            }
        }

        step_idx += u32::new(1);
    }

    // Hintergrund-Kompositing: Hintergrund · T (verbleibende Transmission)
    let bg_r = acc_r + env.bg_color.x * transmittance;
    let bg_g = acc_g + env.bg_color.y * transmittance;
    let bg_b = acc_b + env.bg_color.z * transmittance;

    Vec3::new(bg_r, bg_g, bg_b)
}
