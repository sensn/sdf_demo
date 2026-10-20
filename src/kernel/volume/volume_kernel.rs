//! **TR.md Step 5 — Hardware-Linked Primary Compute Launch Kernel.**
//!
//! Verknüpft hardware-allozierte Invokations-IDs mit physischen Bildschirm-
//! Fragmenten, richtet die Kamera-Vektoren ein, wertet den Integrations-Stack
//! aus und committet den Output direkt in den linearen Speicher (Row-major).
//!
//! Layout: **RGB (3×f32 pro Pixel)** — identisch zum Surface-Kernel und
//! Blit-Vertrag (`shader.wgsl`); Alpha ist implizit 1.0 (D1 im Plan).

use cubecl::prelude::*;

use crate::kernel::types::{CameraSettings, EnvSettings, ArchParams, FoldParams, RenderSettings, Vec3};
use crate::kernel::volume::integrate::volumetric_raymarch_integrated;

/// Volumetrischer Render-Kernel: Pixel → Jitter → Gyroid-Dichte-Feld →
/// Beer-Lambert-Transmittanz → Henyey-Greenstein-In-Scattering → HDR-Buffer.
///
/// Zero-Copy: kein Szenen-Buffer, keine Textur — alle Raumdaten entstehen
/// prozedural in den ALU-Registern (TR §1, §4.1).
#[cube(launch)]
pub fn render_volumetric_kernel(
    output: &mut Tensor<f32>,
    camera: CameraSettings,
    env: EnvSettings,
    arch: ArchParams,
    fold: FoldParams,
    render: RenderSettings,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    // Out-of-Bounds-Sicherheitszaun für nicht teilbare Subgroup-Grenzen
    let w = render.width;
    let h = render.height;
    let in_bounds = x < w && y < h;

    // Farbe (HDR) mit Null initialisieren
    let mut color_x = f32::new(0.0);
    let mut color_y = f32::new(0.0);
    let mut color_z = f32::new(0.0);

    if in_bounds {
        let w_f = f32::cast_from(w);
        let h_f = f32::cast_from(h);

        // Screenspace → zentrierte, aspect-korrigierte UV (TR Step 5;
        // Codebase-Konvention: Höhe als Referenz, y nach oben invertiert —
        // identische Kamera wie im Surface-Modus, (fx−w/2)/h entspricht
        // ((fx/w)·2−1)·(aspect/2), also TRs Projektionskorrektur in 0.5-Skala)
        let fx = f32::cast_from(x);
        let fy = f32::cast_from(y);
        let uv_x = (fx - (w_f / f32::new(2.0))) / h_f;
        let uv_y = ((h_f / f32::new(2.0)) - fy) / h_f;

        // Strahlursprung aus der Kameraposition
        let ray_origin = Vec3::new(camera.position.x, camera.position.y, camera.position.z);

        // Kamera-Basis des Codebase: Yaw/Pitch-Rotation + Normierung
        let rd = Vec3::new(uv_x, uv_y, f32::new(1.2));
        let cos_y = camera.yaw.cos();
        let sin_y = camera.yaw.sin();
        let cos_p = camera.pitch.cos();
        let sin_p = camera.pitch.sin();

        let rd_y1 = rd.y * cos_p - rd.z * sin_p;
        let rd_z1 = rd.y * sin_p + rd.z * cos_p;

        let ray_dir = Vec3::new(
            rd.x * cos_y + rd_z1 * sin_y,
            rd_y1,
            -rd.x * sin_y + rd_z1 * cos_y,
        )
        .normalize();

        // TR Step 4/5-Konstanten: max_dist = 25, steps = 32
        let max_dist = f32::new(25.0);
        let steps = u32::new(32);

        // Den zusammengesetzten volumetrischen Berechnungs-Stack auswerten
        let final_hdr_color = volumetric_raymarch_integrated(
            ray_origin,
            ray_dir,
            max_dist,
            steps,
            x,
            y,
            render,
            env,
        );

        color_x = final_hdr_color.x;
        color_y = final_hdr_color.y;
        color_z = final_hdr_color.z;
    }

    // Memory Address Offset Serialization: Row-major lineare Byte-Zuweisung
    // (direkt als usize, wie im Surface-Kernel)
    let x_idx = usize::cast_from(x);
    let y_idx = usize::cast_from(y);
    let w_idx = usize::cast_from(w);
    let pixel_index = (y_idx * w_idx + x_idx) * usize::new(3); // RGB: 3×f32/Pixel (D1)

    output[pixel_index] = color_x;
    output[pixel_index + usize::new(1)] = color_y;
    output[pixel_index + usize::new(2)] = color_z;
}
