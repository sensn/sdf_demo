Um das Jitter-Rauschen pro Pixel korrekt zu berechnen, müssen wir die globalen 2D-Pixelkoordinaten auf dem Bildschirm ermitteln. In CubeCL geschieht dies hocheffizient über das integrierte cubecl::prelude::*-Modul.
Anstatt wie in anderen Grafik-APIs (wie wgpu/WGSL) Variablen wie global_invocation_id manuell zu parsen, stellt CubeCL das native Konstrukt LAUNCH_ID_X und LAUNCH_ID_Y zur Verfügung. Diese Abfragen werden beim Kompilieren in die schnellstmöglichen Hardware-Register der jeweiligen GPU-Architektur (z. B. CUDA threadIdx/blockIdx oder WebGPU global_invocation_id) übersetzt – komplett Zero-Cost.
Hier ist die Implementierung deines Haupt-Einstiegskernels (#[cube(launch)]), der die Pixel-IDs ausliest, den Kamerastrahl berechnet und das volumetrische Raymarching startet:
rust
#![allow(warnings)]
use cubecl::prelude::*;

// Wir nehmen an, dass dein Ausgabebuffer ein flaches f32- oder u32-Array für die Pixelfarben ist.
// params1-5 konform übergeben wir die typisierten Structs direkt als Launch-Parameter.
#[cube(launch)]
pub fn render_kernel(
    output_image: &mut Tensor<f32>, // Der finale Framebuffer (Größe: width * height * 4 für RGBA)
    camera: CameraSettings,
    env: EnvSettings,
    arch: ArchParams,
    fold: FoldParams,
    settings: RenderSettings,
) {
    // 1. 🟢 Globale Pixelkoordinaten direkt aus den GPU-Hardware-Registern auslesen
    let pixel_x = LAUNCH_ID_X;
    let pixel_y = LAUNCH_ID_Y;

    // Sicherheits-Check: Falls die Grid-Größe nicht exakt aufgeht, außerhalb des Bildschirms abbrechen
    if pixel_x >= settings.width || pixel_y >= settings.height {
        return;
    }

    // 2. Normalisierte Bildschirmkoordinaten (UV) berechnen (-1.0 bis 1.0)
    let fx = f32::cast(pixel_x);
    let fy = f32::cast(pixel_y);
    let width_f = f32::cast(settings.width);
    let height_f = f32::cast(settings.height);

    // UVs zentrieren und das Seitenverhältnis (Aspect Ratio) ausgleichen
    let uv_x = (fx / width_f) * f32::new(2.0) - f32::new(1.0);
    let uv_y = (fy / height_f) * f32::new(2.0) - f32::new(1.0);
    let aspect = width_f / height_f;
    let uv_x_adjusted = uv_x * aspect;

    // 3. Setup des Kamerastrahls (Ray Origin & Ray Direction)
    // Einfaches mathematisches Setup basierend auf deinen CameraSettings (Yaw/Pitch)
    let ray_origin = Vec3::new(camera.position.x, camera.position.y, camera.position.z);
    
    // Einfacher Richtungsvektor (für eine vollumfängliche Matrix-Kamera müsste hier mit Yaw/Pitch rotiert werden)
    let ray_dir = Vec3::new(uv_x_adjusted, uv_y, f32::new(-1.0)).normalize();

    // 4. Aufruf des zuvor gebauten volumetrischen Jitter-Raymarchers
    let max_dist = f32::new(20.0);
    let steps = u32::new(32); // 32 Schritte reichen dank Jitter vollkommen aus!
    
    let pixel_color = volumetric_raymarch_jittered(
        ray_origin,
        ray_dir,
        max_dist,
        steps,
        pixel_x,     // 🟢 Hier übergeben
        pixel_y,     // 🟢 Hier übergeben
        settings,
        env,
    );

    // 5. Index im flachen Ausgabe-Tensor berechnen (Linearer Offset für RGBA)
    let pixel_index = (pixel_y * settings.width + pixel_x) * u32::new(4);

    // Farbe in den Framebuffer schreiben (mit rudimentärem Alpha = 1.0)
    output_image[pixel_index + u32::new(0)] = pixel_color.x; // R
    output_image[pixel_index + u32::new(1)] = pixel_color.y; // G
    output_image[pixel_index + u32::new(2)] = pixel_color.b; // B
    output_image[pixel_index + u32::new(3)] = f32::new(1.0);  // A
}
Verwende Code mit Vorsicht.

Wie das auf dem Host (CPU-Seite) aufgerufen wird:

Wenn du diesen Kernel von deiner Hauptanwendung in Rust aus startest, nutzt du den von CubeCL automatisch generierten Launch-Wrapper. Dank des CubeLaunch-Traits deiner Parameter-Structs sieht das extrem sauber aus:
rust
// Typischer Host-Aufruf in deinem main.rs / gpu.rs
render_kernel::launch(
    &device,
    // CubeCL berechnet die benötigten Workgroups basierend auf der Bildschirmauflösung
    CubeCount::Static(settings.width / 16, settings.height / 16, 1), 
    TensorArg::from_raw_parts(&output_buffer, ...),
    camera_settings,
    env_settings,
    arch_params,
    fold_params,
    render_settings,
);
Verwende Code mit Vorsicht.
Damit schließt sich der Kreis: Deine CPU sendet typisierte, speicherausgerichtete Datenblöcke, die GPU liest über ihre Hardware-Register die exakten Pixel-IDs aus, berechnet ein zeitlich und räumlich gejittertes Dichte-Volumen und schreibt das Ergebnis ohne jegliche Speicherallokations-Overheads (Zero-Copy) direkt in den Grafikspeicher zurück.
Möchtest du nun das Licht-Setup erweitern, um beispielsweise eine atmosphärische Phasenfunktion wie die Henyey-Greenstein-Anisotropie (aus dem Shader Bits-Blog für realistisches Vorwärtsstreuen des Lichts bei Gegenlicht) zu integrieren?
