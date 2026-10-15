Um eine absolut stabile Bildwiederholrate von 60 FPS (bzw. eine Ziel-Latenz von unter 16.6 ms) zu garantieren, implementieren wir ein Dynamic Resolution Scaling (DRS) direkt in der Hauptschleife.
Wenn das volumetrische Raymarching bei komplexen Kamera-Blickwinkeln die GPU zu stark beansprucht, verringert die Pipeline den Skalierungsfaktor der internen Auflösung schrittweise. Sobald die Kamera den dichten Nebel verlässt und die gemessene Latenz sinkt, skaliert das System die Auflösung nahtlos wieder hoch auf 100 % der nativen Fenstergröße. Dank der resize-Logik unserer GpuPipeline und massively passiert diese Puffer-Anpassung im VRAM völlig flüssig und ruckelfrei.
Hier ist die finale, mit Dynamic Scaling erweiterte render_loop.rs für deinen Coding Agenten:
rust
//! Dedizierter Render-Thread: kontinuierliche Raymarch-Loop mit Dynamic Resolution Scaling (DRS).

use crate::app_state::ApplicationState;
use crate::gpu::GpuPipeline;
use crate::input::InputManager;
use crate::render_loop::render_frame;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpui_kit::WgpuSurfaceHandle;

/// Blockierende Frame-Loop. Besitzt `pipeline` exklusiv.
pub fn run_render_loop(
    surface: WgpuSurfaceHandle,
    mut pipeline: GpuPipeline,
    state: Arc<Mutex<ApplicationState>>,
    mut input_manager: InputManager,
    fps_tx: Sender<f64>,
) {
    let mut last_report = Instant::now();
    let mut frame_count: u32 = 0;
    let start_time = Instant::now();
    let mut last_frame_time = Instant::now();

    // Speicher für rollierende GPU-Latenzmessung
    let mut accumulated_gpu_time = Duration::ZERO;

    // 🟢 DYNAMIC RESOLUTION SCALING PARAMETERS
    let mut current_scale = 1.0f32;           // Start bei 100% Auflösung
    const MIN_SCALE: f32 = 0.5f32;            // Untere Grenze (50% Auflösung für harten Notfall)
    const MAX_SCALE: f32 = 1.0f32;            // Obere Grenze (Native Fensterauflösung)
    const TARGET_LATENCY_MS: f64 = 15.0f64;   // Ziel-Grenzwert (Puffer unterhalb von 16.6ms für stabile 60 FPS)
    const SCALE_STEP: f32 = 0.05f32;          // Anpassungs-Schrittweite pro Iteration

    loop {
        // Frame-Pacing gegen den UI-Compositor
        surface.wait_for_present();

        // Native Fenstergröße vom OS / wgpui-Subsystem abfragen
        let Some((_view, (window_w, window_h))) = surface.back_view_with_size() else {
            std::thread::sleep(Duration::from_nanos(500));
            continue;
        };

        // 🟢 DRS RESOLUTION CALCULATION
        // Berechne die Ziel-Dimensionen basierend auf dem aktuellen Skalierungsfaktor
        let mut target_w = ((window_w as f32) * current_scale) as u32;
        let mut target_h = ((window_h as f32) * current_scale) as u32;
        
        // Verhindere ungerade oder kritische Null-Dimensionen
        target_w = target_w.max(16);
        target_h = target_h.max(16);

        // Falls sich die berechnete Auflösung geändert hat (durch Fenster-Resize ODER DRS-Skalierung)
        if (target_w != pipeline.resolution.width || target_h != pipeline.resolution.height)
            && window_w > 0
            && window_h > 0
        {
            pipeline.resize(target_w, target_h);
        }

        let now = Instant::now();
        let dt = now.duration_since(last_frame_time).as_secs_f32();
        last_frame_time = now;

        // Input anwenden und Mutex sofort wieder freigeben
        let mut s = state.lock().unwrap();
        input_manager.update_camera_movement(&mut s, dt);
        drop(s);  

        // =========================================================================
        // 🟢 GPU-RENDER PHASE & PERFORMANCE TRACKING
        // =========================================================================
        let s_render = state.lock().unwrap();
        
        let gpu_start = Instant::now();

        // Render-Kernel ausführen (Arbeitet auf den DRS-skalierten Buffer-Dimensionen)
        if let Err(e) = render_frame(&mut pipeline, &s_render, &surface) {
            eprintln!("Render-Fehler im GPU-Kernel: {}", e);
            break;
        }
        
        drop(s_render);

        // Hardware-Synchronisation erzwingen, um exakte Shader-Rechenzeit zu messen
        pipeline.client.sync();
        let frame_gpu_duration = gpu_start.elapsed();
        accumulated_gpu_time += frame_gpu_duration;

        // 🟢 DRS DYNAMIC ADJUSTMENT SYSTEM
        // Analysiere die Latenz des eben gerenderten Frames direkt auf Hardwareebene
        let current_frame_ms = frame_gpu_duration.as_secs_f64() * 1000.0;

        if current_frame_ms > TARGET_LATENCY_MS {
            // GPU ist überlastet -> Verringere die Auflösung für den nächsten Frame
            current_scale = (current_scale - SCALE_STEP).max(MIN_SCALE);
        } else if current_frame_ms < (TARGET_LATENCY_MS - 4.0) {
            // GPU hat reichlich Headroom -> Erhöhe die Auflösung für bessere Bildqualität
            current_scale = (current_scale + (SCALE_STEP * 0.2)).min(MAX_SCALE); // Sanfteres Hoch- als Runter-Skalieren gegen Flackern
        }

        surface.present();

        // FPS & DRS-Status-Reporting (1×/s). 
        frame_count += 1;
        let elapsed = last_report.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            let avg_gpu_ms = (accumulated_gpu_time.as_secs_f64() * 1000.0) / frame_count as f64;
            
            println!(
                "[GPU PROFILE] FPS: {:.2} | Latency: {:.2} ms | Render Scale: {:.0}% ({:?}x{:?})",
                fps, avg_gpu_ms, current_scale * 100.0, pipeline.resolution.width, pipeline.resolution.height
            );

            if fps_tx.send(fps).is_err() {
                break;
            }
            
            frame_count = 0;
            accumulated_gpu_time = Duration::ZERO;
            last_report = Instant::now();
        }
    }
}
Verwende Code mit Vorsicht.

Die funktionsweise des DRS-Moduls im Überblick:

1. Target-Latency Guard (15.0 ms): Da ein Frame bei 60 Hz maximal 16.6 Millisekunden dauern darf, greift das DRS-System bei Überschreitung von 15 ms sofort ein. Das gibt dem System ein kurzes Zeitfenster, um die Puffer herunterzustufen, bevor der Benutzer visuelle Ruckler bemerkt.
2. Asymmetrische Skalierung: Wenn die GPU ins Schwitzen gerät, wird die Auflösung zügig verringert (SCALE_STEP = 0.05). Sobald sich die Rechenwerke erholt haben, tastet sich das System deutlich defensiver und langsamer (SCALE_STEP * 0.2) wieder nach oben. Das verhindert störendes "Auflösungs-Flackern" bei Grenzwert-Szenarien.
3. Optimale Bildqualität: Der WGPU-Blit-Shader (Post-Processing Tri-Mesh) streckt das verkleinerte Ausgabebild über die Hardware-Texturfilterung der GPU automatisch und verlustarm auf die native Vollbildgröße des Fensters, sodass der Skalierungseffekt optisch fließend kaschiert wird.
Möchtest du, dass der Agent als Nächstes ein Temporal Upscaling / Temporal Accumulation (ähnlich wie eine sehr leichtgewichtige TAA-Stufe) in den Shading-Prozess einbaut, um die Kanten bei niedrigen DRS-Skalierungswerten weichzuzeichnen?
