//! Dedizierter Render-Thread: kontinuierliche Raymarch-Loop.
//!
//! Warum ein OS-Thread statt `cx.background_executor().spawn`? Der Raymarch
//! ist keine endliche Task, sondern eine dauerhafte ~60-FPS-Loop; ein
//! Executor-Worker wäre dauerhaft blockiert. Das etablierte wgpui-Pattern
//! für kontinuierliche GPU-Animationen ist ein eigener Thread, der
//! `back_view_with_size()` / `present()` / `wait_for_present()` pollt.
//!
//! Shutdown: `fps_tx` wird einmal pro Sekade gesendet — fällt der Send fehl
//! (Receiver im View wurde mit dem Fenster gedroppt), beendet sich die Loop.

use crate::app_state::ApplicationState;
use crate::gpu::GpuPipeline;
use crate::input::InputManager;
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

    loop {
        // Frame-Pacing gegen den UI-Compositor (blockt, bis der letzte
        // präsentierte Frame konsumiert wurde).
        surface.wait_for_present();

        // Back-Buffer existiert erst nach dem ersten Layout-Pass → Retry.
        let Some((view, (dw, dh))) = surface.back_view_with_size() else {
            std::thread::sleep(Duration::from_nanos(500));
            continue;
        };

        // Resize: nur bei gültiger Größe re-allozieren + Resolution neu
        // schreiben. (`||` — nicht `&&`: jede geänderte Dimension zählt.)
        if (dw != pipeline.resolution.width || dh != pipeline.resolution.height)
            && dw > 0
            && dh > 0
        {
            pipeline.resize(dw, dh);
        }

        let now = Instant::now();
        let dt = now.duration_since(last_frame_time).as_secs_f32();
        let time = start_time.elapsed().as_secs_f32();
        last_frame_time = now;

        // State-Snapshot: Input anwenden, dann GPU-Register schreiben.
        let mut s = state.lock().unwrap();
        input_manager.update_camera_movement(&mut s, dt);
        
       /* //TEST
        // =========================================================================
// 🔬 VISUELLER PBR-BELASTUNGSTEST (Erzwingt extremes Material-Morphing)
// =========================================================================
// Wir modifizieren den State direkt vor dem Schreiben in den VRAM,
// um zu sehen, ob die GPU-Akkumulatoren reaktiv antworten.
let time = start_time.elapsed().as_secs_f32();

// 1. Lass den Kristall (Slot 0) rhythmisch wie ein Plasma-Reaktor glühen (0.0 bis 8.0)
s.slot_emissive[0] = (time * 3.0).sin().abs() * 8.0;

// 2. Morphing der Oberflächenstruktur (Wechselt zwischen Chrom und Sandstein)
// Sinus schwingt flüssig zwischen 0.0 und 1.0
let morph_wave = (time * 2.0).sin().abs(); 

if morph_wave > 0.5 {
    // 💎 SPIEGELNDES CHROM (Erzwingt stechende Blinn-Phong/PBR-Reflexe)
    s.slot_roughness[0] = 0.02; // Spiegelglatt
    s.slot_metallic[0]  = 1.0;  // Vollmetall
    s.slot_specular[0]  = 1.0;  // Maximale Reflektivität
} else {
    // 🪵 DIFFUSER SANDSTEIN (Erzwingt flache, matte Lichtstreuung)
    s.slot_roughness[0] = 0.90; // Extrem rau
    s.slot_metallic[0]  = 0.0;  // Nicht-Metall
    s.slot_specular[0]  = 0.1;  // Stumpf
}
// =========================================================================

        //TEST END
        */
        pipeline.write_state_buffers(&s);
        pipeline.render_frame(&view, dw, dh, &s, time);
        drop(s);

        surface.present();

        // FPS-Reporting (1×/s). Fehler = Receiver weg → sauber beenden.
        frame_count += 1;
        let elapsed = last_report.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            if fps_tx.send(fps).is_err() {
                break;
            }
            frame_count = 0;
            last_report = Instant::now();
        }
    }
}
