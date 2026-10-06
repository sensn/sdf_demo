//! sdf_demo — wgpui-Bootstrap.
//!
//! Architektur (siehe REFACTORING_PLAN.md):
//! - `app_state.rs`   : geteilter CPU-State + Tensor-Packing
//! - `gpu.rs`         : GpuPipeline (cubecl-Init, Blit-Pipeline, Frame-Render)
//! - `render_loop.rs` : dedizierter Render-Thread (kontinuierlicher Raymarch)
//! - `view.rs`        : SurfaceExample-View (Layout, Keys, FPS-Overlay)
//! - `gui.rs`         : Sidebar (Tabs, Accordions, Slider)
//! - `input.rs`       : WASD-Kamerabewegung
//! - `kernel.rs`      : cubecl-Raymarch-Kernel

pub mod app_state;
pub mod gpu;
pub mod gui;
pub mod input;
mod kernel;
pub mod render_loop;
pub mod view;

use app_state::ApplicationState;
use gpu::GpuPipeline;
use gui::GuiState;
use input::InputManager;
use render_loop::run_render_loop;
use std::sync::{Arc, Mutex};
use std::thread;

use wgpui_kit::{component::theme::ThemeMode, App, AppContext, WindowOptions};

/// Initiale Surface-Größe (Back-Buffer; Resize wird in der Loop erkannt).
const SURFACE_WIDTH: u32 = 1720;
const SURFACE_HEIGHT: u32 = 1080;

fn main() {
    wgpui_kit::application().run(|cx: &mut App| {
        wgpui_kit::init(cx);
        // Dark-Theme: wgpui_kit::init defaultet auf Light (weiße Accordion-
        // Hintergründe). Unsere Sidebar ist dunkel — Theme umschalten.
        wgpui_kit::component::theme::Theme::change(ThemeMode::Dark, None, cx);

        cx.spawn(async move |cx| {
            let opts = WindowOptions::default();

            cx.open_window(opts, |window, cx| {
                let surface = window
                    .create_wgpu_surface(
                        SURFACE_WIDTH,
                        SURFACE_HEIGHT,
                        wgpu::TextureFormat::Rgba8UnormSrgb,
                    )
                    .expect("WgpuSurface not supported on this platform");

                let shared_state = Arc::new(Mutex::new(ApplicationState::default()));

                // 🟢 FIX (Refactoring): GPU-Init SYNCHRON im View-Lifecycle —
                // device()/queue() stehen sofort bereit, kein Sleep-Loop mehr.
                // Schlägt er fehl, beenden wir mit klarer Fehlermeldung
                // statt still schwarz einzufrieren.
                let pipeline = match GpuPipeline::new(&surface, SURFACE_WIDTH, SURFACE_HEIGHT) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("sdf_demo: GPU-Init fehlgeschlagen: {e}");
                        std::process::exit(1);
                    }
                };

                // FPS-Channel: Render-Thread → View-Overlay. Der Drop des
                // Receivers (mit dem Fenster) signalisiert der Loop das Ende.
                let (fps_tx, fps_rx) = std::sync::mpsc::channel::<f64>();

                // Render-Thread: besitzt die Pipeline exklusiv. Eigene
                // Surface-Referenz — `surface` selbst bleibt beim View.
                let render_state = shared_state.clone();
                let render_surface = surface.clone();
                thread::spawn(move || {
                    run_render_loop(
                        render_surface,
                        pipeline,
                        render_state,
                        InputManager::new(8.5), // Camera speed  //OLD mouse handling viy input.rs - now directly via wgpui in view.rs !
                        fps_tx,
                    );
                });

                cx.new(|cx| {
                    // 🟢 GUI: kompletten UI-Zustand bauen (Slider-Entities mit
                    // min/max/step/default aus den Original-Konstanten,
                    // Subscriptions Slider -> ApplicationState, Tab-/Accordion-
                    // Zustand). Alles Weitere lebt in src/gui.rs.
                    let gui = GuiState::new(cx, &shared_state);

                    SurfaceExample {
                        surface,
                        state: shared_state.clone(),
                        fps_rx,
                        display_fps: 0.0,
                        focus_handle: cx.focus_handle(),
                        gui,
                        // 🟢 INITIALISIERUNG FÜR DIE NEUEN STRUKTURFELDER:
                        last_mouse_x: 0.0,
                        last_mouse_y: 0.0,
                        mouse_sensitivity: 0.002,
                        show_gui:true,
                    }
                })
            })
            .expect("Failed to open window");
        })
        .detach();
    });
}

// Re-Export, damit gui.rs/input.rs ihre Pfade behalten (crate::SurfaceExample).
pub use view::SurfaceExample;
