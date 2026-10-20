//! Headless-Volumetric-Test (TR.md): kompiliert den volumetrischen Kernel über
//! CubeCL (WGSL-Codegen) und führt einen echten Launch aus — ohne Fenster.
//!
//! Prüft den kompletten TR-Pfad: Gyroid-Dichtefeld, FBM-Wind, Beer-Lambert-
//! Transmittanz, Henyey-Greenstein-In-Scattering, Schattenstrahl, Temporal
//! Dithering, HDR-Emission und Early-Ray-Termination. Fängt Compile-Fehler,
//! die `cargo check` nicht sieht (Runtime-Codegen von CubeCL).

#[path = "../app_state.rs"]
mod app_state;
#[path = "../gpu.rs"]
mod gpu;
#[path = "../kernel/mod.rs"]
mod kernel;
#[path = "../sdf_preprocessor.rs"]
mod sdf_preprocessor;

use app_state::ApplicationState;
use gpu::GpuPipeline;
use std::sync::{Arc, Mutex};

fn main() {
    // Headless-Instanz (kein Surface) — Adapter/Device wie im View.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("Kein GPU-Adapter (headless)");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("headless-volumetric-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("Device-Anfrage fehlgeschlagen");

    let width = 64u32;
    let height = 64u32;
    let mut pipeline =
        GpuPipeline::new_headless(&device, &queue, width, height).expect("Pipeline-Init fehlgeschlagen");

    let state = Arc::new(Mutex::new(ApplicationState::default()));
    {
        let s = state.lock().unwrap();
        // render_mode = 1: volumetrischer Pfad (Default der ApplicationState).
        assert_eq!(s.render_mode, 1, "Volumetric Mode muss Default sein");
        // 🌌 Zero-Copy: KEIN write_state_buffers — alle Raumdaten entstehen
        // prozedural in den ALU-Registern (TR §1, §4.1).
        // Ein Kernel-Launch — triggert den CubeCL-Kernel-Compile (WGSL).
        pipeline.launch_kernel(width, height, &s, 0.0);
    }
    // Readback: Output-Buffer lesen und Pixel-Statistik prüfen.
    let pixels = pipeline
        .client
        .read(vec![pipeline.output_handle.clone()])
        .expect("Output-Readback fehlgeschlagen");
    let bytes = &pixels[0];
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    let n = floats.len() / 3;
    let mut non_black = 0usize;
    let mut sum = 0.0f32;
    let mut max_v = 0.0f32;
    for px in 0..n {
        let (r, g, b) = (floats[3 * px], floats[3 * px + 1], floats[3 * px + 2]);
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        sum += lum;
        if lum > max_v {
            max_v = lum;
        }
        if lum > 0.01 {
            non_black += 1;
        }
    }
    let avg = sum / n as f32;
    println!(
        "Pixel: {n}, nicht-schwarz: {non_black} ({:.1}%), Ø-Luminanz: {avg:.4}, Max: {max_v:.3}",
        100.0 * non_black as f32 / n as f32
    );
    // Das Volumen muss sichtbar sein: Dichtefeld + Emission + Hintergrund
    // dürfen nicht alles schwarz liefern.
    assert!(max_v > 0.05, "Output komplett schwarz — Volume-Kernel defekt?");
    println!("✅ Volumetric Headless-Launch OK — Zero-Copy-Kernel kompilierte fehlerfrei (WGSL-Codegen)");
}
