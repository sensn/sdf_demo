//! Headless-Runtime-Test: kompiliert den Raymarch-Kernel über CubeCL
//! (WGSL-Codegen) und führt einen echten Launch aus — ohne Fenster.
//! Fängt Compile-Fehler, die `cargo check` nicht sieht (Runtime-Codegen).

#[path = "../app_state.rs"]
mod app_state;
#[path = "../gpu.rs"]
mod gpu;
#[path = "../kernel.rs"]
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
        label: Some("headless-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("Device-Anfrage fehlgeschlagen");

    // Pipeline ohne Surface initialisieren — kompiliert den Kernel.
    let width = 64u32;
    let height = 64u32;
    let mut pipeline =
        GpuPipeline::new_headless(&device, &queue, width, height).expect("Pipeline-Init fehlgeschlagen");

    let state = Arc::new(Mutex::new(ApplicationState::default()));
    {
        let s = state.lock().unwrap();
        // Scene-Buffer (inkl. SDF-Grid) schreiben wie im Render-Loop.
        pipeline.write_state_buffers(&s);
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
    assert!(max_v > 0.05, "Output ist komplett schwarz — Kernel defekt?");
    println!("✅ Headless-Launch OK — Kernel kompilierte fehlerfrei (WGSL-Codegen)");
}
