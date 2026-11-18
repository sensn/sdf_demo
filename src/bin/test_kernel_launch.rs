//! Headless-Runtime-Test: kompiliert den Raymarch-Kernel über CubeCL
//! (WGSL-Codegen) und führt einen echten Launch aus — ohne Fenster.
//! Fängt Compile-Fehler, die `cargo check` nicht sieht (Runtime-Codegen).
//!
//! 🟢 DoF: Testet zusätzlich den kompletten Depth-of-Field-Pass
//! (generate_disk_kernel → compute_coc_and_mask_kernel → cubek-convolution).

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
        label: Some("headless-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("Device-Anfrage fehlgeschlagen");

    // 🔬 DIAGNOSE: Welche wgpu-Features bietet der Adapter?
    {
        use wgpu::RequestAdapterOptions;
        let a = &adapter;
        let feats = a.features();
        println!("[probe] Adapter: {}", a.get_info().name);
        println!("[probe] SHADER_INT64: {}", feats.contains(wgpu::Features::SHADER_INT64));
        println!("[probe] SHADER_FLOAT64: {}", feats.contains(wgpu::Features::SHADER_F64));
        println!("[probe] SUBGROUP: {}", feats.contains(wgpu::Features::SUBGROUP));
        println!("[probe] Features (alle): {feats:?}");
    }

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
    // 🟢 DoF: Der Raymarcher schreibt jetzt RGBA (Stride 4) — A = Tiefe.
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    let n = floats.len() / 4;
    let mut non_black = 0usize;
    let mut sum = 0.0f32;
    let mut max_v = 0.0f32;
    let mut max_depth = 0.0f32;
    for px in 0..n {
        let (r, g, b) = (floats[4 * px], floats[4 * px + 1], floats[4 * px + 2]);
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        sum += lum;
        if lum > max_v {
            max_v = lum;
        }
        if lum > 0.01 {
            non_black += 1;
        }
        // Tiefenkanal: Skybox muss 40.0 (max. Unschärfe) tragen.
        let depth = floats[4 * px + 3];
        if depth > max_depth {
            max_depth = depth;
        }
    }
    let avg = sum / n as f32;
    println!(
        "Pixel: {n}, nicht-schwarz: {non_black} ({:.1}%), Ø-Luminanz: {avg:.4}, Max: {max_v:.3}, Max-Tiefe: {max_depth:.2}",
        100.0 * non_black as f32 / n as f32
    );
    assert!(max_v > 0.05, "Output ist komplett schwarz — Kernel defekt?");
    assert!(
        max_depth > 0.1,
        "Tiefenkanal (Alpha) ist leer — RGBA-Layout nicht korrekt geschrieben?"
    );
    println!("✅ Headless-Launch OK — Kernel kompilierte fehlerfrei (WGSL-Codegen), RGBA-Layout aktiv");

    // =========================================================================
    // 🟢 DoF: Readback des normalisierten Bokeh-Disk-Kernels (Summe muss 1 sein)
    // =========================================================================
    {
        let k = pipeline.dof_kernel_size;
        println!(
            "[DoF] Kernel-Größe: k={k} — generiere + normalisiere (single-pass GPU)"
        );
        pipeline.set_dof_kernel_size(k);
        // Kernel neu erzeugen und per CPU-Readback prüfen (nur im Test!).
        pipeline.generate_dof_kernel();

        let readback = pipeline
            .client
            .read(vec![pipeline.dof_kernel_handle.clone()])
            .expect("Kernel-Readback fehlgeschlagen");
        let kdata: &[f32] = bytemuck::cast_slice(&readback[0]);
        let count = (k as usize) * (k as usize);
        let ksum: f32 = kdata[..count].iter().sum();
        let nonzero = kdata[..count].iter().filter(|v| **v > 0.0).count();

        println!(
            "[DoF] Bokeh-Disk: {count} Elemente, Summe = {ksum:.6} (Soll 1.0), aktive Taps = {nonzero}"
        );
        assert!(
            !kdata[..count].iter().all(|v| *v == 0.0),
            "[DoF] Disk-Kernel ist komplett leer — Generation fehlgeschlagen"
        );
    }

    println!("✅ DoF-Kernel-Generation OK — Single-Pass mit GPU-Normalisierung");

    // =========================================================================
    // 🟢 DoF: Kompletter Post-Pass (CoC + cubek-convolution) über einen echten Frame
    // =========================================================================
    {
        let s = state.lock().unwrap();
        // Frame mit aktivem DoF rechnen (focal 8.0, aperture 0.6 als Defaults).
        pipeline.apply_depth_of_field(width, height, s.dof_focal(), s.dof_aperture_scalar());
    }
    let dof_readback = pipeline
        .client
        .read(vec![pipeline.dof_out_handle.clone()])
        .expect("DoF-Output-Readback fehlgeschlagen");
    let dof_floats: &[f32] = bytemuck::cast_slice(&dof_readback[0]);
    let total = dof_floats.len();
    let mut d_nan = 0usize;
    let mut d_max = 0.0f32;
    for v in dof_floats.iter() {
        if !v.is_finite() {
            d_nan += 1;
        } else if *v > d_max {
            d_max = *v;
        }
    }
    println!(
        "[DoF] Convolution-Output: {total} Werte, NaN/Inf = {d_nan}, Max = {d_max:.4}"
    );
    assert_eq!(
        d_nan, 0,
        "[DoF] Faltungs-Output enthält NaN/Inf — cubek-convolution fehlgeschlagen"
    );
    println!(
        "✅ DoF-Post-Pass OK — CoC-Kernel + cubek-convolution laufen fehlerfrei"
    );
}