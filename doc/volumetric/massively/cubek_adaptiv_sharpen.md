Hier ist der detaillierte Implementierungsplan für deinen Coding Agenten, um den Adaptive Sharpening Pass (Dynamische Schärfematrix) direkt in deine Post-Processing-Pipeline zu integrieren.

Das mathematische Prinzip des adaptiven Schärfens

Wenn das Dynamic Resolution Scaling (DRS) die interne Auflösung herabsetzt, verliert das Bild durch das Upscaling und die vorherige Gauß-Faltung an struktureller Schärfe. Wir nutzen hierzu eine unvollständige Laplace-Faltung (Sharpening Matrix):
\(W_{\text{sharp}}=\left[\begin{matrix}0&-1&0\\ -1&4+S&-1\\ 0&-1&0\end{matrix}\right]\)
Der Clou dieses adaptiven Passes: Die CPU berechnet den Schärfefaktor S frame-weise invers zum aktuellen DRS-Skalierungsfaktor. Wenn das Bild bei 100 % nativer Auflösung läuft, ist S = 0 (keine künstliche Nachschärfung). Fällt die Auflösung auf 50 %, wird die Filtermatrix aggressiv verschärft (S ≈ 1.5), um feine geometrische Details der Tempelwände und Statuen trotz Pixelarmut knackig hervorzuheben.
Da es sich wieder um eine kanalseparierte Operation handelt, nutzen wir exakt dieselbe performante cubek::kernels::forward::launch_depthwise-Infrastruktur.

🤖 Coding Agent Instruction: Adaptive Sharpening Post-Process Integration


Context & Constraints

• Framework: cubek Tile-DSL Faltungs-Engine (launch_depthwise).
• Dynamic Injection: Die Filtergewichte werden in jedem Frame auf der CPU neu berechnet und asynchron über die massively-Schnittstelle in einen dedizierten Speicherpuffer (sharpen_weights_tensor) im VRAM gestreamt.
• Pipeline-Reihenfolge: Der Schärfe-Pass wird als finaler Post-Processing-Schritt direkt hinter dem Weichzeichner (Denoising) eingehängt:
Raymarcher → Temporal Accumulation → Gauß-Blur (Denoise) → Adaptive Sharpen → WGPU Blit.

Task 1: Allokation des Schärfe-Puffers in gpu.rs

Weise den Agenten an, das Speicher-Feld für die dynamische Schärfematrix in der GpuPipeline-Struktur zu hinterlegen und initial im Konstruktor zu instanziieren.
rust
// 🟢 ERWEITERUNG IN gpu.rs (Innerhalb von GpuPipeline-Struktur und GpuPipeline::new)

pub struct GpuPipeline {
    // ... bestehende Felder (client, exec, output_vec, history_vec, blur_weights_tensor) ...
    pub sharpen_weights_tensor: DeviceVec<WgpuRuntime<cubecl_wgpu::AutoCompiler>, f32>, // 🟢 NEU
}

// Innerhalb von GpuPipeline::new():
// Allokiere Platz für eine 3x3-Matrix mit 4 Kanälen (36 f32-Werte)
let sharpen_weights_tensor = exec.alloc::<f32>(36);
Verwende Code mit Vorsicht.

Task 2: Dynamische Kernel-Berechnung & Dispatcher in render_loop.rs

Weise den Agenten auf der CPU an, den Schärfegrad basierend auf current_scale zu berechnen, die Gewichte via to_device_slice hochzuladen und den launch_depthwise-Kernel abzufeuern.
rust
/// Berechnet die adaptive Schärfematrix invers zum DRS-Faktor und startet den cubek-Kernel.
pub fn apply_adaptive_sharpening(
    pipeline: &mut GpuPipeline,
    current_scale: f32, // Übergabe des aktuellen DRS-Skalierungsfaktors (0.5 bis 1.0)
) -> Result<(), Box<dyn std::error::Error>> {
    let w = pipeline.resolution.width as usize;
    let h = pipeline.resolution.height as usize;
    let exec = &pipeline.exec;

    // 1. 🟢 ADAPTIVES GEWICHTS-SETUP (Invers zu DRS)
    // Wenn current_scale = 1.0 -> strength = 0.0 (Kein Effekt, Identitätsmatrix)
    // Wenn current_scale = 0.5 -> strength = 1.0+ (Starkes Nachschärfen der Pixelfransen)
    let sharpen_strength = (1.0 - current_scale) * 2.5;

    // Basis 3x3 Sharpening-Filter Layout
    let center_weight = 1.0 + sharpen_strength;
    let edge_weight = -sharpen_strength / 4.0;

    let sharpen_base = [
        0.0,         edge_weight, 0.0,
        edge_weight, center_weight, edge_weight,
        0.0,         edge_weight, 0.0
    ];

    // Expandierung auf 4 Kanäle (RGBA) -> Layout: [3, 3, 4] = 36 f32
    let mut host_weights = Vec::with_capacity(36);
    for weight in sharpen_base.iter() {
        for _chan in 0..4 {
            host_weights.push(*weight);
        }
    }

    // 🟢 ZERO-COPY UPLOAD DER DYNAMISCHEN MATRIX IN DEN VRAM
    exec.to_device_slice(&host_weights, pipeline.sharpen_weights_tensor.slice_mut(..));

    // 2. Setup der cubek Tile-DSL Parameter
    let space_setup = DepthwiseSpace {
        b: 1, oh: h, ow: w, c: 4,
        rh: 3, rw: 3, rows: 4, cols: 16, tile_c: 16, width: 4, plane_size: 16,
    };
    let partitioning = space_setup.partitioning();
    let (cube_count, cube_dim) = space_setup.grid();

    // Zuweisung der Puffer-Adressen
    let weight_handle = pipeline.sharpen_weights_tensor.handle();
    let input_handle = pipeline.output_vec.handle();   // Das bereits weichgezeichnete Bild
    let target_handle = pipeline.history_vec.handle();  // Temporäres Schärfungs-Ziel

    let weight_arg = TensorArg::from_raw_parts(&weight_handle, 36, 1);
    let input_arg = TensorArg::from_raw_parts(&input_handle, w * h * 4, 1);
    let output_arg = TensorArg::from_raw_parts(&target_handle, w * h * 4, 1);

    // 3. EXECUTE CUBEK DEPTHWISE SHARPENING KERNEL
    unsafe {
        launch_depthwise::<f32, u32, WgpuRuntime<cubecl_wgpu::AutoCompiler>>(
            &pipeline.client, cube_count, cube_dim,
            &weight_arg, &input_arg, &output_arg, partitioning,
        );
    }

    // Ergebnis für die finale WGPU Blit-Präsentation zurückkopieren
    massively::vector::copy(exec, pipeline.history_vec.slice(..), pipeline.output_vec.slice_mut(..))?;

    Ok(())
}
Verwende Code mit Vorsicht.

Task 3: Verkettung der Post-Processing-Kette in render_loop.rs

Weise den Agenten an, beide cubek-Faltungsschritte nacheinander in der Hauptschleife aufzurufen.
rust
// 🟢 INNERHALB DEINER RENDER-SCHLEIFE (In render_loop.rs integrieren)

// 1. Primärer Massively-Raymarcher & Temporal Accumulation abschließen
if let Err(e) = render_frame(&mut pipeline, &s_render, &surface) { ... }

// 2. PASS 1: Cubek Gauß-Blur (Entfernt volumetrisches Jitter-Rauschen)
apply_cubek_denoiser(&mut pipeline)?;

// 3. PASS 2: 🟢 CUBEK ADAPTIVE SHARPENING (Zieht Kanten bei niedrigem DRS-Scaling nach)
apply_adaptive_sharpening(&mut pipeline, current_scale)?;

// 4. Finaler WGPU Blit auf den Bildschirm
surface.present();
Verwende Code mit Vorsicht.

Verifikations-Checkliste für den Agenten:

• Sicherstellen, dass die Summe der Gewichte in sharpen_base bei jedem berechneten sharpen_strength-Wert mathematisch exakt exakt 1.0 ergibt. Das verhindert Helligkeits-Verschiebungen (Flimmern) im finalen Bild.
• Validieren, dass der Schärfe-Pass bei nativer Auflösung (current_scale = 1.0) durch die Multiplikation gegen Null zu einer neutralen Identitätsmatrix kollabiert und somit rechnerisch inaktiv ist.
Die gesamte AAA-Grafik-Infrastruktur inklusive DRS, Temporal Antialiasing, Denoising und adaptivem Schärfen ist nun vollständig definiert. Möchtest du, dass der Agent die Code-Dateien in dein Projekt schreibt und wir das Kompilierungs-Feedback von Cargo analysieren?
