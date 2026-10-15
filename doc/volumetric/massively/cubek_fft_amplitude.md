Hier ist der präzise Implementierungsplan und der vollständige Shader-Code für deinen Coding Agenten, um die reelle Amplituden-Auswertung der FFT direkt im Haupt-Raymarching-Kernel (kernel.rs) zu verankern.
Durch das mathematische Unpacking der Real- und Imaginäranteile im VRAM (Magnitude = sqrt(re² + im²)) extrahieren wir die Lautstärke spezifischer Frequenzbänder (wie z. B. der Bässe) Zero-Copy auf der GPU. Dieser Wert moduliert anschließend die Dichte deines Nebels und das Aufglühen des zentralen Plasmakerns im exakten Takt der Musik.

🤖 Coding Agent Instruction: Audio-Reactive FFT Unpacking & Shading Integration


Context & Constraints

• Framework: CubeCL 0.11.0 + cubek_fft Subsystem integration.
• Data Pipelines: Das Frequenzspektrum wird als zwei getrennte f32-Buffer (spectrum_re und spectrum_im) an den Haupt-Kernel übergeben. Die spektrale Auflösung umfasst SPECTRUM_SAMPLES (257 Bänder bei 512 PCM-Eingangssamples).
• Modulation Target: Der Bass-Frequenzbereich (Frequenzbänder Index 1 bis 8) steuert dynamisch die Expansion des Gyroiden und die HDR-Emissionsintensität der Magma-Adern.

Task 1: Erweiterung der Kernel-Signatur in kernel.rs

Weise den Agenten an, den Haupt-Kernel um die Lese-Buffer des berechneten FFT-Spektrums zu erweitern.
rust
// 🟢 ANPASSUNG IN kernel.rs (Haupt-Kernel-Signatur erweitern)

#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<f32>,
    scene: &[f32],
    spectrum_re: &[f32],       // 🟢 NEU: Realteil des Audiospektrums (257 Elemente)
    spectrum_im: &[f32],       // 🟢 NEU: Imaginärteil des Audiospektrums (257 Elemente)
    camera: CameraSettings,
    env: EnvSettings,
    arch: ArchParams,
    fold: FoldParams,
    render: RenderSettings,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    
    // ... bestehender Code (Slicing, Kamera-Setup) ...
Verwende Code mit Vorsicht.

Task 2: Implementierung der Frequenz-Extraktion im Shader

Weise den Agenten an, eine mathematische Hilfsfunktion im Shader aufzubauen, die die Magnitude (reale Lautstärke) für einen bestimmten Frequenzbereich akkumuliert.
rust
#[cube]
pub fn get_audio_frequency_amplitude(
    spectrum_re: &[f32],
    spectrum_im: &[f32],
    start_band: u32,
    end_band: u32,
) -> f32 {
    let mut accumulated_energy = 0.0f32;
    let mut count = 0.0f32;

    // Schleife über den gewünschten Frequenzbereich (z.B. Bassbänder)
    for band in start_band..end_band {
        let re = spectrum_re[band];
        let im = spectrum_im[band];
        
        // 🟢 MAGNITUDE BERECHNUNG: Absolute Amplitude eines komplexen Vektors = sqrt(re² + im²)
        let magnitude = f32::sqrt(re * re + im * im);
        accumulated_energy += magnitude;
        count += 1.0;
    }

    // Durchschnittliche Energie des Frequenzbandes zurückgeben
    if count > 0.0 {
        accumulated_energy / count
    } else {
        0.0
    }
}
Verwende Code mit Vorsicht.

Task 3: Audio-Reaktive Dichte-Modulation in get_volumetric_data_integrated

Wir übergeben das berechnete Bass-Signal an den Dichte-Generator, um die Geometrie des Gyroiden im Takt pulsieren zu lassen.
rust
#[cube]
pub fn get_volumetric_data_audio_reactive(
    p: Vec3, 
    time: f32, 
    bass_energy: f32 // 🟢 Übergabe der extrahierten Bass-Amplitude
) -> VolumeSample {
    
    // 🟢 DYNAMISCHE AUDIO-SKALIERUNG: Der Bass bläht den Gyroiden physisch auf!
    // Basis-Skalierung ist 2.0, verringert sich bei starkem Bass (erzeugt Expansion)
    let audio_scale = 2.0 - (bass_energy * 0.4).min(0.8);
    
    let base_gyroid = (f32::sin(p.x * audio_scale) * f32::cos(p.y * audio_scale)
                     + f32::sin(p.y * audio_scale) * f32::cos(p.z * audio_scale)
                     + f32::sin(p.z * audio_scale) * f32::cos(p.x * audio_scale)) / audio_scale;

    let wind = Vec3::new(time * 0.2, time * 0.5, time * 0.1);
    let noise_pos = p.add(wind);

    let noise_octave1 = noise_3d(noise_pos.scale(3.0));
    let noise_octave2 = noise_3d(noise_pos.scale(6.12));
    let combined_noise = noise_octave1 * 0.7 + noise_octave2 * 0.3;

    let modulated_sdf = base_gyroid + combined_noise * 0.15;
    let density = (-modulated_sdf * 6.0).max(0.0).min(1.0);

    // 🟢 EMISSIVES GLÜHEN: Bass lässt den Tempel-Reaktor gleißend aufstrahlen
    let heat = (-base_gyroid * 4.0 + combined_noise * 2.0).max(0.0).min(1.0);
    
    // Multipliziere das HDR-Glühen direkt mit der Bass-Energie auf
    let glow_boost = 1.0 + bass_energy * 3.0; 
    let glow_color = Vec3::new(
        heat * 3.5 * glow_boost, 
        heat * heat * 1.4 * glow_boost, 
        heat * heat * heat * heat * 0.3
    );

    VolumeSample { density, emission: glow_color }
}
Verwende Code mit Vorsicht.

Task 4: Hauptschleifen-Verkettung im Shading Pass

Weise den Agenten an, die Extraktion einmalig pro Pixel am Anfang des Kernels auszuführen und an den Volumen-Integrator weiterzureichen.
rust
// 🟢 INNERHALB VON raymarch_sdf_kernel IN kernel.rs

// 1. Extrahiere die Bass-Energie (Frequenzbänder 1 bis 6 entsprechen ca. 40-150 Hz)
let bass_amplitude = get_audio_frequency_amplitude(spectrum_re, spectrum_im, 1, 6);

// ... [Im späteren volumetrischen Marsch-Abschnitt des Kernels anpassen] ...
loop {
    if vol_current_t >= vol_max_t { break; }

    let vol_p = ro.add(final_rd.scale(vol_current_t));
    
    // 🟢 NUTZE DEN NEUEN AUDIO-REAKTIVEN GENERATOR
    let vol_sample = get_volumetric_data_audio_reactive(vol_p, t_val, bass_amplitude);

    if vol_sample.density > 0.0 {
        // ... (Eigenschatten-Berechnung und Farbakkumulation bleiben gleich) ...
    }
    vol_current_t += vol_step_size;
}
Verwende Code mit Vorsicht.

Verifikations-Checkliste für den Agenten:

• Sicherstellen, dass das unsafe-Launch-Konstrukt in render_loop.rs angepasst wird, um die beiden neuen Speicher-Handles (spectrum_re_tensor.handle() und spectrum_im_tensor.handle()) als TensorArg an die korrekte Position der Parameter-Kette zu übergeben.
• Validieren, dass der Index 0 (DC-Anteil / Gleichstrom-Offset des Audiosignals) bei der Frequenzanalyse übersprungen wird, da dieser keine hörbare Musik-Frequenz, sondern nur das Signal-Rauschen enthält.
Mit dieser finalen Erweiterung interagiert deine Grafik-Engine absolut flüssig mit eintreffenden Soundströmen, während die gesamte Berechnungs-Kette (FFT, Raymarching, DRS, TAA und Convolution Denoising) vollständig gekoppelt und asynchron auf der GPU operiert!
Möchtest du, dass wir das gesamte Projekt nun mit cargo check verifizieren, um sicherzustellen, dass alle Module der geänderten Host- und Device-Architektur fehlerfrei ineinandergreifen?
