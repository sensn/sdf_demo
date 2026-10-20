Hier ist der strukturierte Implementation Plan zur Integration der volumetrischen Features basierend auf der CubeCL 0.11.0 Spezifikation. [1] (https://github.com/tracel-ai/cubecl/releases)
In CubeCL 0.11.0 werden globale Thread-IDs über topologische built-in Konstanten wie ABSOLUTE_POS_X statt älterer Launch-Makros angesprochen. Schleifen über dynamische Arrays nutzen dedizierte Lese-Muster, um optimale Kompilierung für CUDA, Metal und Vulkan zu garantieren. [1] (https://crates.io/crates/cubecl-ir/range/=0.11.0-pre.4)

Implementation Plan: Volumetric Integration Stack


Phase 1: Datentypen & Uniform-Strukturen (Host/Device Bridge)

• Ziel: Sicherstellung des korrekten Byte-Alignments und der Speicher-Layouts für das CubeLaunch-Trait zur fehlerfreien CPU-GPU Serialisierung.
• Schritte:
	1. Erstellung der PBR-Erweiterung für Volumen-Strukturen (VolumeSample).
	2. Einbetten expliziter Padding-Felder in Vektoren (GpuVec3), um die 16-Byte-Ausrichtung nativer GPU-Uniform-Buffer einzuhalten.
rust
use cubecl::prelude::*;

#[derive(CubeType, Copy, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(CubeType, Copy, Clone)]
pub struct VolumeSample {
    pub density: f32,
    pub emission: Vec3,
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32, // Strenges 16-Byte Alignment für Uniforms
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct VolumetricRenderSettings {
    pub time: f32,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub fog_density: f32,
    pub anisotropy_g: f32,
}
Verwende Code mit Vorsicht.

Phase 2: Prozedurale Register-Generatoren (Zero-Copy Noise)

• Ziel: Implementierung des mathematischen 3D-Rauschens direkt auf Rechenwerkebene (ALU) ohne Speicher-Lookups. [1] (https://docs.rs/crate/cubecl/latest)
• Schritte:
	1. Mathematischen 3D-Hash auf Basis von Primzahl-Skalarprodukten aufbauen.
	2. Trilineare Hermite-Interpolation (Smoothstep) für kontinuierliche Übergänge implementieren.
rust
#[cube]
pub fn hash_3d(p: Vec3) -> f32 {
    let dot_product = p.x * 127.1 + p.y * 311.7 + p.z * 74.7;
    let sine_val = f32::sin(dot_product);
    sine_val * 43758.5453 - f32::floor(sine_val * 43758.5453)
}

#[cube]
pub fn noise_3d(p: Vec3) -> f32 {
    let i = Vec3 { x: f32::floor(p.x), y: f32::floor(p.y), z: f32::floor(p.z) };
    let f = Vec3 { x: p.x - f32::floor(p.x), y: p.y - f32::floor(p.y), z: p.z - f32::floor(p.z) };

    // Cubic Hermite Interpolation (Smoothstep)
    let u = Vec3 {
        x: f.x * f.x * (3.0 - 2.0 * f.x),
        y: f.y * f.y * (3.0 - 2.0 * f.y),
        z: f.z * f.z * (3.0 - 2.0 * f.z),
    };

    let i000 = hash_3d(i);
    let i100 = hash_3d(Vec3 { x: i.x + 1.0, y: i.y, z: i.z });
    let i010 = hash_3d(Vec3 { x: i.x, y: i.y + 1.0, z: i.z });
    let i110 = hash_3d(Vec3 { x: i.x + 1.0, y: i.y + 1.0, z: i.z });
    let i001 = hash_3d(Vec3 { x: i.x, y: i.y, z: i.z + 1.0 });
    let i101 = hash_3d(Vec3 { x: i.x + 1.0, y: i.y, z: i.z + 1.0 });
    let i011 = hash_3d(Vec3 { x: i.x, y: i.y + 1.0, z: i.z + 1.0 });
    let i111 = hash_3d(Vec3 { x: i.x + 1.0, y: i.y + 1.0, z: i.z + 1.0 });

    let x00 = i000 + u.x * (i100 - i000);
    let x10 = i010 + u.x * (i110 - i010);
    let x01 = i001 + u.x * (i101 - i001);
    let x11 = i011 + u.x * (i111 - i011);

    let y0 = x00 + u.y * (x10 - x00);
    let y1 = x01 + u.y * (x11 - x01);

    y0 + u.z * (y1 - y0)
}
Verwende Code mit Vorsicht.

Phase 3: Volumetrische Physik (Streuung & Schatten)

• Ziel: Mathematische Integration der Lichtabsorption (Beer-Lambert) und richtungsabhängiger Streuung (Henyey-Greenstein). [1] (https://shaderbits.com/blog/creating-volumetric-ray-marcher)
• Schritte:
	1. Phasefunktion berechnen, um gleißendes Gegenlicht (Forward Scattering) zu ermöglichen.
	2. Sekundäre Shadow-Märsche mit minimaler Schrittanzahl zur Performance-Schonung aufbauen. [1] (https://shaderbits.com/blog/creating-volumetric-ray-marcher)
rust
#[cube]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let pi = 3.14159265;
    let numerator = 1.0 - g2;
    let denom_inner = 1.0 + g2 - 2.0 * g * cos_theta;
    let denominator = 4.0 * pi * denom_inner * f32::sqrt(denom_inner);
    numerator / denominator
}
Verwende Code mit Vorsicht.

Phase 4: Der Integrierte Integrations-Loop (Raymarch-Engine)

• Ziel: Zusammenführung von Dithering, In-Scattering, Emissionen und der kritischen Early-Ray-Termination Optimierung.
• Schritte:
	1. Ray-Origin per Pseudo-Zufall vor-jitten (Rauschunterdrückung).
	2. Strahl akkumulieren und bei Unterschreitung der Sichtbarkeit (transmittance < 0.001) via break abbrechen. [1] (https://shaderbits.com/blog/creating-volumetric-ray-marcher)
rust
#[cube]
pub fn noise_2d_time(x: u32, y: u32, time: f32) -> f32 {
    let dot_product = f32::cast(x) * 12.9898 + f32::cast(y) * 78.233 + time * 43758.5453;
    let sine_val = f32::sin(dot_product);
    sine_val * 43758.5453 - f32::floor(sine_val * 43758.5453)
}

// Haupt-Integrationslogik (wird aus dem Launch-Kernel gerufen)
Verwende Code mit Vorsicht.

Phase 5: CubeCL 0.11.0 Launch-Kernel & Hardware-Binding

• Ziel: Anbindung an das physische GPU-Grid über topologische 0.11.0 built-in Variablen. [1] (https://crates.io/crates/cubecl-ir/range/=0.11.0-pre.4)
• Schritte:
	1. Nutzung von ABSOLUTE_POS_X und ABSOLUTE_POS_Y anstelle alter Makros.
	2. Transformation in den normalisierten UV-Bildschirmraum. [1] (https://crates.io/crates/cubecl-ir/range/=0.11.0-pre.4)
rust
#[cube(launch)]
pub fn render_volumetric_kernel(
    output_image: &mut Tensor<f32>,
    settings: VolumetricRenderSettings,
    light_pos: GpuVec3,
    light_color: GpuVec3,
) {
    // 🟢 CubeCL 0.11.0 Konforme Hardware-Register Abfragen
    let pixel_x = ABSOLUTE_POS_X;
    let pixel_y = ABSOLUTE_POS_Y;

    // Out-of-bounds Sicherheitszaun
    if pixel_x >= settings.width || pixel_y >= settings.height {
        return;
    }

    let fx = f32::cast(pixel_x);
    let fy = f32::cast(pixel_y);
    let width_f = f32::cast(settings.width);
    let height_f = f32::cast(settings.height);

    // Normalisierter UV-Raum Setup
    let uv_x = (fx / width_f) * 2.0 - 1.0;
    let uv_y = (fy / height_f) * 2.0 - 1.0;
    
    // Hier folgt der integrierte Raymarcher-Aufruf und das Schreiben in den Tensor
}
Verwende Code mit Vorsicht.
Soll ich dir zeigen, wie du für diesen Plan die passenden Cargo.toml Abhängigkeiten und das Host-Setup (Wgpu oder CUDA Runtime) in CubeCL 0.11.0 aufsetzt? [1] (https://burn.dev/books/cubecl/getting-started/installation.html), [2] (https://crates.io/crates/cubecl-ir/range/=0.11.0-pre.4)
