//! `kernel/types.rs` — Basis-Typen & Helfer: Vec3/Vec2/SdfResult/GpuVec3, smin, smin_material, Launch-Structs.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct SdfResult<F: Float> {

   pub d: F,
    pub r: F,
    pub g: F,
    pub b: F,
    pub roughness: F,
    pub metallic: F,
    pub emissive: F,
    pub specular: F,
}

#[cube]
pub fn smin_material<F: Float>(a: SdfResult<F>, b: SdfResult<F>, k: F) -> SdfResult<F> {
    let h = (k - (a.d - b.d).abs()).max(F::new(0.0)) / k;

    // Berechne die interpolierte Distanz
    let mixed_d = a.d.min(b.d) - h * h * k * F::new(0.25);

    // Lineare Farbmischung basierend auf der Nähe zum jeweiligen Objekt
    // Wenn h nah an 1 ist, befinden wir uns in der Übergangszone
    let mix_factor = (F::new(0.5) + F::new(0.5) * (b.d - a.d) / k)
        .min(F::new(1.0))
        .max(F::new(0.0));

    let mixed_r = a.r + mix_factor * (b.r - a.r);
    let mixed_g = a.g + mix_factor * (b.g - a.g);
    let mixed_b = a.b + mix_factor * (b.b - a.b);
    
    // 🟢 NEU: Weiches Überblenden der PBR-Materialien in der Mischzone!
    let mixed_rough = a.roughness + mix_factor * (b.roughness - a.roughness);
    let mixed_metal = a.metallic + mix_factor * (b.metallic - a.metallic);
    let mixed_emiss = a.emissive + mix_factor * (b.emissive - a.emissive);
    let mixed_spec = a.specular + mix_factor * (b.specular - a.specular);

    SdfResult::<F> {
        d: mixed_d,
        r: mixed_r,
        g: mixed_g,
        b: mixed_b,
        roughness: mixed_rough,
        metallic: mixed_metal,
        emissive: mixed_emiss,
        specular: mixed_spec,
    }
}

// 1. Zuerst die normalen Rust-Derives
#[derive(CubeType, Copy, Clone)]
#[cube(derive(Copy, Clone))]
pub struct Vec3<F: Float> {
    pub x: F,
    pub y: F,
    pub z: F,
}

#[cube]
impl<F: Float> Vec3<F> {
    pub fn new(x: F, y: F, z: F) -> Self {
        Vec3::<F> { x, y, z }
    }
    
    pub fn add(self, other: Vec3<F>) -> Vec3<F> {
        Vec3::<F>::new(self.x + other.x, self.y + other.y, self.z + other.z) // 💡 KORREKTUR: ::<F> hinzugefügt
    }
    
    pub fn sub(self, other: Vec3<F>) -> Vec3<F> {
        Vec3::<F>::new(self.x - other.x, self.y - other.y, self.z - other.z) // 💡 KORREKTUR: ::<F> hinzugefügt
    }
    
    pub fn scale(self, factor: F) -> Vec3<F> {
        Vec3::<F>::new(self.x * factor, self.y * factor, self.z * factor)    // 💡 KORREKTUR: ::<F> hinzugefügt
    }
    
    pub fn dot(self, other: Vec3<F>) -> F {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    
    pub fn length(self) -> F {
        // 💡 KORREKTUR: .sqrt() muss als Methode auf dem Wert aufgerufen werden!
        self.dot(self).sqrt() 
    }
    
    pub fn normalize(self) -> Vec3<F> {
        let len = self.length();
        let mut x = self.x;
        let mut y = self.y;
        let mut z = self.z;
        if len > F::new(0.0) {
            let factor = F::new(1.0) / len;
            x = self.x * factor;
            y = self.y * factor;
            z = self.z * factor;
        }
        Vec3::<F>::new(x, y, z) // 💡 KORREKTUR: ::<F> hinzugefügt
    }
}

#[cube]
pub fn smin<F: Float>(a: F, b: F, k: F) -> F {
    let h = (k - (a - b).abs()).max(F::new(0.0)) / k;
    a.min(b) - h * h * k * F::new(0.25)
}

#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct Vec2<F: Float> {
    pub d: F,
    pub mat_id: F,
}
impl<F: Float> Vec2<F> {
    pub fn new(d: F, mat_id: F) -> Self {
        Self { d, mat_id }
    }
}

/// 16-Byte-Alignment fuer vec3-Parameter (params5): vec3 + explizites Pad.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct GpuVec3<F: Float> {
    pub x: F,
    pub y: F,
    pub z: F,
    pub _pad: F,
}

/// Kamera-Parameter (params5: "camera settings struct").
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct CameraSettings<F: Float> {
    pub position: GpuVec3<F>,
    pub yaw: F,
    pub pitch: F,
}

/// Umwelt-Parameter (ersetzt Tensor 4 / env_data()[16]).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct EnvSettings<F: Float> {
    pub key_light_pos: GpuVec3<F>, // war env[0..2]
    pub light_intensity: F,   // war env[3]
    pub key_color: GpuVec3<F>,     // war env[4..6]
    pub ambient_strength: F,  // war env[7]
    pub bg_color: GpuVec3<F>,      // war env[8..10]
    // 🟢 NEU: Dedizierte Nebelfarbe für physikalisch korrektes Blending
    pub fog_color: GpuVec3<F>,  
    pub fog_density: F,       // war env[11]
    pub fog_enabled: u32,       // war env[12] (f32) -> params3: u32-Flag 0/1
    // 🟢 PARAM-GUI: Atmosphäre & Lighting
    pub volumetric_fog_steps: u32, // Nebel-Integrationsschritte (4..64, default 16)
    pub fog_density_falloff: F,  // Höhenprofil-Falloff (0.01..0.5, default 0.08)
     // 🟢 NEU: Ausgelagerte Nebel-Parameter (Fehler behoben & erweitert)
    pub min_height_falloff: F,    // Minimale Nebeldichte in der Höhe (0.0..1.0, default 0.1)
    pub max_optical_depth: F,     // Grenzwert für Opazitäts-Abbruch (1.0..10.0, default 4.605 für ~1% Sicht)
    pub fog_sample_offset: F,     // Sampling-Position im Strahlsegment (0.0..1.0, default 0.5 = Mitte)
    pub ao_iterations: u32,        // AO-Sampling-Iterationen (1..10, default 5)
    pub shadow_softness_k: F,     // Weichheits-Konstante (1.0..64.0, default 8.0)
}


/// Tempel-Architektur (ersetzt Tensor 5 / arch_data()[12]).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct ArchParams<F: Float> {
    pub pillar_dist: F,   // war arch[0]
    pub pillar_thick: F,  // war arch[1]
    pub room_height: F,   // war arch[2]
    pub ceiling_thick: F, // war arch[3]
    pub arch_radius: F,   // war arch[4]
    pub arch_height: F,   // war arch[5]
    pub decor_freq: F,    // war arch[6]
    pub decor_depth: F,   // war arch[7]
    pub decor_thick: F,   // war arch[8]
}

/// Unendliche Raumfaltung (ersetzt Tensor 6 / fold_data()[4]).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct FoldParams<F: Float> {
    pub cell_size: F,  // war fold[0]
    pub half_cell: F,  // war fold[1]
    pub fold_speed: F, // war fold[2]
    // 🟢 PARAM-GUI: Infinite Repetition (Pfeiler-Faltung)
    pub pillar_spacing: F, // XZ-Wiederholungs-Abstand (2.0..20.0, default 6.0)
    pub pillar_height: F,  // Pfeilerhöhe (1.0..10.0, default 4.0)
}

/// Frame-Parameter & Mode-Flags (ersetzt Meta-Tensor + 12 lose Skalare).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct RenderSettings<F: Float> {
    pub time: F,
    pub width: u32,
    pub height: u32,
    pub blend_factor: F,
    pub active_slots: u32, // ersetzt meta[0] (f32) -> params3: Count als u32
    pub shadow_mode: u32,
    pub enable_ao_mode: u32,
    pub enable_key: u32,
    pub enable_fill: u32,
    pub enable_rim: u32,
    // 🟢 PARAM-GUI: Materialien & Reflexionen
    pub reflection_max_steps: u32,   // Sekundär-Schritte (8..64, default 32)
    pub reflection_roughness_gate: F, // Aktivierungsschwelle (0.05..1.0, default 0.25)
    pub fresnel_strength_factor: F,  // Reflexions-Stärke (0.0..1.0, default 0.35)
    pub statue_metallic: F,          // Statue PBR (ID 2), default 0.8
    pub pillar_roughness: F,         // Pfeiler PBR (ID 1), default 0.8
    // 🟢 PARAM-GUI: Objekt / Statue-Konfiguration
    pub sdf_grid_size: F,             // Grid-Ausdehnung (1.0..16.0, default 4.0)
    pub statue_center: GpuVec3<F>,     // Statue-Zentrum (default [0, -1.4, 0])
    pub bounding_sphere_radius_offset: F, // Proximity-Schranke (default 0.1)
}


