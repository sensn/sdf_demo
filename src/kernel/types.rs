//! `kernel/types.rs` — Basis-Typen & Helfer: Vec3/Vec2/SdfResult/GpuVec3, smin, smin_material, Launch-Structs.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct SdfResult {
    pub d: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    // 🟢 NEU: Physikalische Oberflächen-Eigenschaften
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: f32,
    pub specular: f32,
}

#[cube]
pub fn smin_material(a: SdfResult, b: SdfResult, k: f32) -> SdfResult {
    let h = (k - (a.d - b.d).abs()).max(f32::new(0.0)) / k;

    // Berechne die interpolierte Distanz
    let mixed_d = a.d.min(b.d) - h * h * k * f32::new(0.25);

    // Lineare Farbmischung basierend auf der Nähe zum jeweiligen Objekt
    // Wenn h nah an 1 ist, befinden wir uns in der Übergangszone
    let mix_factor = (f32::new(0.5) + f32::new(0.5) * (b.d - a.d) / k)
        .min(f32::new(1.0))
        .max(f32::new(0.0));

    let mixed_r = a.r + mix_factor * (b.r - a.r);
    let mixed_g = a.g + mix_factor * (b.g - a.g);
    let mixed_b = a.b + mix_factor * (b.b - a.b);
    // 🟢 NEU: Weiches Überblenden der PBR-Materialien in der Mischzone!
    let mixed_rough = a.roughness + mix_factor * (b.roughness - a.roughness);
    let mixed_metal = a.metallic + mix_factor * (b.metallic - a.metallic);
    let mixed_emiss = a.emissive + mix_factor * (b.emissive - a.emissive);
    let mixed_spec = a.specular + mix_factor * (b.specular - a.specular);

    SdfResult {
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
// HIER: Damit wird das Trait CloneExpand automatisch für Vec3Expand generiert!
#[cube(derive(Copy, Clone))]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[cube]
impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }
    pub fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
    pub fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
    pub fn scale(self, factor: f32) -> Vec3 {
        Vec3::new(self.x * factor, self.y * factor, self.z * factor)
    }
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn length(self) -> f32 {
        f32::sqrt(self.dot(self))
    }
    pub fn normalize(self) -> Vec3 {
        let len = self.length();
        let mut x = self.x;
        let mut y = self.y;
        let mut z = self.z;
        if len > 0.0 {
            let factor = 1.0 / len;
            x = self.x * factor;
            y = self.y * factor;
            z = self.z * factor;
        }
        Vec3::new(x, y, z)
    }
}

#[cube]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(f32::new(0.0)) / k;
    a.min(b) - h * h * k * f32::new(0.25)
}

#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct Vec2 {
    pub d: f32,
    pub mat_id: f32,
}
/// 16-Byte-Alignment fuer vec3-Parameter (params5): vec3 + explizites Pad.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

/// Kamera-Parameter (params5: "camera settings struct").
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct CameraSettings {
    pub position: GpuVec3,
    pub yaw: f32,
    pub pitch: f32,
}

/// Umwelt-Parameter (ersetzt Tensor 4 / env_data()[16]).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct EnvSettings {
    pub key_light_pos: GpuVec3, // war env[0..2]
    pub light_intensity: f32,   // war env[3]
    pub key_color: GpuVec3,     // war env[4..6]
    pub ambient_strength: f32,  // war env[7]
    pub bg_color: GpuVec3,      // war env[8..10]
    pub fog_density: f32,       // war env[11]
    pub fog_enabled: u32,       // war env[12] (f32) -> params3: u32-Flag 0/1
    // 🟢 PARAM-GUI: Atmosphäre & Lighting
    pub volumetric_fog_steps: u32, // Nebel-Integrationsschritte (4..64, default 16)
    pub fog_density_falloff: f32,  // Höhenprofil-Falloff (0.01..0.5, default 0.08)
    pub ao_iterations: u32,        // AO-Sampling-Iterationen (1..10, default 5)
    pub shadow_softness_k: f32,     // Weichheits-Konstante (1.0..64.0, default 8.0)
}

/// Tempel-Architektur (ersetzt Tensor 5 / arch_data()[12]).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct ArchParams {
    pub pillar_dist: f32,   // war arch[0]
    pub pillar_thick: f32,  // war arch[1]
    pub room_height: f32,   // war arch[2]
    pub ceiling_thick: f32, // war arch[3]
    pub arch_radius: f32,   // war arch[4]
    pub arch_height: f32,   // war arch[5]
    pub decor_freq: f32,    // war arch[6]
    pub decor_depth: f32,   // war arch[7]
    pub decor_thick: f32,   // war arch[8]
}

/// Unendliche Raumfaltung (ersetzt Tensor 6 / fold_data()[4]).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct FoldParams {
    pub cell_size: f32,  // war fold[0]
    pub half_cell: f32,  // war fold[1]
    pub fold_speed: f32, // war fold[2]
    // 🟢 PARAM-GUI: Infinite Repetition (Pfeiler-Faltung)
    pub pillar_spacing: f32, // XZ-Wiederholungs-Abstand (2.0..20.0, default 6.0)
    pub pillar_height: f32,   // Pfeilerhöhe (1.0..10.0, default 4.0)
}

/// Frame-Parameter & Mode-Flags (ersetzt Meta-Tensor + 12 lose Skalare).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[cube(derive(Copy, Clone))]
pub struct RenderSettings {
    pub time: f32,
    pub width: u32,
    pub height: u32,
    pub blend_factor: f32,
    pub active_slots: u32, // ersetzt meta[0] (f32) -> params3: Count als u32
    pub shadow_mode: u32,
    pub enable_ao_mode: u32,
    pub enable_key: u32,
    pub enable_fill: u32,
    pub enable_rim: u32,
    // 🟢 PARAM-GUI: Materialien & Reflexionen
    pub reflection_max_steps: u32,   // Sekundär-Schritte (8..64, default 32)
    pub reflection_roughness_gate: f32, // Aktivierungsschwelle (0.05..1.0, default 0.25)
    pub fresnel_strength_factor: f32,  // Reflexions-Stärke (0.0..1.0, default 0.35)
    pub statue_metallic: f32,          // Statue PBR (ID 2), default 0.8
    pub pillar_roughness: f32,         // Pfeiler PBR (ID 1), default 0.8
    // 🟢 PARAM-GUI: Objekt / Statue-Konfiguration
    pub sdf_grid_size: f32,             // Grid-Ausdehnung (1.0..16.0, default 4.0)
    pub statue_center: GpuVec3,         // Statue-Zentrum (default [0, -1.4, 0])
    pub bounding_sphere_radius_offset: f32, // Proximity-Schranke (default 0.1)
}

