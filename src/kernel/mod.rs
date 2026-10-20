//! `kernel/` — Modularisierter cubecl-Raymarch-Kernel.
//!
//! Die ehemals monolithische `kernel.rs` (1670 Zeilen) ist in Feature-Module
//! aufgeteilt (Funktionalität unverändert):
//! - `types`       : Vec3/Vec2/SdfResult/GpuVec3 + smin/smin_material + Launch-Structs
//! - `objects`     : dynamische SDF-Objekte (Crystal, Gyroid, Torus)
//! - `scene`       : scene_sdf (Slots + Arch) + scene_sdf_normal
//! - `statue_grid` : SDF-Grid-Sampling (Statue) + scene_map_with_material + calculate_normal
//! - `fog`         : volumetrischer Nebel (Beer-Lambert-Integration)
//! - `reflection`  : 1-Bounce-Reflexion (trace_reflection)
//! - `shadows`     : weiche Schatten (calculate_soft_shadow)
//! - `ao`          : Ambient Occlusion (calculate_ao)
//! - `pbr`         : Cook-Torrance-PBR (fresnel, GGX, Smith, ACES, cook_torrance_light)
//! - `main_kernel` : raymarch_sdf_kernel (Haupt-Render-Pfad)
//!
//! Alle öffentlichen Symbole werden hier re-exportiert, damit externe Aufrufer
//! (gpu.rs, app_state.rs) unverändert bleiben.

pub mod ao;
pub mod fog;
pub mod main_kernel;
pub mod objects;
pub mod pbr;
pub mod reflection;
pub mod scene;
pub mod shadows;
pub mod statue_grid;
pub mod types;
pub mod volume;

// Re-Exports: alter `crate::kernel::X`-Pfad bleibt gültig
pub use types::{
    CameraSettings, CameraSettingsLaunch, EnvSettings, EnvSettingsLaunch, FoldParams,
    FoldParamsLaunch, GpuVec3, GpuVec3Launch, RenderSettings, RenderSettingsLaunch, ArchParams,
    ArchParamsLaunch, SdfResult, Vec2, Vec3,
};
pub use objects::{evaluate_dynamic_crystal, evaluate_dynamic_gyroid, evaluate_dynamic_torus};
pub use scene::{scene_sdf, scene_sdf_normal};
pub use statue_grid::{calculate_normal, sample_sdf_grid, scene_map_with_material};
pub use fog::integrate_volumetric_fog;
pub use reflection::trace_reflection;
pub use shadows::calculate_soft_shadow;
pub use ao::calculate_ao;
pub use pbr::{aces_tonemap, cook_torrance_light, d_ggx, fresnel_schlick, g_smith};
pub use main_kernel::raymarch_sdf_kernel;
