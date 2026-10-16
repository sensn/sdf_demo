//! `kernel/statue_grid.rs` — SDF-Grid-Sampling: Statue (trilinear), scene_map_with_material, calculate_normal.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::types::{ArchParams, GpuVec3, Vec2 as OtherVec2, Vec3};
// ============================================================================
// 🟢 PLAN-IMPLEMENTATION (Agent_Impl_Plan.md, Steps 3–7)
// ============================================================================
/// Trilineares Sampling des 3D-SDF-Grids (Step 3).
///
/// `p` liegt im Welt-Raum; das Grid spannt `[-size/2, size/2]^3` auf und
/// ist in ZYX-Reihenfolge geflacht (index = x + res*(y + res*z)).
/// 8 Voxel werden nach den fraktionalen Anteilen gemischt.
#[cube]
pub fn sample_sdf_grid(p: Vec3, grid: &[f32], res: u32, size: f32) -> f32 {
    let res_f = f32::cast_from(res);
    let res_us = usize::cast_from(res);
    let half = size * f32::new(0.5);
    // In Grid-Koordinaten [0, res-1] mappen.
    let local_x = (p.x + half) / size * (res_f - f32::new(1.0));
    let local_y = (p.y + half) / size * (res_f - f32::new(1.0));
    let local_z = (p.z + half) / size * (res_f - f32::new(1.0));
    // Auf [0, res-2] klemmen, damit x0+1/y0+1/z0+1 valide Indizes sind.
    let max_bound = res_f - f32::new(2.0);
    let lp_x = local_x.max(f32::new(0.0)).min(max_bound);
    let lp_y = local_y.max(f32::new(0.0)).min(max_bound);
    let lp_z = local_z.max(f32::new(0.0)).min(max_bound);
    // Basis-Voxel + fraktionale Gewichte.
    let x0 = u32::cast_from(lp_x);
    let y0 = u32::cast_from(lp_y);
    let z0 = u32::cast_from(lp_z);
    let tx = lp_x - f32::cast_from(x0);
    let ty = lp_y - f32::cast_from(y0);
    let tz = lp_z - f32::cast_from(z0);
    let x0_us = usize::cast_from(x0);
    let y0_us = usize::cast_from(y0);
    let z0_us = usize::cast_from(z0);
    // 8 Ecken des Voxels (ZYX-Flattening).
    let c000 = grid[x0_us + res_us * (y0_us + res_us * z0_us)];
    let c100 = grid[(x0_us + usize::new(1)) + res_us * (y0_us + res_us * z0_us)];
    let c010 = grid[x0_us + res_us * ((y0_us + usize::new(1)) + res_us * z0_us)];
    let c110 = grid[(x0_us + usize::new(1)) + res_us * ((y0_us + usize::new(1)) + res_us * z0_us)];
    let c001 = grid[x0_us + res_us * (y0_us + res_us * (z0_us + usize::new(1)))];
    let c101 = grid[(x0_us + usize::new(1)) + res_us * (y0_us + res_us * (z0_us + usize::new(1)))];
    let c011 = grid[x0_us + res_us * ((y0_us + usize::new(1)) + res_us * (z0_us + usize::new(1)))];
    let c111 = grid[(x0_us + usize::new(1)) + res_us * ((y0_us + usize::new(1)) + res_us * (z0_us + usize::new(1)))];
    // Trilineare Mischung entlang X, dann Y, dann Z.
    let x00 = c000 * (f32::new(1.0) - tx) + c100 * tx;
    let x10 = c010 * (f32::new(1.0) - tx) + c110 * tx;
    let x01 = c001 * (f32::new(1.0) - tx) + c101 * tx;
    let x11 = c011 * (f32::new(1.0) - tx) + c111 * tx;
    let y0v = x00 * (f32::new(1.0) - ty) + x10 * ty;
    let y1v = x01 * (f32::new(1.0) - ty) + x11 * ty;
    y0v * (f32::new(1.0) - tz) + y1v * tz
}
// HINWEIS: `op_repetition_xz` + `sdf_fractal_pillar` (Plan Step 4)
// wurden entfernt — sie kollidierten mit der Arch-Zellen-Faltung aus
// `scene_sdf` (Phantom-Pfeiler in Nachbarräumen, kaputte Schatten).
// Die Architektur generiert ihre eigenen Pfeiler datengetrieben
// (ArchParams: pillar_dist/thick, room_height, ...).
/// Szenen-Map mit Material-ID (Step 5): das Statuen-SDF-Grid mit
/// Bounding-Sphere-Distance-Culling für das teure Grid-Sampling.
/// Rückgabe: `Vec2 { d, mat_id }` (mat_id: 2=Statue).
///
/// ⚠️ Die Pfeiler des Plans wurden ENTFERNT: sie kollidierten mit der
/// bestehenden Arch-Zelle (anderes Folding, andere Höhe) und erzeugten
/// Phantom-Pfeiler in Nachbarräumen + kaputte Schatten. Die Architektur
/// hat ihre eigenen, korrekt gefalteten Pfeiler (siehe `scene_sdf`).
#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct Vec2 {
    pub d: f32,
    pub mat_id: f32,
}
#[cube]
pub fn scene_map_with_material(
    p: Vec3,
    spacing: f32,
    pillar_height: f32,
    grid: &[f32],
    grid_res: u32,
    grid_size: f32,
    use_sphere_bound: bool,
    statue_center: GpuVec3,
    bounding_sphere_radius_offset: f32,
) -> Vec2 {
    let mut d_scene = f32::new(1000.0);
    let mut mat_id = f32::new(2.0);
    // Statuen-Grid: Bounding-Sphere-Culling — nur samplen, wenn der
    // Punkt nahe an der Statuen-Kugel liegt (Radius = grid_size/2).
    // WICHTIG für den Raymarch: außerhalb der Kugel ist die exakte
    // Distanz zur Kugel eine gültige untere Schranke (Lipschitz-
    // Bedingung) — der Marcher kann nicht durch die Statue tunneln.
    // `use_sphere_bound`: Primär-March JA (sicheres Stepping), Schatten/
    // AO NEIN (sonst wirft die unsichtbare Kugel einen Phantom-Schatten
    // — dunkler Ring um die Statue).
    let statue_center_v = Vec3::new(
        statue_center.x,
        statue_center.y,
        statue_center.z,
    );
    let statue_pos = p.sub(statue_center_v);
    let sphere_d = statue_pos.length() - grid_size * f32::new(0.5);
    if sphere_d < bounding_sphere_radius_offset {
        let d_statue = sample_sdf_grid(statue_pos, grid, grid_res, grid_size);
        if d_statue < d_scene {
            d_scene = d_statue;
        }
    } else if use_sphere_bound {
        // Außerhalb: Distanz zur Bounding-Kugel als sichere Schranke
        // (nur für den Primär-March — Schatten/AO sehen das nicht!).
        // mat_id bleibt 0 — die Kugel ist KEIN Objekt, nur eine
        // Schritt-Schranke (kein Material, keine Normalen von hier).
        d_scene = sphere_d;
        mat_id = f32::new(0.0);
    }
    Vec2 { d: d_scene, mat_id }
}
/// Normale via zentrale Differenzen über `scene_map_with_material`
/// (Step 5): 6 SDF-Auswertungen entlang der Hauptachsen, ε = 0.001.
#[cube]
pub fn calculate_normal(
    p: Vec3,
    spacing: f32,
    pillar_height: f32,
    grid: &[f32],
    grid_res: u32,
    grid_size: f32,
    statue_center: GpuVec3,
    bounding_sphere_radius_offset: f32,
) -> Vec3 {
    let eps = f32::new(0.001);
    // Normalen brauchen die reale Distanz (kein Sphere-Bound — sonst
    // gäbe es an der unsichtbaren Kugelgrenze falsche Normalen).
    let d = scene_map_with_material(p.clone(), spacing, pillar_height, grid, grid_res, grid_size, false, statue_center, bounding_sphere_radius_offset)
        .d;
    let nx = scene_map_with_material(
        Vec3::new(p.x + eps, p.y, p.z),
        spacing,
        pillar_height,
        grid,
        grid_res,
        grid_size,
        false,
        statue_center,
        bounding_sphere_radius_offset,
    )
    .d - d;
    let ny = scene_map_with_material(
        Vec3::new(p.x, p.y + eps, p.z),
        spacing,
        pillar_height,
        grid,
        grid_res,
        grid_size,
        false,
        statue_center,
        bounding_sphere_radius_offset,
    )
    .d - d;
    let nz = scene_map_with_material(
        Vec3::new(p.x, p.y, p.z + eps),
        spacing,
        pillar_height,
        grid,
        grid_res,
        grid_size,
        false,
        statue_center,
        bounding_sphere_radius_offset,
    )
    .d - d;
    Vec3::new(nx, ny, nz).normalize()
}

