//! `kernel/reflection.rs` — 1-Bounce-Reflexion: trace_reflection (march + PBR-Shading wie Primär-Pfad).
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::ao::calculate_ao;
use super::pbr::cook_torrance_light;
use super::scene::{scene_sdf, scene_sdf_normal};
use super::shadows::calculate_soft_shadow;
use super::statue_grid::{calculate_normal, scene_map_with_material};
use super::types::{ArchParams, EnvSettings, FoldParams, GpuVec3, RenderSettings, Vec3};
#[cube]
pub fn trace_reflection(
    ro: Vec3,
    rd: Vec3,
    time: f32,
    blend_factor: f32,
    active_slots: u32,
    slots: &[f32],
    materials: &[f32],
    rotations: &[f32],
    arch: ArchParams,
    fold: FoldParams,
    sdf_grid: &[f32],
    grid_res: u32,
    grid_size: f32,
    bg_color: Vec3,
    max_dist: f32,
    reflection_max_steps: u32,
    statue_center: GpuVec3,
    bounding_sphere_radius_offset: f32,
    env: EnvSettings,
    render: RenderSettings,
) -> Vec3 {
    let mut t = f32::new(0.02);
    let mut hit_dist = max_dist;
    let mut i = u32::new(0);
    // Material-Register des Reflektions-Treffers (Albedo + PBR + Emissive).
    let mut hit_r = bg_color.x;
    let mut hit_g = bg_color.y;
    let mut hit_b = bg_color.z;
    let mut hit_rough = f32::new(0.5);
    let mut hit_metal = f32::new(0.0);
    let mut hit_emiss = f32::new(0.0);
    let mut hit_spec = f32::new(0.5);
    // 🟢 PLAN STEP 6: Glow-Akkumulator — sammelt Emissive-Anteile entlang
    // des Reflektionsstrahls (wie im Primär-Pfad): glow += 0.01/(d+0.02).
    let mut glow_r = f32::new(0.0);
    let mut glow_g = f32::new(0.0);
    let mut glow_b = f32::new(0.0);
    loop {
        if i >= reflection_max_steps {
            break;
        }
        let p = ro.add(rd.scale(t));
        // Komplette Szene: Slots + Arch (scene_sdf) …
        let sdf_res = scene_sdf(p.clone(), time, blend_factor, active_slots, slots, materials,
          rotations, arch, fold);
        let mut d = sdf_res.d;
        let mut r = sdf_res.r;
        let mut g = sdf_res.g;
        let mut b = sdf_res.b;
        let mut rough = sdf_res.roughness;
        let mut metal = sdf_res.metallic;
        let mut emiss = sdf_res.emissive;
        let mut spec = sdf_res.specular;
        // … plus Statue (ohne Sphere-Bound — sonst Phantom-Reflexionen).
        let plan_res = scene_map_with_material(
            p.clone(), f32::new(0.0), f32::new(0.0), sdf_grid, grid_res, grid_size, false, statue_center, bounding_sphere_radius_offset,
        );
        if plan_res.d < d {
            d = plan_res.d;
            // Statue: gold-bronze (PBR wie im Primär-Pfad).
            let is_statue = plan_res.mat_id > f32::new(1.5);
            r = select(is_statue, f32::new(0.85), f32::new(0.55));
            g = select(is_statue, f32::new(0.7), f32::new(0.52));
            b = select(is_statue, f32::new(0.3), f32::new(0.5));
            rough = select(is_statue, f32::new(0.35), render.pillar_roughness);
            metal = select(is_statue, render.statue_metallic, f32::new(0.0));
            emiss = f32::new(0.0);
            spec = f32::new(0.5);
        }
        // 🟢 FIX BUG 2: Glow-Akkumulation entlang des Reflektionsstrahls —
        // Leuchten bleibt sichtbar, auch wenn der Strahl die Quelle verfehlt.
        let glow_kernel = f32::new(0.01) / (d + f32::new(0.02));
        glow_r += glow_kernel * emiss * r;
        glow_g += glow_kernel * emiss * g;
        glow_b += glow_kernel * emiss * b;
        if d < f32::new(0.001) {
            hit_dist = t;
            hit_r = r;
            hit_g = g;
            hit_b = b;
            hit_rough = rough;
            hit_metal = metal;
            hit_emiss = emiss;
            hit_spec = spec;
            break;
        }
        t += d;
        if t > max_dist {
            break;
        }
        i += u32::new(1);
    }
    // 🟢 FIX BUG 1: Ergebnis mit dem VOLLSTÄNDIGEN Licht-Setup des
    // Primär-Pfads schattiert (statt hardcoded N·L) — nur so stimmen
    // Direktansicht und Spiegelbild überein.
    let mut out_r = bg_color.x;
    let mut out_g = bg_color.y;
    let mut out_b = bg_color.z;
    if hit_dist < max_dist {
        let p = ro.add(rd.scale(hit_dist));
        let mut n_x = f32::new(0.0);
        let mut n_y = f32::new(1.0);
        let mut n_z = f32::new(0.0);
        // Normale über die Map, die den Treffer geliefert hat.
        let sdf_n = scene_sdf_normal(p.clone(), time, blend_factor, active_slots, slots, materials,
          rotations, arch, fold);
        let plan_n = calculate_normal(p.clone(), f32::new(0.0), f32::new(0.0), sdf_grid, grid_res, grid_size, statue_center, bounding_sphere_radius_offset);
        // Wähle per Distanz-Vergleich (keine if-else-Expr für Structs).
        let sdf_d = scene_sdf(p.clone(), time, blend_factor, active_slots, slots, materials,
          rotations, arch, fold).d;
        let plan_d = scene_map_with_material(
            p.clone(), f32::new(0.0), f32::new(0.0), sdf_grid, grid_res, grid_size, false, statue_center, bounding_sphere_radius_offset,
        ).d;
        let use_plan = plan_d < sdf_d;
        n_x = select(use_plan, plan_n.x, sdf_n.x);
        n_y = select(use_plan, plan_n.y, sdf_n.y);
        n_z = select(use_plan, plan_n.z, sdf_n.z);
        let normal = Vec3::new(n_x, n_y, n_z);

        // 🟢 Licht-Setup identisch zum Primär-Pfad: Key-Licht dynamisch aus
        // EnvSettings, Fill/Rim fix.
        let key_light_pos = Vec3::new(
            env.key_light_pos.x,
            env.key_light_pos.y,
            env.key_light_pos.z,
        );
        let l_intensity = env.light_intensity;
        let key_color = Vec3::new(env.key_color.x, env.key_color.y, env.key_color.z);
        let fill_light_pos = Vec3::new(f32::new(-5.0), f32::new(3.0), f32::new(-3.0));
        let rim_light_pos = Vec3::new(f32::new(0.0), f32::new(6.0), f32::new(5.0));

        let key_dir = key_light_pos.sub(p.clone()).normalize();
        let fill_dir = fill_light_pos.sub(p.clone()).normalize();
        let rim_dir = rim_light_pos.sub(p.clone()).normalize();

        // View-Richtung: vom Trefferpunkt zurück entlang des reflektierten
        // Strahls (rd zeigt VOM Spiegel weg).
        let view_dir = rd.scale(f32::new(-1.0)).normalize();

        // 🟢 Schatten nur aufs Key-Licht (wie im Primär-Pfad).
        let mut key_visibility = f32::new(1.0);
        if render.shadow_mode == u32::new(1) {
            let offset_p = p.add(normal.scale(f32::new(0.02)));
            let shadow_factor = calculate_soft_shadow(
                offset_p, key_dir.clone(), time, blend_factor, active_slots, slots, materials,
                rotations, arch, fold,
                f32::new(0.0), f32::new(0.0), sdf_grid, grid_res, grid_size,
                statue_center, bounding_sphere_radius_offset, env.shadow_softness_k,
            );
            key_visibility = shadow_factor;
        }

        let mut ao_factor = f32::new(1.0);
        if render.enable_ao_mode == u32::new(1) {
            ao_factor = calculate_ao(p, normal.clone(), time, blend_factor, active_slots, slots, materials,
                rotations, arch, fold,
                f32::new(0.0), f32::new(0.0), sdf_grid, grid_res, grid_size,
                statue_center, bounding_sphere_radius_offset, env.ao_iterations);
        }

        let fill_color = Vec3::new(f32::new(0.25), f32::new(0.40), f32::new(0.60));
        let rim_color = Vec3::new(f32::new(0.50), f32::new(0.70), f32::new(1.00));

        let key_on = render.enable_key != u32::new(0);
        let fill_on = render.enable_fill != u32::new(0);
        let rim_on = render.enable_rim != u32::new(0);

        let albedo = Vec3::new(hit_r, hit_g, hit_b);

        // 🟢 Cook-Torrance-PBR mit den echten Material-Werten des Treffers.
        let mut light_r = f32::new(0.0);
        let mut light_g = f32::new(0.0);
        let mut light_b = f32::new(0.0);

        if key_on {
            let contrib = cook_torrance_light(
                albedo.clone(), normal.clone(), key_dir.clone(), view_dir.clone(),
                key_color, l_intensity, hit_rough, hit_metal, hit_spec,
            );
            light_r += contrib.x * key_visibility;
            light_g += contrib.y * key_visibility;
            light_b += contrib.z * key_visibility;
        }
        if fill_on {
            let contrib = cook_torrance_light(
                albedo.clone(), normal.clone(), fill_dir.clone(), view_dir.clone(),
                fill_color, f32::new(1.0), hit_rough, hit_metal, hit_spec,
            );
            light_r += contrib.x;
            light_g += contrib.y;
            light_b += contrib.z;
        }
        if rim_on {
            let contrib = cook_torrance_light(
                albedo.clone(), normal.clone(), rim_dir.clone(), view_dir.clone(),
                rim_color, f32::new(1.0), hit_rough, hit_metal, hit_spec,
            );
            light_r += contrib.x;
            light_g += contrib.y;
            light_b += contrib.z;
        }

        // 🟢 Ambient — identisch zum Primär-Pfad.
        let ambient_r = env.ambient_strength * hit_r;
        let ambient_g = env.ambient_strength * hit_g;
        let ambient_b = env.ambient_strength * hit_b;

        // 🟢 FIX BUG 2: Emissive des Reflektions-Treffers addieren —
        // glühende Objekte erscheinen jetzt auch im Spiegel.
        let emissive_r = hit_emiss * hit_r;
        let emissive_g = hit_emiss * hit_g;
        let emissive_b = hit_emiss * hit_b;

        out_r = (light_r + ambient_r) * ao_factor + emissive_r;
        out_g = (light_g + ambient_g) * ao_factor + emissive_g;
        out_b = (light_b + ambient_b) * ao_factor + emissive_b;
    }
    // 🟢 FIX BUG 2: Glow-Anteil addieren (unabhängig vom Treffer).
    out_r += glow_r;
    out_g += glow_g;
    out_b += glow_b;
    Vec3::new(out_r, out_g, out_b)
}
