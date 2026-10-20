//! `kernel/main_kernel.rs` — Haupt-Render-Kernel: raymarch_sdf_kernel (March, Licht, Fog, Reflexion, Output).
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;


use crate::gpu::{MATERIALS_COUNT, ROTATIONS_COUNT, SDF_GRID_COUNT, SDF_GRID_RES, SLOTS_COUNT};

use super::ao::calculate_ao;
use super::fog::integrate_volumetric_fog;
use super::pbr::{aces_tonemap, cook_torrance_light};
use super::reflection::trace_reflection;
use super::scene::{scene_sdf, scene_sdf_normal};
use super::shadows::calculate_soft_shadow;
use super::statue_grid::{calculate_normal, scene_map_with_material};
use super::types::{ArchParams, CameraSettings, EnvSettings, FoldParams, RenderSettings, Vec3};
//
// ALTE IMPORTS ENTFERNEN UND DURCH DIESE ERSETZEN:
use cubek::tile::{
    Axis, Space,          // Kam aus space
    Partitioning,         // Kam aus partition
    Launcher, TileArg,  // Kam aus launch
    Tile, Scratch,        // Direkt re-exportiert aus dem Tile-Modul
};

//
// =========================================================================
// BLUEPRINT & HOST-SIDE CONFIGURATION
// =========================================================================

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct RaymarchBlueprint {
    pub tile_width: u32,
    pub tile_height: u32,
    pub max_active_slots: u32,
    pub use_advanced_fusing: bool,
}

pub enum RaymarchStrategy {
    StandardFolded,
    LinearCoalesced,
}

pub struct DerivedKernelConfig {
    pub strategy: RaymarchStrategy,
    pub max_slots: usize,
    pub vector_width: u32,
}

impl DerivedKernelConfig {
    pub fn new(blueprint: RaymarchBlueprint, vector_width: u32, props: &DeviceProperties) -> Self {
        // Entscheide die Strategie anhand der Hardwareeigenschaften (Adaptation Workflow)
let strategy = if vector_width >= 4 {
            RaymarchStrategy::LinearCoalesced
        } else {
            RaymarchStrategy::StandardFolded
        };

        Self {
            strategy,
            max_slots: usize::cast_from(blueprint.max_active_slots),
            vector_width,
        }
    }
}

//LAUNCH
// 💡 KORREKTUR: launch_unchecked erlaubt voll-generische Shader-Einstiegspunkte!
#[cube(launch)] // 💡 Nutzt das Standard-Launch-Makro
pub fn raymarch_sdf_kernel<F: Float + LaunchArg>( // 💡 KORREKTUR: Bringe das Generic zurück!
    output: &mut Tensor<F>,
    scene: &Tensor<F>, // 📦 Der gepackte Mega-Buffer wird als Tensor-Referenz durchgereicht
    camera: CameraSettings<F>,
    env: EnvSettings<F>,
    arch: ArchParams<F>,
    fold: FoldParams<F>,
    render: RenderSettings<F>,

) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    let w_val = render.width;
    let h_val = render.height;
    let t_val = render.time;
    let b_factor = render.blend_factor;
/*
    const SLOTS_START: usize = 0;
    const MATERIALS_START: usize = SLOTS_COUNT;                      // 800
    const ROTATIONS_START: usize = SLOTS_COUNT + MATERIALS_COUNT;    // 1200
    const TOTAL_SIZE: usize = ROTATIONS_START + ROTATIONS_COUNT;     // 1500
    const SDF_GRID_START: usize = TOTAL_SIZE;                        // 1500
    const SDF_GRID_END: usize = SDF_GRID_START + SDF_GRID_COUNT;     // 263644

    // 💡 Die Slices werden als f32-Views erzeugt
    let slots: &[f32] = &scene.slice(SLOTS_START, MATERIALS_START);
    let materials: &[f32] = &scene.slice(MATERIALS_START, ROTATIONS_START);
    let rotations: &[f32] = &scene.slice(ROTATIONS_START, TOTAL_SIZE);
    let sdf_grid: &[f32] = &scene.slice(SDF_GRID_START, SDF_GRID_END);
    */
    let grid_res = u32::cast_from(SDF_GRID_RES);
    let grid_size = render.sdf_grid_size;
    let pillar_spacing = fold.pillar_spacing;
    let pillar_height = fold.pillar_height;
    let statue_center = render.statue_center;
    let bounding_sphere_radius_offset = render.bounding_sphere_radius_offset;

  if x < w_val && y < h_val {
        // =========================================================================
        // 🟢 PARAMETER-STRUCTS: Umwelt direkt aus EnvSettings (params2/params4)
        // =========================================================================
        // Block 0: Licht-Vektor und Intensität
        let key_light_pos = Vec3::<F>::new(
            env.key_light_pos.x,
            env.key_light_pos.y,
            env.key_light_pos.z,
        );
        let l_intensity = env.light_intensity;

        // Block 1: Licht-Farben und globale Umgebung
        let key_r = env.key_color.x;
        let key_g = env.key_color.y;
        let key_b = env.key_color.z;
        let a_strength = env.ambient_strength;

        // Block 2: Hintergrund-Farbe und Nebel-Dichte
        let bg_r = env.bg_color.x;
        let bg_g = env.bg_color.y;
        let bg_b = env.bg_color.z;
        let fog_density = env.fog_density;

        // Block 3: Schalter (params3: u32-Flag 0/1)
        let fog_enabled = env.fog_enabled;
        
        // 🟢 NEU - Block 4: Dedizierte Nebelfarbe (Komponenten-Entpackung)
        let fog_r = env.fog_color.x;
        let fog_g = env.fog_color.y;
        let fog_b = env.fog_color.z;

        // 🟢 NEU - Block 5: Erweiterte Nebel-Parameter für den Kernel-Aufruf
        let fog_steps = env.volumetric_fog_steps;
        let fog_falloff = env.fog_density_falloff;
        let fog_min_falloff = env.min_height_falloff;
        let fog_max_depth = env.max_optical_depth;
        let fog_sample_offset = env.fog_sample_offset;

        let w_f = F::cast_from(w_val);
        let h_f = F::cast_from(h_val);

        let uv_x = (F::cast_from(x) - (w_f / F::new(2.0))) / h_f;
        let uv_y = ((h_f / F::new(2.0)) - F::cast_from(y)) / h_f;

        let ro = Vec3::<F>::new(camera.position.x, camera.position.y, camera.position.z);
        let rd = Vec3::<F>::new(uv_x, uv_y, F::new(1.2));
        
        let cos_y = camera.yaw.cos();
        let sin_y = camera.yaw.sin();
        let cos_p = camera.pitch.cos();
        let sin_p = camera.pitch.sin();

        let rd_y1 = rd.y * cos_p - rd.z * sin_p;
        let rd_z1 = rd.y * sin_p + rd.z * cos_p;

        let final_rd = Vec3::<F>::new(
            rd.x * cos_y + rd_z1 * sin_y,
            rd_y1,
            -rd.x * sin_y + rd_z1 * cos_y,
        )
        .normalize();

        let mut t = F::new(0.1);
        let mut hit_dist = F::new(100.0);
        
        // 🟢 PLAN STEP 5: Welches Objekt hat den Strahl getroffen?
        // false = Slot-Objekt (scene_sdf), true = Plan-Objekt (Pfeiler/Statue).
        let mut hit_is_plan_obj = false;

        let mut hit_d = F::new(1000.0);
        let mut hit_r = F::new(0.0);
        let mut hit_g = F::new(0.0);
        let mut hit_b = F::new(0.0);

        let mut hit_rough = F::new(0.5);
        let mut hit_metal = F::new(0.0);
        let mut hit_emiss = F::new(0.0);
        let mut hit_spec = F::new(0.5);

        let mut ray_step = u32::new(0);
        
        // 🟢 PLAN STEP 6: Glow-Akkumulator
        let mut glow_r = F::new(0.0);
        let mut glow_g = F::new(0.0);
        let mut glow_b = F::new(0.0);
        
        loop {
            if ray_step >= u32::new(100) {
                break;
            }

            let current_ro = ro.clone();
            let current_rd = final_rd.clone();
            let p = current_ro.add(current_rd.scale(t));

            // 💡 KORREKTUR: Wir übergeben anstelle der Slices nun direkt den `scene`-Tensor!
            let step_res = scene_sdf::<F>(
                p.clone(), t_val, b_factor, render.active_slots, scene, arch.clone(), fold.clone()
            );
            
            hit_d = step_res.d;
            hit_r = step_res.r;
            hit_g = step_res.g;
            hit_b = step_res.b;
            hit_rough = step_res.roughness;
            hit_metal = step_res.metallic;
            hit_emiss = step_res.emissive;
            hit_spec = step_res.specular;
            
            // 🟢 FIX (Bugfix-Runde 3): Flag JEDEM Schritt zurücksetzen
            hit_is_plan_obj = false;

            
            // 🟢 PLAN STEP 5: Statue in die Primär-Map mischen.
                       // 💡 KORREKTUR: Übergabe von `scene` anstelle des alten `sdf_grid` Slices
            let plan_res = scene_map_with_material::<F>(
                p.clone(), pillar_spacing, pillar_height, scene, grid_res, grid_size, true,
                statue_center.clone(), bounding_sphere_radius_offset,
            );
            if plan_res.d < hit_d {
                hit_d = plan_res.d;
                // Nur als Statuen-Treffer zählen, wenn die Statue selbst die nächste Fläche ist.
                hit_is_plan_obj = plan_res.mat_id > F::new(1.5);
                let is_statue = plan_res.mat_id > F::new(1.5);
                hit_r = select(is_statue, F::new(0.85), F::new(0.55));
                hit_g = select(is_statue, F::new(0.7), F::new(0.52));
                hit_b = select(is_statue, F::new(0.3), F::new(0.5));
                hit_rough = select(is_statue, F::new(0.35), render.pillar_roughness);
                hit_metal = select(is_statue, render.statue_metallic, F::new(0.0));
                hit_emiss = F::new(0.0);
                hit_spec = F::new(0.5);
            }
            
            // 🟢 PLAN STEP 6: Glow-Akkumulation
            let glow_kernel = F::new(0.01) / (hit_d + F::new(0.02));
            glow_r += glow_kernel * step_res.emissive * step_res.r;
            glow_g += glow_kernel * step_res.emissive * step_res.g;
            glow_b += glow_kernel * step_res.emissive * step_res.b;

            if hit_d < F::new(0.001) {
                hit_dist = t;
                break;
            }
            t += hit_d;

            if t > F::new(40.0) {
                break;
            }
            ray_step += u32::new(1);
        }

        // 🟢 TENSOR 4: Hintergrund-Farbe dynamisch aus dem Umwelt-Register
        let bg_r = bg_r - uv_y * F::new(0.10);
        let bg_g = bg_g - uv_y * F::new(0.12);
        let bg_b = bg_b - uv_y * F::new(0.15);

        let mut final_color_x = bg_r;
        let mut final_color_y = bg_g;
        let mut final_color_z = bg_b;

        if hit_dist < F::new(40.0) {
            let p = ro.add(final_rd.scale(hit_dist));
            
            // 🟢 PLAN STEP 5: Normale über die Map, die den Treffer geliefert hat
            let mut n_x = F::new(0.0);
            let mut n_y = F::new(0.0);
            let mut n_z = F::new(0.0);
            
            if hit_is_plan_obj {
                // 💡 KORREKTUR: `scene` anstelle des Slices übergeben
                let n = calculate_normal::<F>(
                    p.clone(), pillar_spacing, pillar_height, scene, grid_res, grid_size,
                    statue_center.clone(), bounding_sphere_radius_offset,
                );
                n_x = n.x;
                n_y = n.y;
                n_z = n.z;
            } else {
                // 💡 KORREKTUR: Signaturen-Anpassung, reicht `scene` statt der Slices nach unten weiter
                let n = scene_sdf_normal::<F>(
                    p.clone(), t_val, b_factor, render.active_slots, scene, arch.clone(), fold.clone()
                );
                n_x = n.x;
                n_y = n.y;
                n_z = n.z;
            }
            let normal = Vec3::<F>::new(n_x, n_y, n_z);

            // 🟢 TENSOR 4: Key-Licht-Position dynamisch aus dem Umwelt-Register
                                 let fill_light_pos = Vec3::<F>::new(F::new(-5.0), F::new(3.0), F::new(-3.0));
            let rim_light_pos = Vec3::<F>::new(F::new(0.0), F::new(6.0), F::new(5.0));

            let key_dir = key_light_pos.sub(p.clone()).normalize();
            let fill_dir = fill_light_pos.sub(p.clone()).normalize();
            let rim_dir = rim_light_pos.sub(p.clone()).normalize();

            let view_dir = final_rd.scale(F::new(-1.0)).normalize();

            // 🟢 Schatten nur aufs Key-Licht
            let mut key_visibility = F::new(1.0);
            if render.shadow_mode == u32::new(1) {
                let offset_p = p.add(normal.scale(F::new(0.02)));
                // 💡 KORREKTUR: Übergabe von `scene` statt der Slices slots/materials/rotations/sdf_grid
                let shadow_factor = calculate_soft_shadow::<F>(
                    offset_p, key_dir.clone(), t_val, b_factor, render.active_slots, scene,
                    arch.clone(), fold.clone(), pillar_spacing, pillar_height, grid_res, grid_size,
                    statue_center.clone(), bounding_sphere_radius_offset, env.shadow_softness_k,
                );
                key_visibility = shadow_factor;
            }

            let mut ao_factor = F::new(1.0);
            if render.enable_ao_mode == u32::new(1) {
                // 💡 KORREKTUR: Übergabe von `scene` statt der Slices slots/materials/rotations/sdf_grid
                ao_factor = calculate_ao::<F>(
                    p.clone(), normal.clone(), t_val, b_factor, render.active_slots, scene,
                    arch.clone(), fold.clone(), pillar_spacing, pillar_height, grid_res, grid_size,
                    statue_center.clone(), bounding_sphere_radius_offset, env.ao_iterations
                );
            }

            // 🟢 TENSOR 4: Licht-Farben (Key dynamisch, Fill/Rim fix)
            let key_color = Vec3::<F>::new(key_r, key_g, key_b);
            let fill_color = Vec3::<F>::new(F::new(0.25), F::new(0.40), F::new(0.60));
            let rim_color = Vec3::<F>::new(F::new(0.50), F::new(0.70), F::new(1.00));

            let key_on = render.enable_key != u32::new(0);
            let fill_on = render.enable_fill != u32::new(0);
            let rim_on = render.enable_rim != u32::new(0);

            let albedo = Vec3::<F>::new(hit_r, hit_g, hit_b);

            // =========================================================================
            // 🟢 COOK-TORRANCE PBR
            // =========================================================================
            let mut light_r = F::new(0.0);
            let mut light_g = F::new(0.0);
            let mut light_b = F::new(0.0);

            if key_on {
                let contrib = cook_torrance_light::<F>(
                    albedo.clone(), normal.clone(), key_dir.clone(), view_dir.clone(),
                    key_color, l_intensity, hit_rough, hit_metal, hit_spec,
                );
                light_r += contrib.x * key_visibility;
                light_g += contrib.y * key_visibility;
                light_b += contrib.z * key_visibility;
            }
            if fill_on {
                let contrib = cook_torrance_light::<F>(
                    albedo.clone(), normal.clone(), fill_dir.clone(), view_dir.clone(),
                    fill_color, F::new(1.0), hit_rough, hit_metal, hit_spec,
                );
                light_r += contrib.x;
                light_g += contrib.y;
                light_b += contrib.z;
            }
            if rim_on {
                let contrib = cook_torrance_light::<F>(
                    albedo.clone(), normal.clone(), rim_dir.clone(), view_dir.clone(),
                    rim_color, F::new(1.0), hit_rough, hit_metal, hit_spec,
                );
                light_r += contrib.x;
                light_g += contrib.y;
                light_b += contrib.z;
            }

            // 🟢 Ambient: konstant
            let ambient_r = a_strength * hit_r;
            let ambient_g = a_strength * hit_g;
            let ambient_b = a_strength * hit_b;

            // 🟢 Emissive: farbig statt weiß (Slot-Farbe × Emissive-Skalar)
            let emissive_r = hit_emiss * hit_r;
            let emissive_g = hit_emiss * hit_g;
            let emissive_b = hit_emiss * hit_b;

            // Finaler Farb-Kanal-Zusammenbau inklusive AO & Emission
            final_color_x = (light_r + ambient_r) * ao_factor + emissive_r;
            final_color_y = (light_g + ambient_g) * ao_factor + emissive_g;
            final_color_z = (light_b + ambient_b) * ao_factor + emissive_b;
            
            // 🟢 PLAN STEP 7: 1-Bounce-Reflexion
            if hit_rough < render.reflection_roughness_gate {
                let refl_dir = final_rd.sub(normal.scale(F::new(2.0) * final_rd.dot(normal.clone()))).normalize();

                               // 💡 KORREKTUR: Reicht `scene` statt der Slices slots/materials/rotations/sdf_grid an trace_reflection weiter
                let refl_color = trace_reflection::<F>(
                    p.add(normal.scale(F::new(0.02))),
                    refl_dir,
                    t_val,
                    b_factor,
                    render.active_slots,
                    scene,
                    arch.clone(),
                    fold.clone(),
                    grid_res,
                    grid_size,
                    Vec3::<F>::new(bg_r, bg_g, bg_b), // 💡 KORREKTUR: Turbofisch hinzugefügt
                    F::new(40.0),
                    render.reflection_max_steps,
                    statue_center.clone(),
                    bounding_sphere_radius_offset,
                    env.clone(),
                    render.clone(),
                );
                
                // Fresnel-artige Mischung: glatte Flächen spiegeln stärker.
                let refl_strength = (F::new(1.0) - hit_rough) * render.fresnel_strength_factor;
                final_color_x += refl_color.x * refl_strength;
                final_color_y += refl_color.y * refl_strength;
                final_color_z += refl_color.z * refl_strength;
            }
        }
        // 🟢 PLAN STEP 6: Glow-Anteil addieren
        final_color_x += glow_r;
        final_color_y += glow_g;
        final_color_z += glow_b;

        // =====================================================================
        // 🟢 PLAN STEP 7: VOLUMETRISCHER NEBEL (Beer-Lambert-Integration)
        // =====================================================================
        let mut fog_transmittance = F::new(1.0);
        
        if fog_enabled > u32::new(0) {
            fog_transmittance = integrate_volumetric_fog::<F>(
                ro.clone(),
                final_rd.clone(),
                hit_dist,
                F::new(40.0), // Hardcoded 40.0
                fog_density,
                fog_steps,          // Lokal entpackt
                fog_falloff,        // Lokal entpackt
                fog_min_falloff,    // Neu hinzugefügt (lokal entpackt)
                fog_max_depth,      // Neu hinzugefügt (lokal entpackt)
                fog_sample_offset,  // Neu hinzugefügt (lokal entpackt)
            );
        }

        // Korrektes Blending unter Verwendung der dedizierten Nebelfarbe
        let fog_absorption = F::new(1.0) - fog_transmittance;

        final_color_x = final_color_x * fog_transmittance + fog_r * fog_absorption;
        final_color_y = final_color_y * fog_transmittance + fog_g * fog_absorption;
        final_color_z = final_color_z * fog_transmittance + fog_b * fog_absorption;

        // 🟢 ACES-TONE-MAPPING
        final_color_x = aces_tonemap::<F>(final_color_x);
        final_color_y = aces_tonemap::<F>(final_color_y);
        final_color_z = aces_tonemap::<F>(final_color_z);

        // 🟢 LINEARE ZUWEISUNG IM GENERISCHEN VRAM-PUFFER
        // 1. Hole die Thread-Koordinaten direkt als usize
        let x_idx = usize::cast_from(x);
        let y_idx = usize::cast_from(y);
        let w_idx = usize::cast_from(w_val);

        // 2. Berechne den Index direkt in einem Rutsch als usize
        let pixel_index = (y_idx * w_idx + x_idx) * usize::new(3);

        // 3. Direktes Schreiben ohne Typprobleme
        output[pixel_index] = final_color_x;
        output[pixel_index + usize::new(1)] = final_color_y;
        output[pixel_index + usize::new(2)] = final_color_z;
    }
}
