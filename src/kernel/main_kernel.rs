//! `kernel/main_kernel.rs` — Haupt-Render-Kernel: raymarch_sdf_kernel (March, Licht, Fog, Reflexion, Output).
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;


use crate::gpu_iced::{SDF_GRID_RES, MATERIALS_COUNT, ROTATIONS_COUNT, SLOTS_COUNT};

use super::ao::calculate_ao;
use super::fog::integrate_volumetric_fog;
use super::pbr::{aces_tonemap, cook_torrance_light};
use super::reflection::trace_reflection;
use super::scene::{scene_sdf, scene_sdf_normal};
use super::shadows::calculate_soft_shadow;
use super::statue_grid::{calculate_normal, objects_min_bound_distance, scene_map_with_material, sdf_object_pbr};
use super::types::{ArchParams, CameraSettings, EnvSettings, FoldParams, RenderSettings, Vec3};
//
// ALTE IMPORTS ENTFERNEN UND DURCH DIESE ERSETZEN:
//use cubek::tile::{
use cubecl::tile::{
    Axis, Space,          // Kam aus space
    Partitioning,         // Kam aus partition
    Launcher,   // Kam aus launch
    Tile, Scratch,        // Direkt re-exportiert aus dem Tile-Modul
};
//use cubecl::tile::launch::TileArg;
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
// 🟢 OPTIMIZATION 2: Control float contraction per kernel for SDF precision
// Disable contraction (AllowContraction) to prevent surface acne/overshoot artifacts
// while keeping other fast math optimizations.
#[cube(launch, fast_math = FastMath::all().difference(FastMath::AllowContraction.into()))]
pub fn raymarch_sdf_kernel<F: Float + LaunchArg>( // 💡 KORREKTUR: Bringe das Generic zurück!
    output: &mut Tensor<F>,
    slots: &Tensor<F>,           // 🟢 OPTIMIZATION 1: Flat f32 buffer for SlotData (100 * 8 = 800 f32)
    materials: &Tensor<F>,       // 🟢 OPTIMIZATION 1: Flat f32 buffer for MaterialData (100 * 4 = 400 f32)
    rotations: &Tensor<F>,       // 🟢 OPTIMIZATION 1: Flat f32 buffer for RotationData (100 * 3 = 300 f32)
    objects: &Tensor<F>,         // SDF-Objekt-Tensor [header*N|grid*N]
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
    let grid_res = u32::cast_from(SDF_GRID_RES);
    let bound_offset = render.bounding_sphere_radius_offset;
    let pillar_spacing = fold.pillar_spacing;
    let pillar_height = fold.pillar_height;
    // 🟢 SDF-OBJEKT-TENSOR: Zentrum/Größe/PBR kommen jetzt aus den
    // per-Slot-Headern im Objekt-Tensor (statue_center/sdf_grid_size/
    // statue_metallic aus RenderSettings entfallen).

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
        
        // 🟢 OPTIMIZATION (Hot-Loop): Welches Objekt hat den Strahl getroffen?
        //   1 = Slot-Objekt (scene_sdf), 2 = Grid-Bounding-Shell, 3 = Grid selbst.
        // Statt 8 PBR-Felder pro Schritt zu entpacken, merken wir uns nur die
        // Quelle und holen das Material EINMAL am Trefferpunkt (S3).
        let mut hit_mat_source = u32::new(0);
        let mut hit_mat_id = F::new(0.0);

        let mut ray_step = u32::new(0);
        
        // 🟢 OPTIMIZATION (Hot-Loop): Glow-Akkumulator
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

            // 1) Slot-SDF (schnelle analytische Formen) — immer nötig.
            let step_res = scene_sdf::<F>(
                p.clone(), t_val, b_factor, render.active_slots, slots, materials, rotations,
                arch.clone(), fold.clone()
            );
            let mut d = step_res.d;
            let mut step_source = u32::new(1);

            // 2) Grid-Slots NUR sampeln, wenn sie überhaupt näher sein könnten.
            //    `objects_min_bound_distance` liest nur 4×24 Header-Werte statt
            //    ~4 MB Voxel-Daten. Liegt der Strahl außerhalb aller Kugeln,
            //    entfällt der teure Speicherzugriff komplett.
            let grid_bound_d = objects_min_bound_distance::<F>(
                p.clone(), objects, bound_offset,
            );
            if grid_bound_d < d {
                let plan_res = scene_map_with_material::<F>(
                    p.clone(), objects, bound_offset, true,
                );
                if plan_res.d < d {
                    d = plan_res.d;
                    // 3 = echtes Grid (Header-PBR), 2 = nur Bounding-Shell.
                    step_source = select(plan_res.mat_id > F::new(1.5), u32::new(3), u32::new(2));
                    hit_mat_id = plan_res.mat_id;
                }
            }

            // 3) Glow-Akkumulation: nur Slot-Objekte, die wirklich die nächste
            //    Fläche sind (sonst leuchtete ein verdecktes Objekt durch).
            if step_source == u32::new(1) && step_res.emissive > F::new(0.0) {
                let glow_kernel = F::new(0.01) / (d + F::new(0.02));
                glow_r += glow_kernel * step_res.emissive * step_res.r;
                glow_g += glow_kernel * step_res.emissive * step_res.g;
                glow_b += glow_kernel * step_res.emissive * step_res.b;
            }

            // 4) NaN-sicherer Hit-Test (CubeCL #2121): `!(d >= eps)` ist wahr,
            //    wenn d < eps ODER d == NaN — der Strahl bricht sofort ab,
            //    statt alle 100 Schritte leer zu drehen.
            if !(d >= F::new(0.001)) {
                hit_dist = t;
                hit_mat_source = step_source;
                break;
            }
            t += d;

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
            
            // 🟢 OPTIMIZATION (S3): Material EINMAL am Trefferpunkt ermitteln
            // (vorher wurden 8 PBR-Felder in JEDEM Iterationsschritt entpackt).
            // hit_mat_source: 1 = Slot-Objekt, 2 = Bounding-Shell, 3 = Grid.
            let mut hit_r = F::new(0.55);
            let mut hit_g = F::new(0.52);
            let mut hit_b = F::new(0.5);
            let mut hit_rough = render.pillar_roughness;
            let mut hit_metal = F::new(0.0);
            let mut hit_emiss = F::new(0.0);
            let mut hit_spec = F::new(0.5);

            if hit_mat_source == u32::new(1) {
                // Slot-Objekt: vollständiges Material aus der Szene.
                let mat_res = scene_sdf::<F>(
                    p.clone(), t_val, b_factor, render.active_slots, slots, materials, rotations,
                    arch.clone(), fold.clone(),
                );
                hit_r = mat_res.r;
                hit_g = mat_res.g;
                hit_b = mat_res.b;
                hit_rough = mat_res.roughness;
                hit_metal = mat_res.metallic;
                hit_emiss = mat_res.emissive;
                hit_spec = mat_res.specular;
            } else if hit_mat_source == u32::new(3) {
                // Echtes Grid-Treffer: Albedo + PBR aus dem Slot-Header.
                let pbr = sdf_object_pbr::<F>(objects, hit_mat_id);
                hit_r = pbr.r;
                hit_g = pbr.g;
                hit_b = pbr.b;
                hit_rough = pbr.roughness;
                hit_metal = pbr.metallic;
                hit_emiss = pbr.emissive;
                hit_spec = pbr.specular;
            }
            // hit_mat_source == 2 (Bounding-Shell): behält die steinigen
            // Defaults oben — in der Praxis folgt scene_sdf näher.

            // 🟢 OPTIMIZATION (S3): Normale NUR über die Map, die wirklich den
            // Treffer geliefert hat. Der alte Code lief bei JEDEM Treffer in
            // die Grid-Normale (6 × 64³-Voxel-Sampling), sobald irgendwann
            // einmal plan_res.d < hit_d war — auch bei bloßen Pfeiler-Treffern.
            let mut n_x = F::new(0.0);
            let mut n_y = F::new(1.0);
            let mut n_z = F::new(0.0);

            if hit_mat_source >= u32::new(2) {
                let n = calculate_normal::<F>(
                    p.clone(), objects, bound_offset,
                );
                n_x = n.x;
                n_y = n.y;
                n_z = n.z;
            } else {
                let n = scene_sdf_normal::<F>(
                    p.clone(), t_val, b_factor, render.active_slots, slots, materials, rotations,
                    arch.clone(), fold.clone(),
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
                // 💡 KORREKTUR: Übergabe von `slots/materials/rotations` statt der Slices
                let shadow_factor = calculate_soft_shadow::<F>(
                    offset_p, key_dir.clone(), t_val, b_factor, render.active_slots,
                    slots, materials, rotations,
                    arch.clone(), fold.clone(), objects, bound_offset, env.shadow_softness_k,
                );
                key_visibility = shadow_factor;
            }

            let mut ao_factor = F::new(1.0);
            if render.enable_ao_mode == u32::new(1) {
                // 💡 KORREKTUR: Übergabe von `slots/materials/rotations` statt der Slices
                ao_factor = calculate_ao::<F>(
                    p.clone(), normal.clone(), t_val, b_factor, render.active_slots,
                    slots, materials, rotations,
                    arch.clone(), fold.clone(), objects, bound_offset, env.ao_iterations
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

                               // 💡 KORREKTUR: Reicht `slots/materials/rotations` statt der Slices an trace_reflection weiter
                let refl_color = trace_reflection::<F>(
                    p.add(normal.scale(F::new(0.02))),
                    refl_dir,
                    t_val,
                    b_factor,
                    render.active_slots,
                    slots,
                    materials,
                    rotations,
                    arch.clone(),
                    fold.clone(),
                    objects,
                    bound_offset,
                    Vec3::<F>::new(bg_r, bg_g, bg_b), // 💡 KORREKTUR: Turbofisch hinzugefügt
                    F::new(40.0),
                    render.reflection_max_steps,
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
