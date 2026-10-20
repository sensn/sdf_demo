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
#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<f32>, // 🟢 OPTIMIERT: Jetzt f32 statt u32!
    scene: &[f32], // 📦 Der einzige, gepackte Mega-Buffer  // 🟢 TENSOR 7: Objekt-Rotation (Stride 3)
    camera: CameraSettings,   // 🟢 params5: Kamera-Struct (16-Byte-Alignment)
    env: EnvSettings,         // 🟢 TENSOR 4 → Struct: Umwelt-Parameter
    arch: ArchParams,         // 🟢 TENSOR 5 → Struct: Tempel-Architektur
    fold: FoldParams,         // 🟢 TENSOR 6 → Struct: Unendliche Raumfaltung
    render: RenderSettings,   // 🟢 Meta-Tensor + 12 Skalare → Struct
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    let w_val = render.width;
    let h_val = render.height;
    let t_val = render.time;
    let b_factor = render.blend_factor;


//
   // --- Konstanten für das automatische Slicing ---
      // --- Konstanten für das automatische Slicing (Jetzt sauber als usize) ---
  
    // --- 🟢 KORRIGIERT: Nutzt jetzt die exakten Namen aus dem gpu-Import (_COUNT) ---
    const SLOTS_START: usize = 0;
    const MATERIALS_START: usize = SLOTS_COUNT;                      // 800
    const ROTATIONS_START: usize = SLOTS_COUNT + MATERIALS_COUNT;    // 1200
    const TOTAL_SIZE: usize = ROTATIONS_START + ROTATIONS_COUNT;     // 1500

    // Slices für die alten Funktionen herausschneiden
    let slots = &scene[SLOTS_START..MATERIALS_START];
    let materials = &scene[MATERIALS_START..ROTATIONS_START];
    let rotations = &scene[ROTATIONS_START..TOTAL_SIZE];
    // 🟢 PLAN STEP 3: SDF-Statuen-Grid-Slice aus dem Mega-Buffer.
    // Layout: [slots|materials|rotations|sdf_grid(64³)].
    const SDF_GRID_START: usize = TOTAL_SIZE;                        // 1500
    const SDF_GRID_END: usize = SDF_GRID_START + SDF_GRID_COUNT;     // 263644
    let sdf_grid = &scene[SDF_GRID_START..SDF_GRID_END];
    // Grid-Parameter (müssen zu sdf_preprocessor.rs passen).
    let grid_res = u32::cast_from(SDF_GRID_RES);
    // 🟢 PARAM-GUI: Grid-Größe datengetrieben (default 4.0).
    let grid_size = render.sdf_grid_size;
    // Plan-Parameter: Pfeiler-Faltung + Statuen-Platzierung.
    // 🟢 PARAM-GUI: aus FoldParams (datengetrieben statt hardcoded).
    let pillar_spacing = fold.pillar_spacing;
    let pillar_height = fold.pillar_height;
    // 🟢 PARAM-GUI: Statue-Zentrum + Bounding-Offset aus RenderSettings.
    let statue_center = render.statue_center;
    let bounding_sphere_radius_offset = render.bounding_sphere_radius_offset;
//
    if x < w_val && y < h_val {
    
    
        // =========================================================================
        // 🟢 PARAMETER-STRUCTS: Umwelt direkt aus EnvSettings (params2/params4)
        // =========================================================================
        // Block 0: Licht-Vektor und Intensität
        let key_light_pos = Vec3::new(
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

        let w_f = f32::cast_from(w_val);
        let h_f = f32::cast_from(h_val);

        let uv_x = (f32::cast_from(x) - (w_f / f32::new(2.0))) / h_f;
        let uv_y = ((h_f / f32::new(2.0)) - f32::cast_from(y)) / h_f;

        let ro = Vec3::new(camera.position.x, camera.position.y, camera.position.z);
        let rd = Vec3::new(uv_x, uv_y, f32::new(1.2));
        let cos_y = camera.yaw.cos();
        let sin_y = camera.yaw.sin();
        let cos_p = camera.pitch.cos();
        let sin_p = camera.pitch.sin();

        let rd_y1 = rd.y * cos_p - rd.z * sin_p;
        let rd_z1 = rd.y * sin_p + rd.z * cos_p;

        let final_rd = Vec3::new(
            rd.x * cos_y + rd_z1 * sin_y,
            rd_y1,
            -rd.x * sin_y + rd_z1 * cos_y,
        )
        .normalize();

        let mut t = f32::new(0.1);
        let mut hit_dist = f32::new(100.0);
        // 🟢 PLAN STEP 5: Welches Objekt hat den Strahl getroffen?
        // false = Slot-Objekt (scene_sdf), true = Plan-Objekt (Pfeiler/Statue).
        let mut hit_is_plan_obj = false;

        let mut hit_d = f32::new(1000.0);
        let mut hit_r = f32::new(0.0);
        let mut hit_g = f32::new(0.0);
        let mut hit_b = f32::new(0.0);

        let mut hit_rough = f32::new(0.5);
        let mut hit_metal = f32::new(0.0);
        let mut hit_emiss = f32::new(0.0);
        let mut hit_spec = f32::new(0.5);

        let mut ray_step = u32::new(0);
        // 🟢 PLAN STEP 6: Glow-Akkumulator — sammelt Emissive-Anteile
        // entlang des Strahls: glow += 0.01/(d+0.02) × Emissive-Farbe.
        let mut glow_r = f32::new(0.0);
        let mut glow_g = f32::new(0.0);
        let mut glow_b = f32::new(0.0);
        
        //for ray_step in 0..100 {
        loop {
            if ray_step >= u32::new(100) {
                break;
            }

            let current_ro = ro.clone();
            let current_rd = final_rd.clone();
            let p = current_ro.add(current_rd.scale(t));

            let step_res = scene_sdf(p, t_val, b_factor, render.active_slots, slots, materials,
              rotations, arch, fold);
            hit_d = step_res.d;
            hit_r = step_res.r;
            hit_g = step_res.g;
            hit_b = step_res.b;
            hit_rough = step_res.roughness;
            hit_metal = step_res.metallic;
            hit_emiss = step_res.emissive;
            hit_spec = step_res.specular;
            // 🟢 FIX (Bugfix-Runde 3): Flag JEDEM Schritt zurücksetzen —
            // sonst bleibt es hängen, wenn der Sphere-Bound mal gewinnt,
            // und der Boden neben der Statue bekommt Statuen-Normalen
            // (≈0-Gradient → schwarzer Fleck).
            hit_is_plan_obj = false;
            // 🟢 PLAN STEP 5: Statue in die Primär-Map mischen.
            // mat_id 2 = Statue (Gold-Bronze). Sphere-Bound aktiv —
            // der Primär-March braucht die sichere Schritt-Schranke.
            let plan_res = scene_map_with_material(
                p.clone(), pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size, true,
                statue_center, bounding_sphere_radius_offset,
            );
            if plan_res.d < hit_d {
                hit_d = plan_res.d;
                // Nur als Statuen-Treffer zählen, wenn die Statue selbst
                // (nicht nur die unsichtbare Kugel) die nächste Fläche ist.
                hit_is_plan_obj = plan_res.mat_id > f32::new(1.5);
                let is_statue = plan_res.mat_id > f32::new(1.5);
                hit_r = select(is_statue, f32::new(0.85), f32::new(0.55));
                hit_g = select(is_statue, f32::new(0.7), f32::new(0.52));
                hit_b = select(is_statue, f32::new(0.3), f32::new(0.5));
                hit_rough = select(is_statue, f32::new(0.35), render.pillar_roughness);
                hit_metal = select(is_statue, render.statue_metallic, f32::new(0.0));
                hit_emiss = f32::new(0.0);
                hit_spec = f32::new(0.5);
            }
            // 🟢 PLAN STEP 6: Glow-Akkumulation — nur wenn der Strahl
            // nah an einer Emissive-Fläche vorbeistreift.
            let glow_kernel = f32::new(0.01) / (hit_d + f32::new(0.02));
            glow_r += glow_kernel * step_res.emissive * step_res.r;
            glow_g += glow_kernel * step_res.emissive * step_res.g;
            glow_b += glow_kernel * step_res.emissive * step_res.b;

            if hit_d < f32::new(0.001) {
                hit_dist = t;
                break;
            }
            t += hit_d;

            if t > f32::new(40.0) {
                break;
            }
            ray_step += u32::new(1);
        }

        // 🟢 TENSOR 4: Hintergrund-Farbe dynamisch aus dem Umwelt-Register
        let bg_r = bg_r - uv_y * f32::new(0.10);
        let bg_g = bg_g - uv_y * f32::new(0.12);
        let bg_b = bg_b - uv_y * f32::new(0.15);

        let mut final_color_x = bg_r;
        let mut final_color_y = bg_g;
        let mut final_color_z = bg_b;

        if hit_dist < f32::new(40.0) {
            let p = ro.add(final_rd.scale(hit_dist));
            // 🟢 PLAN STEP 5: Normale über die Map, die den Treffer
            // geliefert hat — Slot-Objekte behalten ihre alte Normalen-
            // Funktion, Pfeiler/Statue nutzen die kombinierte Plan-Map.
            // (if-else-Expressions sind für Vec3 nicht unterstützt →
            // Statement-Form mit skalaren Akkumulatoren.)
            let mut n_x = f32::new(0.0);
            let mut n_y = f32::new(0.0);
            let mut n_z = f32::new(0.0);
            if hit_is_plan_obj {
                let n = calculate_normal(
                    p.clone(), pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size,
                    statue_center, bounding_sphere_radius_offset,
                );
                n_x = n.x;
                n_y = n.y;
                n_z = n.z;
            } else {
                let n = scene_sdf_normal(p.clone(), t_val, b_factor, render.active_slots, slots, materials,
                  rotations, arch, fold);
                n_x = n.x;
                n_y = n.y;
                n_z = n.z;
            }
            let normal = Vec3::new(n_x, n_y, n_z);

            // 🟢 TENSOR 4: Key-Licht-Position dynamisch aus dem Umwelt-Register
            let fill_light_pos = Vec3::new(f32::new(-5.0), f32::new(3.0), f32::new(-3.0));
            let rim_light_pos = Vec3::new(f32::new(0.0), f32::new(6.0), f32::new(5.0));

            let key_dir = key_light_pos.sub(p.clone()).normalize();
            let fill_dir = fill_light_pos.sub(p.clone()).normalize();
            let rim_dir = rim_light_pos.sub(p.clone()).normalize();

            let view_dir = final_rd.scale(f32::new(-1.0)).normalize();

            // 🟢 Schatten nur aufs Key-Licht (wie bisher)
            let mut key_visibility = f32::new(1.0);
            if render.shadow_mode == u32::new(1) {
                let offset_p = p.add(normal.scale(f32::new(0.02)));
                let shadow_factor = calculate_soft_shadow(
                    offset_p, key_dir.clone(), t_val, b_factor, render.active_slots, slots, materials,
                     rotations, arch, fold,
                     pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size,
                     statue_center, bounding_sphere_radius_offset, env.shadow_softness_k,
                );
                key_visibility = shadow_factor;
            }

            let mut ao_factor = f32::new(1.0);
            if render.enable_ao_mode == u32::new(1) {
                ao_factor = calculate_ao(p, normal.clone(), t_val, b_factor, render.active_slots, slots, materials,
                 rotations, arch, fold,
                 pillar_spacing, pillar_height, sdf_grid, grid_res, grid_size,
                 statue_center, bounding_sphere_radius_offset, env.ao_iterations);
            }

            // 🟢 TENSOR 4: Licht-Farben (Key dynamisch, Fill/Rim fix)
            let key_color = Vec3::new(key_r, key_g, key_b);
            let fill_color = Vec3::new(f32::new(0.25), f32::new(0.40), f32::new(0.60));
            let rim_color = Vec3::new(f32::new(0.50), f32::new(0.70), f32::new(1.00));

            let key_on = render.enable_key != u32::new(0);
            let fill_on = render.enable_fill != u32::new(0);
            let rim_on = render.enable_rim != u32::new(0);

            let albedo = Vec3::new(hit_r, hit_g, hit_b);

            // =========================================================================
            // 🟢 COOK-TORRANCE PBR (siehe PBR_Optimisation_plan.md):
            // - F0 = mix(0.04*(1+3*specular), albedo, metallic)
            // - GGX-Verteilung, Smith-Geometrie, Schlick-Fresnel
            // - kd = (1-F)*(1-metallic): Metalle haben keine diffuse Farbe
            // =========================================================================
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

            // 🟢 Ambient: konstant — Metalle bekommen ihr Ambient über die
            // Spiegel-Farbe F0 im Cook-Torrance-Specular, nicht über Grau
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
            // 🟢 PLAN STEP 7: 1-Bounce-Reflexion — nur auf glatten,
            // nicht-metallischen Flächen sinnvoll (Roughness-Gate).
            if hit_rough < render.reflection_roughness_gate {
                let refl_dir = final_rd.sub(normal.scale(f32::new(2.0) * final_rd.dot(normal.clone()))).normalize();
                let refl_color = trace_reflection(
                    p.add(normal.scale(f32::new(0.02))),
                    refl_dir,
                    t_val,
                    b_factor,
                    render.active_slots,
                    slots,
                    materials,
                    rotations,
                    arch,
                    fold,
                    sdf_grid,
                    grid_res,
                    grid_size,
                    Vec3::new(bg_r, bg_g, bg_b),
                    f32::new(40.0),
                    render.reflection_max_steps,
                    statue_center,
                    bounding_sphere_radius_offset,
                    env,
                    render,
                );
                // Fresnel-artige Mischung: glatte Flächen spiegeln stärker.
                let refl_strength = (f32::new(1.0) - hit_rough) * render.fresnel_strength_factor;
                final_color_x += refl_color.x * refl_strength;
                final_color_y += refl_color.y * refl_strength;
                final_color_z += refl_color.z * refl_strength;
            }
        }
        // 🟢 PLAN STEP 6: Glow-Anteil addieren (unabhängig vom Treffer —
        // Leuchten sichtbar auch wenn der Strahl die Quelle verfehlt).
        final_color_x += glow_r;
        final_color_y += glow_g;
        final_color_z += glow_b;
        // =====================================================================
        // 🟢 PLAN STEP 7: VOLUMETRISCHER NEBEL (Beer-Lambert-Integration)
        // Ersetzt die alte lineare Distanz-Kompression: schrittweise
        // Transmittanz entlang des Strahls mit Höhen-Profil.
        // =====================================================================
     let mut fog_transmittance = f32::new(1.0);
        
        if fog_enabled > u32::new(0) {
            fog_transmittance = integrate_volumetric_fog(
                ro.clone(),
                final_rd.clone(),
                hit_dist,
                f32::new(40.0), // Hardcoded 40.0 (Optional: durch env.max_fog_dist ersetzen)
                fog_density,
                fog_steps,          // Lokal entpackt
                fog_falloff,        // Lokal entpackt
                fog_min_falloff,    // Neu hinzugefügt (lokal entpackt)
                fog_max_depth,      // Neu hinzugefügt (lokal entpackt)
                fog_sample_offset,  // Neu hinzugefügt (lokal entpackt)
            );
        }

        // Korrektes Blending unter Verwendung der dedizierten Nebelfarbe
        let fog_absorption = f32::new(1.0) - fog_transmittance;

        final_color_x = final_color_x * fog_transmittance + fog_r * fog_absorption;
        final_color_y = final_color_y * fog_transmittance + fog_g * fog_absorption;
        final_color_z = final_color_z * fog_transmittance + fog_b * fog_absorption;

        // 🟢 ACES-TONE-MAPPING: HDR-Werte (>1.0 durch Emissive, hohe
        // Intensitäten, Glanzlichter) werden natürlich komprimiert statt
        // vom 8-Bit-Framebuffer hart geclampt. Behebt das
        // „Emissive > 1.0 = totes Weiß"-Problem.
        final_color_x = aces_tonemap(final_color_x);
        final_color_y = aces_tonemap(final_color_y);
        final_color_z = aces_tonemap(final_color_z);

        // 🟢 NEU: Direkte, lineare Zuweisung im f32 VRAM-Puffer
        // 1. Hole die Thread-Koordinaten direkt als usize
        let x_idx = usize::cast_from(x);
        let y_idx = usize::cast_from(y);
        let w_idx = usize::cast_from(w_val);

        // 2. Berechne den Index direkt in einem Rutsch als usize (Kein u32-Zwischenschritt!)
        let pixel_index = (y_idx * w_idx + x_idx) * usize::new(3);

        // 3. Direktes Schreiben ohne Typprobleme
        output[pixel_index] = final_color_x;
        output[pixel_index + usize::new(1)] = final_color_y;
        output[pixel_index + usize::new(2)] = final_color_z;
    }
}
