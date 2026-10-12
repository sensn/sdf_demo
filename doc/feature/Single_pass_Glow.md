Um magisch leuchtende Runen (Glow / Emission) auf deinen Tempelwänden hinzuzufügen, erweitern wir unsere `scene_map` um eine zweite Information: Neben dem Distanzwert muss uns die Szene an jedem Punkt auch sagen, ob dort eine Rune leuchtet.

In einer High-End-Raymarching-Engine nutzen wir dafür das Prinzip von Glow-Akkumulation. Das bedeutet: Jedes Mal, wenn unser Primärstrahl (oder Nebelstrahl) _nah_ an einer leuchtenden Rune vorbeifliegt, sammeln wir das Leuchten auf. Dadurch entsteht ein wunderschöner, volumetrischer Glüheffekt (Bloom) in der Luft um die Rune herum, ganz ohne teure Post-Processing-Filter!

Hier ist die Implementierung in CubeCL:

## 1. Die erweiterte Szenen-Map mit Runen-Logik

Wir definieren eine mathematische Rune (z. B. dünne, leuchtende Boxen oder Streifen auf den Säulen) und messen deren Distanz separat.

```rust
use cubecl::prelude::*;

/// Hilfsfunktion, die neben der SDF-Distanz auch den Emissions-Wert (Leuchtstärke) zurückgibt
#[cube]
pub fn scene_map_with_emission(
    p: Line<f32, 3>, 
    params: &RaymarchParams, 
    sdf_grid: &Tensor<f32>
) -> Line<f32, 2> { // 💡 Gibt [Distanz, Leuchtstärke] zurück
    // 1. Säulen-Wiederholung (wie gehabt)
    let pillar_spacing = params.arch.pillar_count as f32 * 2.0; 
    let local_p = op_repetition_xz(p, pillar_spacing);
    let pillar_bounds = Line::<f32, 3>::new(1.0, params.arch.temple_height, 1.0);
    let d_pillars = sdf_box(local_p, pillar_bounds);

    // 2. Organische Gitter-Statue (wie gehabt)
    let grid_res = Line::<u32, 3>::new(64, 64, 64);
    let grid_size = Line::<f32, 3>::new(8.0, 8.0, 8.0);
    let statue_pos = p - Line::<f32, 3>::new(0.0, 4.0, 0.0);
    let d_statue = sample_sdf_grid(statue_pos, sdf_grid, grid_res, grid_size);

    // Gesamte Szene-Geometrie verschmelzen
    let d_scene = f32::min(d_pillars, d_statue);

    // 💡 3. MAGISCHE RUNEN GENERIEREN
    // Wir schneiden hauchdünne Streifen (Runen) in die Oberfläche der Säulen
    // Indem wir die Sinus-Welle der Höhe (p.y) und des Winkels nutzen, erzeugen wir ein Muster
    let rune_pattern = f32::sin(p.y * 3.0) * f32::cos(local_p.x * 5.0);
    
    let mut rune_emission = 0.0f32;
    // Wenn wir uns sehr nah an der Säule befinden UND das Muster anschlägt, aktivieren wir das Glühen
    if d_pillars < 0.05f32 && rune_pattern > 0.7f32 {
        rune_emission = 1.0f32; // Volle Leuchtkraft der Rune
    }

    Line::<f32, 2>::new(d_scene, rune_emission)
}

// Kompatibilitäts-Brücke für Funktionen, die nur die Distanz brauchen (z.B. Normalen/Schatten)
#[cube]
pub fn scene_map(p: Line<f32, 3>, params: &RaymarchParams, sdf_grid: &Tensor<f32>) -> f32 {
    scene_map_with_emission(p, params, sdf_grid).x
}
```

---

## 2. Der Hauptkernel mit Glow-Akkumulation

In der Raymarching-Schleife prüfen wir nun in jedem Schritt, wie nah wir einer leuchtenden Rune sind. Wenn wir nah dran sind (kleine Distanz) und die Rune aktiv leuchtet, addieren wir diesen Wert auf unseren `glow_accumulator`.

```rust
#[cube(launch)]
pub fn raymarch_sdf_kernel<C: RaymarchConfig>(
    output: &mut Tensor<f32>,
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>,
    sdf_grid: &Tensor<f32>,
    params: RaymarchParams, 
) {
    let cam_pos = Line::<f32, 3>::new(params.camera.position.x, params.camera.position.y, params.camera.position.z);
    let ray_dir = Line::<f32, 3>::new(0.0, 0.0, 1.0); // Beispiel

    let mut depth = 0.0f32;
    let mut hit = false;
    let mut hit_pos = Line::<f32, 3>::to_line(0.0);
    let mut hit_emission = 0.0f32;
    
    // 💡 Unser Akkumulator für das volumetrische Glühen in der Luft
    let mut glow_accumulator = 0.0f32; 

    // 1. OBERFLÄCHEN-RAYMARCHING MIT GLOW-SUCHE
    Comptime::loop_unrolled(0, C::MAX_STEPS, |i| {
        if !hit && depth < 50.0f32 {
            let current_pos = cam_pos + ray_dir * depth;
            
            // Abfrage von Geometrie UND Emission
            let scene_data = scene_map_with_emission(current_pos, &params, sdf_grid);
            let distance = scene_data.x;
            let emission = scene_data.y;

            // 💡 GLOW-EFFEKT AKKUMULIEREN:
            // Je näher der Strahl an einer leuchtenden Rune vorbeigleitung, 
            // desto mehr "Licht-Aura" sammelt er auf.
            if emission > 0.0f32 {
                // Formel: Emission geteilt durch Distanz (verhindert Division durch Null)
                glow_accumulator += emission * (0.01f32 / (distance + 0.02f32));
            }
            
            if distance < 0.001f32 {
                hit = true;
                hit_pos = current_pos;
                hit_emission = emission;
            }
            depth += distance;
        }
    });

    // 2. KLASSISCHES SHADING (Normalen, Schatten, AO, Nebel - wie vorher)
    let mut final_color = 0.0f32;
    
    if hit {
        let normal = calculate_normal(hit_pos, &params, sdf_grid);
        let light_dir = Line::<f32, 3>::normalize(Line::<f32, 3>::new(0.5, 1.0, -0.5));
        let diffuse = f32::max(Line::<f32, 3>::dot(normal, light_dir), 0.0);
        
        let ao = calculate_ao(hit_pos + normal * 0.005f32, normal, &params, sdf_grid);
        let mut shadow = 1.0f32;
        if Comptime::get::<C::USE_SHADOWS>() {
            shadow = calculate_shadow(hit_pos + normal * 0.01f32, light_dir, 0.01f32, 20.0f32, 16.0f32, &params, sdf_grid);
        }

        let ambient = params.env.ambient_fog * ao;
        let direct = diffuse * params.env.light_intensity * shadow;
        
        // Oberflächenfarbe berechnen
        final_color = direct + ambient;

        // 💡 Wenn die Oberfläche direkt auf einer Rune getroffen wurde, 
        // brennt die Oberfläche im puren Weiß/Eigenlicht aus
        if hit_emission > 0.0f32 {
            final_color += hit_emission * 2.0f32; 
        }
    } else {
        final_color = params.env.ambient_fog * 0.3f32;
    }

    // Volumetrischen Nebel hinzufügen (aus vorherigem Schritt)
    let fog = integrate_volumetric_fog(cam_pos, ray_dir, depth, &params, sdf_grid);
    final_color = final_color * (1.0f32 - fog) + (fog * params.env.light_intensity);

    // 💡 3. FINALE COMPOSITION: Glow-Aura oben auf das fertige Bild addieren
    // Das Magielicht durchbricht selbst dichten Nebel und dunkle AO-Ecken!
    final_color += glow_accumulator * params.env.light_intensity;

    let global_id = cubecl::prelude::absolute_idx();
    output[global_id] = final_color;
}
```

---

## Warum dieser Ansatz für WebGPU & Vulkan so genial ist:

1. Kein Post-Processing-Pass nötig: Normalerweise müsste man für ein Leuchten (Bloom) das Bild in einem zweiten Shader weichzeichnen. Durch die Akkumulation direkt während des Raymarchings erhalten wir eine wunderschöne Licht-Aura völlig gratis in einem einzigen Durchlauf (_Single-Pass Glow_).
2. Magische Lichtdurchbrüche: Wenn eine Säule im Schatten steht, siehst du die Rune und ihre hell glühende Aura im Dunkeln perfekt leuchten.

Möchtest du als Nächstes die Engine finalisieren und sehen, wie wir Reflections (Spiegelungen) berechnen, indem wir den Strahl an den glatten Tempelwänden abprallen lassen, oder sollen wir ein echtes Farb- und Materialsystem (z.B. Marmor-Textur vs. Gold-Rune) einbauen?