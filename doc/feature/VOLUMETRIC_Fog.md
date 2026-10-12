Volumetrischer Nebel und Lichtschächte (God Rays) entstehen, wenn Licht an winzigen Partikeln in der Luft gestreut wird. Im SDF-Raymarching können wir diesen Effekt atemberaubend schön berechnen, indem wir die Atmosphäre entlang des Kamerasichtstrahls abtasten (_In-Scattering_).

Anstatt nur am Endpunkt (der Oberfläche) das Licht zu berechnen, wandern wir den Strahl entlang und prüfen an mehreren Punkten im leeren Raum, ob dieser Punkt im Schatten liegt oder von der Sonne getroffen wird. Wenn er getroffen wird, addieren wir Nebeldichte auf.

Hier ist die Implementierung in CubeCL:

## 1. Der Volumetric Accumulator (Hilfsfunktion)

Wir schreiben eine Funktion `integrate_volumetric_fog`, die das Licht entlang des Sichtstrahls aufsummiert. Sie nutzt direkt unsere `calculate_shadow`- und `scene_map`-Funktionen.

```rust
use cubecl::prelude::*;

#[cube]
pub fn integrate_volumetric_fog(
    ro: Line<f32, 3>,       // Kamera-Position (Strahlursprung)
    rd: Line<f32, 3>,       // Strahlrichtung
    max_depth: f32,         // Distanz bis zur ersten getroffenen Oberfläche
    params: &RaymarchParams,
    sdf_grid: &Tensor<f32>,
) -> f32 {
    let mut fog_accumulation = 0.0f32;
    
    // Festgelegte Anzahl an Nebel-Abtastschritten (z.B. 24 Schritte für gute Performance)
    let sample_count = 24u32;
    let step_size = max_depth / (sample_count as f32);
    
    // Die Lichtrichtung (muss exakt mit deinem Hauptlicht übereinstimmen)
    let light_dir = Line::<f32, 3>::normalize(Line::<f32, 3>::new(0.5, 1.0, -0.5));
    
    // Globale Nebeldichte aus deinen Parametern (z.B. params.env.ambient_fog)
    let base_density = params.env.ambient_fog * 0.1f32; 

    let mut t = 0.0f32;
    
    // Eine einfache Schleife zur Volumen-Integration
    let mut i = 0u32;
    while i < sample_count {
        let current_pos = ro + rd * t;
        
        // 💡 1. Prüfen, ob dieser Punkt im Raum im Schatten liegt
        // Ein k-Wert von z.B. 8.0 sorgt für wunderschön weiche Lichtschächte
        let shadow_factor = calculate_shadow(
            current_pos, 
            light_dir, 
            0.05f32, 
            15.0f32, 
            8.0f32, 
            params, 
            sdf_grid
        );
        
        // 💡 2. Wenn der Punkt von der Sonne getroffen wird, sammelt sich Licht im Nebel
        let light_scattering = shadow_factor * params.env.light_intensity;
        
        // Exponentielle Abschwächung: Je tiefer im Nebel, desto mehr Licht wird akkumuliert
        fog_accumulation += light_scattering * base_density * step_size;
        
        t += step_size;
        i += 1u32;
    }
    
    // Wert für die finale Anzeige begrenzen
    f32::min(fog_accumulation, 1.0f32)
}
```

## 2. Integration in deinen Hauptkernel

Der Hauptkernel führt nun zuerst das reguläre Oberflächen-Raymarching aus. Danach wird das Oberflächen-Shading mit dem volumetrischen Nebel überblendet (_Composite-Schritt_).

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
    let max_ray_distance = 50.0f32; // Maximale Sichtweite des Strahls

    // 1. OBERFLÄCHEN-RAYMARCHING (Wie bisher)
    Comptime::loop_unrolled(0, C::MAX_STEPS, |i| {
        if !hit && depth < max_ray_distance {
            let current_pos = cam_pos + ray_dir * depth;
            let distance = scene_map(current_pos, &params, sdf_grid);
            
            if distance < 0.001f32 {
                hit = true;
                hit_pos = current_pos;
            }
            depth += distance;
        }
    });

    // Wenn nichts getroffen wurde, setzen wir die Tiefe auf das Maximum
    let final_depth = if hit { depth } else { max_ray_distance };

    // 2. OBERFLÄCHEN-SHADING COMPUTE
    let mut surface_color = 0.0f32;
    if hit {
        let normal = calculate_normal(hit_pos, &params, sdf_grid);
        let light_dir = Line::<f32, 3>::normalize(Line::<f32, 3>::new(0.5, 1.0, -0.5));
        let diffuse = f32::max(Line::<f32, 3>::dot(normal, light_dir), 0.0);
        
        let mut shadow_factor = 1.0f32;
        if Comptime::get::<C::USE_SHADOWS>() {
            shadow_factor = calculate_shadow(hit_pos + normal * 0.01f32, light_dir, 0.01f32, 20.0f32, 16.0f32, &params, sdf_grid);
        }
        
        surface_color = (diffuse * params.env.light_intensity * shadow_factor) + params.env.ambient_fog;
    } else {
        // Hintergrund/Himmelsfarbe, wenn kein Treffer erzielt wurde
        surface_color = params.env.ambient_fog * 0.3f32;
    }

    // 3. VOLUMETRISCHEN NEBEL BERECHNEN (Dazwischenschalten)
    // Wir wandern von der Kamera (cam_pos) bis zum Aufprallpunkt (final_depth)
    let fog = integrate_volumetric_fog(cam_pos, ray_dir, final_depth, &params, sdf_grid);

    // 4. FINALES BILD COMPOSITING
    // Wir blenden die Oberflächenfarbe linear mit dem Nebellicht über
    let final_color = surface_color * (1.0f32 - fog) + (fog * params.env.light_intensity);

    let global_id = cubecl::prelude::absolute_idx();
    output[global_id] = final_color;
}
```

## Warum dieser Effekt im Zusammenspiel mit dem Tensor-Grid fantastisch aussieht:

Wenn das Sonnenlicht hinter deinen mathematischen Säulen oder der Gitter-Statue vorbeischeint, blockiert die Geometrie die Lichtstrahlen entlang des Nebelpfads. Dadurch entstehen echte, dreidimensionale Lichtschatten im Raum (Crepuscular Rays). Da wir die Nebelschritte (`sample_count = 24`) relativ gering halten, bleibt der Shader extrem flüssig auf WebGPU und Vulkan, liefert aber dennoch einen epischen, cineastischen Look für deinen Tempel.

Möchtest du als Nächstes sehen, wie du Ambient Occlusion (Umgebungsverdeckung) implementierst, um Ecken, Ritzen und die Spalten deines Säulentempels ohne Lichtquellen realistisch abzudunkeln?