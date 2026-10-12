Hier ist die vollständig ausformulierte und mathematisch präzise `trace_reflection`-Funktion für deine `kernel.rs`.

Um die GPU zu schonen und gleichzeitig maximale grafische Wiedergabetreue zu gewährleisten, nutzt diese Sekundärstrahl-Funktion eine leicht optimierte Version des Cook-Torrance-PBR-Modells. Sie liest die Material-Eigenschaften des gespiegelten Objekts direkt aus deinem `materials`-Tensor und wertet sowohl die diffusen als auch die metallischen Eigenschaften bei der Reflexion aus.

## 1. Die PBR-optimierte `trace_reflection`-Funktion

Füge diese Funktion in deine `kernel.rs` ein. Sie ist flach programmiert (iteratives Tracing über maximal 32 Schritte), um perfekt auf dem WebGPU- und Vulkan-Backend von CubeCL 0.11.0 zu performen.

```rust
use cubecl::prelude::*;

/// Rastet die Szene für einen Sekundärstrahl (Reflexion) ab und berechnet farbechtes PBR-Shading
#[cube]
pub fn trace_reflection(
    ro: Line<f32, 3>,       // Startpunkt der Spiegelung (hit_pos + normal * 0.01)
    rd: Line<f32, 3>,       // Abprallrichtung (Reflected View Vector)
    materials: &Tensor<f32>, // Der globale Material-Eigenschaften Tensor
    sdf_grid: &Tensor<f32>,  // Das 3D-SDF-Grid für die Statue
    params: &RaymarchParams,
) -> Line<f32, 3> {         // Gibt die reflektierte RGB-Farbe zurück
    let mut depth = 0.01f32; // Leicht versetzt starten, um Selbstkollision zu vermeiden
    let mut hit = false;
    let mut hit_pos = Line::<f32, 3>::to_line(0.0);
    let mut hit_mat_id = 0.0f32;
    let max_reflect_dist = 25.0f32; // Kürzere Reichweite schont die Performance massiv

    // 1. ITERATIVE SEKUNDÄRSTRAHL-SCHLEIFE (32 Schritte)
    let mut i = 0u32;
    while i < 32u32 && depth < max_reflect_dist {
        let current_pos = ro + rd * depth;
        let scene_data = scene_map_with_material(current_pos, params, sdf_grid);
        let distance = scene_data.x;

        if distance < 0.001f32 {
            hit = true;
            hit_pos = current_pos;
            hit_mat_id = scene_data.y;
            break;
        }
        
        // Sicherheits-Mindestschritt gegen TDR (GPU-Hänger)
        depth += f32::max(distance, 0.005f32);
        i += 1u32;
    }

    let mut reflected_rgb = Line::<f32, 3>::to_line(0.0);

    // 2. PBR-SHADING FÜR DAS GESPIEGELTE OBJEKT
    if hit {
        // Materialparameter des gespiegelten Objekts aus dem Tensor laden
        let stride = (hit_mat_id as u32) * 6u32;
        let albedo      = Line::<f32, 3>::new(materials[stride], materials[stride + 1u32], materials[stride + 2u32]);
        let roughness   = materials[stride + 3u32];
        let metallic    = materials[stride + 4u32];
        let is_emission = materials[stride + 5u32];

        // Vektoren für die gespiegelte Oberfläche berechnen
        let n = calculate_normal(hit_pos, params, sdf_grid);
        let l = Line::<f32, 3>::normalize(Line::<f32, 3>::new(0.5, 1.0, -0.5)); // Globale Lichtrichtung
        let v = -rd; // Blickrichtung des Spiegels ist die invertierte Strahlrichtung
        let h = Line::<f32, 3>::normalize(l + v);

        let n_dot_l = f32::max(Line::<f32, 3>::dot(n, l), 0.0);
        let n_dot_v = f32::max(Line::<f32, 3>::dot(n, v), 0.0);

        // Schnelle Cook-Torrance Spekular-Komponente für Sekundär-Pass
        let d = ndf_ggx(n, h, roughness);
        let g = geometry_smith(n, l, v, roughness);
        let f0 = mix(Line::<f32, 3>::to_line(0.04), albedo, metallic);
        let f = fresnel_schlick(f32::max(Line::<f32, 3>::dot(h, v), 0.0), f0);

        let k_s = f;
        let k_d = (Line::<f32, 3>::to_line(1.0) - k_s) * (1.0 - metallic);

        let numerator = f * (d * g);
        let denominator = 4.0 * n_dot_v * n_dot_l + 0.001f32;
        let specular_brdf = numerator / denominator;
        
        let pi = 3.14159265f32;
        let diffuse_brdf = albedo / pi;

        // Umgebungsverdeckung (AO) für das gespiegelte Objekt berechnen
        let ao = calculate_ao(hit_pos + n * 0.005f32, n, params, sdf_grid);
        
        // Aus Performancegründen nutzen wir in Reflexionen oft vereinfachte Schatten (Shadow-Ray entfällt hier)
        let sun_color = Line::<f32, 3>::new(1.0, 0.95, 0.88);
        let direct_light = (k_d * diffuse_brdf + specular_brdf) * sun_color * (params.env.light_intensity * n_dot_l);
        let ambient_light = albedo * (params.env.ambient_fog * ao);

        reflected_rgb = direct_light + ambient_light;

        // Falls das reflektierte Objekt eine leuchtende Rune ist -> Volles Eigenlicht spiegeln!
        if is_emission > 0.0f32 {
            reflected_rgb = reflected_rgb + albedo * 3.0f32;
        }
    } else {
        // Wenn der Spiegelstrahl ins Leere fliegt, sieht er den nebligen Himmel (Dunst/Blau)
        reflected_rgb = Line::<f32, 3>::new(0.4, 0.5, 0.6) * params.env.ambient_fog * 0.3f32;
    }

    reflected_rgb
}
```

---

## 2. Anpassung des Hauptkernels (`raymarch_sdf_kernel`)

Im Shading-Zweig deines Hauptkernels rufst du die neue Funktion nun auf. Wichtig ist hierbei das korrekte farbliche Überblenden: Metalle spiegeln farbig (ihre eigene Albedo), Dielektrika (Marmor) spiegeln rein weiß.

```rust
// Im Shading-Block des primären Treffers innerhalb des Hauptkernels:
if hit {
    // ... [Zuvor implementierte primäre PBR-Berechnungen (direct_light, ambient_light)] ...

    // 💡 HIER STARTET DIE SPIEGELUNGS-INTEGRATION:
    // Wir lassen den primären Kamera-Blickstrahl (ray_dir) an der Normalen (n) abprallen
    let reflect_dir = Line::<f32, 3>::reflect(ray_dir, n);
    
    // Startpunkt entlang der Normalen verschieben gegen Selbst-Spiegelungs-Artefakte
    let reflect_origin = hit_pos + n * 0.01f32;
    
    // Rufe die voll-PBR-fähige Reflexionsfunktion auf
    let reflected_color_rgb = trace_reflection(
        reflect_origin, 
        reflect_dir, 
        materials, // Reicht den Materialien-Tensor weiter
        sdf_grid,  // Reicht den SDF-Gitter-Tensor weiter
        &params
    );

    // Reflexionsstärke anhand der Rauheit bestimmen (Je rauer, desto weniger Spiegelung)
    let reflectivity = mix(0.35f32, 0.0f32, roughness);

    // 💡 Einbinden in das PBR-Modell: 
    // Metalle (metallic = 1.0) färben die gespiegelte Farbe mit ihrer eigenen Albedo ein.
    // Nicht-Metalle spiegeln das Licht unverändert weiß/farbneutral.
    let fresnel_reflect_color = mix(Line::<f32, 3>::to_line(1.0), albedo, metallic);
    let final_reflection = reflected_color_rgb * fresnel_reflect_color * reflectivity;

    // Die Spiegelung wird auf die primäre Oberflächenfarbe aufaddiert!
    surface_color = surface_color * (1.0f32 - reflectivity) + final_reflection;
    
    final_rgb = surface_color;
}
```

---

## Warum das optisch ein Meisterwerk wird:

- Goldspiegelung: Wenn sich das goldene Licht der Runen auf einer anderen Goldfläche spiegelt, addieren sich die Farbspektren physikalisch korrekt auf. Wenn es sich auf Marmor spiegelt, siehst du ein klares, goldenes Abbild auf weißem Grund.
- Gedämpfte Reflexion in Spalten: Weil das gespiegelte Objekt in `trace_reflection` sein eigenes `calculate_ao` ausführt, spiegeln sich Säulenzwischenräume wunderschön düster und plastisch wider, anstatt flach zu wirken.

Damit steht die komplette GPU-Pipeline deines High-End-Raymarchers! Sollen wir als Nächstes den vollständigen CPU Host-Code mit dem PNG-Bilder-Exporter aufsetzen, damit der Coding-Agent das gerenderte Pixel-Array aus dem `output`-Tensor auslesen und als Bild auf deiner Festplatte sichern kann?

Wenn du bereit bist, lass uns festlegen:

- Welche Bildauflösung soll der Exporter standardmäßig nutzen (z. B. 1024x768)?
- Soll der Agent die PNG-Bibliothek (`png` oder `image` Crate) nutzen, um das Bild direkt als `output.png` zu schreiben?