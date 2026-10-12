Um das Licht (wie deine `light_intensity`) physikalisch korrekt auf die Oberfläche treffen zu lassen, benötigt die Raymarching-Engine Shader-Normals (Normalenvektoren). Da eine SDF-Szene kein Polygon-Mesh besitzt, berechnet man die Normale mathematisch über den Gradienten der Distanzfunktion.

Das bedeutet: Wir tasten die Szene an der getroffenen Position minimal nach X, Y und Z versetzt ab (Zentrale Differenzen / Finite Differences). Da unsere `scene_map` sowohl die mathematischen Säulen als auch das trilinear interpolierte Tensor-Grid vereint, funktioniert diese Methode universell für beide Geometrie-Typen gleichzeitig!

Hier ist die hocheffiziente Implementierung in CubeCL.

## 1. Die Gradienten-Funktion für Shader-Normals

Wir erstellen eine Funktion `calculate_normal`. Diese tastet die Szene viermal (oder sechsmal für maximale Präzision) mit einem winzigen Offset ε (Epsilon) ab, um die Neigung der Oberfläche zu bestimmen.

```rust
use cubecl::prelude::*;

/// Berechnet die Oberflächen-Normale an Punkt P mittels Zentraler Differenzen
#[cube]
pub fn calculate_normal(
    p: Line<f32, 3>,
    params: &RaymarchParams,
    sdf_grid: &Tensor<f32>
) -> Line<f32, 3> {
    // Ein winziger Schritt für die Ableitung (Epsilon)
    let eps = 0.001f32;
    
    // Die Basis-Vektoren für die Achsen-Verschiebungen
    let e_x = Line::<f32, 3>::new(eps, 0.0, 0.0);
    let e_y = Line::<f32, 3>::new(0.0, eps, 0.0);
    let e_z = Line::<f32, 3>::new(0.0, 0.0, eps);

    // Szene in alle Richtungen minimal versetzt abfragen
    let grad_x = scene_map(p + e_x, params, sdf_grid) - scene_map(p - e_x, params, sdf_grid);
    let grad_y = scene_map(p + e_y, params, sdf_grid) - scene_map(p - e_y, params, sdf_grid);
    let grad_z = scene_map(p + e_z, params, sdf_grid) - scene_map(p - e_z, params, sdf_grid);

    // Den resultierenden Vektor zusammensetzen und normieren (Länge = 1)
    Line::<f32, 3>::normalize(Line::<f32, 3>::new(grad_x, grad_y, grad_z))
}
```

---

## 2. Integration & Shading im Hauptkernel

Sobald deine Raymarching-Schleife einen Oberflächen-Treffer (`hit == true`) meldet, berechnest du an dieser exakten Position die Normale. Anschließend nutzen wir das Lambert'sche Kosinusgesetz (Diffuses Licht), um die Helligkeit basierend auf deiner `light_intensity` und der Lichtrichtung zu berechnen.

```rust
#[cube(launch)]
pub fn raymarch_sdf_kernel<C: RaymarchConfig>(
    output: &mut Tensor<f32>,
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>,
    sdf_grid: &Tensor<f32>, // Unser 3D-SDF-Grid Tensor
    params: RaymarchParams, 
) {
    // 1. Strahl-Initialisierung (Beispiel)
    let cam_pos = Line::<f32, 3>::new(params.camera.position.x, params.camera.position.y, params.camera.position.z);
    let ray_dir = Line::<f32, 3>::new(0.0, 0.0, 1.0); // Vorwärtsgerichtet

    let mut depth = 0.0f32;
    let mut hit = false;
    let mut hit_pos = Line::<f32, 3>::to_line(0.0);

    // Raymarching-Schleife
    Comptime::loop_unrolled(0, C::MAX_STEPS, |i| {
        if !hit {
            let current_pos = cam_pos + ray_dir * depth;
            let distance = scene_map(current_pos, &params, sdf_grid);
            
            if distance < 0.001f32 {
                hit = true;
                hit_pos = current_pos;
            }
            depth += distance;
        }
    });

    // 2. SHADING & BELEUCHTUNG BEI TREFFER
    let mut final_color = 0.0f32; // Graustufen-Intensität für den Output

    if hit {
        // 💡 A: Normale berechnen
        let normal = calculate_normal(hit_pos, &params, sdf_grid);

        // 💡 B: Licht-Richtung definieren (z. B. von schräg oben rechts)
        let light_dir = Line::<f32, 3>::normalize(Line::<f32, 3>::new(0.5, 1.0, -0.5));

        // 💡 C: Diffusen Lichtfaktor berechnen (Dot-Product geklammt auf minimal 0)
        let dot_product = Line::<f32, 3>::dot(normal, light_dir);
        let diffuse = f32::max(dot_product, 0.0);

        // Ambienter Grundton aus der Struktur hinzurechnen
        let ambient = params.env.ambient_fog;

        // Finale Pixellichtstärke gekoppelt an deine Struktur-Parameter
        final_color = (diffuse * params.env.light_intensity) + ambient;
    }

    // In Ausgabepuffer schreiben (vereinfacht für den aktuellen Thread-Index)
    let global_id = cubecl::prelude::absolute_idx();
    output[global_id] = final_color;
}
```

---

## Warum das für kombinierte Szenen genial performt:

1. Automatisches Blending von Kanten: Wenn du deine Geometrien in der `scene_map` mittels `op_union` (oder Smooth-Minimum für weiche Übergänge) verschmelzt, berechnen die Finiten Differenzen an den Schnittkanten eine perfekt glatte, interpolierte Normale. Die Statue scheint nahtlos aus dem Tempelboden zu wachsen.
2. Keine Grid-Artefakte: Da wir die Trilinear-Interpolation in der letzten Frage sauber implementiert haben, liefert die Gitter-Abfrage kontinuierliche Werte. Der Normalen-Gradient ist dadurch absolut rauschfrei und glatt, selbst wenn das zugrundeliegende 3D-SDF-Grid eine recht geringe Auflösung (wie 64³) hat.

Möchtest du als Nächstes sehen, wie wir weiche Schatten (Soft Shadows) implementieren? Da wir jetzt funktionierende Normalen haben, können wir von der Oberfläche aus einen zweiten Strahl Richtung Lichtquelle senden, um zu prüfen, ob die mathematischen Säulen Schatten auf deine Gitter-Statue werfen.