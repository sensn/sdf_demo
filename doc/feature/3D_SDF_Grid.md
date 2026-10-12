Ein 3D-SDF-Grid speichert vorberechnete Distanzwerte an festen Gitterpunkten (Voxeln). Da ein Strahl beim Raymarching jedoch fast nie exakt auf einem dieser Gitterpunkte landet, müssen wir die Werte der 8 umliegenden Voxel auslesen und mittels trilinearer Interpolation (linear in X, Y und Z) verschmelzen==. Das verhindert unschöne Treppenstufen (Aliasing) und sorgt für eine perfekt glatte Oberfläche.

Hier erfährst du, wie du das Auslesen eines 3D-SDF-Grids inklusive trilinearer Interpolation in CubeCL implementierst.

---

## 1. Das mathematische Prinzip im Schnelldurchlauf

Wenn wir eine Position P(x,y,z) im Raum abfragen:

1. Rechnen wir P in die lokale Gitterkoordinate um.
2. Bestimmen wir die Indizes der 8 umliegenden Nachbarpunkte im flachen Tensor-Array.
3. Berechnen wir die Nachkommastellen (Gewichte) für die Interpolation.
4. Verschmelzen wir die Werte schrittweise (erst entlang X, dann Y, dann Z).

---

## 2. Die Implementierung in CubeCL

Wir bauen eine modulare Hilfsfunktion `sample_sdf_grid`, die mit `#[cube]` markiert ist, damit sie nahtlos in deine `scene_map` integriert werden kann.

```rust
use cubecl::prelude::*;

/// Hilfsfunktion zur trilinearen Interpolation (Equivalent zu HLSL/GLSL mix)
#[cube]
fn mix(a: f32, b: f32, weight: f32) -> f32 {
    a * (1.0 - weight) + b * weight
}

/// Liest ein 3D-SDF-Grid aus einem flachen Tensor aus
#[cube]
pub fn sample_sdf_grid(
    p: Line<f32, 3>,
    grid_tensor: &Tensor<f32>,
    grid_res: Line<u32, 3>, // z.B. (64, 64, 64)
    grid_size: Line<f32, 3>, // Physikalische Ausdehnung im Raum, z.B. (10.0, 10.0, 10.0)
) -> f32 {
    // 1. Position von Weltkoordinaten in lokale Gitterkoordinaten umrechnen
    // Wir verschieben und skalieren p, sodass es im Bereich [0, grid_res - 1] liegt
    let half_size = grid_size * 0.5;
    let local_pos = (p + half_size) / grid_size * Line::<f32, 3>::new(
        (grid_res.x - 1u32) as f32,
        (grid_res.y - 1u32) as f32,
        (grid_res.z - 1u32) as f32,
    );

    // Klemmen (Clamp), damit wir nicht außerhalb des Tensors lesen
    let max_bound = Line::<f32, 3>::new(
        (grid_res.x - 2u32) as f32,
        (grid_res.y - 2u32) as f32,
        (grid_res.z - 2u32) as f32,
    );
    let clamped_pos = Line::<f32, 3>::max(
        Line::<f32, 3>::to_line(0.0),
        Line::<f32, 3>::min(local_pos, max_bound)
    );

    // 2. Bestimme die Basis-Indizes (abgerundet)
    let x0 = clamped_pos.x as u32;
    let y0 = clamped_pos.y as u32;
    let z0 = clamped_pos.z as u32;

    let x1 = x0 + 1u32;
    let y1 = y0 + 1u32;
    let z1 = z0 + 1u32;

    // 3. Berechne die Interpolations-Gewichte (Nachkommastellen)
    let tx = clamped_pos.x - (x0 as f32);
    let ty = clamped_pos.y - (y0 as f32);
    let tz = clamped_pos.z - (z0 as f32);

    // Strunktur-Strides für das flache 3D-Array berechnen
    let stride_y = grid_res.x;
    let stride_z = grid_res.x * grid_res.y;

    // Helper-Inlining für flachen Index-Lookup
    // index = x + y * stride_y + z * stride_z
    let c000 = grid_tensor[x0 + y0 * stride_y + z0 * stride_z];
    let c100 = grid_tensor[x1 + y0 * stride_y + z0 * stride_z];
    let c010 = grid_tensor[x0 + y1 * stride_y + z0 * stride_z];
    let c110 = grid_tensor[x1 + y1 * stride_y + z0 * stride_z];
    let c001 = grid_tensor[x0 + y0 * stride_y + z1 * stride_z];
    let c101 = grid_tensor[x1 + y0 * stride_y + z1 * stride_z];
    let c011 = grid_tensor[x0 + y1 * stride_y + z1 * stride_z];
    let c111 = grid_tensor[x1 + y1 * stride_y + z1 * stride_z];

    // 4. TRILINEARE INTERPOLATION
    // Schritt A: Entlang der X-Achse interpolieren (4 Werte übrig)
    let nx00 = mix(c000, c100, tx);
    let nx10 = mix(c010, c110, tx);
    let nx01 = mix(c001, c101, tx);
    let nx11 = mix(c011, c111, tx);

    // Schritt B: Entlang der Y-Achse interpolieren (2 Werte übrig)
    let ny0 = mix(nx00, nx10, ty);
    let ny1 = mix(nx01, nx11, ty);

    // Schritt C: Entlang der Z-Achse interpolieren (Finaler Distanzwert!)
    mix(ny0, ny1, tz)
}
```

---

## 3. Integration in deine Szenen-Map

In deiner `scene_map` kannst du nun dieses organische Grid mit deinen mathematischen Primitiven (wie den Tempelsäulen) via `op_union` kombinieren:

```rust
#[cube]
pub fn scene_map(
    p: Line<f32, 3>, 
    params: &RaymarchParams, 
    sdf_grid: &Tensor<f32>
) -> f32 {
    // 1. Mathematische Säulen (Analytische Geometrie)
    let pillar_spacing = params.arch.pillar_count as f32 * 2.0; 
    let local_p = op_repetition_xz(p, pillar_spacing);
    let pillar_bounds = Line::<f32, 3>::new(1.0, params.arch.temple_height, 1.0);
    let d_pillars = sdf_box(local_p, pillar_bounds);

    // 2. Organische Geometrie aus dem Tensor-Grid lesen
    // Wir nehmen an, das Gitter ist 64x64x64 groß und misst 8x8x8 Einheiten im Raum
    let grid_res = Line::<u32, 3>::new(64, 64, 64);
    let grid_size = Line::<f32, 3>::new(8.0, 8.0, 8.0);
    
    // Wir verschieben die Statue etwas nach oben ins Zentrum des Tempels
    let statue_pos = p - Line::<f32, 3>::new(0.0, 4.0, 0.0);
    let d_statue = sample_sdf_grid(statue_pos, sdf_grid, grid_res, grid_size);

    // 3. Kombinieren: Die Statue steht im Säulentempel!
    op_union(d_pillars, d_statue)
}
```

---

## 4. Datenvorbereitung auf der CPU (Host-Seite)

Damit das reibungslos klappt, generierst du auf der CPU ein flaches `Vec<f32>` (z. B. durch das Parsen einer `.obj`-Datei oder einer mathematischen Noise-Funktion wie Perlin-Noise) und schickst es via `TensorArg` an die GPU:

```rust
// Beispiel: Ein 64^3 Grid mit Noise oder Distanzwerten füllen
let res = 64;
let mut cpu_grid = Vec::with_capacity(res * res * res);

for z in 0..res {
    for y in 0..res {
        for x in 0..res {
            // Hier würdest du die echte Distanz zur Oberfläche speichern.
            // Als simpler Platzhalter: Eine mathematische CPU-Kugel berechnen
            let dx = (x as f32 - 32.0) / 32.0;
            let dy = (y as f32 - 32.0) / 32.0;
            let dz = (z as f32 - 32.0) / 32.0;
            let dist = (dx*dx + dy*dy + dz*dz).sqrt() - 0.5;
            cpu_grid.push(dist);
        }
    }
}

// Tensor auf der GPU erstellen
let sdf_grid_tensor = client.create(cpu_grid.as_slice());
```

Möchtest du als Nächstes sehen, wie du Shader-Normals (Normalenvektoren) für diese kombinierte Szene berechnest? Da wir jetzt ein Tensor-Grid nutzen, gibt es einen Trick, wie man die Normalen direkt aus dem Gitter ableitet, ohne rechenintensive numerische Derivate (_Finite Differences_) für das gesamte Weltall auszuführen.