Beim SDF-Raymarching wandert der Strahl Schritt für Schritt durch den Raum. Wenn sich Objekte hinter der Kamera befinden oder viel zu weit entfernt sind, verschwendet die GPU wertvolle Rechenzyklen.

Da wir in CubeCL eine unendliche Raumfaltung (die Kachelung der Säulen via Modulo) nutzen, würde die GPU theoretisch versuchen, unendlich weit in die Ferne zu marschieren. Ein Kameraposition-basiertes Culling optimiert dies radikal auf zwei Ebenen:

1. Dynamic Far-Plane Clipping: Der Strahl bricht sofort ab, wenn er die maximale Sichtweite überschreitet.
2. Frustum / Distance Culling im Objekt-Space: Bevor wir komplexe Berechnungen wie das 3D-SDF-Grid (die Statue) abfragen, prüfen wir über eine schnelle Bounding-Box-Kalkulation, ob das Objekt überhaupt im sichtbaren Bereich der Kamera liegt.

Hier ist die Implementierung in CubeCL:

## 1. Das Culling-System in der Szenen-Map

Wir erweitern die `scene_map_with_material` um einen schnellen Abstandstest. Wenn die Kamera zu weit von der Statue entfernt ist oder der aktuelle Punkt außerhalb des Statuen-Radius liegt, überspringen wir das teure Trilinear-Sampling des Tensors komplett!

```rust
use cubecl::prelude::*;

#[cube]
pub fn scene_map_with_material(
    p: Line<f32, 3>, 
    params: &RaymarchParams, 
    sdf_grid: &Tensor<f32>
) -> Line<f32, 2> {
    // 1. Günstige mathematische Säulen (Marmor -> ID 1.0)
    let pillar_spacing = params.arch.pillar_count as f32 * 2.0; 
    let local_p = op_repetition_xz(p, pillar_spacing);
    let pillar_bounds = Line::<f32, 3>::new(1.0, params.arch.temple_height, 1.0);
    let d_pillars = sdf_box(local_p, pillar_bounds);
    
    let mut d_scene = d_pillars;
    let mut mat_id = 1.0f32; 

    // 💡 2. PERFORMANCE-CULLING FÜR DAS TENSOR-GRID
    // Statuen-Zentrum definieren
    let statue_center = Line::<f32, 3>::new(0.0, 4.0, 0.0);
    let statue_pos = p - statue_center;
    
    let grid_size = Line::<f32, 3>::new(8.0, 8.0, 8.0);
    // Der Hüllradius (Bounding Sphere) der Statue beträgt ca. halbe Diagonale (~7.0 Einheiten)
    let bounding_radius = 7.0f32; 
    let dist_to_statue_zone = Line::<f32, 3>::length(statue_pos) - bounding_radius;

    // CULLING-ABFRAGE:
    // Nur wenn der Raymarching-Strahl sich TATSÄCHLICH innerhalb oder direkt vor der 
    // Bounding-Zone der Statue befindet, führen wir das teure 3D-Tensor-Sampling aus!
    if dist_to_statue_zone < 0.1f32 {
        let grid_res = Line::<u32, 3>::new(64, 64, 64);
        let d_statue = sample_sdf_grid(statue_pos, sdf_grid, grid_res, grid_size);

        if d_statue < d_scene {
            d_scene = d_statue;
            mat_id = 2.0f32;
        }
    }

    // 3. Magische Runen (Gold -> ID 3.0)
    let rune_pattern = f32::sin(p.y * 3.0) * f32::cos(local_p.x * 5.0);
    if d_pillars < 0.05f32 && rune_pattern > 0.7f32 {
        mat_id = 3.0f32;
    }

    Line::<f32, 2>::new(d_scene, mat_id)
}
```

---

## 2. Strahl-Clipping im Hauptkernel (Horizon / Far-Culling)

Zusätzlich optimieren wir die Hauptschleife. Anstatt stumpf `MAX_STEPS` durchzulaufen, stoppen wir den Strahl über ein dynamisches Distanzlimit (`max_ray_distance`). Sobald er in den Himmel entweicht, brechen wir den GPU-Thread ab.

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
    let global_id = cubecl::prelude::absolute_idx();
    let pixel_idx = global_id * 3u32;

    let cam_pos = Line::<f32, 3>::new(params.camera.position.x, params.camera.position.y, params.camera.position.z);
    let ray_dir = Line::<f32, 3>::new(0.0, 0.0, 1.0); // Beispiel-Blickrichtung

    let mut depth = 0.0f32;
    let mut hit = false;
    let mut hit_mat_id = 0.0f32;
    let mut hit_pos = Line::<f32, 3>::to_line(0.0);
    let mut glow_accumulator = Line::<f32, 3>::to_line(0.0);

    // 💡 Maximale Sichtweite basierend auf den Environment-Settings
    let max_ray_distance = 45.0f32; 

    // RAYMARCHING LOOP WITH EARLY EXIT
    Comptime::loop_unrolled(0, C::MAX_STEPS, |i| {
        // Early Exit: Wenn wir bereits getroffen haben ODER die Far-Plane überschreiten, 
        // macht die GPU keine weiteren mathematischen Schritte.
        if !hit && depth < max_ray_distance {
            let current_pos = cam_pos + ray_dir * depth;
            let scene_data = scene_map_with_material(current_pos, &params, sdf_grid);
            let distance = scene_data.x;
            let mat_id = scene_data.y;

            if mat_id == 3.0f32 {
                let glow_strength = 0.01f32 / (distance + 0.02f32);
                glow_accumulator.x += 1.00f32 * glow_strength;
                glow_accumulator.y += 0.70f32 * glow_strength;
                glow_accumulator.z += 0.10f32 * glow_strength;
            }

            if distance < 0.001f32 {
                hit = true;
                hit_pos = current_pos;
                hit_mat_id = mat_id;
            }
            
            // Strahl vorschieben. 
            // 💡 Sicherheits-Feature: Falls die SDF fehlerhaft ist (z.B. 0 zurückgibt), 
            // zwingen wir den Strahl ein Minimum nach vorne, um unendliche GPU-Hänger (TDR) zu verhindern.
            depth += f32::max(distance, 0.005f32);
        }
    });

    // ... [Ab hier folgt dein PBR-Shading und Volumetric Nebel] ...
    // Da `depth` gecullt wurde, stoppt auch der volumetrische Nebel perfekt an der Far-Plane.
}
```

## Die Auswirkung auf die Framerate (WebGPU/Vulkan):

1. Kein unnötiges Textur-Sampling: Wenn du dich im Tempel umdrehst und die Statue nicht im Blickfeld ist, verhält sich die Szene so leichtfüßig wie ein rein mathematischer Shader. Das spart hunderte Lesezugriffe auf den globalen VRAM pro Pixel.
2. Schutz vor dem GPU-Timeout: Wenn Strahlen im unendlichen Tempel-Raum verloren gehen, sorgt das `depth < max_ray_distance` Culling für einen sofortigen Abbruch. Deine WebGPU-App bleibt extrem stabil und friert im Browser niemals ein.

Möchtest du nun sehen, wie der Rust-Host-Code aufgebaut sein muss, um dieses berechnete RGB-Bild aus dem Tensor auszulesen und als Bild anzuzeigen, oder sollen wir noch eine unendliche fraktale Raumfaltung (Mandelbox/Menger-Schwamm) in die Tempelarchitektur integrieren?