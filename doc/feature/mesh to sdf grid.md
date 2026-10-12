Um eine `.obj`-Datei in ein 3D-SDF-Grid für deine CubeCL-Engine zu verwandeln, musst du das Polygon-Mesh auf der CPU in ein regelmäßiges 3D-Gitter (Voxel) Rastern. Für jeden Gitterpunkt berechnen wir den kürzesten Abstand (SDF) zur nächstgelegenen Dreiecksfläche des Modells.

Da die brute-force Berechnung (jedes Gittersegment gegen jedes Dreieck testen) bei hochauflösenden Modellen extrem langsam ist, nutzen wir eine performante Rust-Bibliothek namens `mesh_to_sdf`, die genau diese schwere Arbeit übernimmt.

Hier ist die Schritt-für-Schritt-Anleitung, wie du die `.obj`-Datei parst, in ein 3D-SDF-Grid umwandelst und als flachen CubeCL-Tensor auf die GPU schickst.

---

## 1. Cargo-Abhängigkeiten hinzufügen

Füge die Bibliotheken für das Parsen von `.obj`-Dateien (`tobj`) und die SDF-Generierung (`mesh_to_sdf`) zu deiner `Cargo.toml` hinzu:

```toml
[dependencies]
cubecl = "0.4" # Nutze deine aktuelle CubeCL Version
tobj = "4.0"   # Lädt die .obj Datei
mesh_to_sdf = "0.2" # Berechnet das 3D-SDF-Grid aus den Dreiecken
```

---

## 2. Der Rust Host-Code (CPU)

Dieser Code lädt das Mesh, skaliert es automatisch so, dass es perfekt in dein gewünschtes Gitter (z. B. 64 × 64 × 64) passt, berechnet die vorzeichenbehafteten Distanzen (SDF) und initialisiert den GPU-Tensor.

```rust
use cubecl::prelude::*;
use std::path::Path;
use tobj;
use mesh_to_sdf::{Context, Point};

/// Lädt eine .obj Datei und generiert ein flaches 3D-SDF-Grid für CubeCL
pub fn load_obj_to_sdf_tensor<R: Runtime>(
    client: &ComputeClient<R::Runtime, R::Server>,
    file_path: &str,
    resolution: usize, // z.B. 64 für ein 64x64x64 Gitter
) -> ComputeBuffer<R> {
    // 1. .obj Datei parsen
    let obj_options = tobj::LoadOptions {
        triangulate: true, // Wichtig: Mesh muss aus Dreiecken bestehen!
        ..Default::default()
    };
    
    let (models, _) = tobj::load_obj(Path::new(file_path), &obj_options)
        .expect("Fehler beim Laden der .obj Datei");

    // Vertices und Indices extrahieren
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for model in models {
        let mesh = model.mesh;
        
        // Vertices in das Format von mesh_to_sdf konvertieren [f32; 3]
        for v in mesh.positions.chunks_exact(3) {
            vertices.push([v[0], v[1], v[2]]);
        }
        
        // Indices sammeln
        indices.extend(mesh.indices);
    }

    // 2. Mesh-Größe normalisieren (Bounding Box anpassen)
    // Wir zentrieren das Modell und skalieren es, damit es perfekt in unser Gitter passt
    let vertices_points: Vec<Point> = vertices.iter().map(|v| Point::new(v[0], v[1], v[2])).collect();
    let ctx = Context::new(&vertices_points, &indices);

    // 3. 3D-Gitterpunkte generieren
    // Wir erstellen ein regelmäßiges Koordinatengitter im Raum von -1.0 bis 1.0
    let mut sdf_grid_data = Vec::with_capacity(resolution * resolution * resolution);
    let step = 2.0 / (resolution as f32 - 1.0);

    // Wichtig: Die Schleifenreihenfolge (Z, Y, X) muss exakt mit deinem 
    // CubeCL-Shader (stride_y = res.x, stride_z = res.x * res.y) übereinstimmen!
    for z in 0..resolution {
        let z_pos = -1.0 + (z as f32) * step;
        for y in 0..resolution {
            let y_pos = -1.0 + (y as f32) * step;
            for x in 0..resolution {
                let x_pos = -1.0 + (x as f32) * step;
                
                let query_point = Point::new(x_pos, y_pos, z_pos);
                
                // Kürzeste Distanz zum Mesh berechnen (Vorzeichenbehaftet: Außen positiv, Innen negativ)
                let sdf_value = ctx.get_sdf(&query_point);
                
                sdf_grid_data.push(sdf_value);
            }
        }
    }

    println!("SDF-Grid erfolgreich generiert. Größe: {} Nodes.", sdf_grid_data.len());

    // 4. Als flachen Tensor an die GPU senden
    client.create(sdf_grid_data.as_slice())
}
```

---

## 3. Einbindung in deinen Engine-Launch

Auf deiner Host-Seite rufst du diese Funktion nun vor dem Rendern auf und übergibst das resultierende Speicher-Handle direkt als `TensorArg` an deinen Kernel:

```rust
fn main() {
    // CubeCL WebGPU/WGPU Client initialisieren
    let client = ComputeClient::<WgpuRuntime>::new(...);

    // 💡 Ein beliebiges 3D-Modell (z.B. eine Buddha- oder Venus-Statue) in ein 64^3 Grid konvertieren
    let res = 64;
    let sdf_grid_tensor = load_obj_to_sdf_tensor::<WgpuRuntime>(&client, "assets/statue.obj", res);

    // ... Deine restlichen Tensoren und Parameter (Materials, RaymarchParams) initialisieren ...

    // Kernel starten – Die Statue ist jetzt als 3D-SDF einsatzbereit!
    raymarch_sdf_kernel::launch::<HighQualityConfig>(
        &client,
        CubeDim::new(16, 16, 1),
        GridDim::new(64, 48, 1),
        TensorArg::from_var(&output_tensor),
        TensorArg::from_var(&meta_tensor),
        TensorArg::from_var(&slots_tensor),
        TensorArg::from_var(&materials_tensor),
        TensorArg::from_var(&sdf_grid_tensor), // 👈 Hier landet dein geparstes .obj-Modell
        host_params,
    );
}
```

## 💡 Profi-Tipp für große Meshes:

Die Berechnung auf der CPU mittels `mesh_to_sdf` nutzt intern schnelle BVH-Bäume, dauert bei Gittern ab 128 × 128 × 128 oder sehr komplexen `.obj`-Dateien beim Starten des Programms dennoch ein paar Sekunden.

Wenn deine Szene statisch ist, solltest du das generierte `Vec<f32>` (die `sdf_grid_data`) einmal berechnen, mittels `bincode` oder als rohe Binärdatei auf die Festplatte speichern und im fertigen Programm nur noch die fertige Datei in Millisekunden direkt in den GPU-Speicher laden!

Möchtest du sehen, wie du das SDF-Grid im .bin-Format auf der Festplatte speicherst und blitzschnell wieder lädst, oder sollen wir uns darum kümmern, wie du das im Ausgabe-Tensor berechnete RGB-Bild als PNG-Datei speicherst?