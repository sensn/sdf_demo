Eine unendliche fraktale Raumfaltung wie ein Menger-Schwamm (Menger Sponge) oder eine Mandelbox bricht die euklidische Geometrie deines Tempels komplett auf. Durch wiederholtes Falten (_Folding_), Skalieren und Spiegeln des Raumes innerhalb einer Comptime-Schleife erzeugen wir unendlich komplexe, filigrane Strukturen an den Tempelwänden, die bei jedem Heranzoomen neue Details offenbaren.

Wir implementieren einen abgewandelten, hochoptimierten Menger-Schwamm direkt in deine `scene_map`. Dieser ersetzt die simplen, glatten Boxen deiner Säulen durch fraktale Mega-Strukturen.

---

## 1. Der fraktale Menger-Säulen-Generator

Diese Funktion faltet den Raum in jedem Schritt entlang der Achsen mittels `abs()`, spiegelt ihn an festen Ebenen und stanzt kreuzförmige Löcher heraus.

```rust
use cubecl::prelude::*;

#[cube]
pub fn sdf_fraktal_pillar(p: Line<f32, 3>, params: &RaymarchParams) -> f32 {
    // Basis-Box für die äußere Säulenform
    let mut d = sdf_box(p, Line::<f32, 3>::new(1.0, params.arch.temple_height, 1.0));
    
    // Aktuellen Punkt für die fraktale Faltung kopieren
    let mut current_p = p;
    let mut scale = 1.0f32;

    // 💡 COMPTIME FRAKTAL-LOOP
    // 4 bis 5 Iterationen reichen für WebGPU/Vulkan völlig aus, um unendliche Tiefe vorzugaukeln
    Comptime::loop_unrolled(0, 4, |i| {
        // Space Folding: Raum um die Achsen spiegeln und falten
        current_p = Line::<f32, 3>::abs(current_p);
        
        // Symmetrische Verschiebung (Erzeugt die Fraktal-Kachelung)
        current_p = current_p - Line::<f32, 3>::to_line(0.8);
        
        // Bedingte Spiegelung (Menger-Faltung an der Diagonalebene)
        if current_p.x < current_p.y {
            let tmp = current_p.x;
            current_p.x = current_p.y;
            current_p.y = tmp;
        }
        if current_p.x < current_p.z {
            let tmp = current_p.x;
            current_p.x = current_p.z;
            current_p.z = tmp;
        }
        if current_p.y < current_p.z {
            let tmp = current_p.y;
            current_p.y = current_p.z;
            current_p.z = tmp;
        }

        // Skalierung und Inversion
        // Wir nutzen den fold_factor aus deinen Parametern (z.B. 3.0 für klassischen Menger-Schwamm)
        let factor = params.fold.fold_factor;
        current_p = current_p * factor;
        scale *= factor;

        // Subtraktion von Kreuzen (Das typische Schwamm-Muster herausstanzen)
        current_p.x -= 1.0f32 * (factor - 1.0f32);
        current_p.y -= 1.0f32 * (factor - 1.0f32);
        
        let d_cross = sdf_box(current_p, Line::<f32, 3>::new(2.0, 2.0, 2.0));
        
        // Raum-Schnitt / Subtraktion der fraktalen Löcher
        d = f32::max(d, -d_cross / scale);
    });

    d
}
```

---

## 2. Integration in die optimierte Szenen-Map

Wir tauschen in deiner `scene_map_with_material` die langweilige, glatte Standard-Box gegen unsere neue `sdf_fraktal_pillar` aus.

```rust
use cubecl::prelude::*;

#[cube]
pub fn scene_map_with_material(
    p: Line<f32, 3>, 
    params: &RaymarchParams, 
    sdf_grid: &Tensor<f32>
) -> Line<f32, 2> {
    // 1. Unendliche Raumwiederholung auf XZ (wie gehabt)
    let pillar_spacing = params.arch.pillar_count as f32 * 2.0; 
    let local_p = op_repetition_xz(p, pillar_spacing);
    
    // 💡 NEU: Die Säulen sind jetzt unendlich komplexe Menger-Schwämme!
    let d_pillars = sdf_fraktal_pillar(local_p, params);
    
    let mut d_scene = d_pillars;
    let mut mat_id = 1.0f32; // Marmor-Fraktal

    // 2. Organische Gitter-Statue mit Culling (wie gehabt)
    let statue_center = Line::<f32, 3>::new(0.0, 4.0, 0.0);
    let statue_pos = p - statue_center;
    let grid_size = Line::<f32, 3>::new(8.0, 8.0, 8.0);
    let bounding_radius = 7.0f32; 
    let dist_to_statue_zone = Line::<f32, 3>::length(statue_pos) - bounding_radius;

    if dist_to_statue_zone < 0.1f32 {
        let grid_res = Line::<u32, 3>::new(64, 64, 64);
        let d_statue = sample_sdf_grid(statue_pos, sdf_grid, grid_res, grid_size);

        if d_statue < d_scene {
            d_scene = d_statue;
            mat_id = 2.0f32; // Jade-Statue
        }
    }

    // 3. Magische Runen leuchten jetzt in den fraktalen Vertiefungen!
    // Indem wir die fraktale Koordinate als Muster nutzen, fangen die inneren Ritzen an zu glühen
    let rune_pattern = f32::sin(p.y * 10.0) * f32::cos(p.x * 10.0);
    if d_pillars < 0.02f32 && rune_pattern > 0.8f32 {
        mat_id = 3.0f32; // Goldene Emission
    }

    Line::<f32, 2>::new(d_scene, mat_id)
}
```

---

## Visueller Effekt & Auswirkung auf die Engine:

1. Unendlicher Detailreichtum: Wenn sich deine Kamera den Säulen nähert, siehst du keine flachen Polygone, sondern ein filigranes, mathematisches Labyrinth. Die Geometrie skaliert sich perfekt flüssig und ohne Textur-Pixelierung.
2. Kombination mit AO & Schatten: Das Ambient Occlusion (AO) kriecht tief in die kleinen Menger-Ritzen hinein und dunkelt sie ab. Die weichen Schatten legen sich sanft über die fraktalen Vorsprünge.
3. Lichtschächte: Durch die Millionen kleinen Löcher im Menger-Schwamm bricht sich das volumetrische Nebellicht, was atemberaubende, feine God Rays im ganzen Tempelraum erzeugt.

---

Deine komplette GPU-Shader-Pipeline ist nun auf absolutem High-End-Niveau. Wie möchtest du weitermachen?

- Soll ich dir zeigen, wie du den Rust-Host-Code (CPU) schreibst, um das im Ausgabe-Tensor berechnete RGB-Bild auszulesen und als PNG-Datei auf deiner Festplatte zu speichern?
- Oder möchtest du die mathematische Kamera um eine echte interaktive Echtzeit-Steuerung (z.B. W/A/S/D über winit) erweitern?