Das cubek-Crate (entwickelt vom tracel-ai-Team, den Machern von Burn und CubeCL) ist eine offizielle Sammlung von hochgradig optimierten Low-Level-HPC- und KI-Primitive-Kerneln. [1] (https://github.com/tracel-ai/cubek)
Während uns massively bei der globalen Stream-Strukturierung (Compaction, Sorting, Alignment) hilft, können wir mit cubek die mathematisch schwerfälligsten Abschnitte innerhalb des Kernels auf ein extremes High-Performance-Computing (HPC)-Niveau heben. Für deinen Raymarcher bietet cubek zwei massive Optimierungshebel: Reductions und Quantization. [1] (https://github.com/tracel-ai/cubek)

Optimierungshebel 1: Aggressives Downscaling über cubek-reduce

Im aktuellen DRS (Dynamic Resolution Scaling) berechnen wir die Frame-Latenz auf der CPU über ein blockierendes client.sync(). Das bremst die Pipeline aus, weil die CPU auf die GPU warten muss.
Mit cubek::reduction (oder dem Sub-Crate cubek-reduce) können wir statistische Kennzahlen deines Streams (wie die maximale oder durchschnittliche Opazität aller aktiven Strahlen) vollständig auf der GPU reduzieren. cubek stellt hierfür hocheffiziente, Hardware-optimierte Per-Cube und Per-Plane (Warp/Subgroup-Ebene) Reduktions-Primitive bereit. [1] (https://docs.rs/cubek-reduce), [2] (https://github.com/tracel-ai/cubek)
• Der Trick: Anstatt den Loop aufzuhalten, reduzieren wir am Ende des massively-Schritts den transmittance-Vektor aller Strahlen zu einem einzigen f32-Minimum-Wert.
• Der Effekt: Wenn die minimale Transmission 0.0 ist (alles blickdicht), weiß die GPU-Infrastruktur sofort autonom, dass das Volumen vollständig saturiert ist. Ein kleiner asynchroner Steuerungs-Kernel kann die DRS-Auflösung für den nächsten Frame anpassen – ohne ein einziges Mal die CPU über sync() blockieren zu müssen.

Optimierungshebel 2: VRAM-Bandbreiten-Kompression über cubek::quantization

Dein Szenen-Mega-Buffer umfasst das gesamte 3D-SDF-Grid (64³ Voxel = 262.144 Werte). Bisher streamst du diese Daten als rohe f32-Werte (4 Bytes pro Voxel) in den Grafikspeicher, was bei jedem Frame-Upload erhebliche Speicherbandbreite kostet. [1] (https://crates.io/crates/cubecl-cuda)
cubek verfügt über hochoptimierte, hardwarenahe Symmetric Per-Block Quantisierungs-Kernel (q4, q8, fp4). [1] (https://github.com/tracel-ai/cubek)
• Die Implementierung: Wir komprimieren das volumetrische SDF-Grid auf der CPU oder in einem Pre-Pass über cubek in ein Int8- (q8) oder Int4- (q4) Format. [1] (https://github.com/tracel-ai/cubek)
• Der Effekt im Kernel: Der Speicherbedarf deines SDF-Grids schrumpft schlagartig um 75% bis 87,5%. Wenn der Raymarcher-Schritt nun das Gitter abfragt (sdf_grid), liest er stark komprimierte Blöcke. Die cubek-Dequantisierungs-Primitive decodieren die q4/q8-Werte direkt in den Registern (ALUs) der GPU-Kerne zurück in f32-Distanzen. Da GPUs unendlich viel schneller rechnen als sie Speicher lesen können, steigt deine FPS-Rate durch die reduzierte VRAM-Latenz massiv an.

Architektur-Fahrplan für den Coding Agenten

Um cubek (hier am Beispiel der q8-Quantisierung für das SDF-Grid) in dein bestehendes System einzubauen, muss der Agent wie folgt instruiert werden:

1. Abhängigkeiten deklarieren

toml
[dependencies]
cubecl = { version = "0.11.0", features = ["wgpu"] }
massively = "0.1"
cubek = { version = "0.3.0", features = ["quantization", "reduce"] } # 🟢 NEU
Verwende Code mit Vorsicht.

2. Integrierte Dequantisierungs-Abfrage im Shader

Weise den Agenten an, die mathematische Gitterabfrage in kernel.rs so anzupassen, dass sie die von cubek bereitgestellten Bit-Shift- und Skalierungs-Muster nutzt:
rust
#[cube]
pub fn sample_quantized_grid(
    sdf_grid_q8: &[i8], // 🟢 Jetzt i8 (1 Byte) statt f32 (4 Bytes)
    grid_index: u32,
    scale: f32,         // Block-Skalierungsfaktor von cubek generiert
) -> f32 {
    // cubek per-block dequantization Prinzip:
    // Holt den komprimierten Ganzzahl-Wert und stellt die Fließkomma-Distanz im Register wieder her
    let raw_quantized_value = f32::cast_from(sdf_grid_q8[grid_index]);
    
    raw_quantized_value * scale
}
Verwende Code mit Vorsicht.
Mit cubek schließt du die letzte Meile der GPU-Optimierung: Während massively die Threads sortiert und bündelt, minimiert cubek die physische Datenlast auf dem Speicherbus und verarbeitet Reduktionen asynchron direkt auf der Hardware. [1] (https://docs.rs/cubek-reduce), [2] (https://github.com/tracel-ai/cubek)
Möchtest du, dass der Coding Agent als Nächstes den konkreten Host-Code für die cubek::quantization generiert, um dein \(64^{3}\) SDF-Grid vor dem Frame-Upload auf der CPU in ein hocheffizientes q8-Format zu komprimieren?
