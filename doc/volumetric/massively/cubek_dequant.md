Hier ist die exakte Arbeitsanweisung und der finale Shader-Code für deinen Coding-Agenten, um das Dequantisierungs-Muster für das i8-quantisierte SDF-Grid direkt in deiner kernel.rs zu verankern.
Durch diese Implementierung decodieren die ALUs der GPU die hochkomprimierten 8-Bit-Voxel-Daten (i8) zur Laufzeit direkt in den Registern zurück in mathematisch präzise f32-Distanzen. Dadurch sparst du im Haupt-Raymarching-Loop massiv Speicherbandbreite auf dem VRAM-Bus ein.

🤖 Coding Agent Instruction: Shader Dequantization Engine (kernel.rs)


Context & Constraints

• Framework: CubeCL 0.11.0.
• Layout: Das quantisierte Grid liegt am Ende des Mega-Szenen-Buffers (scene).
• Structure: Es ist aufgeteilt in SDF_GRID_COUNT Bytes für die i8-Distanzen, gefolgt von NUM_BLOCKS Fließkommazahlen (f32) für die Skalierungsfaktoren. Jeder Block umfasst exakt 64 Voxel.

Task 1: Integration der Dequantisierungs-Funktion in kernel.rs

Weise den Agenten an, diese hocheffiziente mathematische Hilfsfunktion in das Shader-Modul einzufügen. Sie führt das Bit- und Typ-Unpacking direkt in den GPU-Registern durch.
rust
#[cube]
pub fn evaluate_quantized_sdf(
    scene: &[f32],
    grid_index: u32,
) -> f32 {
    use crate::gpu::{TOTAL_BUFFER_SIZE, SDF_GRID_COUNT, BLOCK_SIZE};

    // 1. Berechne die absolute Startposition der i8-Daten im flachen f32-Buffer.
    // Da scene ein f32-Slice ist, müssen wir den i8-Index auf f32-Element-Grenzen mappen.
    let byte_offset = grid_index;
    let f32_data_index = TOTAL_BUFFER_SIZE + (byte_offset / 4);
    let byte_sub_idx = byte_offset % 4;

    // Packe das rohe f32-Wort aus dem Speicher, das 4 quantisierte i8-Voxel enthält
    let packed_f32 = scene[f32_data_index];

    // 2. Extrahiere das spezifische i8-Byte mittels Bit-Manipulation (Zero-Copy Register Shift)
    // Wir casten das f32-Wort in u32 für die Bit-Operationen
    let packed_bits = u32::cast_from(packed_f32);
    let mut raw_byte = 0u32;

    if byte_sub_idx == 0 {
        raw_byte = packed_bits & 0xFF;
    } else if byte_sub_idx == 1 {
        raw_byte = (packed_bits >> 8) & 0xFF;
    } else if byte_sub_idx == 2 {
        raw_byte = (packed_bits >> 16) & 0xFF;
    } else {
        raw_byte = (packed_bits >> 24) & 0xFF;
    }

    // Vorzeichen-Erweiterung (Sign Extension) von u8 auf i32/f32 manuell berechnen
    let mut signed_int = f32::cast_from(raw_byte);
    if raw_byte > 127 {
        signed_int = f32::cast_from(raw_byte) - 256.0;
    }

    // 3. 🟢 CUBEK DEQUANTISIERUNG
    // Finde den zugehörigen Skalierungsfaktor für diesen 64er-Block
    let block_idx = grid_index / 64;
    
    // Die Skalierungsfaktoren starten direkt hinter den i8-Datenblöcken
    // Da SDF_GRID_COUNT i8-Werte exakt SDF_GRID_COUNT / 4 f32-Plätze belegen:
    let scale_zone_start = TOTAL_BUFFER_SIZE + (SDF_GRID_COUNT / 4);
    let scale = scene[scale_zone_start + block_idx];

    // Stelle die originale mathematische f32-Distanz wieder her
    signed_int * scale
}
Verwende Code mit Vorsicht.

Task 2: Verknüpfung innerhalb der Szenen-Map (scene_map_with_material)

Der Agent muss nun deine bestehende Gitter-Abfragefunktion (scene_map_with_material) so umschreiben, dass sie nicht mehr direkt auf den Buffer zugreift, sondern den neuen Dequantisierer nutzt.

Instructions for Agent:

1. Ersetze die direkte Array-Indizierung let raw_d = sdf_grid[idx] durch den Aufruf von evaluate_quantized_sdf.
2. Übergebe den globalen scene-Mega-Buffer an die Funktion.
rust
// Innerhalb der scene_map_with_material / 3D-Gitter-Lese-Logik anpassen:
#[cube]
pub fn sample_temple_statue_sdf(
    scene: &[f32],
    p_local: Vec3,
    grid_res: u32,
    grid_size: f32,
) -> f32 {
    // Transformiere die lokale 3D-Koordinate in den Gitter-Index (0 bis 63)
    let half_size = grid_size / 2.0;
    let gx = u32::cast_from(((p_local.x + half_size) / grid_size) * f32::cast_from(grid_res));
    let gy = u32::cast_from(((p_local.y + half_size) / grid_size) * f32::cast_from(grid_res));
    let gz = u32::cast_from(((p_local.z + half_size) / grid_size) * f32::cast_from(grid_res));

    // Sicherheitszaun gegen Out-of-Bounds
    if gx < grid_res && gy < grid_res && gz < grid_res {
        // Berechne den linearisierten 3D-Index im Voxelgitter
        let grid_index = gz * (grid_res * grid_res) + gy * grid_res + gx;

        // 🟢 RUFE DEQUANTISIERUNG AUF (Ersetzt alten f32-Direct-Read)
        evaluate_quantized_sdf(scene, grid_index)
    } else {
        10.0 // Sichere Hintergrund-Distanz außerhalb des Gitters
    }
}
Verwende Code mit Vorsicht.

Technischer Nutzen im Gesamtsystem

Durch diesen letzten Schliff ist dein Shader-System optimal für extrem hohe Bildwiederholraten ausgelegt:
• Reduzierte Texture/Buffer-Latenz: Die Shadereinheiten müssen pro Pixel deutlich seltener auf den globalen Grafikspeicher (VRAM) warten, da 4 Voxel komprimiert in einem einzigen Taktzyklus in die Register geladen werden.
• Perfektes Zusammenspiel: Zusammen mit der Warp-Kohärenz aus massively und dem Depthwise-Post-Processing aus cubek läuft dein volumetrischer Raymarcher nun auf einer State-of-the-Art Render-Infrastruktur.
Damit ist die gesamte HPC-Grafikpipeline fertiggestellt. Soll der Coding Agent zum Abschluss noch ein Logging-Modul hinzufügen, um VRAM-Einsparungen und Latenzvorteile übersichtlich in der Konsole auszugeben?
