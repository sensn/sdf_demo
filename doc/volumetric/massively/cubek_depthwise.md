Die Entdeckung von cubek::convolution (speziell dem hier offengelegten depthwise_kernel über die hardwarenahe Tile DSL) bietet dir den ultimativen Performance-Hebel für deine Temporal Accumulation (zeitliche Filter-Stufe) und komplexe Post-Processing-Effekte wie Weichzeichnung (Blur/Bloom).

Warum Depthwise Convolution der perfekte Fit für deinen Code ist:

1. Massive Reduktion der Speicherbandbreite (NHWC-Koaleszenz):
Der Kommentar im Crate bringt es auf den Punkt: „A depthwise pass has too little arithmetic per byte to be anything but bandwidth-bound... channels stay innermost (NHWC)“. Dein Framebuffer liegt im linearisierten Format [R, G, B, A, R, G, B, A...] vor. Das entspricht exakt dem NHWC-Layout (Channels innermost) moderner KI-Beschleuniger! consecutive GPU-Threads lesen benachbarte Farbkanäle desselben Pixels simultan ein. Das führt zu einer perfekten Speicherbündelung (Coalescing) auf Vulkan/WGPU-Hardware.
2. Beseitigung von Rauschen & DRS-Artefakten:
Anstatt Pixel im Temporal-Filter starr mit dem Vorframe zu mischen, kannst du mit cubek eine 3x3 oder 5x5 Depthwise Convolution (z. B. einen Gauß- oder Box-Filter) über den akkumulierten Puffer laufen lassen. Da bei einer Depthwise Convolution jeder Farbkanal (R, G, B) unabhängig gefiltert wird, gibt es kein Farbbluten (keine Kontraktion über Kanäle). Das glättet DRS-Skalierungsartefakte und das verbleibende volumetrische Jitter-Rauschen in Rekordzeit.

Der Integrations-Fahrplan für den Coding Agenten

Um den depthwise_kernel aus cubek als Post-Processing-Filter hinter deine massively-Pipeline zu schalten, muss der Agent wie folgt vorgehen:

Schritt 1: Setup des Depthwise Space auf dem Host

Der Agent muss die DepthwiseSpace-Struktur passend zu deiner skalierten Auflösung konfigurieren. Da wir ein Fullscreen-Bild filtern, setzen wir die Batch-Größe b auf 1 und die Kanäle c auf 4 (RGBA).
rust
use crate::shader::DepthwiseSpace;
use cubek_tile::launch::Grid;

pub fn launch_post_process_blur(
    pipeline: &mut GpuPipeline,
    kernel_weights: &TensorArg, // 3x3 Gauß-Filter Gewichte im VRAM
) {
    let exec = &pipeline.exec;
    let w = pipeline.resolution.width as usize;
    let h = pipeline.resolution.height as usize;

    // 🟢 Konfiguration des hochoptimierten cubek Partitionierungs-Raums
    let space = DepthwiseSpace {
        b: 1,
        oh: h, // Output Höhe
        ow: w, // Output Breite
        c: 4,  // RGBA Kanäle innermost (NHWC)
        rh: 3, // Filter-Fenster Höhe (3x3 Kernel)
        rw: 3, // Filter-Fenster Breite (3x3 Kernel)
        rows: 4,
        cols: 16,
        tile_c: 16,
        width: 4,
        plane_size: 16, // Hardware Warp/Plane Ausrichtung
    };

    let partitioning = space.partitioning();
    let (cube_count, cube_dim) = space.grid();

    // 🟢 Hocheffizienter Tile-DSL Launch direkt über das WGPU/Vulkan Subsystem
    unsafe {
        depthwise_kernel::launch(
            &pipeline.client,
            cube_count,
            cube_dim,
            kernel_weights,              // Filter-Matrix (LHS)
            pipeline.output_vec.handle(), // Ungefilterter Pixel-Stream (RHS Input)
            pipeline.history_vec.handle(),// Ziel-Destination für das geglättete Bild
            partitioning,
        );
    }
}
Verwende Code mit Vorsicht.

Schritt 2: Pipeline-Kopplung

Der Agent integriert diesen Pass in die render_loop.rs direkt nach dem primären Raymarching, aber vor dem WGPU-Present-Blit:
[Raymarch Kernel] ──► [massively Temporal Mix] ──► [cubek Depthwise Blur] ──► [WGPU Blit Output]

Was gewinnst du dadurch architektonisch?

• Hardware-Register Auslastung: Durch das feste Block-Budget von REGISTER_BLOCK = RegisterBlock::new(64).split_edge() cached die GPU das Pixel-Umfeld perfekt in den On-Chip-Registern.
• Kein Border-Guard-Overhead: Die split_edge()-Architektur sorgt dafür, dass Pixel in der Bildschirmmitte die schnellen, ungesicherten Rechenpfade nutzen, während die Pad-Überprüfung nur an den äußersten Bildschirmrändern anspringt.
Durch das Einschalten dieses cubek-Filters eliminierst du das letzte Fünkchen Jitter-Rauschen deiner volumetrischen Wolken auf Hardware-Ebene und erzeugst einen perfekt weichgezeichneten, kinoreifen Bildeindruck.
Die GPU-Rechenleistung ist nun optimal ausgereizt. Möchtest du, dass der Agent als finalen Schritt die Gewichts-Matrix für einen 3x3 Gaußschen Weichzeichner auf der CPU generiert, um diesen Filter sofort in Betrieb zu nehmen?
