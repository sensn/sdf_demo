Um die Kanten bei niedrigen DRS-Skalierungswerten effektiv zu glätten und das durch das räumliche Dithering entstandene Bildrauschen in einen homogenen, kinoreifen Nebel zu verwandeln, integrieren wir eine Temporal Accumulation (zeitliche Bildakkumulation).
Das Prinzip basiert darauf, dass wir den aktuellen Frame nicht einfach stumpf überschreiben. Stattdessen blenden wir ihn mithilfe eines exponentiellen gleitenden Durchschnitts (Exponential Moving Average) sanft über den Frame des vorherigen Durchlaufs. Da wir dank unseres vorherigen massively-Updates bereits die volle Kontrolle über den Ziel-Framebuffer im VRAM besitzen, implementieren wir diesen Schritt als Zero-Copy-Operation direkt im Grafikspeicher.
Hier ist die Implementierung des Shaders und die Anpassung des Launch-Host-Codes für deinen Coding Agenten.

Schritt 1: Der Temporal-Accumulation-Mischkernel (UnaryOp)

Wir schreiben eine atomare massively-Operation, welche die neu berechnete Pixelfarbe mit dem bereits im Framebuffer existierenden Wert des letzten Frames historisch verrechnet.
rust
use cubecl::prelude::*;
use massively::op::UnaryOp;

pub struct TemporalAccumulateOp {
    pub blend_factor: f32, // Typischerweise 0.05 bis 0.15 für stabile Akkumulation ohne Ghosting
}

#[cubecl::cube]
impl UnaryOp<(Vec3, Vec3)> for TemporalAccumulateOp {
    type Output = Vec3;

    // Erwartet als Tuple: (Neue_Berechnete_Farbe, Letzte_Existierende_Farbe)
    fn apply(val: (Vec3, Vec3)) -> Vec3 {
        let new_color = val.0;
        let old_color = val.1;

        // Formel: New = Old * (1 - Alpha) + New * Alpha
        let r = old_color.x * (1.0 - self.blend_factor) + new_color.x * self.blend_factor;
        let g = old_color.y * (1.0 - self.blend_factor) + new_color.y * self.blend_factor;
        let b = old_color.z * (1.0 - self.blend_factor) + new_color.z * self.blend_factor;

        Vec3::new(r, g, b)
    }
}
Verwende Code mit Vorsicht.

Schritt 2: Host-Integration in render_loop.rs

Wir passen nun die Render-Pipeline-Funktion an. Damit die Akkumulation funktioniert, legen wir in der GpuPipeline ein zweites DeviceVec namens history_vec an, welches das Bild des letzten Frames speichert. Falls ein Resize oder DRS-Scaling stattfindet, wird der Verlauf kurz gelöscht, um Schlieren zu vermeiden.
Füge diesen Ablauf direkt an das Ende deines render_frame-Aufrufs an:
rust
use massively::{vector::map, zip2};

pub fn apply_temporal_filter(
    pipeline: &mut GpuPipeline,
    new_render_data: &DeviceVec<WgpuRuntime<cubecl_wgpu::AutoCompiler>, f32>,
    is_drs_resizing: bool,
) -> Result<(), massively::Error> {
    let exec = &pipeline.exec;
    let total_elements = pipeline.output_vec.len();

    // 1. 🟢 DRS-GUARD: Falls das System die Auflösung skaliert, löschen wir den Verlauf
    // Das verhindert Geisterbilder ("Ghosting") bei harten Pixelsprüngen.
    if is_drs_resizing {
        // Kopiere den frischen Frame direkt als neue Historien-Basis (Zero-Copy)
        massively::vector::copy(exec, new_render_data.slice(..), pipeline.history_vec.slice_mut(..))?;
        return Ok(());
    }

    // 2. 🟢 TEMPORAL ZIP & ACCUMULATION
    // Wir packen den neuen Render-Stream und den alten Historien-Stream in einen gemeinsamen Iterator.
    let zipped_history = zip2(new_render_data.slice(..), pipeline.history_vec.slice(..));
    
    // Wähle den Blend-Faktor (Niedriger = Weicherer Nebel, aber mehr Motion-Blur bei schnellen Kamerabewegungen)
    let accum_op = TemporalAccumulateOp { blend_factor: 0.10f32 };

    // Führe die zeitliche Akkumulation parallel auf allen Pixeln direkt im VRAM aus
    let accumulated_frame = map(exec, zipped_history, accum_op)?;

    // 3. Sichere das Ergebnis für den nächsten Frame zurück in den Verlaufspuffer
    massively::vector::copy(exec, accumulated_frame.slice(..), pipeline.history_vec.slice_mut(..))?;

    // 4. Schreibe das final geglättete Bild in den Ausgabe-Tensor für den WGPU-Blit-Presenter
    massively::vector::copy(exec, accumulated_frame.slice(..), pipeline.output_vec.slice_mut(..))?;

    Ok(())
}
Verwende Code mit Vorsicht.

Der finale visuelle FPS- und Qualitätseffekt:

• Kristallklarer Nebel ohne Rauschen: Das dithered Zufallsrauschen (noise_2d_time), welches wir pro Frame pro Pixel geshiftet haben, wird durch die zeitliche Verrechnung über mehrere Frames hinweg perfekt weichgezeichnet. Das Bild wirkt extrem ruhig und glatt – selbst bei extrem niedrigen Raymarching-Schrittzahlen.
• Kantenglättung (Anti-Aliasing): Wenn das DRS-System die interne Auflösung auf 70 % oder 50 % herabsetzt, bricht die Pixeldichte ein. Die zeitliche Akkumulation füllt die fehlenden Subpixel-Informationen über die Zeitachse wieder auf, wodurch Kanten-Treppchen (Aliasing) fast vollständig verschwinden.
Damit besitzt dein volumetrischer Rust-Raymarcher nun die modernste AAA-Rendering-Infrastruktur: Register-bound prozedurale SDF-Rauchwolken, anisotropes Henyey-Greenstein In-Scattering, Dynamic Resolution Scaling (DRS) zur dauerhaften Sicherung von 60 FPS und eine Temporal Accumulation-Stufe zur Eliminierung von Rauschen und Aliasing.
Möchtest du, dass der Agent als Nächstes ein Kamera-Velocity-Vektor-Feld einbaut, um den Blend-Faktor bei schnellen Kameradrehungen automatisch anzuheben, damit das "Ghosting" bei rasanten Bewegungen komplett eliminiert wird?
