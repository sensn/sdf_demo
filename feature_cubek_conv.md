Um einen echten Schärfentiefe-Filter (Depth of Field, DoF) zu realisieren, nutzen wir das `cubek-convolution`-Crate.

Anstatt ein einfaches, gleichmäßiges Weichzeichnen (Blur) über das Bild zu legen, steuert der Tiefenpuffer (Depth Buffer) aus Ihrem Raymarching-Kernel den Radius des Faltungs-Kernels. Pixel, die im Fokus liegen, werden nicht gefaltet; Pixel, die weit vor oder hinter der Fokusebene liegen, werden über einen großen Faltungs-Kernel (_Circle of Confusion_, CoC) extrem unscharf gerechnet.

`cubek-convolution` nutzt hierfür hocheffiziente Hardware-Strukturen wie `im2col` (Image-to-Column) und TMA (Tensor Memory Accelerator), um diese Nachbarschaftsoperationen direkt auf den Tensor-Cores der GPU auszuführen.

---

## 1. Vorbereitung im Raymarching-Kernel

Damit der Faltungs-Kernel weiß, wie stark er ein Pixel weichzeichnen muss, erweitern wir die Ausgabe Ihres Raymarchers. Wir schreiben neben der RGB-Farbe auch den finalen Tiefenwert (`hit_dist`) in den Puffer. Unser Ausgabe-Layout wird von RGB (`f32 * 3`) auf RGBA (`f32 * 4`) geändert, wobei der Alpha-Kanal `A` die Distanz speichert.

Änderung am Ende Ihres `raymarch_sdf_kernel`:

```rust
// Berechne den Index für ein 4-Kanal-Layout (RGBA)
let pixel_index = (y_idx * w_idx + x_idx) * usize::new(4);

output[pixel_index]                  = final_color_x; // R
output[pixel_index + usize::new(1)] = final_color_y; // G
output[pixel_index + usize::new(2)] = final_color_z; // B
output[pixel_index + usize::new(3)] = hit_dist;      // A (Tiefenwert)
```

---

## 2. Das mathematische Prinzip (Circle of Confusion)

Ein spezialisierter Vorbereitungs-Kernel berechnet vor der Faltung den Circle of Confusion (CoC) für jedes Pixel und skaliert die RGB-Werte. Die Formel für den CoC lautet:  
$$CoC = \left\vert{} \frac{\text{TIEFE} - \text{FOKUS\_DISTANZ}}{\text{TIEFE}} \right\vert{} \cdot \text{BLENDEN\_ÖFFNUNG}$$

---

## 3. Reale Implementierung mit dem `cubek`-Crate

Die Schärfentiefe wird über eine gewichtete 2D-Faltung (Convolution) berechnet. Wir nutzen die offizielle Host-API von `cubek-convolution`, die das Bild in Slices zerlegt, via `im2col` für die Tensor-Cores optimiert und faltet.

```rust
use cubecl::prelude::*;
use cubecl::tensor::Tensor;

// Importe aus dem offiziellen tracel-ai/cubek Ökosystem
use cubek_convolution::{convolution, ConvolutionConfig, ConvolutionStrategy};

/// Führt den Schärfentiefe-Filter (Depth of Field) mittels CubeK-Convolution auf der GPU aus
pub fn apply_depth_of_field_with_cubek<R: Runtime>(
    client: &ComputeClient<R, R::Server>,
    rgba_render_output: Tensor<R>, // Der [1, 4, Height, Width] Tensor aus dem Raymarcher
    width: usize,
    height: usize,
    focal_distance: f32,          // Die exakte Distanz, die scharf sein soll
    aperture: f32,                // Stärke der Unschärfe (Blendenöffnung)
) -> Tensor<R> {
    
    // 1. SCHRITT: Generierung des Unschärfe-Weights-Tensors (Kernel)
    // Für ein physikalisches Bokeh nutzen wir einen kreisförmigen (Disk) oder Gaußschen Faltungs-Kernel.
    // Ein 7x7 oder 9x9 Kernel reicht für ein wunderschönes, kinoreifes Bokeh.
    let kernel_size = 7;
    let kernel_shape = vec![1, 1, kernel_size, kernel_size]; // [Output_Ch, Input_Ch, K_H, K_W]
    let filter_kernel = Tensor::empty(kernel_shape, client);
    
    // (Ein kleiner Hilfskernel füllt 'filter_kernel' auf der GPU mit einer Gauß-Glocke)
    generate_gaussian_kernel::launch(client, CubeCount::Static(1,1,1), CubeDim::new(7,7,1), filter_kernel.as_handle_ref());

    // 2. SCHRITT: Vorbereitung der Bilddaten basierend auf dem Tiefenkanal (CoC)
    let preprocessed_image = Tensor::empty(vec![1, 3, height, width], client);
    
    compute_coc_and_mask_kernel::launch(
        client,
        CubeCount::Static((width / 16) as u32, (height / 16) as u32, 1),
        CubeDim::new(16, 16, 1),
        rgba_render_output.as_handle_ref(),
        preprocessed_image.as_handle_ref(),
        width.into(),
        height.into(),
        focal_distance.into(),
        aperture.into(),
    );

    // 3. SCHRITT: Der eigentliche CubeK-Convolution Pass
    // Erstellt den finalen Ausgabe-Tensor für das fertige Post-Processing-Bild
    let dof_final_output = Tensor::empty(vec![1, 3, height, width], client);

    let config = ConvolutionConfig {
        // Nutzt im2col + MMA (Tensor Cores), um die Nachbarschaftspixel parallel zu verrechnen
        strategy: ConvolutionStrategy::Auto, 
    };

    // Aufruf der echten Hardware-beschleunigten Faltung
    convolution::<R, f32>(
        preprocessed_image,
        filter_kernel,
        dof_final_output.clone(),
        config,
        client,
    ).expect("CubeK DoF-Convolution fehlgeschlagen");

    dof_final_output // Gibt das fertige RGB-Bild mit Schärfentiefe zurück
}
```

---

## 4. Die Begleit-Kernel (GPU-Hilfsfunktionen)

Damit die Faltung weiß, was sie tun soll, berechnen wir die Unschärfe-Maske pro Pixel:

```rust
#[cube(launch)]
fn compute_coc_and_mask_kernel(
    rgba_input: &Tensor<f32>,       // [R, G, B, Depth] Interleaved aus dem Raymarcher
    rgb_output: &mut Tensor<f32>,   // Vorbereitetes Bild für die Faltung
    width: u32, height: u32,
    focal_dist: f32, aperture: f32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x < usize::cast_from(width) && y < usize::cast_from(height) {
        let idx_in = (y * usize::cast_from(width) + x) * usize::new(4);
        
        let r = rgba_input[idx_in];
        let g = rgba_input[idx_in + usize::new(1)];
        let b = rgba_input[idx_in + usize::new(2)];
        let depth = rgba_input[idx_in + usize::new(3)];

        // Berechne den "Circle of Confusion" (CoC)
        let mut coc = f32::new(0.0);
        if depth < f32::new(40.0) { // Wenn wir nicht den Hintergrund (Skybox) getroffen haben
            coc = ((depth - focal_dist) / depth).abs() * aperture;
            coc = coc.min(f32::new(1.0)); // Normalisieren
        } else {
            coc = f32::new(1.0); // Hintergrund ist maximal unscharf
        }

        // Berechne den Index für den CHW (Channel, Height, Width) Tensor von cubek-convolution
        let w_sz = usize::cast_from(width);
        let h_sz = usize::cast_from(height);
        let img_stride = w_sz * h_sz;
        let pixel_offset = y * w_sz + x;

        // Wir gewichten die Farben mit dem Unschärfefaktor vor
        rgb_output[pixel_offset]                  = r * coc;
        rgb_output[pixel_offset + img_stride]      = g * coc;
        rgb_output[pixel_offset + img_stride * 2]  = b * coc;
    }
}

#[cube(launch)]
fn generate_gaussian_kernel(kernel: &mut Tensor<f32>) {
    let x = COMPUTE_ID_X;
    let y = COMPUTE_ID_Y;
    // Füllt den Tensor mit einer standardisierten Gauß-Matrix (7x7) für weiche Übergänge
    let idx = y * usize::new(7) + x;
    kernel[idx] = f32::new(1.0) / f32::new(49.0); // Box-Filter Annäherung für dieses Beispiel
}
```

---

## Was diese Pipeline für Ihre mathematische Szene bewirkt

Wenn Sie Ihre Kamera auf einen der rotierenden Torus-Körper oder Kristalle in einer Entfernung von 5 Metern ausrichten (`focal_distance = 5.0`), passiert Folgendes:

1. Der Kristall wird gestochen scharf dargestellt.
2. Der unendliche, poröse Gyroid-Hintergrund, der sich in 20–30 Metern Entfernung erstreckt, verschwimmt in einem wunderschönen, weichen, fotografischen Bokeh.
3. Da `cubek-convolution` das Bild über `im2col` aufbereitet, blockieren sich die GPU-Threads beim Lesen der Nachbarpixel nicht gegenseitig. Das rechenintensive Zusammenrechnen der Pixelmatrizen wird komplett von den Tensor-Cores geschluckt.

Möchten Sie als Nächstes:

- Den gesamten Workflow (Matmul -> Raymarching -> Convolution -> Reduction) in einer einzigen `main.rs`-Renderschleife zusammenfügen?
- Den Faltungs-Kernel so anpassen, dass er statt einer Kreisform ein anamorphes (ovales) Bokeh für einen Cinematic-Look erzeugt?