In WebGPU und Vulkan ist das Übergeben von 3D-Vektoren (wie der Kameraposition oder Blickrichtung) in Uniform-Strukturen eine häufige Fehlerquelle. Das liegt am strikten WebGPU/WGSL-Speicher-Alignment: Ein Vektor mit 3 Komponenten (`vec3<f32>`) belegt zwar nur 12 Bytes (3 × 4 Bytes), verhält sich im Speicher aber bezüglich des Alignments wie ein Vektor mit 4 Komponenten (`vec4<f32>`). Er verlangt ein 16-Byte-Alignment.

Wenn du ein `vec3<f32>` in ein Custom Struct packst, musst du zwingend ein 4-Byte-Padding (oder eine vierte Koordinate wie z. B. eine Zoom-Stufe oder Zeitkomponente) direkt dahinter setzen. Alternativ nutzt man in CubeCL direkt Arrays oder man baut sich ein eigenes, sauberes Vektor-Struct.

Hier ist die performante und WebGPU-konforme Implementierung für deine Kamera-Parameter.

## 1. Das krisensichere Struct-Layout (Kamera & Licht)

Wir erstellen ein eigenes `Vec3`-Hilfs-Struct, das wir direkt mit dem nötigen 4-Byte-Padding (`_pad`) ausstatten. So bleibt das Speicher-Layout auf CPU und GPU immer synchron.

```rust
use cubecl::prelude::*;

// Ein sicheres 3D-Vektor Struct für WebGPU Uniforms
#[derive(CubeType, Copy, Clone)]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32, // 💡 WICHTIG: Füllt den Vektor auf 16 Bytes auf!
}

impl GpuVec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, _pad: 0.0 }
    }
}

// Deine neuen Kamera- und Szenen-Einstellungen
#[derive(CubeType, Copy, Clone)]
pub struct CameraSettings {
    pub position: GpuVec3,  // 16 Bytes (inkl. Padding)
    pub target: GpuVec3,    // 16 Bytes (inkl. Padding)
    pub up: GpuVec3,        // 16 Bytes (inkl. Padding)
    pub fov: f32,           // 4 Bytes
    pub _pad: [u32; 3],     // 12 Bytes Padding -> Bringt den gesamten Block wieder auf ein Vielfaches von 16 Bytes (48 Bytes gesamt)
}

// Wir fügen die Kamera in unsere Hauptstruktur ein
#[derive(CubeType, CubeLaunch)]
pub struct RaymarchParams {
    pub camera: CameraSettings,
    pub env: EnvSettings,
    pub arch: ArchParams,
    pub fold: FoldParams,
}
```

---

## 2. Nutzung der Vektoren im CubeCL-Kernel

Im Kernel kannst du nun ganz einfach auf die Vektorkomponenten zugreifen. Um echten mathematischen Komfort zu haben, kannst du dir die Werte im Kernel direkt in CubeCL-eigene Vektor-Typen (wie `Line<f32, 3>`) gießen:

```rust
#[cube(launch)]
pub fn raymarch_sdf_kernel<C: RaymarchConfig>(
    output: &mut Tensor<f32>,
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>,
    params: RaymarchParams, 
) {
    // 💡 Wir lesen die CPU-GpuVec3-Struktur aus und konvertieren sie 
    // direkt in einen nativen GPU-3D-Vektor (Line<f32, 3>)
    let cam_pos = Line::<f32, 3>::new(
        params.camera.position.x,
        params.camera.position.y,
        params.camera.position.z,
    );
    
    let cam_target = Line::<f32, 3>::new(
        params.camera.target.x,
        params.camera.target.y,
        params.camera.target.z,
    );

    // Beispiel: Strahlrichtung berechnen (Grob vereinfacht)
    let ray_dir = Line::<f32, 3>::normalize(cam_target - cam_pos);

    // ... Dein Raymarching Loop nutzt jetzt cam_pos und ray_dir ...
}
```

---

## 3. Der Host-Launch Code (CPU)

Auf der CPU-Seite befüllst du die Vektoren ganz intuitiv mit deinen gewünschten Raum-Koordinaten:

```rust
fn launch_raymarcher(client: &ComputeClient<WgpuRuntime, WgpuServer>) {
    // ... Tensoren initialisieren ...

    let host_params = RaymarchParams {
        camera: CameraSettings {
            position: GpuVec3::new(0.0, 5.0, -10.0), // Kamera steht leicht erhöht und zurückgesetzt
            target: GpuVec3::new(0.0, 0.0, 0.0),    // Blick auf den Ursprung (deinen SDF-Tempel)
            up: GpuVec3::new(0.0, 1.0, 0.0),        // Oben ist die Y-Achse
            fov: 60.0,
            _pad: [0; 3],
        },
        env: EnvSettings {
            light_intensity: 1.5,
            ambient_fog: 0.05,
        },
        arch: ArchParams {
            temple_height: 24.0,
            pillar_count: 8,
            detail_level: 3,
            _pad: 0,
        },
        fold: FoldParams {
            fold_factor: 2.17,
            iterations: 5,
            _pad: [0; 2],
        },
    };

    // Kernel-Start mit Comptime-Config und voll-vektorieller Kamera
    raymarch_sdf_kernel::launch::<HighQualityConfig>(
        client,
        CubeDim::new(16, 16, 1),
        GridDim::new(64, 48, 1),
        TensorArg::from_var(&output_tensor),
        TensorArg::from_var(&meta_tensor),
        TensorArg::from_var(&slots_tensor),
        TensorArg::from_var(&mat_tensor),
        host_params,
    );
}
```

Mit dieser Architektur hast du jetzt das absolute Optimum für WebGPU / WGPU erreicht:

- Die WebGPU Storage-Buffer Limits werden absolut geschont.
- Alle konstanten Einstellungen und Flags fliegen via Comptime komplett aus dem Runtime-Code.
- Dynamische Parameter und 3D-Kameravektoren liegen perfekt ausgerichtet im ultraschnellen Constant-Cache der GPU.

Möchtest du als Nächstes ein konkretes Beispiel sehen, wie du die SDF-Distanzfunktionen (z. B. für Kugeln, Boxen oder unendliche Säulen-Wiederholungen) innerhalb von CubeCL sauber modularisierst, oder möchtest du tiefer in die Material- und Texturabfragen deiner Pipeline einsteigen?