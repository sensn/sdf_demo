Technischer Bericht: Dynamischer GPU-Szenengraph via CubeCL (v0.10)

**Fokus:** Datenfluss-Architektur, layoutfreie API-Signatur-Bypässe und Zero-Cost-Abstraktion in Compute-Shadern.

---

1. Systemübersicht & Design-Philosophie

Klassische Grafik-Pipelines nutzen oft komplexe, dynamische Szenengraphen auf der CPU, die baumartig traversiert und über Byte-Buffer (UBOs/SSBOs) an die GPU übertragen werden. In **CubeCL (v0.10)** stoßen voll-dynamische Strukturen auf der GPU zur Laufzeit an Grenzen, da das `#[cube]`-Makro den Rust-Code zur Kompilierzeit statisch analysiert, optimiert und in festen WGSL/SPIR-V Zwischencode übersetzt.

Dieses System implementiert eine **tabellenbasierte Baukasten-Architektur**. Statt Objekte zur Laufzeit physisch in den Speicher zu streamen oder Shader-Code dynamisch zu parsen, werden alle verfügbaren Geometrie-Algorithmen statisch in den Shader integriert. Der Zustand und die Komposition der Szene werden über ein hocheffizientes **Primitive-Signatur-Mapping** (Value-Passing über Thread-Argumente) gesteuert.

Die Architektur-Brücke

```
  [ CPU (Host Thread via Winit) ] 
        | 
        |  Ereignis: Tastendruck 1, 2, 3, 4 (Slot-Zustand rotiert 0 -> 1 -> 2)
        v
  [ Compute-Launch (WGPU Client Queue) ]
        | 
        |  Übergabe: Skalare u32-Parameter (slot1, slot2, slot3, slot4)
        v
  [ GPU Kernel (scene_sdf Evaluierung) ]
        |
        +---> Evaluierung Slot 1 (Zentrum)   --> smin() / min()
        +---> Evaluierung Slot 2 (Links)     --> smin() / min()
        +---> Evaluierung Slot 3 (Rechts)    --> smin() / min()
        +---> Evaluierung Slot 4 (Hintergrund)--> smin() / min()
```

---

2. Struktur und Syntax der Implementation

2.1 Modul-Kombination & Datenfluss

Die Datenfluss-Kette ist streng hierarchisch aufgebaut. Da Teilsysteme wie Schatten (Ray-Traced Hard/Soft Shadows) und Umgebungsverdeckung (Ambient Occlusion) die Szene an beliebigen Punkten im Raum abfragen müssen, fungiert die Funktion `scene_sdf` als zentrales mathematisches Orakel. Jede Parametererweiterung der Szene muss zwingend durch alle Submodule durchgereicht werden.

2.2 Syntax-Spezifika in CubeCL

- **Verbot von vorzeitigen `return`-Anweisungen:** Innerhalb des `#[cube]`-Makros führt ein vorzeitiges `return` in konditionalen Zweigen (`if`) zu Compiler-Fehlern. Daten müssen über mutierbare lokale Variablen (`let mut res = ...;`) gesammelt und am Funktionsende implizit zurückgegeben werden.
- **Explizite Typinferenz bei `.into()`-Konvertierungen:** Host-Arrays, die für `TensorArg::from_raw_parts` genutzt werden, verlieren ohne Inferenz den Typ. Leere Arrays für Strides müssen zwingend als `Vec::<usize>::new().into()` deklarielt werden.
- **Ganzzahl-Casting für Modulo-Operationen:** Da Fließkomma-Modulo-Operationen (`%`) auf GPUs plattformabhängig zu Instabilitäten neigen, nutzt die Implementierung ein sicheres i32/u32 Bit-Casting, um die Parität (Gerade/Ungerade) der unendlichen Gitterzellen zu bestimmen.

---

3. Der Code-Blueprint für den Coding-Agenten

Die folgenden Code-Auszüge zeigen die exakte syntaktische Struktur, die ein nachfolgender AI-Agent direkt lesen und erweitern kann.

3.1 Das mathematische Herzstück: `src/kernel.rs` (Auszug)

rust

```
#[cube]
pub fn scene_sdf(p: Vec3, time: f32, blend_factor: f32, s1: u32, s2: u32, s3: u32, s4: u32) -> f32 {
    let cell_size = 10.0f32;
    let half_cell = cell_size * 0.5;
    
    // Unendliches 3D Space-Folding (Raum-Klonierung)
    let cell_id_x = f32::floor((p.x + half_cell) / cell_size);
    let cell_id_z = f32::floor((p.z + half_cell) / cell_size);
    let grid_p_x = p.x - cell_size * cell_id_x;
    let grid_p_y = p.y - cell_size * f32::floor((p.y + half_cell) / cell_size); 
    let grid_p_z = p.z - cell_size * cell_id_z;
    let local_p = Vec3::new(grid_p_x, grid_p_y, grid_p_z);

    // Initialisierung des SDF-Abstandsfeldes im Unendlichen
    let mut core_system = 1000.0f32;

    // Evaluierung Slot 1: Zentrum (0.0, 0.0, 0.0)
    if s1 == 1 { core_system = smin(core_system, evaluate_crystal(local_p, time), blend_factor); }
    if s1 == 2 { core_system = smin(core_system, evaluate_gyroid(local_p, time), blend_factor); }

    // Evaluierung Slot 2: Links versetzt (-1.8, 0.0, 0.0)
    let p_slot2 = local_p.sub(Vec3::new(-1.8, 0.0, 0.0));
    if s2 == 1 { core_system = smin(core_system, evaluate_crystal(p_slot2, time), blend_factor); }
    if s2 == 2 { core_system = smin(core_system, evaluate_gyroid(p_slot2, time), blend_factor); }

    // Evaluierung Slot 3: Rechts versetzt (1.8, 0.0, 0.0)
    let p_slot3 = local_p.sub(Vec3::new(1.8, 0.0, 0.0));
    if s3 == 1 { core_system = smin(core_system, evaluate_crystal(p_slot3, time), blend_factor); }
    if s3 == 2 { core_system = smin(core_system, evaluate_gyroid(p_slot3, time), blend_factor); }

    // Evaluierung Slot 4: Hintergrund versetzt (0.0, 0.0, 1.8)
    let p_slot4 = local_p.sub(Vec3::new(0.0, 0.0, 1.8));
    if s4 == 1 { core_system = smin(core_system, evaluate_crystal(p_slot4, time), blend_factor); }
    if s4 == 2 { core_system = smin(core_system, evaluate_gyroid(p_slot4, time), blend_factor); }

    // Statische Tempel-Architektur hinzufügen
    let mut architecture = evaluate_architecture(local_p);
    
    // CSG Kombination (Das Minimum fusioniert den Kern mit der Tempelhalle)
    f32::min(core_system, architecture)
}
```

Verwende Code mit Vorsicht.

3.2 Die Pipeline-Verschaltung: `src/shadows.rs` (Auszug)

rust

```
#[cube]
pub fn calculate_soft_shadow(
    p: Vec3, light_dir: Vec3, time: f32, k: f32, blend_factor: f32,
    s1: u32, s2: u32, s3: u32, s4: u32 // Zwingendes Durchreichen der Steuerung
) -> f32 {
    let mut t = 0.15; // Hoher Bias gegen Self-Shadowing Acne an Tempelkanten
    let mut res = 1.0;
    // ... Raymarching-Schleife zur Lichtquelle
    let d = scene_sdf(current_pos, time, blend_factor, s1, s2, s3, s4);
    // ... Penumbra-Berechnung
    res.min(1.0).max(0.25) // Harte Kappung bei 0.25 verhindert unnatürliche klumpige Finsternis
}
```

Verwende Code mit Vorsicht.

---

4. Technische Leistung & Wirtschaftlichkeit (Zero-Cost Analysis)

Warum ist diese Methode "Zero-Cost"?

1. **Keine dynamischen Speicher-Allokationen:** Es werden keine Arrays oder Vektoren zur Laufzeit auf der GPU erzeugt oder vergrößert. Die Slot-Abfrage basiert auf primitiven Registern (`u32`).
2. **Keine CPU-GPU-Synchronisationsbarrieren:** Da die Werte direkt als Skalare im Kernel-Launch mitgeschickt werden, entfällt das Schreiben in Uniform-Buffer (UBOs) und das damit verbundene Pipeline-Stalling.
3. **Perfekte Branch-Prediction der GPU-Hardware:** Moderne GPUs verarbeiten `if`-Abfragen extrem effizient, wenn der Zustand (`s1`, `s2`, etc.) für alle Threads innerhalb eines Warps/Subgroups identisch ist (_Uniform Branching_). Da jeder Pixel auf dem Bildschirm im exakt gleichen Frame die gleiche Szenenkonfiguration nutzt, gibt es keine Divergenz (Branch Divergence) innerhalb der Rechenkerne. Die inaktiven Zweige kosten absolut keine Rechenzeit.

---

5. Visionäre Anwendungsfälle (Advanced Use Cases)

Ein nachfolgender Entwicklungs-Agent kann dieses Framework als Fundament für hochkomplexe Systeme nutzen:

5.1 Prozedurales Endlos-RGP (Chunk-basiertes Spawnen)

Anstatt die Slots starr an relative Offsets im Raum zu binden, kann die CPU die Slot-Argumente an die absolute Position des Spielers koppeln. [[1](https://github.com/tracel-ai/cubecl/releases), [2](https://jonngan.medium.com/rust-burn-c-u-m-d49d2a04350e)]

- **Mechanismus:** Wenn der Spieler die Grenze einer unendlichen Zelle überschreitet (`p.x > 8.0`), berechnet Tokio im Hintergrund über einen Seed ein neues prozedurales Muster.
- **Resultat:** Der Hauptthread füttert die Slot-IDs beim nächsten Launch mit völlig neuen Werten. Für den Nutzer entsteht die optische Illusion einer sich prozedural verändernden unendlichen Welt, obwohl der Shader-Code statisch bleibt.

5.2 Physik-gesteuerte Partikel-Kollision (SDF-to-CPU Feedback)

Da die Slots über einfache IDs gesteuert werden, kann eine separate Tokio-Hintergrund-Task eine einfache CPU-Physikberechnung (z. B. Gravitation oder Kollision) durchführen.

- **Mechanismus:** Tokio berechnet die Anziehungskraft zwischen Slot 2 und Slot 3.
- **Resultat:** Sobald sich zwei Objekte mathematisch berühren, sendet Tokio ein Signal, das den `blend_factor` im Hauptthread nach oben schraubt. Die Objekte fangen an, sich im Shader visuell flüssig wie Quecksilber ineinander zu saugen.

5.3 Audio-Reaktive Geometrie-Synthese (Vj-Tooling)

Die Slot-Zustände und der `blend_factor` können direkt an ein Audio-Inhalts-Analysesystem (FFT - Fast Fourier Transformation) gekoppelt werden.

- **Mechanismus:** Bässe triggern das Umschalten von Slot 1 (Kristall) auf Slot 2 (Gyroid), während hohe Frequenzen die Wellenbewegung des Diamanten (`waves`) beschleunigen.
- **Resultat:** Es entsteht ein vollständig GPU-beschleunigtes, interaktives Fraktal-Musik-Visualisierungstool, das mit konstanten 60 FPS läuft.

Das System-Scaffold ist nun lückenlos dokumentiert und für jeden nachfolgenden Coding-Agenten perfekt lesbar aufbereitet. Welches dieser fortgeschrittenen Szenarien (z. B. die Audio-Reaktivität oder das chunkspezifische Spawnen beim Durchfliegen der Räume) wollen wir als Nächstes implementieren?