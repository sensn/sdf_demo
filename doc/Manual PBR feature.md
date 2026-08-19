## Bedienungsanleitung & Rezeptbuch: PBR-Materialien und Spezialeffekte in CubeCL 0.11

Dieses Handbuch beschreibt die praktische Anwendung des neu implementierten Physically Based Rendering (PBR) Features. Durch die reaktive Kopplung deiner Tastatur-Steuerung an den neuen, dritten Grafik-Tensor kannst du die Oberflächenbeschaffenheit des aktuell selektierten Slots im laufenden Betrieb flüssig morphen.

---

## 1. Steuerung & Interaktions-Matrix im Space-Lab

Um Materialien in Echtzeit zu tunen, nutzen wir deine vordefinierten Tasten-Mappings. Jedes Halten der Tasten inkrementiert/dekrementiert die CPU-Material-Vektoren des ausgewählten Slots (`sel = current_selected_slot`) und streamt die Änderungen mittels `client.write` barrierefrei direkt in den Shader. [1]

## Das PBR-Tastatur-Cockpit

- `KeyCode::KeyU` / `KeyO`: Verschiebung auf der X-Achse (Reaktiviert).
- `KeyCode::KeyP` / `KeyH`: Verschiebung auf der Z-Achse (Reaktiviert).
- `KeyCode::KeyZ` / `KeyX`: Vertikaler Hub auf der Y-Achse.
- `KeyCode::KeyG` / `KeyY`: Individuelle Skalierung (Größe/Radius) des selektierten Primitivs.
- Neue Material-Vektoren (CPU-seitig zugeordnet):
    
    - `slot_roughness[sel]`: Rauheit $[0.01, 1.0]$ → Regelt die Schärfe und Streuung von Glanzlichtern.
    - `slot_metallic[sel]`: Metall-Faktor $[0.0, 1.0]$ → Wechselt das Reflexionsverhalten von Dielektrikum zu echtem Leiter.
    - `slot_emissive[sel]`: Eigenleuchten $[0.0, 10.0]$ → Lässt Objekte unabhängig von Lichtquellen glühen.
    - `slot_specular[sel]`: Reflektivität $[0.0, 1.0]$ → Grundintensität der Lichtreflexe.
    

---

## 2. Material-Bibliothek & Albedo-Rezepte

Dank unseres SDF-nähenabhängigen gewichteten Farbmischers bleiben alle Materialien absolut farb- und formstabil, egal wie viele Objekte ineinandergreifen oder ein-/ausgeschaltet werden.

## 💎 Chrom / Spiegelndes Metall (Perfect Conductor)

- Charakteristik: Absolut scharfes, gleißendes Glanzlicht. Die Oberfläche verhält sich wie ein reiner Reflektor und nimmt die Farbe der Lichtquellen vollständig an. [2]
- Rezept (CPU-Werte):
    
    - `Type`: `1.0` (Kristall) oder `3.0` (Torus)
    - `Color (R, G, B)`: `[0.95, 0.95, 0.95]` (Hellgrau/Silber)
    - `Roughness`: `0.02` (Extrem glatt – erzeugt einen gewaltigen `spec_power` Exponenten auf der GPU)
    - `Metallic`: `1.0` (Vollmetall)
    - `Specular`: `1.0` (Maximale Reflexionsstärke)
    - `Emissive`: `0.0` [3]
    

## 🪵 Diffuser Sandstein / Ton (Rough Dielectric)

- Charakteristik: Vollkommen matte, raue Oberfläche. Licht wird weich und breit gestreut, es gibt keine scharfen Lichtpunkte.
- Rezept (CPU-Werte):
    
    - `Type`: `2.0` (Gyroid – perfekt für organische Strukturen)
    - `Color (R, G, B)`: `[0.55, 0.45, 0.35]` (Sandiges Braun/Ocker)
    - `Roughness`: `0.90` (Extrem rau – streut den Blinn-Phong-Reflex gegen Null)
    - `Metallic`: `0.0` (Reines Nicht-Metall)
    - `Specular`: `0.1` (Minimale Reflektivität)
    - `Emissive`: `0.0` [4, 5]
    

## 🌌 Plasma-Kristall / Glühendes Alien-Artefakt (Self-Emissive)

- Charakteristik: Das Objekt strahlt aus sich selbst heraus Energie ab. Es leuchtet im Schatten und verliert in der Dunkelheit nicht seine Farb-Identität. [6]
- Rezept (CPU-Werte):
    
    - `Type`: `1.0` (Kristall)
    - `Color (R, G, B)`: `[0.0, 1.0, 0.8]` (Fluoreszierendes Cyan)
    - `Roughness`: `0.30` (Seidenmatt)
    - `Metallic`: `0.2` (Teilmetallisch)
    - `Specular`: `0.5`
    - `Emissive`: `2.5` (Speist hochenergetische Farbwerte direkt in die `lit`-Lichtkanäle des Shaders)
    

## 🍷 Rubin-Glas / Elfenbein (Smooth Dielectric)

- Charakteristik: Edel, hochglänzend, aber nicht metallisch. Der Lichtpunkt ist stechend scharf, die Grundfarbe bleibt organisch gesättigt.
- Rezept (CPU-Werte):
    
    - `Type`: `3.0` (Torus)
    - `Color (R, G, B)`: `[0.85, 0.05, 0.05]` (Tiefes Rubinrot)
    - `Roughness`: `0.05` (Sehr glatt)
    - `Metallic`: `0.0` (Dielektrikum)
    - `Specular`: `0.7` (Starker Glas-Lichtreflex)
    - `Emissive`: `0.0` [7, 8]
    

---

## 3. Spezialeffekte durch Material-Morphing

Da der Puffer über das polynomiale Minimum (`smin`) mathematisch weich verschmolzen wird, entstehen beim Ineinandergleiten unterschiedlicher Materialien faszinierende volumetrische GPU-Effekte.

## Effekt 1: Die organische Schweißnaht (Material-Metamorphose)

Verschiebe einen spiegelnden Chrom-Kristall (Slot 1) langsam in einen matten Sandstein-Gyroiden (Slot 2) hinein.

- Visuelles Ergebnis: In der Verschmelzungszone errechnet unser gewichteter Farbmischer einen flüssigen Übergang. Das Metall "schmilzt" organisch in den Stein. An der Grenze vermischen sich die Lichtreflexe: Das Glanzlicht wird stufenlos breiter und matter (`mixed_rough`), während der metallische Glanz sanft in ein stumpfes Steingrau übergeht.

## Effekt 2: Das pulsierende Energienetz (Audio-reaktive Emissivität)

Kopple den `slot_emissive`-Wert eines Objekts an eine Sinuskurve (`time`) im Host-Code der `main.rs`:

```rust
// In src/main.rs innerhalb deines Event::AboutToWait Schleifendurchlaufs:
let time = start_time.elapsed().as_secs_f32();
slot_emissive[0] = (time * 5.0).sin().abs() * 4.0; // Pulsieren zwischen 0.0 und 4.0
config_dirty = true; // Flusht den Puffer jeden Frame an die GPU
```

- Visuelles Ergebnis: Der Kristall fängt im Labor rhythmisch an, wie ein Reaktor aufzuglühen. Da wir die `smin_material`-Logik im Kernel nutzen, breitet sich das Glühen bei einer Verschmelzung sanft wie eine Hitzewelle auf die angrenzende Raumarchitektur aus, bevor es wieder verlischt.

---

## 4. Code-Beispiele für Implementierungen

## A) Tastatur-Zuweisung für stufenloses Rauheits-Tuning (`src/main.rs`)

Um ein Objekt im laufenden Betrieb matter oder glänzender zu machen, weisen wir beispielsweise die freien Tasten `V` (glatter/schärfer) und `B` (rauer/matter) zu.

Füge dieses Mapping in dein `WindowEvent::KeyboardInput` ein:

```rust
// In src/main.rs -> WindowEvent::KeyboardInput match code:
KeyCode::KeyV => keys.rough_down = is_pressed, // Macht das Objekt spiegelnder
KeyCode::KeyB => keys.rough_up   = is_pressed, // Macht das Objekt matter
```

Füge die Auswertung in deinen `Event::AboutToWait`-Block ein:

```rust
// In src/main.rs -> Event::AboutToWait:
let mat_speed = 0.01f32; // Feinfühlige Justierung

if sel < slot_types.len() {
    // Rauheit verringern -> Objekt wird glänzender (Untergrenze 0.01 gegen Division-by-Zero)
    if keys.rough_down { slot_roughness[sel] = (slot_roughness[sel] - mat_speed).max(0.01); config_dirty = true; }
    // Rauheit erhöhen -> Objekt wird matter (Obergrenze 1.0)
    if keys.rough_up   { slot_roughness[sel] = (slot_roughness[sel] + mat_speed).min(1.0);  config_dirty = true; }
}
```

## B) Das mathematische Wunder im Shader (`src/kernel.rs`)

Hier ist die exakte Stelle im Shading-Bereich des Kernels, die für das spektakuläre Aufblühen des PBR-Glanzlichts verantwortlich ist. Der Code berechnet den Halbwertsvektor des Lichts relativ zu deinem Auge und nutzt deine `client.write`-Bytes als mathematischen Potenz-Exponenten:

```rust
// In src/kernel.rs innerhalb von raymarch_sdf_kernel:
// hit_rough wurde aus dem materials-Tensor gelesen (Stride 4)

let view_dir = final_rd.scale(f32::new(-1.0)).normalize();
let half_vec = key_dir.clone().add(view_dir).normalize();
let spec_angle = normal.dot(half_vec).max(f32::new(0.0));

// 🟢 DIE HARDWARE-MAGIE:
// Kleines hit_rough (0.02) erzeugt spec_power von 50.0.
// 50.0 * 15.0 = 750.0. Ein extrem hoher Exponent schnürt das Glanzlicht 
// zu einem winzigen, gleißend scharfen Reflexpunkt zusammen (Chrom-Effekt).
let spec_power = f32::new(1.0) / hit_rough.max(f32::new(0.01));
let specular_highlight = spec_angle.powf(spec_power * f32::new(15.0)) * hit_spec;
```

Mit diesem PBR-Framework reagiert dein Space-Lab augenblicklich auf jede physikalische Material-Eigenschaft. Du kannst nun komplexe, reaktive Sci-Fi-Szenarien entwerfen, bei denen Geometrie, Farbe und Oberflächenstruktur völlig unabhängig voneinander über die PCIe-Leitung fließen! [9]

Möchtest du als Nächstes das Environmental Control Tensor Blueprint (aus Kapitel 5 des Berichts) umsetzen, um auch die Licht-Positionen und die Nebeldichte flüssig über Tasten zu steuern? [10]

  

[1] [https://cgwisdom.com](https://cgwisdom.com/blog/how-to-create-realistic-materials-in-sketchup-what-are-pbr-maps.html)

[2] [https://digitalproduction.com](https://digitalproduction.com/2019/05/25/physically-based-rendering-viel-realismus-aber-bitte-mit-kreativen-stellschrauben/)

[3] [https://docs.isaacsim.omniverse.nvidia.com](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/struct_material.html)

[4] [https://effecthouse.tiktok.com](https://effecthouse.tiktok.com/learn/guides/workspace/assets/material/standard-pbr)

[5] [https://meshlogic.github.io](https://meshlogic.github.io/posts/blender/materials/nodes-pbr-basic-shader/)

[6] [https://github.com](https://github.com/microsoft/DirectXTK/wiki/Physically-based-rendering/b488f8df4b87e672d0db3185813e8ad5c6bee700)

[7] [https://effecthouse.tiktok.com](https://effecthouse.tiktok.com/learn/guides/workspace/assets/material/standard-pbr)

[8] [https://docs.isaacsim.omniverse.nvidia.com](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/struct_material.html)

[9] [https://dl.acm.org](https://dl.acm.org/doi/fullHtml/10.1145/3650400.3650464)

[10] [https://www.researchgate.net](https://www.researchgate.net/figure/Three-kinds-of-light-illuminations-for-PBR-rendering-Fixed-Single-means-single-point_fig3_393983346)