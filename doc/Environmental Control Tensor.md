
Hier ist der detaillierte Schritt-für-Schritt-Leitfaden, um den Environmental Control Tensor (Umwelt-Steuerungs-Tensor) in dein CubeCL 0.11-Projekt zu integrieren.

Durch dieses Design lagern wir globale Parameter wie Lichtpositionen, Lichtfarben und Nebeldichte in einen vierten, separaten Tensor aus. Dieser nutzt ein exaktes 16-Float Layout (verteilt auf 4 perfekte `Vec4`-Register), um maximale Hardware-Performance und unbrechbares Speicher-Alignment zu garantieren.

---

## Schritt 1: Das 16-Float Speicher-Layout festlegen (CPU & GPU)

Um das WGSL/SPIR-V Alignment-Gesetz (16-Byte-Grenze) perfekt zu bedienen, strukturieren wir den Puffer als Matrix aus genau vier `Vec4`-Blöcken (insgesamt 16 Floats).

|Index|Variable|Funktion|Register-Block|
|---|---|---|---|
|`0, 1, 2`|`light_pos_x, y, z`|Position des Hauptlichts im 3D-Raum|Block 0 (`Vec4`)|
|`3`|`light_intensity`|Stärke des Hauptlichts|Block 0 (`Vec4`)|
|`4, 5, 6`|`key_r, key_g, key_b`|RGB-Farbe des Hauptlichts|Block 1 (`Vec4`)|
|`7`|`ambient_strength`|Grundhelligkeit der Szene (Umgebungslicht)|Block 1 (`Vec4`)|
|`8, 9, 10`|`bg_r, bg_g, bg_b`|RGB-Farbe des Hintergrunds / Nebels|Block 2 (`Vec4`)|
|`11`|`fog_density`|Dichte des volumetrischen Nebels|Block 2 (`Vec4`)|
|`12`|`fog_enabled`|Nebel-Schalter (`1.0` = AN, `0.0` = AUS)|Block 3 (`Vec4`)|
|`13,14,15`|`padding`|Ungenutzte Füllwerte (müssen auf `0.0` stehen)|Block 3 (`Vec4`)|

---

## Schritt 2: Host-Initialisierung in `src/main.rs`

Wir erstellen den Vektor vor der Event-Loop und allozieren den vierten Tensor persistent im VRAM.

```rust
    // =========================================================================
    // INITIALISIERUNG DES ENVIRONMENTAL CONTROL TENSORS (4. Tensor)
    // =========================================================================
    // Wir packen deine bestehenden Werte in ein flaches, 16 Elemente langes Array
    let mut env_payload = vec![0.0f32; 16];
    
    // Block 0: Lichtposition (4.0, 7.0, -4.0) + Intensität (Startwert z.B. 1.0)
    env_payload[0] = 4.0f32;  
    env_payload[1] = 7.0f32;  
    env_payload[2] = -4.0f32; 
    env_payload[3] = 1.0f32;  // light_intensity variabel machen
    
    // Block 1: Lichtfarbe (Weiß) + Ambient Strength
    env_payload[4] = 1.0f32;  // key_r
    env_payload[5] = 0.95f32; // key_g
    env_payload[6] = 0.85f32; // key_b
    env_payload[7] = 0.2f32;  // ambient_strength variabel machen
    
    // Block 2: Hintergrund/Nebel-Farbe + Dichte
    env_payload[8] = 0.35f32; // bg_r
    env_payload[9] = 0.45f32; // bg_g
    env_payload[10] = 0.60f32; // bg_b
    env_payload[11] = 1.0f32;  // fog_density (Normalisierungs-Faktor)
    
    // Block 3: Schalter (1.0 = AN)
    env_payload[12] = 1.0f32; // fog_enabled
    // Indizes 13, 14, 15 bleiben 0.0 (Padding)

    // Einmalige VRAM-Allokation für die Umweltsteuerung
    let env_handle = client.create(cubecl::bytes::Bytes::from_elems(env_payload.clone()));
```

---

## Schritt 3: Fluss-Update im Render-Loop (`src/main.rs`)

Erweitere deine Tastaturabfragen im `Event::AboutToWait`-Block. Wenn du jetzt `E/Q` (für Lichtstärke) oder neue Tasten für den Nebel drückst, schreiben wir die Werte direkt in das CPU-Array `env_payload` und jagen sie per `client.write` hoch:

```rust
                // Innerhalb von Event::AboutToWait bei den Rendering-Parametern:
                let mut env_changed = false;

                // Tasten für Lichtintensität (Q und E laut deinem Mapping)
                if keys.q { 
                    env_payload[3] = (env_payload[3] - 0.05f32).max(0.0f32); 
                    env_changed = true; 
                }
                if keys.e { 
                    env_payload[3] = (env_payload[3] + 0.05f32).min(5.0f32); 
                    env_changed = true; 
                }

                // Tasten für Ambient Strength (F und R laut deinem Mapping)
                if keys.f { 
                    env_payload[7] = (env_payload[7] - 0.01f32).max(0.0f32); 
                    env_changed = true; 
                }
                if keys.r { 
                    env_payload[7] = (env_payload[7] + 0.01f32).min(1.0f32); 
                    env_changed = true; 
                }

                // BEISPIEL: Neue Tasten für Nebeldichte (z.B. Numpad 4 und 6 oder KeyO/KeyL falls frei)
                if keys.fog_less { 
                    env_payload[11] = (env_payload[11] - 0.05f32).max(0.1f32); 
                    env_changed = true; 
                }
                if keys.fog_more { 
                    env_payload[11] = (env_payload[11] + 0.05f32).min(10.0f32); 
                    env_changed = true; 
                }

                // Wenn sich ein globaler Parameter geändert hat, flushen wir NUR diesen winzigen 16-Float-Tensor!
                if env_changed {
                    client.write(&env_handle, cubecl::bytes::Bytes::from_elems(env_payload.clone()));
                }
```

---

## Schritt 4: Pipeline-Launch anpassen (`src/main.rs`)

Direkt vor deinem `unsafe`-Launch Block deklarieren wir die Shape für den vierten Tensor (feste 16 Elemente) und reichen das `env_handle` an den Kernel weiter.

```rust
        // Shapes für den Launch deklarieren
        let env_shape: Vec<usize> = vec![16];
        let env_strides: Vec<usize> = Vec::<usize>::new();

        unsafe {
            kernel::raymarch_sdf_kernel::launch(
                &client,
                CubeCount::Static((width + 15) / 16, (height + 3) / 4, 1),
                cube_dim,
                
                TensorArg::from_raw_parts(output_handle.clone(), shape.into(), strides.into()),
                TensorArg::from_raw_parts(meta_handle.clone(), meta_shape.into(), meta_strides.into()),
                TensorArg::from_raw_parts(slots_handle.clone(), slots_shape.into(), slots_strides.into()),
                TensorArg::from_raw_parts(materials_handle.clone(), materials_shape.into(), materials_strides.into()),
                // 🟢 HIER: Der 4. Tensor wird als 5. Argument injiziert (nach output, meta, slots, materials)
                TensorArg::from_raw_parts(env_handle.clone(), env_shape.into(), env_strides.into()),
                
                // Ab hier folgen deine Skalare (time, width, height... Die alten Skalare für Lichtstärke/Ambient können jetzt gelöscht werden!)
                time, width, height, current_shadow_mode, cam_x, cam_y, cam_z, dynamic_blend_factor, enable_ao_mode, cam_yaw, cam_pitch,
                enable_key, enable_fill, enable_rim,
            );
        }
```

---

## Schritt 5: GPU-Kernel anpassen (`src/kernel.rs`)

Wir fügen den neuen `env_settings`-Tensor in den Funktionskopf von `raymarch_sdf_kernel` ein und extrahieren die Werte über saubere primitive Indizes direkt im Shader.

```rust
#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<u32>,
    meta: &Tensor<f32>,
    slots: &Tensor<f32>,
    materials: &Tensor<f32>,
    env_settings: &Tensor<f32>, // 🟢 NEU: Als 5. Argument empfangen
    time: f32,
    width: u32,
    height: u32,
    shadow_mode: u32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    blend_factor: f32,
    enable_ao_mode: u32,
    cam_yaw: f32,
    cam_pitch: f32,
    // light_intensity und ambient_strength wurden aus den Skalaren entfernt!
    enable_key: u32,   
    enable_fill: u32,  
    enable_rim: u32,   
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    // ... (deine UV- und Kamera-Rotationen wie gehabt) ...

    if x < width && y < height {
        
        // =========================================================================
        // DYNAMISCHE EXTRAKTION AUS DEM UMWELT-TENSOR (Perfektes Register-Mapping)
        // =========================================================================
        // Block 0: Licht-Vektor und Intensität
        let key_light_pos = Vec3::new(env_settings[usize::new(0)], env_settings[usize::new(1)], env_settings[usize::new(2)]);
        let l_intensity   = env_settings[usize::new(3)];

        // Block 1: Licht-Farben und globale Umgebung
        let key_r         = env_settings[usize::new(4)];
        let key_g         = env_settings[usize::new(5)];
        let key_b         = env_settings[usize::new(6)];
        let a_strength    = env_settings[usize::new(7)];

        // Block 2: Hintergrund-Farbe und Nebel-Dichte
        let bg_r          = env_settings[usize::new(8)];
        let bg_g          = env_settings[usize::new(9)];
        let bg_b          = env_settings[usize::new(10)];
        let fog_density   = env_settings[usize::new(11)];

        // Block 3: Schalter
        let fog_enabled   = env_settings[usize::new(12)];

        // ... (deine Raymarching loop läuft unverändert durch, reich meta, slots, materials weiter) ...

        if hit_dist < f32::new(40.0) {
            // ... (Normalenberechnung, Diffuse-Anteile wie gehabt) ...

            // Die Lichtberechnung nutzt nun völlig flüssig die gestreamten Werte!
            let lit_r = (key_r * diff_key * l_intensity) + (fill_r * diff_fill * f32::new(0.7)) + (rim_r * diff_rim.powf(f32::new(3.0)) * f32::new(1.5)) + specular_highlight + hit_emiss;
            // ... lit_g, lit_b berechnen ...

            let final_r = mat_r * (a_strength * ao_factor + lit_r * ao_factor);
            // ... final_g, final_b berechnen ...

            // 🟢 NEU: Dynamische, Puffer-gesteuerte Nebelberechnung
            let mut fog = f32::new(1.0);
            if fog_enabled == f32::new(1.0) {
                // Nutzt fog_density dynamisch als Teiler für die Nebel-Kompression
                fog = (f32::new(40.0) - hit_dist) / (f32::new(40.0) - (f32::new(15.0) * fog_density));
                if fog > f32::new(1.0) { fog = f32::new(1.0); }
                if fog < f32::new(0.0) { fog = f32::new(0.0); }
            }

            final_color_x = final_r * fog + bg_r * (f32::new(1.0) - fog);
            final_color_y = final_g * fog + bg_g * (f32::new(1.0) - fog);
            final_color_z = final_b * fog + bg_b * (f32::new(1.0) - fog);
        }
        
        // ... (Pixel Packen und in output[pixel_index] schreiben wie gehabt) ...
    }
}
```

_Hinweis:_ Reiche den `materials`-Tensor im Funktionskopf deiner Hilfsfunktionen (`scene_sdf`, `scene_sdf_normal`, `calculate_soft_shadow`, `calculate_ao`) wie in der vorherigen Nachricht gelernt weiter, aber du musst `env_settings` nicht in `scene_sdf` einspeisen, da die Umwelt-Parameter ausschließlich im Haupt-Shading des Kernels benötigt werden!

---

## Schritt 6: Cache bereinigen und flüssig steuern

Da wir das Kernel-Interface erneut verändert und Skalare gelöscht haben, jagen wir den JIT-Cache ein letztes Mal durch den Schredder:

```bash
cargo clean && rm -rf ~/.cache/cubecl* && RUSTFLAGS="-A warnings" cargo run --release
```

## Das Ergebnis:

Wenn du jetzt die Tasten `E` oder `Q` gedrückt hältst, erhöht oder verringert sich die Lichtintensität absolut stufenlos. Da das System nicht mehr auf träge Skalar-Injektionen oder teure Neu-Kompilierungen angewiesen ist, streamt `client.write` die 16 Gleitkommazahlen ohne jegliche Frametime-Verzögerung direkt in die Grafik-Register. Dein Labor reagiert in Echtzeit und die Nebeldichte lässt sich butterweich im Raum kalibrieren!

Möchtest du als Nächstes die genauen Tastatur-Verdrahtungen für das Nebel-Tuning (`fog_less`, `fog_more`) in der `main.rs` einrichten?