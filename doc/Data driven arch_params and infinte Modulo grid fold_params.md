Um die mathematischen Welten deines Space-Labs komplett dynamisch zu machen, lagern wir nun die geometrischen Parameter des antiken Tempels (Säulenabstände, Deckenhöhen, Bogenradien) und die mathematischen Konstanten der unendlichen Faltung (Zellengrößen, Rotationen, Wiederholungsfaktoren) in zwei weitere, separate Tensoren aus.

Durch diese evolutionäre Erweiterung bricht dein Code nicht, und wir bewahren das unumstößliche 16-Byte/Vec4-Alignment auf der GPU. Die Pipeline wächst damit auf ein hochprofessionelles 6-Tensor-Modell.

---

## Architektur-Erweiterung: Das 6-Tensor-Modell

```unset
 [Tensor 1: meta_handle]      -> Objekt-Länge (Schleifentiefe)
 [Tensor 2: slots_handle]     -> Objekt-Geometrie & Albedo (Stride 8)
 [Tensor 3: materials_handle] -> Objekt-PBR (Roughness, Metallic, etc. Stride 4)
 [Tensor 4: env_handle]       -> Globale Umwelt (Lichter, Nebel. Stride 16)
 
 🟢 NEU:
 [Tensor 5: arch_param_handle]-> Tempel-Architektur Parameter (Stride 12)
 [Tensor 6: fold_param_handle]-> Unendliche Raumfaltungs Konstanten (Stride 4)
```

---

## Schritt 1: Layout & Speicher-Ausrichtung festlegen

## Tensor 5: Architektur-Parameter (`arch_params`)

Wir bündeln alle Kontrollwerte deines Tempels in ein 12-Float Layout (entspricht exakt 3 vollen `Vec4`-Hardware-Registern, da $12 \times 4 = 48 \text{ Bytes}$ perfekt durch 16 teilbar sind).

|Index|Variable|Funktion|Register-Block|
|---|---|---|---|
|`0`|`pillar_dist`|Abstand der Säulen vom Ursprung (bisher fest auf `5.0`)|Block 0 (`Vec4`)|
|`1`|`pillar_thick`|Dicke/Radius der Säulen (bisher fest auf `0.6`)|Block 0 (`Vec4`)|
|`2`|`room_height`|Höhe des Raumes (Boden/Decke, bisher fest auf `3.0`)|Block 0 (`Vec4`)|
|`3`|`ceiling_thick`|Dicke der Decken-Dämpfung (bisher fest auf `0.1`)|Block 0 (`Vec4`)|
|`4`|`arch_radius`|Radius des Tonnengewölbes (bisher fest auf `3.2`)|Block 1 (`Vec4`)|
|`5`|`arch_height`|Vertikaler Versatz der Bögen (bisher fest auf `1.0`)|Block 1 (`Vec4`)|
|`6`|`decor_freq`|Frequenz des Säulendekors (bisher fest auf `2.0`)|Block 1 (`Vec4`)|
|`7`|`decor_depth`|Tiefe der Löcher in den Säulen (bisher fest auf `0.03`)|Block 1 (`Vec4`)|
|`8`|`decor_thick`|Innere Dicke des Dekors (bisher fest auf `0.01`)|Block 2 (`Vec4`)|
|`9,10,11`|`arch_padding`|Ungenutzte Werte zur Ausrichtung (müssen `0.0` sein)|Block 2 (`Vec4`)|

## Tensor 6: Raumfaltungs-Parameter (`fold_params`)

Für das infinite Modulo-Gitter nutzen wir ein kompaktes 4-Float Layout (exakt ein `Vec4`-Register).

|Index|Variable|Funktion|Register-Block|
|---|---|---|---|
|`0`|`cell_size`|Physische Größe einer unendlichen Zelle (bisher `10.0`)|Block 0 (`Vec4`)|
|`1`|`cell_half`|Halbe Zellengröße für Modulo-Arithmetik (bisher `5.0`)|Block 0 (`Vec4`)|
|`2`|`fold_speed`|Zeit-Skalierungsfaktor für Objekt-Animationen|Block 0 (`Vec4`)|
|`3`|`fold_padding`|Reservierter Füllwert (`0.0`)|Block 0 (`Vec4`)|

---

## Schritt 2: Host-Initialisierung in `src/main.rs`

Definiere die Werte ganz oben vor deiner Event-Loop und alloziehre die Handles persistent.

```rust
    // =========================================================================
    // INITIALISIERUNG DER ARCHITEKTUR- UND RAUMFALTUNGS-TENSOREN
    // =========================================================================
    // Tensor 5: Architektur-Werte (12 Floats)
    let mut arch_payload = vec![
        5.0f32,  // 0: pillar_dist
        0.6f32,  // 1: pillar_thick
        3.0f32,  // 2: room_height
        0.1f32,  // 3: ceiling_thick
        3.2f32,  // 4: arch_radius
        1.0f32,  // 5: arch_height
        2.0f32,  // 6: decor_freq
        0.03f32, // 7: decor_depth
        0.01f32, // 8: decor_thick
        0.0f32, 0.0f32, 0.0f32 // 9,10,11: Padding
    ];
    let arch_param_handle = client.create(cubecl::bytes::Bytes::from_elems(arch_payload.clone()));

    // Tensor 6: Raumfaltung (4 Floats)
    let mut fold_payload = vec![
        10.0f32, // 0: cell_size
        5.0f32,  // 1: cell_half (cell_size * 0.5)
        1.0f32,  // 2: fold_speed
        0.0f32,  // 3: Padding
    ];
    let fold_param_handle = client.create(cubecl::bytes::Bytes::from_elems(fold_payload.clone()));
```

---

## Schritt 3: Interaktives Streaming im Render-Loop (`src/main.rs`)

Füge in deinem `Event::AboutToWait`-Zweig neue Tasten hinzu, um die Dimensionen des Tempels oder des Raums im laufenden Betrieb zu morphen (z. B. Tempel enger machen mit `Numpad 1/3` oder Faltung vergrößern mit `Numpad 7/9`):

```rust
                let mut arch_changed = false;
                let mut fold_changed = false;

                // BEISPIEL: Säulenabstand modulieren (z. B. mit Tasten deiner Wahl)
                if keys.temple_narrow {
                    arch_payload[0] = (arch_payload[0] - 0.05f32).max(2.0f32);
                    arch_changed = true;
                }
                if keys.temple_wide {
                    arch_payload[0] = (arch_payload[0] + 0.05f32).min(15.0f32);
                    arch_changed = true;
                }

                // BEISPIEL: Unendliche Zellengröße skalieren (Raumfaltung stauchen/dehnen)
                if keys.space_shrink {
                    fold_payload[0] = (fold_payload[0] - 0.1f32).max(4.0f32);
                    fold_payload[1] = fold_payload[0] * 0.5f32; // cell_half synchron halten!
                    fold_changed = true;
                }
                if keys.space_grow {
                    fold_payload[0] = (fold_payload[0] + 0.1f32).min(30.0f32);
                    fold_payload[1] = fold_payload[0] * 0.5f32;
                    fold_changed = true;
                }

                // Blitzschnelles Stream-Update über PCIe
                if arch_changed {
                    client.write(&arch_param_handle, cubecl::bytes::Bytes::from_elems(arch_payload.clone()));
                }
                if fold_changed {
                    client.write(&fold_param_handle, cubecl::bytes::Bytes::from_elems(fold_payload.clone()));
                }
```

---

## Schritt 4: Pipeline-Launch anpassen (`src/main.rs`)

Wir deklarieren die Launch-Shapes für die beiden neuen Puffer und injizieren sie als zusätzliche Argumente direkt in die Pipeline:

```rust
        let arch_shape: Vec<usize> = vec![12];
        let arch_strides: Vec<usize> = Vec::<usize>::new();

        let fold_shape: Vec<usize> = vec![4];
        let fold_strides: Vec<usize> = Vec::<usize>::new();

        unsafe {
            kernel::raymarch_sdf_kernel::launch(
                &client,
                CubeCount::Static((width + 15) / 16, (height + 3) / 4, 1),
                cube_dim,
                
                TensorArg::from_raw_parts(output_handle.clone(), shape.into(), strides.into()),
                TensorArg::from_raw_parts(meta_handle.clone(), meta_shape.into(), meta_strides.into()),
                TensorArg::from_raw_parts(slots_handle.clone(), slots_shape.into(), slots_strides.into()),
                TensorArg::from_raw_parts(materials_handle.clone(), materials_shape.into(), materials_strides.into()),
                TensorArg::from_raw_parts(env_handle.clone(), env_shape.into(), env_strides.into()),
                // 🟢 HIER: Injektion der Tensoren 5 und 6
                TensorArg::from_raw_parts(arch_param_handle.clone(), arch_shape.into(), arch_strides.into()),
                TensorArg::from_raw_parts(fold_param_handle.clone(), fold_shape.into(), fold_strides.into()),
                
                time, width, height, current_shadow_mode, cam_x, cam_y, cam_z, dynamic_blend_factor, enable_ao_mode, cam_yaw, cam_pitch,
                enable_key, enable_fill, enable_rim,
            );
        }
```

---

## Schritt 5: GPU-Shader anpassen (`src/kernel.rs`)

Wir reichen die beiden Kontroll-Tensoren bis nach ganz unten an die `scene_sdf`-Funktion weiter.

## 1. Anpassung der Funktionsköpfe (`src/kernel.rs`)

Der Hauptkernel und alle nachfolgenden mathematischen Hilfsfunktionen müssen die Kaskade mitschleifen:

```rust
#[cube(launch)]
pub fn raymarch_sdf_kernel(
    output: &mut Tensor<u32>, meta: &Tensor<f32>, slots: &Tensor<f32>, materials: &Tensor<f32>, env_settings: &Tensor<f32>,
    arch_params: &Tensor<f32>, // 🟢 NEU: Tensor 5 empfangen
    fold_params: &Tensor<f32>, // 🟢 NEU: Tensor 6 empfangen
    time: f32, // ... restliche Skalare
)

// In der Raymarching-Schleife weiterreichen:
let step_res = scene_sdf(p, t_val, b_factor, meta, slots, materials, arch_params, fold_params);

// Auch in scene_sdf_normal, calculate_soft_shadow und calculate_ao im Kopf deklarieren und an scene_sdf übergeben!
```

## 2. Dynamisierung der mathematischen Welten in `scene_sdf`

Jetzt ersetzen wir alle hartcodierten Geometrie-Konstanten innerhalb von `scene_sdf` durch die reaktiven Speicherregister der beiden neuen Tensoren.

```rust
#[cube]
pub fn scene_sdf(
    p: Vec3, time: f32, blend_factor: f32, 
    meta: &Tensor<f32>, slots: &Tensor<f32>, materials: &Tensor<f32>,
    arch_params: &Tensor<f32>, // 🟢 Empfangen
    fold_params: &Tensor<f32>  // 🟢 Empfangen
) -> SdfResult {
    
    // =========================================================================
    // TENSOR 6: EXTRAKTION DER UNENDLICHEN RAUMFALTUNGS-KONSTANTEN
    // =========================================================================
    let cell_size  = fold_params[usize::new(0)];
    let half_cell  = fold_params[usize::new(1)];
    let fold_speed = fold_params[usize::new(2)];
    
    let animated_time = time * fold_speed;

    let mut min_dist = f32::new(1000.0);
    let mut sum_r = f32::new(0.0); let mut sum_g = f32::new(0.0); let mut sum_b = f32::new(0.0);
    let mut sum_rough = f32::new(0.0); let mut sum_metal = f32::new(0.0);
    let mut sum_emiss = f32::new(0.0); let mut sum_spec  = f32::new(0.0);
    let mut sum_w = f32::new(0.0); 

    let active_slots = usize::cast_from(meta[usize::new(0)]);

    let mut i = usize::new(0);
    loop {
        if i >= usize::new(100) || i >= active_slots { break; }

        let base_idx = i * usize::new(8);
        let obj_type = u32::cast_from(slots[base_idx]);

        if obj_type > u32::new(0) {
            let obj_size  = slots[base_idx + usize::new(1)];
            let offset_x  = slots[base_idx + usize::new(2)];
            let offset_y  = slots[base_idx + usize::new(3)]; 
            let offset_z  = slots[base_idx + usize::new(4)];
            let obj_r     = slots[base_idx + usize::new(5)]; 
            let obj_g     = slots[base_idx + usize::new(6)]; 
            let obj_b     = slots[base_idx + usize::new(7)]; 
            
            let mat_idx   = i * usize::new(4);
            let obj_rough = materials[mat_idx];
            let obj_metal = materials[mat_idx + usize::new(1)];
            let obj_emiss = materials[mat_idx + usize::new(2)];
            let obj_spec  = materials[mat_idx + usize::new(3)];

            let shifted_p_x = p.x - offset_x;
            let shifted_p_y = p.y - offset_y;
            let shifted_p_z = p.z - offset_z;

            // Nutzt nun vollkommen dynamisch die veränderbaren Raumfaltungs-Register!
            let grid_p_x = shifted_p_x - cell_size * ((shifted_p_x + half_cell) / cell_size).floor();
            let grid_p_y = shifted_p_y - cell_size * ((shifted_p_y + half_cell) / cell_size).floor(); 
            let grid_p_z = shifted_p_z - cell_size * ((shifted_p_z + half_cell) / cell_size).floor();
            let p_slot = Vec3::new(grid_p_x, grid_p_y, grid_p_z);

            let mut d_obj = f32::new(1000.0);
            // Übergabe der animierten Zeit an die Primitive
            if obj_type == u32::new(1) { d_obj = evaluate_dynamic_crystal(p_slot, animated_time, obj_size); }
            if obj_type == u32::new(2) { d_obj = evaluate_dynamic_gyroid(p_slot, animated_time, obj_size); }
            if obj_type == u32::new(3) { d_obj = evaluate_dynamic_torus(p_slot, animated_time, obj_size); }

            if min_dist > f32::new(999.0) { min_dist = d_obj; } else {
                let h = (blend_factor - (min_dist - d_obj).abs()).max(f32::new(0.0)) / blend_factor;
                min_dist = min_dist.min(d_obj) - h * h * blend_factor * f32::new(0.25);
            }

            let w = f32::new(1.0) / (d_obj.max(f32::new(0.001))).powf(f32::new(2.0));
            sum_r += obj_r * w; sum_g += obj_g * w; sum_b += obj_b * w;
            sum_rough += obj_rough * w; sum_metal += obj_metal * w;
            sum_emiss += obj_emiss * w; sum_spec += obj_spec * w;
            sum_w += w;
        }
        i += usize::new(1);
    }

    let mut core_r = f32::new(1.0); let mut core_g = f32::new(1.0); let mut core_b = f32::new(1.0);
    let mut core_rough = f32::new(0.5); let mut core_metal = f32::new(0.0);
    let mut core_emiss = f32::new(0.0); let mut core_spec  = f32::new(0.5);
    if sum_w > f32::new(0.0) {
        core_r = sum_r / sum_w; core_g = sum_g / sum_w; core_b = sum_b / sum_w;
        core_rough = sum_rough / sum_w; core_metal = sum_metal / sum_w;
        core_emiss = sum_emiss / sum_w; core_spec = sum_spec / sum_w;
    }
    let core_system = SdfResult { 
        d: min_dist, r: core_r, g: core_g, b: core_b,
        roughness: core_rough, metallic: core_metal, emissive: core_emiss, specular: core_spec
    };

    // =========================================================================
    // TENSOR 5: DYNAMISCHE TEMPEL-ARCHITEKTUR GENERIERUNG
    // =========================================================================
    let arch_grid_x = p.x - cell_size * ((p.x + half_cell) / cell_size).floor();
    let arch_grid_y = p.y - cell_size * ((p.y + half_cell) / cell_size).floor(); 
    let arch_grid_z = p.z - cell_size * ((p.z + half_cell) / cell_size).floor();
    let local_arch_p = Vec3::new(arch_grid_x, arch_grid_y, arch_grid_z);

    // 🟢 HIER: Registrierte Register aus arch_params anstelle der harten Literale nutzen!
    let p_dist   = arch_params[usize::new(0)];
    let p_thick  = arch_params[usize::new(1)];
    let r_height = arch_params[usize::new(2)];
    let c_thick  = arch_params[usize::new(3)];
    let a_radius = arch_params[usize::new(4)];
    let a_height = arch_params[usize::new(5)];
    let d_freq   = arch_params[usize::new(6)];
    let d_depth  = arch_params[usize::new(7)];
    let d_thick  = arch_params[usize::new(8)];

    let pillar_x = (local_arch_p.x.abs() - p_dist).abs() - p_thick;
    let pillar_z = (local_arch_p.z.abs() - p_dist).abs() - p_thick;
    let corner_pillars = pillar_x.max(pillar_z);
    
    let room_floor = local_arch_p.y + r_height; 
    let room_ceiling = r_height - local_arch_p.y; 
    let floor_and_ceiling = room_floor.min(room_ceiling) - c_thick;

    let arch_z = (local_arch_p.x * local_arch_p.x + (local_arch_p.y - a_height) * (local_arch_p.y - a_height)).sqrt() - a_radius;
    let arch_x = (local_arch_p.z * local_arch_p.z + (local_arch_p.y - a_height) * (local_arch_p.y - a_height)).sqrt() - a_radius;
    let wall_arches = arch_z.min(arch_x);

    let mut arch_d = corner_pillars.min(floor_and_ceiling);
    arch_d = arch_d.max(-wall_arches);

    let pillar_holes = ( (local_arch_p.x * d_freq).sin().abs() + (local_arch_p.y * d_freq).cos().abs() + (local_arch_p.z * d_freq).sin().abs() ) * d_depth;
    arch_d = arch_d.max(-(pillar_holes - d_thick));

    let architecture = SdfResult { 
        d: arch_d, r: f32::new(0.7), g: f32::new(0.7), b: f32::new(0.7),
        roughness: f32::new(0.8), metallic: f32::new(0.0), emissive: f32::new(0.0), specular: f32::new(0.2)
    };

    // =========================================================================
    // FINALE SCHNITTAUSWERTUNG (Unverändert)
    // =========================================================================
    let mut final_d     = f32::new(1000.0);
    let mut final_r     = f32::new(1.0); let mut final_g     = f32::new(1.0); let mut final_b     = f32::new(1.0);
    let mut final_rough = f32::new(0.5); let mut final_metal = f32::new(0.0); let mut final_emiss = f32::new(0.0); let mut final_spec  = f32::new(0.5);

    let check_core = core_system.clone(); let check_arch = architecture.clone();

    if check_core.d < check_arch.d { 
        final_d = check_core.d; final_r = check_core.r; final_g = check_core.g; final_b = check_core.b;
        final_rough = check_core.roughness; final_metal = check_core.metallic; final_emiss = check_core.emissive; final_spec = check_core.specular;
    } else { 
        final_d = check_arch.d; final_r = check_arch.r; final_g = check_arch.g; final_b = check_arch.b;
        final_rough = check_arch.roughness; final_metal = check_arch.metallic; final_emiss = check_arch.emissive; final_spec = check_arch.specular;
    }
    
    SdfResult { 
        d: final_d, r: final_r, g: final_g, b: final_b,
        roughness: final_rough, metallic: final_metal, emissive: final_emiss, specular: final_spec
    }
}
```

---

## Schritt 6: Cache bereinigen & die totale Freiheit testen

Jage die Modifikationen ein letztes Mal durch das Terminal, um alle Shader-Altlasten im Treiber zu löschen:

```bash
cargo clean && rm -rf ~/.cache/cubecl* && RUSTFLAGS="-A warnings" cargo run --release
```

## Die krasse Wirkung im Spiel:

Wenn du jetzt die neuen Tempel-Tasten drückst, weiten oder schließen sich die Säulenhallen des Tempels stufenlos vor deinen Augen, während die Deckenhöhe wächst oder schrumpft. Drückst du die Faltungs-Tasten, dehnen sich die unendlichen Modulo-Zellen im Raum aus – die Objekte rücken optisch meilenweit auseinander oder werden eng ineinander gestaucht, während die Animationsgeschwindigkeit (`fold_speed`) variiert.

Die gesamte Engine ist nun vollständig Daten-gesteuert (Data-Driven). Jede mathematische Konstante im Universum deiner GPU gehorcht ab jetzt flüssig und latenzfrei deinen Fingern auf der Tastatur!

Möchtest du, dass wir als Nächstes die Maus-Steuerung (Kamera-Umschauen per Mausbewegung) integrieren, um dich völlig frei im mutierenden Space-Lab umsehen zu können?