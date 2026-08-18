mod kernel;
//pub mod shadows;
//pub mod ambient_occlusion;

use cubecl::prelude::*;

//use cubecl::wgpu::WgpuRuntime;
use cubecl::client::ComputeClient; // Expliziter Import des Client-Typs
use cubecl::wgpu::{WgpuRuntime, AutoCompiler};

//use cubecl::wgpu::AutoCompiler; // Make sure this is in scope
use softbuffer::{Context, Surface};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Instant, Duration};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, KeyEvent, ElementState};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

enum BackgroundMessage {
    LevelDataLoaded { level_id: u32, blend_factor: f32 },
}

struct KeyboardState {
    w: bool,
    s: bool,
    a: bool,
    d: bool,
    space: bool,
    c: bool,
    i: bool,
    k: bool,
    j: bool,
    l: bool,
    u: bool, // HIER HINZUGEFÜGT
    o: bool, // HIER HINZUGEFÜGT
    p: bool, // HIER HINZUGEFÜGT
    h: bool, // HIER HINZUGEFÜGT
    //
    // FIX: Die 4 neuen Felder für die Transformation hinzufügen!
    grow: bool,
    shrink: bool,
    y_up: bool,
    y_down: bool,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
   // let width = 800u32;
   // let height = 600u32;
   // let width = 1920u32;
   // let height = 1080u32;
    let width = 960u32;
    let height = 540u32;
    let total_pixels = (width * height) as usize;
    let byte_size = total_pixels * std::mem::size_of::<u32>();

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<BackgroundMessage>();
    let (_trigger_tx, mut trigger_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();

    rt.spawn(async move {
        while let Some(level_to_load) = trigger_rx.recv().await {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let _ = tx.send(BackgroundMessage::LevelDataLoaded {
                level_id: level_to_load,
                blend_factor: 99.0,
            });
        }
    });

    let device = Default::default();
    //let client = WgpuRuntime::client(&device); // 0.10
    //let client = WgpuRuntime::client::<AutoCompiler>(&device); //0.11.0-pre.2
    // Specifying the compiler generic directly on WgpuRuntime or letting type inference handle it:
// Wir geben dem Compiler den konkreten Typ vor, damit er weiß, dass AutoCompiler genutzt wird
let client: ComputeClient<cubecl::wgpu::WgpuRuntime<AutoCompiler>> = 
    cubecl::wgpu::WgpuRuntime::client(&device);

    let output_handle = client.empty(byte_size);

    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Temple of Wisdom | Optimized State-Change Buffer")
            .with_inner_size(LogicalSize::new(width, height))
            .with_resizable(false)
            .build(&event_loop)?
    );

      let context = Context::new(window.clone())?;
    let mut surface = Surface::new(&context, window.clone())?;
    surface.resize(NonZeroU32::new(width).unwrap(), NonZeroU32::new(height).unwrap())?;

    let mut cam_x = 0.0f32; let mut cam_y = 0.0f32; let mut cam_z = -4.5f32;
    let mut cam_yaw = 0.0f32; let mut cam_pitch = 0.0f32; 
    let mut current_shadow_mode = 2u32;
    let mut enable_ao_mode = 1u32; 
    let mut dynamic_blend_factor = 0.2f32;
    let mut light_intensity = 1.0f32; let mut ambient_strength = 0.15f32; 
    let mut enable_key = 1u32; let mut enable_fill = 1u32; let mut enable_rim = 1u32;

       // =========================================================================
    // DYNAMISCHE SLOT-KONFIGURATION (Vektoren für CPU-Logik)
    // =========================================================================
    let mut active_slots_count = 3.0f32; // Zählt nur die ungleich 0.0 (Typ 1, 2, 3)
    
    let mut slot_types = vec![1.0f32, 2.0f32, 3.0f32, 0.0f32, 0.0f32]; 
    let mut slot_sizes = vec![1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32]; 

    let mut slot_offsets_x = vec![0.0f32, -1.8f32, 1.8f32, 0.0f32, 0.0f32];  
    let mut slot_offsets_z = vec![0.0f32, 0.0f32, 0.0f32, 1.8f32, -1.8f32];
    let mut slot_offsets_y = vec![0.0f32; 5]; // Start-Höhe für alle 5 Slots ist 0.0

    // Dieses Flag signalisiert, ob eine Transformation den Puffer überschreiben muss
    let mut config_dirty = true;

    // =========================================================================
    // INITIALISIERUNG DES STATISCHEN GPU-CONFIG-FIELDS (Zero Allocation Lifecycle)
    // =========================================================================
    const MAX_SLOTS: usize = 100;
    //const CONFIG_BUFFER_FLOATS: usize = 1 + MAX_SLOTS * 4; // Header + 100 Slots * 4 Felder
    const CONFIG_BUFFER_FLOATS: usize = 1 + MAX_SLOTS * 5; // 5 Floats pro Slot!

    // Erzeuge ein fixes Speicherfeld auf der CPU, initialisiert mit Nullen
    let mut raw_config = vec![0.0f32; CONFIG_BUFFER_FLOATS];
    
    // Header an Index 0 schreiben
    raw_config[0] = active_slots_count;
    
    // Verpackt die initialen Slots sequentiell im interleaved Format in das feste Feld
    for i in 0..slot_types.len() {
        if i >= MAX_SLOTS { break; }
        let base = 1 + i * 4;
        raw_config[base]     = slot_types[i];
        raw_config[base + 1] = slot_sizes[i];
        raw_config[base + 2] = slot_offsets_x[i];
        raw_config[base + 3] = slot_offsets_z[i];
    }
    
    // Invariant: Die Elementanzahl für den Tensor-Launch bleibt starr auf dem Maximum!
    let total_elements = CONFIG_BUFFER_FLOATS;
    
    // DIE EINMALIGE ALLOKATION: Das Handle wird ab jetzt NIEMALS wieder neu instanziiert!
    let config_handle = client.create(cubecl::bytes::Bytes::from_elems(raw_config.clone()));

    // BEHOBEN: Alle Felder werden direkt innerhalb des Struct-Initialisierers auf false gesetzt!
    let mut keys = KeyboardState {
        w: false,
        s: false,
        a: false,
        d: false,
        space: false,
        c: false,
        i: false,
        k: false,
        j: false,
        l: false,
        u: false, // Direkt im Struct gesetzt
        o: false,
        p: false,
        h: false,
        // FIX: Die 4 neuen Felder für die Transformation hinzufügen!
    grow: false,
    shrink: false,
    y_up: false,
    y_down: false,
    };

    // HIER die fehlende Selektionsvariable direkt darunter anlegen (behebt E0425)
    let mut current_selected_slot: usize = 0; 

        let start_time = Instant::now();
    println!("[Main] Dynamische Engine gestartet. Reaktive Puffer-Steuerung aktiv.");

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            // Fenster-Events abfangen und Tastatur-Eingaben sauber entpacken
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(code), state, .. }, .. } => {
                    let is_pressed = state == ElementState::Pressed;
                    
                    match code {
                        // 1. KAMERA-NAVIGATION & ROTATION (Dauerhaftes Halten der Tasten)
                        KeyCode::KeyW => keys.w = is_pressed,
                        KeyCode::KeyS => keys.s = is_pressed,
                        KeyCode::KeyA => keys.a = is_pressed,
                        KeyCode::KeyD => keys.d = is_pressed,
                        KeyCode::Space => keys.space = is_pressed,
                        KeyCode::KeyC => keys.c = is_pressed,
                        KeyCode::KeyI => keys.i = is_pressed, // Kamera Pitch Up
                        KeyCode::KeyK => keys.k = is_pressed, // Kamera Pitch Down
                        KeyCode::KeyJ => keys.j = is_pressed, // Kamera Yaw Left
                        KeyCode::KeyL => keys.l = is_pressed, // Kamera Yaw Right
                        //
                        // FREIE TASTEN GEFUNDEN: Objekt-Transformation (Y-Achse auf E und Q)
                        KeyCode::KeyZ => keys.y_up = is_pressed,
                        KeyCode::KeyX => keys.y_down = is_pressed,

                        // GLOBALE SKALIERUNG / GRÖSSE (Plus und Minus)
                                                // SAUBERES FIX FÜR NORMALE TASTATUREN (+ / - ohne Numpad)
                        KeyCode::KeyG => keys.grow = is_pressed,
                        KeyCode::KeyY => keys.shrink = is_pressed,

                                               // 2. DYNAMISCHES SLOT-SPAWNING & SELEKTION (Slots 1 bis 5)
                                               // =========================================================================
                        // DYNAMISCHES SLOT-SPAWNING & SELEKTION (Slots 1 bis 6+)
                        // =========================================================================
                        KeyCode::Digit1 if is_pressed => { 
                            current_selected_slot = 0; 
                            slot_types[0] = if slot_types[0] == 0.0 { 1.0 } else if slot_types[0] == 1.0 { 2.0 } else if slot_types[0] == 2.0 { 3.0 } else { 0.0 }; 
                            // BUGFIX: Schicke der GPU die tatsächliche Länge, nicht die Anzahl der gefüllten Slots!
                            active_slots_count = slot_types.len() as f32; 
                            config_dirty = true; 
                            println!("[Space-Lab] Slot 1 ausgewählt & geändert: {}", slot_types[0]); 
                        }
                        KeyCode::Digit2 if is_pressed => { 
                            current_selected_slot = 1; 
                            slot_types[1] = if slot_types[1] == 0.0 { 1.0 } else if slot_types[1] == 1.0 { 2.0 } else if slot_types[1] == 2.0 { 3.0 } else { 0.0 }; 
                            active_slots_count = slot_types.len() as f32; 
                            config_dirty = true; 
                            println!("[Space-Lab] Slot 2 ausgewählt & geändert: {}", slot_types[1]); 
                        }
                        KeyCode::Digit3 if is_pressed => { 
                            current_selected_slot = 2; 
                            slot_types[2] = if slot_types[2] == 0.0 { 1.0 } else if slot_types[2] == 1.0 { 2.0 } else if slot_types[2] == 2.0 { 3.0 } else { 0.0 }; 
                            active_slots_count = slot_types.len() as f32; 
                            config_dirty = true; 
                            println!("[Space-Lab] Slot 3 ausgewählt & geändert: {}", slot_types[2]); 
                        }
                        KeyCode::Digit4 if is_pressed => { 
                            current_selected_slot = 3; 
                            slot_types[3] = if slot_types[3] == 0.0 { 1.0 } else if slot_types[3] == 1.0 { 2.0 } else if slot_types[3] == 2.0 { 3.0 } else { 0.0 }; 
                            active_slots_count = slot_types.len() as f32; 
                            config_dirty = true; 
                            println!("[Space-Lab] Slot 4 ausgewählt & geändert: {}", slot_types[3]); 
                        }
                        KeyCode::Digit5 if is_pressed => { 
                            current_selected_slot = 4; 
                            slot_types[4] = if slot_types[4] == 0.0 { 1.0 } else if slot_types[4] == 1.0 { 2.0 } else if slot_types[4] == 2.0 { 3.0 } else { 0.0 }; 
                            active_slots_count = slot_types.len() as f32; 
                            config_dirty = true; 
                            println!("[Space-Lab] Slot 5 ausgewählt & geändert: {}", slot_types[4]); 
                        }

                        // GEFIXT: Vollkommen dynamischer 6. Slot (Erzeugung UND unendliches Umschalten)
                                               // GEFIXT FÜR 5-FLOAT LAYOUT: Vollkommen dynamischer 6. Slot
                        KeyCode::Digit6 if is_pressed => {
                            current_selected_slot = 5; 
                            
                            if slot_types.len() <= current_selected_slot {
                                slot_types.push(1.0f32);       
                                slot_sizes.push(1.0f32);       
                                slot_offsets_x.push(0.0f32);   
                                slot_offsets_y.push(0.0f32);   // NEU: Initialer Y-Offset für Slot 6
                                slot_offsets_z.push(0.0f32);
                                println!("[Space-Lab] 🚀 Slot 6 NEU ERZEUGT! Typ: Kristall (1.0)");
                            } else {
                                let idx = current_selected_slot;
                                slot_types[idx] = if slot_types[idx] == 0.0 { 1.0 } else if slot_types[idx] == 1.0 { 2.0 } else if slot_types[idx] == 2.0 { 3.0 } else { 0.0 };
                                println!("[Space-Lab] Slot 6 Zustand gewechselt auf: {}", slot_types[idx]); 
                            }
                            
                            active_slots_count = slot_types.len() as f32; 
                            config_dirty = true;
                        }


                        // 3. ECHTZEIT-OBJEKT-TRANSFORMATION (Nutzt U, O, P, H)
                        KeyCode::KeyU => keys.u = is_pressed, 
                        KeyCode::KeyO => keys.o = is_pressed, 
                        KeyCode::KeyP => keys.p = is_pressed, 
                        KeyCode::KeyH => keys.h = is_pressed,
                        ////

                        // 4. GLOBALE SKALIERUNG DER OBJEKTE
                        KeyCode::KeyM if is_pressed => {
                            for s in slot_sizes.iter_mut() { *s = (*s + 0.1f32).min(2.5f32); }
                            config_dirty = true;
                            println!("[Space-Lab] Alle Objekte vergrößert.");
                        }
                        KeyCode::KeyN if is_pressed => {
                            for s in slot_sizes.iter_mut() { *s = (*s - 0.1f32).max(0.2f32); }
                            config_dirty = true;
                            println!("[Space-Lab] Alle Objekte verkleinert.");
                        }

                        // 5. RENDERING-PARAMETER (Licht, Schatten, Ambient Occlusion)
                        KeyCode::Digit0 if is_pressed => { 
                            enable_ao_mode = if enable_ao_mode == 1 { 0 } else { 1 }; 
                            println!("[Space-Lab] Ambient Occlusion gewechselt: {}", if enable_ao_mode == 1 { "EIN" } else { "AUS" });
                        }
                        KeyCode::KeyQ if is_pressed => { light_intensity = (light_intensity - 0.1f32).max(0.0f32); }
                        KeyCode::KeyE if is_pressed => { light_intensity = (light_intensity + 0.1f32).min(3.0f32); }
                        KeyCode::KeyF if is_pressed => { ambient_strength = (ambient_strength - 0.05f32).max(0.0f32); }
                        KeyCode::KeyR if is_pressed => { ambient_strength = (ambient_strength + 0.05f32).min(1.0f32); }
                        
                        KeyCode::Digit7 if is_pressed => { enable_key = if enable_key == 1 { 0 } else { 1 }; }
                        KeyCode::Digit8 if is_pressed => { enable_fill = if enable_fill == 1 { 0 } else { 1 }; }
                        KeyCode::Digit9 if is_pressed => { enable_rim = if enable_rim == 1 { 0 } else { 1 }; }
                        
                        KeyCode::ArrowRight if is_pressed => { if current_shadow_mode < 2 { current_shadow_mode += 1; } }
                        KeyCode::ArrowLeft if is_pressed => { if current_shadow_mode > 0 { current_shadow_mode -= 1; } }
                        _ => {}
                    }
                }
                _ => {}
            }, 

                        Event::AboutToWait => {
                while let Ok(message) = rx.try_recv() {
                    match message { BackgroundMessage::LevelDataLoaded { blend_factor, .. } => { dynamic_blend_factor = blend_factor; } }
                }

                // 1. KAMERA-NAVIGATION
                let look_speed = 0.02f32;
                if keys.i { cam_pitch += look_speed; } if keys.k { cam_pitch -= look_speed; }
                if keys.j { cam_yaw -= look_speed; } if keys.l { cam_yaw += look_speed; }
                cam_pitch = cam_pitch.clamp(-1.4, 1.4);

                let move_speed = 0.29f32;
                let cos_y = f32::cos(cam_yaw); let sin_y = f32::sin(cam_yaw);
                if keys.w { cam_x += sin_y * move_speed; cam_z += cos_y * move_speed; }
                if keys.s { cam_x -= sin_y * move_speed; cam_z -= cos_y * move_speed; }
                if keys.a { cam_x -= cos_y * move_speed; cam_z -= -sin_y * move_speed; }
                if keys.d { cam_x += cos_y * move_speed; cam_z += -sin_y * move_speed; }
                if keys.space { cam_y += move_speed; } if keys.c { cam_y -= move_speed; }

                // 2. OBJEKT-TRANSFORMATIONEN (Stufenloser Echtzeit-Stream)
                let obj_speed = 0.05f32;
                let scale_speed = 0.02f32;
                let sel = current_selected_slot; 
                let mut object_changed = false;

                // Sicherheitsabfrage: Nur transformieren, wenn der Slot existiert
                if sel < slot_types.len() {
                    // X-Achse (U / O)
                    if keys.u { slot_offsets_x[sel] += obj_speed; object_changed = true; } 
                    if keys.o { slot_offsets_x[sel] -= obj_speed; object_changed = true; } 
                    
                    // Z-Achse (P / H)
                    if keys.p { slot_offsets_z[sel] += obj_speed; object_changed = true; } 
                    if keys.h { slot_offsets_z[sel] -= obj_speed; object_changed = true; } 

                    // NEU: Y-Achse (Hoch / Runter via Pfeiltasten oder Tasten deiner Wahl, z.B. KeyN / KeyM)
                    // Stelle sicher, dass du diese Tasten in deinem KeyboardInput-Match abfängst!
                    if keys.y_up   { slot_offsets_y[sel] += obj_speed; object_changed = true; }
                    if keys.y_down { slot_offsets_y[sel] -= obj_speed; object_changed = true; }

                    // NEU: Skalierung / Größe (Größer / Kleiner via KeyPlus / KeyMinus oder z.B. KeyG / KeyT)
                    if keys.grow   { slot_sizes[sel] = (slot_sizes[sel] + scale_speed).min(5.0); object_changed = true; }
                    if keys.shrink { slot_sizes[sel] = (slot_sizes[sel] - scale_speed).max(0.1); object_changed = true; }
                }

                if object_changed {
                    config_dirty = true;
                }

                let time = start_time.elapsed().as_secs_f32();
                
                // 3. HOCHEFFIZIENTES GPU-STREAMING (CubeCL 0.11 Ready)
                if config_dirty {
                    let mut dynamic_raw_config = vec![0.0f32; CONFIG_BUFFER_FLOATS];
                    dynamic_raw_config[0] = active_slots_count;
                    
                    // Befüllen des neuen 5-Float Interleaved Rasters
                    for i in 0..slot_types.len() {
                        if i >= MAX_SLOTS { 
                            break; 
                        }
                        let base = 1 + i * 5; // Multiplikator auf 5 erhöht!
                        dynamic_raw_config[base]     = slot_types[i];
                        dynamic_raw_config[base + 1] = slot_sizes[i];
                        dynamic_raw_config[base + 2] = slot_offsets_x[i];
                        dynamic_raw_config[base + 3] = slot_offsets_y[i]; // NEU im Puffer!
                        dynamic_raw_config[base + 4] = slot_offsets_z[i];
                    }
                    
                    let config_bytes = cubecl::bytes::Bytes::from_elems(dynamic_raw_config);
                    client.write(&config_handle, config_bytes);
                    
                    config_dirty = false;
                }




// 1. Beide Metadaten-Vektoren MÜSSEN in CubeCL 0.11 usize nutzen
let shape = vec![total_pixels].into();
let strides = Vec::<usize>::new().into(); 

let config_shape = vec![total_elements].into();
let config_strides = Vec::<usize>::new().into(); // Von u32 auf usize korrigiert

// 2. CubeDim Initialisierung für 0.11 angepasst

// ✅ Der offizielle und sauberste Weg für 3D-Dimensionen in CubeCL 0.11
let cube_dim = CubeDim::new_3d(16, 4, 1);


unsafe {
    kernel::raymarch_sdf_kernel::launch(
        &client,
        CubeCount::Static((width + 15) / 16, (height + 3) / 4, 1),
        cube_dim,
        TensorArg::from_raw_parts(output_handle.clone(), shape, strides),
        TensorArg::from_raw_parts(config_handle.clone(), config_shape, config_strides),
        
        // Die Skalare werden als normale Rust-Primitives (f32/u32) übergeben,
        // da wir im Kernel-Kopf die Referenzen `&` entfernt haben!
        time, 
        width, 
        height, 
        current_shadow_mode,
        cam_x, 
        cam_y, 
        cam_z, 
        dynamic_blend_factor, 
        enable_ao_mode,
        cam_yaw, 
        cam_pitch, 
        light_intensity, 
        ambient_strength,
        enable_key, 
        enable_fill, 
        enable_rim,
    );
}

// 3. Synchronisation und Read-Back
let _ = client.sync();
let mut bytes_vec = client.read(vec![output_handle.clone()]);

if let Some(bytes) = bytes_vec.pop() {
    let raw_bytes: &[u8] = &*bytes;
    if raw_bytes.len() >= byte_size {
        let results: &[u32] = bytemuck::cast_slice(&raw_bytes[..byte_size]);
        let mut buffer = surface.buffer_mut().unwrap();
        buffer.copy_from_slice(results);
        buffer.present().unwrap();
    }
}

window.request_redraw();
std::thread::sleep(Duration::from_millis(8));

}_ => {}}})?;Ok(())}