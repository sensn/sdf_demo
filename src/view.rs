//! wgpui-View: `SurfaceExample` — Host der wgpu-Surface + Sidebar + Overlay.
//!
//! Der View ist rein deklarativ: Er rendert das Layout (Surface-Element,
//! FPS-Overlay, Sidebar), draint den FPS-Channel und fängt Key-Events.
//! Das eigentliche GPU-Rendering macht der Render-Thread (render_loop.rs).

use crate::app_state::ApplicationState;
// Am Anfang von src/view.rs zu deinen anderen Imports hinzufügen:
use wgpui_kit::prelude::FluentBuilder;

use crate::gui::{self, GuiState};
use std::sync::{Arc, Mutex};
use wgpui_kit::{
    div, px, rgb, wgpu_surface, Context, FocusHandle, IntoElement, InteractiveElement,
    ParentElement, Render, Styled, Window, WgpuSurfaceHandle,
};

pub struct SurfaceExample {
    pub surface: WgpuSurfaceHandle,
    pub state: Arc<Mutex<ApplicationState>>,
    pub fps_rx: std::sync::mpsc::Receiver<f64>,
    pub display_fps: f64,
    /// wgpui delivers key events only to the focused element (dispatch path:
    /// window root -> focused node). Without focus, `on_key_down`/`on_key_up`
    /// never fire, so we track + claim focus on the root div.
    pub focus_handle: FocusHandle,
    /// 🟢 GUI: kompletter UI-Zustand (Slider-Entities, Tabs, Accordions)
    pub gui: GuiState,
    // 🟢 NEU: Speichert den Maus-Status direkt im View-Objekt
    pub last_mouse_x: f32,
    pub last_mouse_y: f32,
    pub mouse_sensitivity: f32,
     // 🟢 NEU: Steuert, ob das GUI-Overlay sichtbar ist
    pub show_gui: bool,
    
}

/// Key-Down-Logik (WASD + Toggles). `held == true` bei OS-Key-Repeat →
/// Toggles nur bei `!held` feuern lassen.
fn handle_key_down(mut s: &mut ApplicationState, key: &str, held: bool) {
let update_slot_color = |s: &mut ApplicationState, idx: usize| {
    match s.slot_types[idx] as u32 {
        1 => { s.slot_r[idx] = 1.0; s.slot_g[idx] = 0.0; s.slot_b[idx] = 0.0; } // Kristall -> Rot
        2 => { s.slot_r[idx] = 0.0; s.slot_g[idx] = 1.0; s.slot_b[idx] = 0.0; } // Gyroid -> Grün
        3 => { s.slot_r[idx] = 0.0; s.slot_g[idx] = 0.0; s.slot_b[idx] = 1.0; } // Torus -> Blau
        _ => { s.slot_r[idx] = 1.0; s.slot_g[idx] = 1.0; s.slot_b[idx] = 1.0; } // Inaktiv -> Weiß
    }
};
    match key {
        "w" | "W" => s.w_pressed = true,
        "a" | "A" => s.a_pressed = true,
        "s" | "S" => s.s_pressed = true,
        "d" | "D" => s.d_pressed = true,
       /* "1" if !held => s.enable_key = if s.enable_key == 1 { 0 } else { 1 },
        "2" if !held => s.enable_fill = if s.enable_fill == 1 { 0 } else { 1 },
        "3" if !held => s.enable_rim = if s.enable_rim == 1 { 0 } else { 1 },
        "4" if !held => { s.current_shadow_mode = if s.current_shadow_mode == 1 { 0 } else { 1 } }
        */
        // 2️⃣ Slots 1 bis 5: Typen rotieren & Farben synchronisieren
        
        "1" if !held => { 
            s.current_selected_slot = 0; 
            s.slot_types[0] = if s.slot_types[0] == 3.0 { 0.0 } else { s.slot_types[0] + 1.0 }; 
            update_slot_color(s, 0); 
        }
        "2" if !held => { 
            s.current_selected_slot = 1; 
            s.slot_types[1] = if s.slot_types[1] == 3.0 { 0.0 } else { s.slot_types[1] + 1.0 }; 
            update_slot_color(s, 1); 
        }
        "3" if !held => { 
            s.current_selected_slot = 2; 
            s.slot_types[2] = if s.slot_types[2] == 3.0 { 0.0 } else { s.slot_types[2] + 1.0 }; 
            update_slot_color(s, 2); 
        }
        "4" if !held => { 
            s.current_selected_slot = 3; 
            s.slot_types[3] = if s.slot_types[3] == 3.0 { 0.0 } else { s.slot_types[3] + 1.0 }; 
            update_slot_color(s, 3); 
        }
        "5" if !held => { 
            s.current_selected_slot = 4; 
            s.slot_types[4] = if s.slot_types[4] == 3.0 { 0.0 } else { s.slot_types[4] + 1.0 }; 
            update_slot_color(s, 4); 
        }

        // 3️⃣ 🚀 TASTE "6": Vollkommen synchronisiertes Spawning zur Laufzeit
        "6" if !held => {
            s.current_selected_slot = 5;
            if s.slot_types.len() <= 5 {
                s.slot_types.push(1.0);     // Start als Kristall
                s.slot_sizes.push(1.0);     
                s.slot_offsets_x.push(0.0);
                s.slot_offsets_y.push(0.0); 
                s.slot_offsets_z.push(0.0);
                
                s.slot_r.push(1.0); s.slot_g.push(0.0); s.slot_b.push(0.0);

                s.slot_roughness.push(0.3);
                s.slot_metallic.push(0.0);
                s.slot_emissive.push(0.0);
                s.slot_specular.push(0.5);

                s.active_slots_count = s.slot_types.len() as f32;
                println!("[Space-Lab] 🚀 Slot 6 vollumfänglich im PBR-Verbund generiert!");
            } else {
                s.slot_types[5] = if s.slot_types[5] == 3.0 { 0.0 } else { s.slot_types[5] + 1.0 };
                update_slot_color(s, 5);
            }
        }
        
        // 🟢 TENSOR 4: Umwelt-Steuerung (Blueprint: Environmental Control)
        // Q/E : Licht-Intensität runter/hoch
        "q" if !held => s.light_intensity = (s.light_intensity - 0.1).max(0.0),
        "e" if !held => s.light_intensity = (s.light_intensity + 0.1).min(3.0),
        // F/R : Ambient-Stärke runter/hoch
        "f" if !held => s.ambient_strength = (s.ambient_strength - 0.05).max(0.0),
        "r" if !held => s.ambient_strength = (s.ambient_strength + 0.05).min(1.0),
        // O/L : Nebel-Dichte runter/hoch
        "o" if !held => s.fog_density = (s.fog_density - 0.1).max(0.0),
        "l" if !held => s.fog_density = (s.fog_density + 0.1).min(2.0),
        // N : Nebel an/aus
        "n" if !held => s.fog_enabled = if s.fog_enabled == 1 { 0 } else { 1 },
        _ => {}
    }
}

/// Key-Up-Logik (nur Bewegungs-Flags).
fn handle_key_up(s: &mut ApplicationState, key: &str) {
    match key {
        "w" | "W" => s.w_pressed = false,
        "a" | "A" => s.a_pressed = false,
        "s" | "S" => s.s_pressed = false,
        "d" | "D" => s.d_pressed = false,
        _ => {}
    }
}

impl Render for SurfaceExample {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        while let Ok(f) = self.fps_rx.try_recv() {
            self.display_fps = f;
        }

        window.request_animation_frame();

        // wgpui only dispatches key events along the path root -> focused node.
        // Claim focus on our root div so the WASD listeners actually receive
        // key events. Re-claim whenever focus was lost (e.g. after clicking a
        // Button, which grabs focus for itself).
        if window.focused(_cx).as_ref() != Some(&self.focus_handle) {
            window.focus(&self.focus_handle, _cx);
        }

        // If the OS window lost focus while a key was held (alt-tab), the
        // key-up never arrives: clear stuck movement keys.
        if !window.is_window_active() {
            if let Ok(mut s) = self.state.lock() {
                s.w_pressed = false;
                s.a_pressed = false;
                s.s_pressed = false;
                s.d_pressed = false;
            }
        }

        let state_key_down = self.state.clone();
        let state_key_up = self.state.clone();

        div()
            .id("root")
            .track_focus(&self.focus_handle)
            .size_full()
            .relative() // 🟢 WICHTIG: Erlaubt absoluten Kindern, sich an diesem Root auszurichten
            //.flex()
            //.flex_row()
            .bg(rgb(0x10121a))
            //.on_key_down(move |event, _win, _cx| {
             // 🟢 Nutzung von _cx.listener für exklusiven Zugriff auf `this` und `cx`
// 🟢 Typ-Annotation hinzugefügt, um E0282 endgültig zu lösen
            // --- 1. KEY DOWN HANDLER (Vollständig isoliert via _cx.listener) ---
                 // --- 1. KEY DOWN HANDLER (Isoliert via _cx.listener) ---
            .on_key_down(_cx.listener(|this: &mut Self, event: &wgpui_kit::KeyDownEvent, _win, cx| {
                let key = event.keystroke.key.as_str();

                // 1. Zuerst das GUI-Overlay umschalten, wenn "h" gedrückt wird
                if (key == "h" || key == "H") && !event.is_held {
                    this.show_gui = !this.show_gui;
                    cx.notify(); 
                    return; // Event vollständig verarbeitet, wir brechen hier ab
                }

                // 2. KORREKTUR E0373: Wir nutzen `this.state` direkt aus dem Struct.
                // Dadurch wird keine äußere Variable (wie state_key_down) mehr gefangen.
                if let Ok(mut s) = this.state.lock() {
                    handle_key_down(&mut s, key, event.is_held);
                }
            })) // Hier schließt der Listener syntaktisch und typsicher

            .on_key_up(move |event, _win, _cx| {
                if let Ok(mut s) = state_key_up.lock() {
                    handle_key_up(&mut s, event.keystroke.key.as_str());
                }
            })
            
                 // --- 1. DIE 3D-SZENEN-BASIS (Füllt das gesamte Fenster aus) ---
            .child(
                div()
                    .size_full()
                    .relative()  // Basis für den absoluten FPS-Zähler
                    //
                    // --- IDIOMATISCHES MAUS-HANDLING IN WGPUI_KIT ---
            // 🟢 Typ-Annotation direkt über den exportierten Root-Typen aufgelöst
            .on_mouse_move(_cx.listener(|this: &mut Self, event: &wgpui_kit::MouseMoveEvent, _win, _cx| {
                // Holt die f32-Werte sicher aus der gekapselten Pixels-Struktur
                let current_x = f32::from(event.position.x);
                let current_y = f32::from(event.position.y);

                // Delta-Berechnung (rein im f32-Raum)
                let dx = current_x - this.last_mouse_x;
                let dy = current_y - this.last_mouse_y;

                if dx != 0.0 || dy != 0.0 {
                    if let Ok(mut s) = this.state.lock() {
                        let max_pitch = 89.0f32.to_radians();
                        
                        s.cam_yaw += dx * this.mouse_sensitivity;
                        s.cam_pitch -= dy * this.mouse_sensitivity; 
                        s.cam_pitch = s.cam_pitch.clamp(-max_pitch, max_pitch);
                    }
                }

                // Werte für den nächsten Frame im Struct puffern
                this.last_mouse_x = current_x;
                this.last_mouse_y = current_y;
            }))
           // --- 2. TRACKPAD / MAUSRAD SCROLL HANDLING (KONSISTENT AN MAUS ANGEGLICHEN) ---
            .on_scroll_wheel(_cx.listener(|this: &mut Self, event: &wgpui_kit::ScrollWheelEvent, _win, _cx| {
                let (dx, dy) = match event.delta {
                    // 1. Pixelgenaues Trackpad-Scrolling (Zwei-Finger-Geste)
                    wgpui_kit::ScrollDelta::Pixels(point) => {
                        // Vorzeichen gespiegelt, um der Bewegung des Mauszeigers zu entsprechen
                        (-f32::from(point.x), -f32::from(point.y))
                    }
                    // 2. Klassisches Mausrad-Scrolling (in Zeilen gerastert)
                    wgpui_kit::ScrollDelta::Lines(point) => {
                        // Auch hier die Achsen an das visuelle Mausdelta anpassen
                        (-point.x * 20.0, -point.y * 20.0)
                    }
                };

                if dx != 0.0 || dy != 0.0 {
                    if let Ok(mut s) = this.state.lock() {
                        let max_pitch = 89.0f32.to_radians();
                        
                        // Sensitivität für Scroll-Events
                        let scroll_sensitivity = this.mouse_sensitivity * 1.5; 

                        s.cam_yaw += dx * scroll_sensitivity;
                        s.cam_pitch -= dy * scroll_sensitivity; 
                        s.cam_pitch = s.cam_pitch.clamp(-max_pitch, max_pitch);
                    }
                }
            }))
                    //
                    .child(wgpu_surface(self.surface.clone()).absolute().inset_0())
                    .child(
                        div()
                            .absolute()
                            .top(px(16.0))
                            .left(px(16.0))
                            .text_color(rgb(0x00ffcc))
                            .text_xl()
                            .child(format!("FPS: {:.1}", self.display_fps)),
                    ),
            )
            // =========================================================
            // 🟢 GUI OVERLAY: Schwebende Sidebar auf der rechten Seite
            // =========================================================
            // Dank des Imports von FluentBuilder funktioniert `.when` nun fehlerfrei!
             .when(self.show_gui, |root| {
                root.child(
                    div()
                        .absolute() // Löst das Element aus dem normalen UI-Fluss
                        .top_0()
                        .right_0()
                        .bottom_0()
                        .w(px(320.0)) // Feste Breite für dein Slider-Menü
                        .h_full()
                         // 🟢 NATIVE GPUI-LÖSUNG: Schützt die darunterliegende 3D-Fläche vor Maus-Events
                        .occlude() 
                        // Ein edler, semitransparenter Hintergrund
                        .bg(wgpui_kit::rgba(0x10121ae6)) 
                        // 🟢 KORREKTUR: Schatten durch feine, dunkle Trennlinie links ersetzen
                        .border_l(px(1.0))
                        .border_color(rgb(0x1c1e26))
                         // 🟢 NEU: Blockiert, dass Mausbewegungen an das Root-div (Kamera) durchgereicht werden
                        // 1. 🟢 Fängt Klicks (Slider-Drücken) ab, bevor sie das Root-div erreichen
                        //.capture_any_mouse_down(|_event, _win, _cx| {})
                         // 🟢 KORREKTUR E0061: MouseButton::Left explizit übergeben
                       // .on_mouse_down(wgpui_kit::MouseButton::Left, _cx.listener(|this: &mut Self, _event, win, _cx| {
                       //     win.focus(&this.focus_handle, _cx);
                       // }))
                        // 2. 🟢 Fängt reine Mausbewegungen über der GUI ab
                        //.on_mouse_move(|_event, _win, _cx| {})
                        // Hier wird deine bestehende Sidebar eingebettet
                        .child(gui::sidebar(&mut self.gui, &self.state, window, _cx))
                )
            })
    }
}