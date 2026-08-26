use std::time::Instant;
use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::ActiveEventLoop;
use crate::Application;
// ❌ ALT: use crossbeam_channel::{Receiver, try_recv};
// 🟢 NEU: Nur den Receiver importieren
use crossbeam_channel::Receiver;


pub struct InputManager {
    w_pressed: bool,
    a_pressed: bool,
    s_pressed: bool,
    d_pressed: bool,
    pub camera_speed: f32,
    pub mouse_sensitivity: f32,
    
    // Empfänger für die Hardware-Maus-Deltas aus dem Linux-Thread
    mouse_receiver: Receiver<(f32, f32)>,
}

impl InputManager {
    pub fn new(camera_speed: f32, mouse_sensitivity: f32) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();

        // 🟢 LINUX KERNEL THREAD: Liest die rohe System-Maus aus (/dev/input/mice)
        std::thread::spawn(move || {
            // Öffnet den standardisierten, virtuellen Multiplex-Mauspuffer von Linux
            if let Ok(mut file) = std::fs::File::open("/dev/input/mice") {
                let mut buffer = [0u8; 3];
                loop {
                    if std::io::Read::read_exact(&mut file, &mut buffer).is_ok() {
                        // Das Linux-Mausprotokoll (PS/2) liefert relative Deltas im 2. und 3. Byte
                        let dx = buffer[1] as i8 as f32;
                        let dy = buffer[2] as i8 as f32;
                        
                        if dx != 0.0 || dy != 0.0 {
                            let _ = sender.send((dx, dy));
                        }
                    }
                }
            }
        });

        Self {
            w_pressed: false,
            a_pressed: false,
            s_pressed: false,
            d_pressed: false,
            camera_speed,
            mouse_sensitivity,
            mouse_receiver: receiver,
        }
    }

    // Tastatur-Events (Bleiben unverändert)
    pub fn handle_key_event(&mut self, key_event: &KeyEvent, elwt: &ActiveEventLoop, app: &mut Application) {
        if let PhysicalKey::Code(keycode) = key_event.physical_key {
            let is_pressed = key_event.state == ElementState::Pressed;
            
            match keycode {
                KeyCode::KeyW => self.w_pressed = is_pressed,
                KeyCode::KeyA => self.a_pressed = is_pressed,
                KeyCode::KeyS => self.s_pressed = is_pressed,
                KeyCode::KeyD => self.d_pressed = is_pressed,
                KeyCode::Escape => elwt.exit(),
                
                KeyCode::Digit1 => { if is_pressed { app.enable_key = if app.enable_key == 1 { 0 } else { 1 }; } }
                KeyCode::Digit2 => { if is_pressed { app.enable_fill = if app.enable_fill == 1 { 0 } else { 1 }; } }
                KeyCode::Digit3 => { if is_pressed { app.enable_rim = if app.enable_rim == 1 { 0 } else { 1 }; } }
                KeyCode::Digit4 => { if is_pressed { app.current_shadow_mode = if app.current_shadow_mode == 1 { 0 } else { 1 }; } }
                _ => {}
            }
        }
    }

    // Berechnet Bewegung UND sammelt die blockierungsfreien Linux-Mausdaten
    pub fn update_camera_movement(&self, app: &mut Application) {
        // --- 1. ROTATION (Aus dem Linux-Kernel-Thread entleeren) ---
        let mut total_dx = 0.0;
        let mut total_dy = 0.0;

        // Lese alle aufgelaufenen Mausbewegungen aus dem Kanal
        while let Ok((dx, dy)) = self.mouse_receiver.try_recv() {
            total_dx += dx;
            // Unter Linux ist das Delta-Y invertiert, wir drehen es für die Kamera um
            total_dy -= dy; 
        }

        // Winkel berechnen und flüssig anwenden
        app.cam_yaw += total_dx * self.mouse_sensitivity;
        app.cam_pitch += total_dy * self.mouse_sensitivity;

        let max_pitch = 89.0f32.to_radians();
        app.cam_pitch = app.cam_pitch.clamp(-max_pitch, max_pitch);

        // --- 2. BEWEGUNG (W, A, S, D) ---
        let now = Instant::now();
        let dt = now.duration_since(app.last_frame_time).as_secs_f32();
        let move_dist = self.camera_speed * dt;

        let cos_yaw = app.cam_yaw.cos();
        let sin_yaw = app.cam_yaw.sin();

        if self.w_pressed {
            app.cam_x += sin_yaw * move_dist;
            app.cam_z += cos_yaw * move_dist;
        }
        if self.s_pressed {
            app.cam_x -= sin_yaw * move_dist;
            app.cam_z -= cos_yaw * move_dist;
        }
        if self.a_pressed {
            app.cam_x -= cos_yaw * move_dist;
            app.cam_z += sin_yaw * move_dist;
        }
        if self.d_pressed {
            app.cam_x += cos_yaw * move_dist;
            app.cam_z -= sin_yaw * move_dist;
        }
    }
}
