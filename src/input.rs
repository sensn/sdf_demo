use crossbeam_channel::Receiver;
use crate::ApplicationState;

/// Handles camera movement. Keyboard state (WASD) is owned by
/// `ApplicationState` and updated via wgpui key events in `main.rs`
/// (wgpui owns the winit event loop, so there is no direct winit access).
///OLD: Mouse rotation is read from a raw evdev thread (`/dev/input/mice`).
///NEW: Mouse rotation via wgpui in view.rs ( .on_mouse_move(_cx.listener(|this: &mut Self, event: &wgpui_kit::MouseMoveEvent, _win, _cx| {) and .on_scroll_wheel(_cx.listener(|this: &mut Self, event: &wgpui_kit::ScrollWheelEvent, _win, _cx| { )
pub struct InputManager {
    pub camera_speed: f32,
    // Platzhalter für zukünftige Gamepad-Events (z.B. Buttons/Achsen)
    gamepad_receiver: Receiver<()>, 
}

impl InputManager {
    pub fn new(camera_speed: f32) -> Self {
        // Wir geben dem Channel explizit den leeren Typ `()`, damit E0282 verschwindet
        let (_sender, receiver) = crossbeam_channel::unbounded::<()>();

        // HIER KANNST DU SPÄTER DEINEN GILRS-GAMEPAD-THREAD SPAWNEN:
        // std::thread::spawn(move || { ... });

        Self {
            camera_speed,
            gamepad_receiver: receiver,
        }
    }


/*
pub struct InputManager {
    pub camera_speed: f32,
    //pub mouse_sensitivity: f32,
    mouse_receiver: Receiver<(f32, f32)>,
}

impl InputManager {
    pub fn new(camera_speed: f32, mouse_sensitivity: f32) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();

        std::thread::spawn(move || {
            if let Ok(mut file) = std::fs::File::open("/dev/input/mice") {
                let mut buffer = [0u8; 3];
                loop {
                    if std::io::Read::read_exact(&mut file, &mut buffer).is_ok() {
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
            camera_speed,
          //  mouse_sensitivity,
          //  mouse_receiver: receiver,
        }
    }
*/
    pub fn update_camera_movement(&mut self, state: &mut ApplicationState, dt: f32) {
       //OLD LINUX KERNEL MOUSE
     /*
             // --- 1. ROTATION (drain the raw mouse thread) ---
        let mut total_dx = 0.0;
        let mut total_dy = 0.0;

        while let Ok((dx, dy)) = self.mouse_receiver.try_recv() {
            total_dx += dx;
            total_dy -= dy;
        }

        state.cam_yaw += total_dx * self.mouse_sensitivity;
        state.cam_pitch += total_dy * self.mouse_sensitivity;

        let max_pitch = 89.0f32.to_radians();
        state.cam_pitch = state.cam_pitch.clamp(-max_pitch, max_pitch);
*/
        // --- 2. MOVEMENT (W, A, S, D flags from wgpui key events) ---
        let move_dist = self.camera_speed * dt;

        let cos_yaw = state.cam_yaw.cos();
        let sin_yaw = state.cam_yaw.sin();

        if state.w_pressed {
            state.cam_x += sin_yaw * move_dist;
            state.cam_z += cos_yaw * move_dist;
        }
        if state.s_pressed {
            state.cam_x -= sin_yaw * move_dist;
            state.cam_z -= cos_yaw * move_dist;
        }
        if state.a_pressed {
            state.cam_x -= cos_yaw * move_dist;
            state.cam_z += sin_yaw * move_dist;
        }
        if state.d_pressed {
            state.cam_x += cos_yaw * move_dist;
            state.cam_z -= sin_yaw * move_dist;
        }
        // 🟢 NEU: Kamera hoch/runter (Q/E) — unabhängig von Yaw/Pitch
        if state.q_pressed {
            state.cam_y += move_dist;
        }
        if state.e_pressed {
            state.cam_y -= move_dist;
        }
    }
}
