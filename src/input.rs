use std::time::Instant;
use crossbeam_channel::Receiver;
use crate::ApplicationState;

pub struct InputManager {
    w_pressed: bool,
    a_pressed: bool,
    s_pressed: bool,
    d_pressed: bool,
    pub camera_speed: f32,
    pub mouse_sensitivity: f32,
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
            w_pressed: false,
            a_pressed: false,
            s_pressed: false,
            d_pressed: false,
            camera_speed,
            mouse_sensitivity,
            mouse_receiver: receiver,
        }
    }

    pub fn update_camera_movement(&mut self, state: &mut ApplicationState, dt: f32) {
        // 1. Rotation aus dem Linux‑Kernel‑Thread entleeren
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

        // 2. Bewegung verarbeiten (W, A, S, D)
        let move_dist = self.camera_speed * dt;

        let cos_yaw = state.cam_yaw.cos();
        let sin_yaw = state.cam_yaw.sin();

        if self.w_pressed {
            state.cam_x += sin_yaw * move_dist;
            state.cam_z += cos_yaw * move_dist;
        }
        if self.s_pressed {
            state.cam_x -= sin_yaw * move_dist;
            state.cam_z -= cos_yaw * move_dist;
        }
        if self.a_pressed {
            state.cam_x -= cos_yaw * move_dist;
            state.cam_z += sin_yaw * move_dist;
        }
        if self.d_pressed {
            state.cam_x += cos_yaw * move_dist;
            state.cam_z -= sin_yaw * move_dist;
        }
    }
}
