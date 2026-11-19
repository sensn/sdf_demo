//! sdf_demo - iced 0.15 port.
//!
//! Architecture:
//! - `SdfScene` holds the application state and implements `shader::Program`.
//! - `SdfPrimitive` carries per-frame data, implements `shader::Primitive`.
//! - `SdfPipeline` owns the wgpu pipeline + CubeCL client, implements
//!   `shader::Pipeline`.
//! - The GUI is built inline with iced widgets (see custom_shader example).
//!
//! Blueprint: iced/examples/custom_shader

mod app_state;
mod gpu_iced;
mod kernel;
mod sdf_preprocessor;
mod shader_program;

use shader_program::SdfScene;

use iced::time::Instant;
use iced::widget::{center, column, shader};
use iced::{Fill, Subscription, Widget};

fn main() -> iced::Result {
    iced::application(IcedSdf::default, IcedSdf::update, IcedSdf::view)
        .subscription(IcedSdf::subscription)
        .run()
}

struct IcedSdf {
    start: Instant,
    scene: SdfScene,
}

impl Default for IcedSdf {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            scene: SdfScene::new(),
        }
    }
}

#[derive(Debug, Clone)]
enum Message {
    Tick(Instant),
}

impl IcedSdf {
    fn update(&mut self, message: Message) {
        match message {
            Message::Tick(time) => {
                self.scene.elapsed = (time - self.start).as_secs_f32();
            }
        }
    }

    fn view(&self) -> impl Widget<Message> {
        let shader_widget = shader(&self.scene).width(Fill).height(Fill);

        center(column![shader_widget].align_x(iced::Center))
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::window::frames().map(Message::Tick)
    }
}