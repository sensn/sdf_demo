//! Shader program for the SDF raymarcher.
//!
//! `SdfScene` holds the application state and implements `shader::Program`.
//! `SdfPrimitive` carries the per-frame data (cloned state + resolution/time)
//! and implements `shader::Primitive`.
//! `SdfPipeline` (gpu_iced.rs) implements `shader::Pipeline`.
//!
//! Blueprint: iced/examples/custom_shader (scene.rs).

use crate::app_state::ApplicationState;

use iced::wgpu;

use iced::mouse;
use iced::widget::shader::{self, Viewport};
use iced::Rectangle;

/// The scene: application state + camera mode flags.
#[derive(Clone)]
pub struct SdfScene {
    pub state: ApplicationState,
    /// Vergangene Zeit seit Start (Sekunden) - treibt `fold_speed`.
    pub elapsed: f32,
    /// Kameramodus (Maus-Look aktiv).
    pub camera_mode: bool,
}

impl Default for SdfScene {
    fn default() -> Self {
        Self {
            state: ApplicationState::default(),
            elapsed: 0.0,
            camera_mode: false,
        }
    }
}

impl SdfScene {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<Message> shader::Program<Message> for SdfScene {
    type State = ();
    type Primitive = SdfPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        SdfPrimitive {
            state: self.state.clone(),
            time: self.elapsed,
        }
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if self.camera_mode {
            mouse::Interaction::Grabbing
        } else {
            mouse::Interaction::Crosshair
        }
    }
}

/// Per-frame render data handed to the pipeline.
#[derive(Debug)]
pub struct SdfPrimitive {
    state: ApplicationState,
    time: f32,
}

impl shader::Primitive for SdfPrimitive {
    type Pipeline = crate::gpu_iced::SdfPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        let physical = viewport.physical_size();
        let width = physical.width.max(1);
        let height = physical.height.max(1);

        if pipeline.resolution.width != width || pipeline.resolution.height != height {
            pipeline.resize(width, height);
        }

        // 1. Upload state registers + object tensor.
        // 2. Launch the raymarch kernel (ordered on the same queue).
        pipeline.run_frame(&self.state, width, height, self.time);
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        _clip_bounds: &Rectangle<u32>,
    ) {
        pipeline.render(target, encoder);
    }
}
