pub mod input;
mod kernel;

use bytemuck::{Pod, Zeroable};
use cubecl::client::ComputeClient;
use cubecl::frontend::TensorArg;
use cubecl::Runtime;
use cubecl_wgpu::{init_device, AutoCompiler, RuntimeOptions, WgpuRuntime, WgpuSetup};
use cubecl_zspace::{Shape, Strides};

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// Offizielle wgpui-kit 0.6.1 Imports
use wgpui_kit::{
    component::{button::Button, Root},
    div,
    prelude::*,
    px, rgb, wgpu_surface, App, Context, FocusHandle, Render, Styled, WgpuSurfaceHandle, Window,
    WindowOptions,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ResolutionUniform {
    width: u32,
    height: u32,
    _padding: [u32; 2],
}

pub struct ApplicationState {
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_z: f32,
    pub cam_yaw: f32,
    pub cam_pitch: f32,

    pub w_pressed: bool,
    pub a_pressed: bool,
    pub s_pressed: bool,
    pub d_pressed: bool,

    pub light_intensity: f32,
    pub ambient_strength: f32,
    pub enable_ao_mode: u32,
    pub current_shadow_mode: u32,
    pub enable_key: u32,
    pub enable_fill: u32,
    pub enable_rim: u32,
    pub dynamic_blend_factor: f32,
}

struct SurfaceExample {
    surface: WgpuSurfaceHandle,
    state: Arc<Mutex<ApplicationState>>,
    fps_rx: std::sync::mpsc::Receiver<f64>,
    display_fps: f64,
    /// wgpui delivers key events only to the focused element (dispatch path:
    /// window root -> focused node). Without focus, `on_key_down`/`on_key_up`
    /// never fire, so we track + claim focus on the root div.
    focus_handle: FocusHandle,
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
        let state_ao_toggle = self.state.clone();
        let state_rim_toggle = self.state.clone();
        let state_shadow_toggle = self.state.clone();

        let state_read = self.state.lock().unwrap();

        let cam_x_val = state_read.cam_x;
        let cam_y_val = state_read.cam_y;
        let cam_z_val = state_read.cam_z;
        let ao_on = state_read.enable_ao_mode == 1;
        let rim_on = state_read.enable_rim == 1;
        let shadow_on = state_read.current_shadow_mode == 1;

        div()
            .id("root")
            .track_focus(&self.focus_handle)
            .size_full()
            .flex()
            .flex_row()
            .bg(rgb(0x10121a))
            .on_key_down(move |event, _win, _cx| {
                if let Ok(mut s) = state_key_down.lock() {
                    match event.keystroke.key.as_str() {
                        "w" | "W" => s.w_pressed = true,
                        "a" | "A" => s.a_pressed = true,
                        "s" | "S" => s.s_pressed = true,
                        "d" | "D" => s.d_pressed = true,
                        // Ignore OS key-repeat for toggles (`is_held` is true on repeats).
                        "1" if !event.is_held => {
                            s.enable_key = if s.enable_key == 1 { 0 } else { 1 }
                        }
                        "2" if !event.is_held => {
                            s.enable_fill = if s.enable_fill == 1 { 0 } else { 1 }
                        }
                        "3" if !event.is_held => {
                            s.enable_rim = if s.enable_rim == 1 { 0 } else { 1 }
                        }
                        "4" if !event.is_held => {
                            s.current_shadow_mode = if s.current_shadow_mode == 1 { 0 } else { 1 }
                        }
                        _ => {}
                    }
                }
            })
            .on_key_up(move |event, _win, _cx| {
                if let Ok(mut s) = state_key_up.lock() {
                    match event.keystroke.key.as_str() {
                        "w" | "W" => s.w_pressed = false,
                        "a" | "A" => s.a_pressed = false,
                        "s" | "S" => s.s_pressed = false,
                        "d" | "D" => s.d_pressed = false,
                        _ => {}
                    }
                }
            })
            .child(
                div()
                    .flex_grow(1.0)
                    .h_full()
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
            .child(
                div()
                    .w(px(320.0))
                    .h_full()
                    .bg(rgb(0x151a29))
                    .p_4()
                    .flex()
                    .flex_col()
                    .gap(px(16.0))
                    .child(
                        div()
                            .text_color(rgb(0xffffff))
                            .text_lg()
                            .child("SDF Engine Controls"),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(4.0))
                            .child(
                                div()
                                    .text_color(rgb(0x8a92a6))
                                    .text_sm()
                                    .child(format!("Cam X: {:.2}", cam_x_val)),
                            )
                            .child(
                                div()
                                    .text_color(rgb(0x8a92a6))
                                    .text_sm()
                                    .child(format!("Cam Y: {:.2}", cam_y_val)),
                            )
                            .child(
                                div()
                                    .text_color(rgb(0x8a92a6))
                                    .text_sm()
                                    .child(format!("Cam Z: {:.2}", cam_z_val)),
                            ),
                    )
                    .child(div().h(px(1.0)).bg(rgb(0x2a2f42)))
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .justify_between()
                            .items_center()
                            .child(div().text_color(rgb(0xffffff)).child("Ambient Occlusion"))
                            .child(
                                Button::new("ao_btn")
                                    .label(if ao_on { "ACTIVE" } else { "INACTIVE" })
                                    .on_click(move |_, _, _| {
                                        if let Ok(mut s) = state_ao_toggle.lock() {
                                            s.enable_ao_mode =
                                                if s.enable_ao_mode == 1 { 0 } else { 1 };
                                        }
                                    }),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .justify_between()
                            .items_center()
                            .child(div().text_color(rgb(0xffffff)).child("Rim Lighting"))
                            .child(
                                Button::new("rim_btn")
                                    .label(if rim_on { "ACTIVE" } else { "INACTIVE" })
                                    .on_click(move |_, _, _| {
                                        if let Ok(mut s) = state_rim_toggle.lock() {
                                            s.enable_rim = if s.enable_rim == 1 { 0 } else { 1 };
                                        }
                                    }),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .justify_between()
                            .items_center()
                            .child(div().text_color(rgb(0xffffff)).child("Shadow Mode"))
                            .child(
                                Button::new("shadow_btn")
                                    .label(if shadow_on { "HARD" } else { "SOFT" })
                                    .on_click(move |_, _, _| {
                                        if let Ok(mut s) = state_shadow_toggle.lock() {
                                            s.current_shadow_mode =
                                                if s.current_shadow_mode == 1 { 0 } else { 1 };
                                        }
                                    }),
                            ),
                    )
                    .child(
                        div()
                            .bg(rgb(0x1a2333))
                            .p_3()
                            .rounded_md()
                            .flex()
                            .flex_col()
                            .gap(px(4.0))
                            .child(
                                div()
                                    .text_color(rgb(0x00ffcc))
                                    .text_sm()
                                    .child("Keyboard Shortcuts:"),
                            )
                            .child(
                                div()
                                    .text_color(rgb(0x8a92a6))
                                    .text_xs()
                                    .child("• WASD : Fly through Scene"),
                            )
                            .child(
                                div()
                                    .text_color(rgb(0x8a92a6))
                                    .text_xs()
                                    .child("• Key 1 : Toggle Key Light"),
                            )
                            .child(
                                div()
                                    .text_color(rgb(0x8a92a6))
                                    .text_xs()
                                    .child("• Key 2 : Toggle Fill Light"),
                            )
                            .child(
                                div()
                                    .text_color(rgb(0x8a92a6))
                                    .text_xs()
                                    .child("• Key 3 : Toggle Rim Light"),
                            )
                            .child(
                                div()
                                    .text_color(rgb(0x8a92a6))
                                    .text_xs()
                                    .child("• Key 4 : Toggle Hard Shadows"),
                            ),
                    ),
            )
    }
}

fn main() {
    wgpui_kit::application().run(|cx: &mut App| {
        wgpui_kit::init(cx);

        cx.spawn(async move |cx| {
            let width = 1720;
            let height = 1080;
            let format = wgpu::TextureFormat::Rgba8UnormSrgb;
            let opts = WindowOptions::default();

            cx.open_window(opts, |window, cx| {
                let surface = window
                    .create_wgpu_surface(width, height, format)
                    .expect("WgpuSurface not supported on this platform");

                let surface_thread = surface.clone();

                // 🟢 FIX: Expliziter Typ-Parameter für den Channel‑Aufruf
                let (fps_tx, fps_rx) = std::sync::mpsc::channel::<f64>();

                let shared_state = Arc::new(Mutex::new(ApplicationState {
                    cam_x: 0.0,
                    cam_y: 0.0,
                    cam_z: -5.0,
                    cam_yaw: 0.0,
                    cam_pitch: 0.0,
                    w_pressed: false,
                    a_pressed: false,
                    s_pressed: false,
                    d_pressed: false,
                    light_intensity: 1.0,
                    ambient_strength: 0.1,
                    enable_ao_mode: 1,
                    current_shadow_mode: 1,
                    enable_key: 1,
                    enable_fill: 1,
                    enable_rim: 1,
                    dynamic_blend_factor: 0.0,
                }));
                let render_state = shared_state.clone();
                thread::spawn(move || {
                    loop {
                        if surface_thread.back_buffer_view().is_some() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(10));
                    }
                    thread::sleep(Duration::from_millis(100));
                    let device = surface_thread.device();
                    let queue = surface_thread.queue();
                    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                        backends: wgpu::Backends::VULKAN,
                        flags: wgpu::InstanceFlags::empty(),
                        backend_options: wgpu::BackendOptions::default(),
                        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                        display: None,
                    });
                    let adapter = pollster::block_on(instance.request_adapter(
                        &wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            compatible_surface: None,
                            force_fallback_adapter: false,
                            apply_limit_buckets: false,
                        },
                    ))
                    .unwrap();
                    let wgpu_setup = WgpuSetup {
                        instance,
                        adapter,
                        device: device.clone(),
                        queue: queue.clone(),
                        backend: wgpu::Backend::Vulkan,
                    };
                    let cubecl_device_id = init_device(wgpu_setup, RuntimeOptions::default());
                    let client: ComputeClient<WgpuRuntime> = WgpuRuntime::client(&cubecl_device_id);
                    let total_pixels = (width * height) as usize; // 🟢 FIX: Ersetzt durch die Konstante 4 zur Vermeidung von Turbofish-Ausgabefehlern
                    let byte_size = total_pixels * 3 * 4;
                    let mut output_handle = client.empty(byte_size);
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Screen Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
                    });
                    let mut resolution_uniform = ResolutionUniform {
                        width,
                        height,
                        _padding: [0; 2],
                    };
                    use wgpu::util::DeviceExt;
                    let resolution_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Resolution Uniform"),
                            contents: bytemuck::cast_slice(&[resolution_uniform]),
                            usage: wgpu::BufferUsages::UNIFORM
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::STORAGE,
                        });
                    let bind_group_layout =
                        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Screen Bind Group Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                    let pipeline_layout =
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("Screen Pipeline Layout"),
                            bind_group_layouts: &[Some(&bind_group_layout)],
                            immediate_size: 0,
                        });
                    let render_pipeline =
                        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some("Screen Render Pipeline"),
                            layout: Some(&pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some("vs_main"),
                                buffers: &[],
                                compilation_options: wgpu::PipelineCompilationOptions::default(),
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some("fs_main"),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format,
                                    blend: Some(wgpu::BlendState::REPLACE),
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                                compilation_options: wgpu::PipelineCompilationOptions::default(),
                            }),
                            primitive: wgpu::PrimitiveState {
                                topology: wgpu::PrimitiveTopology::TriangleList,
                                strip_index_format: None,
                                front_face: wgpu::FrontFace::Ccw,
                                cull_mode: None,
                                unclipped_depth: false,
                                polygon_mode: wgpu::PolygonMode::Fill,
                                conservative: false,
                            },
                            depth_stencil: None,
                            multisample: wgpu::MultisampleState::default(),
                            multiview_mask: None,
                            cache: None,
                        });
                    let mut last_report = Instant::now();
                    let mut frame_count: u32 = 0;
                    let start_time = Instant::now();
                    let mut last_frame_time = Instant::now();
                    let mut input_manager = input::InputManager::new(2.5, 0.002);
                    loop {
                        surface_thread.wait_for_present();
                        let (view, (dw, dh)) = match surface_thread.back_view_with_size() {
                            Some(tuple) => tuple,
                            None => {
                                thread::sleep(Duration::from_nanos(500));
                                continue;
                            }
                        };
                        if dw != resolution_uniform.width || dh != resolution_uniform.height {
                            if dw > 0 && dh > 0 {
                                resolution_uniform.width = dw;
                                resolution_uniform.height = dh;
                                queue.write_buffer(
                                    &resolution_buffer,
                                    0,
                                    bytemuck::cast_slice(&[resolution_uniform]),
                                );
                                output_handle = client.empty((dw * dh) as usize * 3 * 4);
                            }
                        }
                        let now = Instant::now();
                        let dt = now.duration_since(last_frame_time).as_secs_f32();
                        let time = start_time.elapsed().as_secs_f32();
                        last_frame_time = now;
                        let mut s = render_state.lock().unwrap();
                        //input_manager.handle_key_event(&key_event, event_loop, app);
                        input_manager.update_camera_movement(&mut *s, dt);
                        let total_pixels = (dw * dh) as usize;
                        let cube_count = cubecl::CubeCount::Static((dw + 15) / 16, (dh + 3) / 4, 1);
                        let cube_dim = cubecl::CubeDim::new_3d(16, 4, 1); // 🟢 FIX: Ersetzt durch Konstanten zur Vermeidung von Turbofish-Fehlern
                        let meta_handle = client.empty(4);
                        let slots_handle = client.empty(800 * 4);
                        let materials_handle = client.empty(100 * 4);
                        let output_arg = unsafe {
                            TensorArg::from_raw_parts(
                                output_handle.clone(),
                                Strides::from(&[1usize]),
                                Shape::from(&[total_pixels]),
                            )
                        };
                        let meta_arg = unsafe {
                            TensorArg::from_raw_parts(
                                meta_handle.clone(),
                                Strides::from(&[1usize]),
                                Shape::from(&[1usize]),
                            )
                        };
                        let slots_arg = unsafe {
                            TensorArg::from_raw_parts(
                                slots_handle.clone(),
                                Strides::from(&[1usize]),
                                Shape::from(&[800usize]),
                            )
                        };
                        let materials_arg = unsafe {
                            TensorArg::from_raw_parts(
                                materials_handle.clone(),
                                Strides::from(&[1usize]),
                                Shape::from(&[100usize]),
                            )
                        };
                        unsafe {
                            kernel::raymarch_sdf_kernel::launch(
                                &client,
                                cube_count,
                                cube_dim,
                                output_arg,
                                meta_arg,
                                slots_arg,
                                materials_arg,
                                time,
                                dw,
                                dh,
                                s.current_shadow_mode,
                                s.cam_x,
                                s.cam_y,
                                s.cam_z,
                                s.dynamic_blend_factor,
                                s.enable_ao_mode,
                                s.cam_yaw,
                                s.cam_pitch,
                                s.light_intensity,
                                s.ambient_strength,
                                s.enable_key,
                                s.enable_fill,
                                s.enable_rim,
                            );
                        }
                        let _ = client.sync();
                        let managed_resource = client.get_resource(output_handle.clone()).unwrap();
                        let wgpu_resource = managed_resource.resource();
                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Screen Bind Group"),
                            layout: &bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &wgpu_resource.buffer,
                                        offset: wgpu_resource.offset,
                                        size: None,
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &resolution_buffer,
                                        offset: 0,
                                        size: None,
                                    }),
                                },
                            ],
                        });
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Render Encoder"),
                            });
                        {
                            let mut render_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                        depth_slice: None,
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                    label: Some("Screen Render Pass"),
                                });
                            render_pass.set_pipeline(&render_pipeline);
                            render_pass.set_bind_group(0, &bind_group, &[]);
                            render_pass.draw(0..3, 0..1);
                        }
                        queue.submit(std::iter::once(encoder.finish()));
                        surface_thread.present();
                        frame_count += 1;
                        let elapsed = last_report.elapsed();
                        if elapsed >= Duration::from_secs(1) {
                            let fps = frame_count as f64 / elapsed.as_secs_f64();
                            let _ = fps_tx.send(fps);
                            frame_count = 0;
                            last_report = Instant::now();
                        }
                    }
                });
                let view = cx.new(|cx| SurfaceExample {
                    surface,
                    state: shared_state,
                    fps_rx,
                    display_fps: 0.0,
                    focus_handle: cx.focus_handle(),
                });
                cx.new(|cx| Root::new(view, window, cx))
            })
            .expect("Failed to open window");
        })
        .detach();
    });
}
