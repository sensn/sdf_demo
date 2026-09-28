mod kernel;
pub mod input;

use cubecl::client::ComputeClient;
use cubecl::frontend::TensorArg;
use cubecl::Runtime; 
use cubecl_wgpu::{WgpuRuntime, AutoCompiler, WgpuSetup, init_device, RuntimeOptions};
use cubecl_zspace::{Strides, Shape};
use bytemuck::{Pod, Zeroable};

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// Exakte wgpui-kit 0.6.1 Imports aus dem Hello-World Template
use wgpui_kit::{
    prelude::*,
    component::Root,
    wgpu_surface, WgpuSurfaceHandle, Window, WindowOptions, App, Context, Render, Styled, div, rgb, px
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
}

impl Render for SurfaceExample {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        while let Ok(f) = self.fps_rx.try_recv() {
            self.display_fps = f;
        }
        
        window.request_animation_frame();

        let state_key_down = self.state.clone();

        div()
            .size_full()
            .flex()
            .flex_row()
            .bg(rgb(0x10121a))
            .child(
                div()
                    .flex_grow(1.0)
                    .h_full()
                    .on_key_down(move |event, _win, _cx| {
                        if let Ok(mut s) = state_key_down.lock() {
                            match event.keystroke.key.as_str() {
                                "1" => s.enable_key = if s.enable_key == 1 { 0 } else { 1 },
                                "2" => s.enable_fill = if s.enable_fill == 1 { 0 } else { 1 },
                                "3" => s.enable_rim = if s.enable_rim == 1 { 0 } else { 1 },
                                "4" => s.current_shadow_mode = if s.current_shadow_mode == 1 { 0 } else { 1 },
                                _ => {}
                            }
                        }
                    })
                    .child(
                        wgpu_surface(self.surface.clone()).absolute().inset_0(),
                    )
                    .child(
                        div()
                            .absolute()
                            .top(px(16.0))
                            .left(px(16.0))
                            .text_color(rgb(0x00ffcc))
                            .text_xl()
                            .child(format!("FPS: {:.1}", self.display_fps)),
                    )
            )
            .child(
                div()
                    .w(px(320.0))
                    .h_full()
                    .bg(rgb(0x151a29))
                    .p_4()
                    .flex()
                    .flex_col()
                    .gap(px(12.0))
                    .child(
                        div()
                            .text_color(rgb(0xffffff))
                            .text_lg()
                            .child("SDF Engine Controls")
                    )
            )
    }
}

fn main() {
    wgpui_kit::application().run(|cx: &mut App| {
        wgpui_kit::init(cx);

        // 1. Fenster asynchron auf der UI‑Schleife spawnen (WGPUI‑Kit Vorschrift)
        cx.spawn(async move |cx| {
            let width = 1720;
            let height = 1080;
            let format = wgpu::TextureFormat::Rgba8UnormSrgb;
            let opts = WindowOptions::default();

            cx.open_window(opts, |window, cx| {
                // Oberfläche (Surface Handle) erzeugen
                let surface = window.create_wgpu_surface(width, height, format)
                    .expect("WgpuSurface not supported on this platform");
                
                let surface_thread = surface.clone();
                let (fps_tx, fps_rx) = std::sync::mpsc::channel::<f64>();

                let shared_state = Arc::new(Mutex::new(ApplicationState {
                    cam_x: 0.0,
                    cam_y: 0.0,
                    cam_z: -5.0,
                    cam_yaw: 0.0,
                    cam_pitch: 0.0,
                    light_intensity: 1.0,
                    ambient_strength: 0.1,
                    enable_ao_mode: 1,
                    current_shadow_mode: 1,
                    enable_key: 1,
                    enable_fill: 1,
                    enable_rim: 1,
                    dynamic_blend_factor: 0.0,
                }));

                // 2. Dedizierten Render-Thread starten
                let render_state = shared_state.clone();
                thread::spawn(move || {
                    loop {
                        if surface_thread.back_buffer_view().is_some() {
                            break;
                        }
                        thread::sleep(Duration::from_millis(10));
                    }
                    
                     // 🟢 NEU: Ein zusätzlicher, kleiner Sicherheits-Puffer (100ms), 
    // damit das OS-Fenster sich vollständig einpendeln kann, bevor die Loops loslegen.
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
                    
                    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                        apply_limit_buckets: false,
                    })).unwrap();

                    let wgpu_setup = WgpuSetup {
                        instance,
                        adapter,
                        device: device.clone(),
                        queue: queue.clone(),
                        backend: wgpu::Backend::Vulkan,
                    };
                    let cubecl_device_id = init_device(wgpu_setup, RuntimeOptions::default());
                    let client: ComputeClient<WgpuRuntime<AutoCompiler>> = WgpuRuntime::client(&cubecl_device_id);

                    let total_pixels = (width * height) as usize;
                    let f32_size = std::mem::size_of::<f32>();
                    let byte_size = total_pixels * 3 * f32_size;
                    let mut output_handle = client.empty(byte_size);

                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Screen Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
                    });

                    let mut resolution_uniform = ResolutionUniform { width, height, _padding: [0; 2] };
                    use wgpu::util::DeviceExt;
                    let resolution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Resolution Uniform"),
                        contents: bytemuck::cast_slice(&[resolution_uniform]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                    });

                    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

                let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Screen Pipeline Layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

                let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Screen Render Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
fragment: Some(wgpu::FragmentState {module: &shader,entry_point: Some("fs_main"),targets: &[Some(wgpu::ColorTargetState {format,blend: Some(wgpu::BlendState::REPLACE),write_mask: wgpu::ColorWrites::ALL,})],compilation_options: wgpu::PipelineCompilationOptions::default(),}),primitive: wgpu::PrimitiveState {topology: wgpu::PrimitiveTopology::TriangleList,strip_index_format: None,front_face: wgpu::FrontFace::Ccw,cull_mode: None,unclipped_depth: false,polygon_mode: wgpu::PolygonMode::Fill,conservative: false,},depth_stencil: None,multisample: wgpu::MultisampleState::default(),multiview_mask: None,cache: None,});let mut last_report = Instant::now();let mut frame_count: u32 = 0;let start_time = Instant::now();let mut last_frame_time = Instant::now();let mut input_manager = input::InputManager::new(2.5, 0.002);loop {surface_thread.wait_for_present();let (view, (dw, dh)) = match surface_thread.back_view_with_size() {Some(tuple) => tuple,None => {thread::sleep(Duration::from_nanos(500));continue;}};
if dw != resolution_uniform.width || dh != resolution_uniform.height {
// 🟢 NEU: Ungültige Rendergrößen abfangen (verhindert Abstürze beim Resizen)
    if dw > 0 && dh > 0 {
resolution_uniform.width = dw;
resolution_uniform.height = dh;
queue.write_buffer(&resolution_buffer, 0, bytemuck::cast_slice(&[resolution_uniform]));
let total_px = (dw * dh) as usize;
output_handle = client.empty(total_px * 3 * f32_size);}}
let now = Instant::now();let dt = now.duration_since(last_frame_time).as_secs_f32();let time = start_time.elapsed().as_secs_f32();last_frame_time = now;let mut s = render_state.lock().unwrap();input_manager.update_camera_movement(&mut *s, dt);let total_pixels = (dw * dh) as usize;let cube_count = cubecl::CubeCount::Static((dw + 15) / 16, (dh + 3) / 4, 1);let cube_dim = cubecl::CubeDim::new_3d(16, 4, 1);let meta_handle = client.empty(f32_size);let slots_handle = client.empty(800 * f32_size);let materials_handle = client.empty(100 * f32_size);let output_arg = unsafe { TensorArg::from_raw_parts(output_handle.clone(), Strides::from(&[1usize]), Shape::from(&[total_pixels])) };let meta_arg = unsafe { TensorArg::from_raw_parts(meta_handle.clone(), Strides::from(&[1usize]), Shape::from(&[1usize])) };let slots_arg = unsafe { TensorArg::from_raw_parts(slots_handle.clone(), Strides::from(&[1usize]), Shape::from(&[800usize])) };let materials_arg = unsafe { TensorArg::from_raw_parts(materials_handle.clone(), Strides::from(&[1usize]), Shape::from(&[100usize])) };unsafe {kernel::raymarch_sdf_kernel::launch(&client, cube_count, cube_dim, output_arg, meta_arg, slots_arg, materials_arg,time, dw, dh, s.current_shadow_mode, s.cam_x, s.cam_y, s.cam_z, s.dynamic_blend_factor,s.enable_ao_mode, s.cam_yaw, s.cam_pitch, s.light_intensity, s.ambient_strength, s.enable_key, s.enable_fill, s.enable_rim,);}let _ = client.sync();let managed_resource = client.get_resource(output_handle.clone()).unwrap();let wgpu_resource = managed_resource.resource();let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {label: Some("Screen Bind Group"),layout: &bind_group_layout,entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &wgpu_resource.buffer, offset: wgpu_resource.offset, size: None }) },wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &resolution_buffer, offset: 0, size: None }) },],});let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });{let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {color_attachments: &[Some(wgpu::RenderPassColorAttachment {view: &view, resolve_target: None,ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },depth_slice: None,})],depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,label: Some("Screen Render Pass"),});render_pass.set_pipeline(&render_pipeline);render_pass.set_bind_group(0, &bind_group, &[]);render_pass.draw(0..3, 0..1);}queue.submit(std::iter::once(encoder.finish()));surface_thread.present();frame_count += 1;let elapsed = last_report.elapsed();if elapsed >= Duration::from_secs(1) {let fps = frame_count as f64 / elapsed.as_secs_f64();let _ = fps_tx.send(fps);frame_count = 0;last_report = Instant::now();}}});// 3. 🟢 FIX: View über cx.new erstellen und im Root-Element ans Fenster zurückgeben
let view = cx.new(|_| SurfaceExample {surface,state: shared_state,fps_rx,display_fps: 0.0,});cx.new(|cx| Root::new(view, window, cx))}).expect("Failed to open window");}).detach();});}