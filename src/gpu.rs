//! GPU-Pipeline: cubecl-Compute-Setup + wgpu-Blit-Pipeline.
//!
//! `GpuPipeline::new` läuft **synchron** im View-Konstruktor — `device()`/
//! `queue()` stehen direkt nach `window.create_wgpu_surface()` bereit
//! (wgpui-0.3.6, elements/wgpu_surface.rs:82–88). Nur der erste Back-Buffer
//! existiert erst nach dem ersten Layout-Pass; damit umgeht die Frame-Loop
//! über `back_view_with_size()`-Retry (siehe render_loop.rs).
//!
//! Die Pipeline gehört exklusiv dem Render-Thread (kein Arc um wgpu-Objekte).

use crate::app_state::ApplicationState;
use bytemuck::{Pod, Zeroable};
//use cubecl::client::ComputeClient; // old cubecl syntax
use cubecl::client::Client; // 🟢 KORREKTUR: client:: hinzufügen cubecl 0.11.0 
use cubecl::frontend::TensorArg;
//use cubecl::Runtime;
use cubecl_runtime::runtime::Runtime; // RICHTIG
use cubecl_runtime::server::Handle;
use cubecl_wgpu::{init_device, RuntimeOptions, WgpuRuntime, WgpuSetup};
use cubecl_zspace::{Shape, Strides};
use wgpui_kit::WgpuSurfaceHandle;

/// Auflösungs-Uniform für den Screen-Shader (fs_main).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ResolutionUniform {
    pub width: u32,
    pub height: u32,
    pub _padding: [u32; 2],
}

/// Fehler bei der GPU-Initialisierung (Adapter/Device/Shader).
#[derive(Debug)]
pub enum GpuInitError {
    NoAdapter,
}

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "kein passender Vulkan-Adapter gefunden"),
        }
    }
}

/// Cubecl-Workgroup-Layout des Raymarch-Kernels (16×4 Threads pro Cube).
const CUBE_DIM_X: u32 = 16;
const CUBE_DIM_Y: u32 = 4;

/// Buffer-Größen (Bytes) für die Kernel-Register.
/// meta = 1 f32, slots = 800 f32, materials = 100 f32,
/// env = 16 f32 (Tensor 4), arch = 12 f32 (Tensor 5), fold = 4 f32 (Tensor 6).
const SLOTS_BYTES: usize = 800 * 4;
const MATERIALS_BYTES: usize = 100 * 4 * 4; //(was *4)
const ROTATIONS_BYTES: usize = 100 * 3 * 4; // 🟢 TENSOR 7: 100 Slots × Stride 3 (rotX, rotY, rotZ)
const ENV_BYTES: usize = 16 * 4;
const ARCH_BYTES: usize = 12 * 4;
const FOLD_BYTES: usize = 4 * 4;

/// Alles, was der Render-Thread zum Zeichnen eines Frames braucht.
pub struct GpuPipeline {
    pub client: Client,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub render_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub resolution_buffer: wgpu::Buffer,
    pub resolution: ResolutionUniform,
    /// Raymarch-Output (RGB f32): w*h*3*4 Bytes. Wird bei Resize ersetzt.
    pub output_handle: Handle,
    pub meta_handle: Handle,
    pub slots_handle: Handle,
    pub materials_handle: Handle,
    pub rotations_handle: Handle, // 🟢 TENSOR 7: Objekt-Rotation
    pub env_handle: Handle,
    pub arch_handle: Handle,
    pub fold_handle: Handle,
    bind_group: Option<wgpu::BindGroup>,
}

impl GpuPipeline {
    /// Synchron, ohne Sleeps. `width`/`height` sind die initialen Werte der
    /// wgpui-Surface (Back-Buffer-Größe wird in der Loop per Resize erkannt).
    pub fn new(
        surface: &WgpuSurfaceHandle,
        width: u32,
        height: u32,
    ) -> Result<Self, GpuInitError> {
        let device = surface.device().clone();
        let queue = surface.queue().clone();

        // --- cubecl: eigener Vulkan-Adapter, aber Device+Queue der Surface ---
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
        }))
        .map_err(|_| GpuInitError::NoAdapter)?;
        let wgpu_setup = WgpuSetup {
            instance,
            adapter,
            device: device.clone(),
            queue: queue.clone(),
            backend: wgpu::Backend::Vulkan,
        };
        let cubecl_device_id = init_device(wgpu_setup, RuntimeOptions::default());
// WgpuRuntime::client erfordert nun das importierte Runtime-Trait
// Nutzt den AutoCompiler (empfohlen)
let client = cubecl_wgpu::WgpuRuntime::<cubecl_wgpu::AutoCompiler>::client(&cubecl_device_id);

        // --- Kernel-Register-Buffer ---
        let output_handle = client.empty(output_byte_size(width, height));
        let meta_handle = client.empty(4);
        let slots_handle = client.empty(SLOTS_BYTES);
        let materials_handle = client.empty(MATERIALS_BYTES);
        let rotations_handle = client.empty(ROTATIONS_BYTES); // 🟢 TENSOR 7
        let env_handle = client.empty(ENV_BYTES);
        let arch_handle = client.empty(ARCH_BYTES);
        let fold_handle = client.empty(FOLD_BYTES);

        // --- wgpu-Blit: Raymarch-Buffer → Back-Buffer ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Screen Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let resolution = ResolutionUniform {
            width,
            height,
            _padding: [0; 2],
        };
        use wgpu::util::DeviceExt;
        let resolution_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Resolution Uniform"),
                contents: bytemuck::cast_slice(&[resolution]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
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

        let mut pipeline = Self {
            client,
            device,
            queue,
            render_pipeline,
            bind_group_layout,
            resolution_buffer,
            resolution,
            output_handle,
            meta_handle,
            slots_handle,
            materials_handle,
            rotations_handle, // 🟢 TENSOR 7
            env_handle,
            arch_handle,
            fold_handle,
            bind_group: None,
        };
        pipeline.rebuild_bind_group();
        Ok(pipeline)
    }

    /// Baut die BindGroup neu (nötig, wenn `output_handle` ersetzt wurde).
    fn rebuild_bind_group(&mut self) {
        // Ergänzen Sie die Turbofish-Syntax mit dem WgpuServer-Typen:
let managed_resource = self.client
    .get_resource::<cubecl_wgpu::WgpuServer<cubecl_wgpu::AutoCompiler>>(self.output_handle.clone())
    .unwrap();

let wgpu_resource: &cubecl_wgpu::WgpuResource = managed_resource.resource();
//let src_wgpu_buffer = &wgpu_resource.buffer;

        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen Bind Group"),
            layout: &self.bind_group_layout,
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
                        buffer: &self.resolution_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        }));
    }

    /// Re-Allokiert den Output-Buffer bei Größenänderung und schreibt die
    /// Resolution neu. Nur bei `w>0 && h>0` aufrufen.
    pub fn resize(&mut self, w: u32, h: u32) {
        self.resolution.width = w;
        self.resolution.height = h;
        self.queue.write_buffer(
            &self.resolution_buffer,
            0,
            bytemuck::cast_slice(&[self.resolution]),
        );
        self.output_handle = self.client.empty(output_byte_size(w, h));
        self.rebuild_bind_group();
    }

    /// Schreibt die Tensor-4/5/6-Register + Meta in die cubecl-Buffer.
    /// Nutzt dieselbe wgpu-Queue wie cubecl → Schreibreihenfolge vor
    /// Kernel-Launch ist garantiert.
    pub fn write_state_buffers(&self, s: &ApplicationState) {
        let write = |handle: &Handle, data: &[u8]| {
          // 1. Dem Server-Typ beim get_resource-Aufruf mitgeben (Typ-Inferenz für das Backend)
        let resource = self.client
            .get_resource::<cubecl_wgpu::WgpuServer<cubecl_wgpu::AutoCompiler>>(handle.clone())
            .unwrap();
        
        // 2. Den echten Typ "cubecl_wgpu::WgpuResource" statt des Platzhalters "Resource" nutzen
        let wgpu_res: &cubecl_wgpu::WgpuResource = resource.resource();
        
        self.queue.write_buffer(&wgpu_res.buffer, wgpu_res.offset, data);
    };
        write(&self.env_handle, bytemuck::cast_slice(&s.env_data()));
        write(&self.arch_handle, bytemuck::cast_slice(&s.arch_data()));
        write(&self.fold_handle, bytemuck::cast_slice(&s.fold_data()));
        write(&self.meta_handle, bytemuck::cast_slice(&s.meta_data()));
        // 🚀 NEU: Schreibt die interleaved gepackten Slot-Daten in den VRAM-Buffer
        write(&self.slots_handle, bytemuck::cast_slice(&s.slots_data()));
         // 🚀 BEHOBEN: Schreibt die physischen PBR-Materialdaten in den VRAM-Buffer (Stride 4)
        write(&self.materials_handle, bytemuck::cast_slice(&s.materials_data()));
        // 🟢 TENSOR 7: Schreibt die Rotations-Daten (Euler-Winkel) in den VRAM-Buffer (Stride 3)
        write(&self.rotations_handle, bytemuck::cast_slice(&s.rotations_data()));
    }

    /// Ein kompletter Frame: Raymarch-Kernel-Launch + Blit auf den
    /// Back-Buffer. `time`/`dt` in Sekunden.
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &mut self,
        view: &wgpu::TextureView,
        dw: u32,
        dh: u32,
        s: &ApplicationState,
        time: f32,
    ) {
        let total_pixels = (dw * dh) as usize;
        let cube_count = cubecl::CubeCount::Static(
            dw.div_ceil(CUBE_DIM_X),
            dh.div_ceil(CUBE_DIM_Y),
            1,
        );
        let cube_dim = cubecl::CubeDim::new_3d(CUBE_DIM_X, CUBE_DIM_Y, 1);

        // --- Kernel-Args (unsafe: Handles sind korrekt sized, Strides/Shape 1D) ---
        let output_arg = unsafe {
            TensorArg::from_raw_parts(
                self.output_handle.clone(),
                Strides::from(&[1usize]),
                Shape::from(&[total_pixels]),
            )
        };
        let meta_arg = unsafe {
            TensorArg::from_raw_parts(
                self.meta_handle.clone(),
                Strides::from(&[1usize]),
                Shape::from(&[1usize]),
            )
        };
        let slots_arg = unsafe {
            TensorArg::from_raw_parts(
                self.slots_handle.clone(),
                Strides::from(&[1usize]),
                Shape::from(&[800usize]),
            )
        };
        let materials_arg = unsafe {
            TensorArg::from_raw_parts(
                self.materials_handle.clone(),
                Strides::from(&[1usize]),
                Shape::from(&[400usize]),  //was 100
            )
        };
        let rotations_arg = unsafe {
            TensorArg::from_raw_parts(
                self.rotations_handle.clone(),
                Strides::from(&[1usize]),
                Shape::from(&[300usize]), // 🟢 TENSOR 7: 100 Slots × Stride 3
            )
        };
        let env_arg = unsafe {
            TensorArg::from_raw_parts(
                self.env_handle.clone(),
                Strides::from(&[1usize]),
                Shape::from(&[16usize]),
            )
        };
        let arch_arg = unsafe {
            TensorArg::from_raw_parts(
                self.arch_handle.clone(),
                Strides::from(&[1usize]),
                Shape::from(&[12usize]),
            )
        };
        let fold_arg = unsafe {
            TensorArg::from_raw_parts(
                self.fold_handle.clone(),
                Strides::from(&[1usize]),
                Shape::from(&[4usize]),
            )
        };

        crate::kernel::raymarch_sdf_kernel::launch(
            &self.client,
            cube_count,
            cube_dim,
            output_arg,
            meta_arg,
            slots_arg,
            materials_arg,
           // rotations_arg, // 🟢 TENSOR 7
            env_arg,
            arch_arg,
            fold_arg,
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
            s.enable_key,
            s.enable_fill,
            s.enable_rim,
        );

        // --- Blit: Raymarch-Buffer → Surface-Back-Buffer ---
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            render_pass.set_pipeline(&self.render_pipeline);
            let bind_group = self
                .bind_group
                .as_ref()
                .expect("bind_group is built in GpuPipeline::new");
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

/// RGB f32-Output: 3 Kanäle × 4 Bytes pro Pixel.
fn output_byte_size(w: u32, h: u32) -> usize {
    (w * h) as usize * 3 * 4
}
