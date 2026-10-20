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

/// 🟢 ZENTRALE STEUERUNG FÜR DIE GPU-PRÄZISION
/// Hier kannst du den Typ global auf `f32`, `f64` oder (falls vom Backend unterstützt) `f16` ändern.
pub type GpuFloat = f32;  // Centralized precision switch

// =========================================================================
// 📐 GLOBALE GPU-PIPELINE KONSTANTEN (STRUCTURED BUFFER PACKING)
// =========================================================================
pub const MAX_SLOTS: usize = 100;

// Element-Counts (Anzahl der f32-Werte)
pub const SLOTS_COUNT: usize = MAX_SLOTS * 8;      // 800
pub const MATERIALS_COUNT: usize = MAX_SLOTS * 4;  // 400
pub const ROTATIONS_COUNT: usize = MAX_SLOTS * 3;  // 300
pub const TOTAL_BUFFER_SIZE: usize = SLOTS_COUNT + MATERIALS_COUNT + ROTATIONS_COUNT; // 1500
// 🟢 PLAN STEP 2/3: SDF-Statuen-Grid (64³ Voxel) wird an den
// Mega-Scene-Buffer angehängt. Layout: [slots|materials|rotations|sdf_grid].
pub const SDF_GRID_RES: usize = crate::sdf_preprocessor::SDF_GRID_RES; // 64
pub const SDF_GRID_COUNT: usize = SDF_GRID_RES * SDF_GRID_RES * SDF_GRID_RES; // 262144
pub const SDF_GRID_START: usize = TOTAL_BUFFER_SIZE; // 1500
pub const TOTAL_SCENE_FLOATS: usize = TOTAL_BUFFER_SIZE + SDF_GRID_COUNT; // 263644

// Byte-Größen (1x f32 = 4 Bytes)
pub const SLOTS_BYTES: usize = SLOTS_COUNT * 4;      // 3200 Bytes
pub const MATERIALS_BYTES: usize = MATERIALS_COUNT * 4;  // 1600 Bytes
pub const ROTATIONS_BYTES: usize = ROTATIONS_COUNT * 4;  // 1200 Bytes
pub const TOTAL_SCENE_BYTES: usize = TOTAL_SCENE_FLOATS * 4; // ~1.05 MB
// =========================================================================

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
/// Nur noch große dynamische Daten als Storage-Buffer (params1/params4):
/// slots = 800 f32, materials = 400 f32, rotations = 300 f32.
/// meta/env/arch/fold sind entfallen → CubeLaunch-Structs im Info-Uniform.
//const SLOTS_BYTES: usize = 800 * 4;
//const MATERIALS_BYTES: usize = 100 * 4 * 4;
//const ROTATIONS_BYTES: usize = 100 * 3 * 4; // 🟢 TENSOR 7: 100 Slots × Stride 3 (rotX, rotY, rotZ)

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
    // 📦 REFAKTORIERT: Ein einziger Mega-Buffer statt 3 Einzelslots!
    pub scene_handle: Handle, 

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
      // --- Kernel-Register-Buffer ---
let output_handle = client.empty(output_byte_size(width, height));

// 📦 2. Nur noch EIN Handle allokieren
let scene_handle = client.empty(TOTAL_SCENE_BYTES);
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
            scene_handle, // 🟢 Jetzt kompakt verpackt
            bind_group: None,
        };
        pipeline.rebuild_bind_group();
        Ok(pipeline)
    }

    /// 🟢 Headless-Konstruktor (Tests/CI): gleiche Pipeline, aber ohne
    /// wgpui-Surface — Adapter/Device/Queue kommen vom Caller. Der
    /// CubeCL-Teil (Kernel-Compile + Launch) läuft identisch.
    pub fn new_headless(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) -> Result<Self, GpuInitError> {
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
        let client = cubecl_wgpu::WgpuRuntime::<cubecl_wgpu::AutoCompiler>::client(&cubecl_device_id);
        let output_handle = client.empty(output_byte_size(width, height));
        let scene_handle = client.empty(TOTAL_SCENE_BYTES);
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
            device: device.clone(),
            queue: queue.clone(),
            render_pipeline,
            bind_group_layout,
            resolution_buffer,
            resolution,
            output_handle,
            scene_handle,
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
        // 🚀 Nur noch echte dynamische Arrays als Storage-Buffer (params4).
        // meta/env/arch/fold/camera/render reisen als CubeLaunch-Structs im
        // Info-Uniform — kein write_buffer mehr nötig (params2/params4).
        // 📦 REFAKTORIERT & OPTIMIERT: 
    // Alle 3 Register-Tabelle (Slots, Materials, Rotations) reisen jetzt 
    // voll-ausgerichtet als ein einziger Mega-Buffer (6000 Bytes) zur GPU!
    let packed_scene = s.scene_data();
    write(&self.scene_handle, bytemuck::cast_slice(&packed_scene));
}

    /// Ein kompletter Frame: Raymarch-Kernel-Launch + Blit auf den
    /// Back-Buffer. `time`/`dt` in Sekunden.
    #[allow(clippy::too_many_arguments)]
    /// 🟢 Headless-fähiger Kernel-Launch: kompiliert den CubeCL-Kernel
    /// (WGSL-Codegen) und startet den Compute-Pass. Getrennt vom Blit,
    /// damit Tests den Kernel ohne Surface treiben können.
pub fn launch_kernel(&mut self, dw: u32, dh: u32, s: &ApplicationState, time: GpuFloat) {
    let total_pixels = (dw * dh) as usize;
    let cube_count = cubecl::CubeCount::Static(
        dw.div_ceil(CUBE_DIM_X),
        dh.div_ceil(CUBE_DIM_Y),
        1,
    );
    let cube_dim = cubecl::CubeDim::new_3d(CUBE_DIM_X, CUBE_DIM_Y, 1);
    
    // --- Kernel-Args ---
    let output_arg = unsafe {
        TensorArg::from_raw_parts(
            self.output_handle.clone(),
            (&[1usize]).into(),
            (&[total_pixels]).into(),
        )
    };
    
    let scene_arg = unsafe {
        cubecl::prelude::TensorArg::from_raw_parts(
            self.scene_handle.clone(),
            (&[TOTAL_SCENE_FLOATS]).into(), 
            (&[1usize]).into(),                  
        )
    };
    
    // 🟢 KORREKTUR: Nutzt jetzt den generischen Typ-Alias statt des hartcodierten f32
    crate::kernel::raymarch_sdf_kernel::launch::<GpuFloat>(
        &self.client,
        cube_count,
        cube_dim,
        output_arg,
        scene_arg,
        s.camera_launch(),
        s.env_launch(),
        s.arch_launch(),
        s.fold_launch(),
        s.render_launch(time, dw, dh),
    );
}
    pub fn render_frame(
        &mut self,
        view: &wgpu::TextureView,
        dw: u32,
        dh: u32,
        s: &ApplicationState,
        time: f32,
    ) {
        self.launch_kernel(dw, dh, s, time);
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
