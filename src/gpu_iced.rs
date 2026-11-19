//! GPU Pipeline for the iced shader widget.
//!
//! Adapted 1:1 from the wgpui `GpuPipeline` (src/gpu.rs): CubeCL init, blit
//! render pipeline, state buffers, kernel launch. The only differences are
//! the constructor signature (`Pipeline::new(device, queue, format)`) and the
//! missing surface — iced owns the surface and hands us device/queue/format.
//!
//! Blueprint: iced/examples/custom_shader (impl shader::Pipeline).

use crate::app_state::ApplicationState;
use crate::kernel::raymarch_sdf_kernel;
use bytemuck::{Pod, Zeroable};
use cubecl::client::Client;
use cubecl::prelude::TensorArg;
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::Handle;
use cubecl_wgpu::{init_device, RuntimeOptions, WgpuRuntime, WgpuSetup};

/// GPU float type — centralized precision control (see gpu.rs).
pub type GpuFloat = f32;

// =========================================================================
// Constants (ported from gpu.rs)
// =========================================================================
pub const MAX_SLOTS: usize = 100;
pub const SLOTS_COUNT: usize = MAX_SLOTS * 8; // 800
pub const MATERIALS_COUNT: usize = MAX_SLOTS * 4; // 400
pub const ROTATIONS_COUNT: usize = MAX_SLOTS * 3; // 300

pub const MAX_SDF_OBJECTS: usize = 4;
pub const SDF_HEADER_STRIDE: usize = 24;
pub const SDF_GRID_RES: usize = crate::sdf_preprocessor::SDF_GRID_RES; // 64
pub const SDF_GRID_COUNT: usize = SDF_GRID_RES * SDF_GRID_RES * SDF_GRID_RES; // 262144
pub const SDF_OBJECTS_FLOATS: usize =
    MAX_SDF_OBJECTS * SDF_HEADER_STRIDE + MAX_SDF_OBJECTS * SDF_GRID_COUNT; // 1_048_672

pub const SLOTS_BYTES: usize = SLOTS_COUNT * 4;
pub const MATERIALS_BYTES: usize = MATERIALS_COUNT * 4;
pub const ROTATIONS_BYTES: usize = ROTATIONS_COUNT * 4;
pub const SDF_OBJECTS_BYTES: usize = SDF_OBJECTS_FLOATS * 4;

const CUBE_DIM_X: u32 = 16;
const CUBE_DIM_Y: u32 = 4;

/// Resolution uniform for the screen shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ResolutionUniform {
    pub width: u32,
    pub height: u32,
    pub _padding: [u32; 2],
}

/// RGB f32-Output: 3 Kanäle × 4 Bytes pro Pixel.
fn output_byte_size(w: u32, h: u32) -> usize {
    (w * h) as usize * 3 * 4
}

/// Alles, was zum Zeichnen eines Frames nötig ist (analog gpu.rs).
pub struct SdfPipeline {
    pub client: Client,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub render_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub resolution_buffer: wgpu::Buffer,
    pub resolution: ResolutionUniform,
    /// Raymarch-Output (RGB f32): w*h*3*4 Bytes. Wird bei Resize ersetzt.
    pub output_handle: Handle,
    pub slots_handle: Handle,
    pub materials_handle: Handle,
    pub rotations_handle: Handle,
    pub sdf_objects_handle: Handle,
    bind_group: Option<wgpu::BindGroup>,
}

impl SdfPipeline {
    /// Baut die BindGroup neu (nötig, wenn `output_handle` ersetzt wurde).
    fn rebuild_bind_group(&mut self) {
        let managed_resource = self
            .client
            .get_resource::<cubecl_wgpu::WgpuServer<cubecl_wgpu::AutoCompiler>>(
                self.output_handle.clone(),
            )
            .unwrap();

        let wgpu_resource: &cubecl_wgpu::WgpuResource = managed_resource.resource();

        self.bind_group = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            }),
        );
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

    /// Schreibt die Tensor-Register + Objekt-Tensor in die cubecl-Buffer.
    /// Nutzt dieselbe wgpu-Queue wie cubecl → Schreibreihenfolge vor
    /// Kernel-Launch ist garantiert.
    pub fn write_state_buffers(&self, s: &ApplicationState) {
        let write = |handle: &Handle, data: &[u8]| {
            let resource = self
                .client
                .get_resource::<cubecl_wgpu::WgpuServer<cubecl_wgpu::AutoCompiler>>(handle.clone())
                .unwrap();
            let wgpu_res: &cubecl_wgpu::WgpuResource = resource.resource();
            self.queue.write_buffer(&wgpu_res.buffer, wgpu_res.offset, data);
        };

        let packed_slots = s.slots_data();
        write(&self.slots_handle, bytemuck::cast_slice(&packed_slots));
        let packed_materials = s.materials_data();
        write(&self.materials_handle, bytemuck::cast_slice(&packed_materials));
        let packed_rotations = s.rotations_data();
        write(&self.rotations_handle, bytemuck::cast_slice(&packed_rotations));
        let packed_objects = s.sdf_objects_data();
        write(&self.sdf_objects_handle, bytemuck::cast_slice(&packed_objects));
    }

    /// Kompiliert den CubeCL-Kernel (WGSL-Codegen) und startet den Compute-Pass.
    pub fn launch_kernel(&mut self, dw: u32, dh: u32, s: &ApplicationState, time: GpuFloat) {
        let total_pixels = (dw * dh) as usize;
        let cube_count = cubecl::CubeCount::Static(
            dw.div_ceil(CUBE_DIM_X),
            dh.div_ceil(CUBE_DIM_Y),
            1,
        );
        let cube_dim = cubecl::CubeDim::new_3d(CUBE_DIM_X, CUBE_DIM_Y, 1);

        let output_arg = unsafe {
            TensorArg::from_raw_parts(
                self.output_handle.clone(),
                (&[1usize]).into(),
                (&[total_pixels]).into(),
            )
        };

        let slots_arg = unsafe {
            TensorArg::from_raw_parts(
                self.slots_handle.clone(),
                (&[SLOTS_COUNT]).into(),
                (&[1usize]).into(),
            )
        };

        let materials_arg = unsafe {
            TensorArg::from_raw_parts(
                self.materials_handle.clone(),
                (&[MATERIALS_COUNT]).into(),
                (&[1usize]).into(),
            )
        };

        let rotations_arg = unsafe {
            TensorArg::from_raw_parts(
                self.rotations_handle.clone(),
                (&[ROTATIONS_COUNT]).into(),
                (&[1usize]).into(),
            )
        };

        let objects_arg = unsafe {
            TensorArg::from_raw_parts(
                self.sdf_objects_handle.clone(),
                (&[SDF_OBJECTS_FLOATS]).into(),
                (&[1usize]).into(),
            )
        };

        raymarch_sdf_kernel::launch::<GpuFloat>(
            &self.client,
            cube_count,
            cube_dim,
            output_arg,
            slots_arg,
            materials_arg,
            rotations_arg,
            objects_arg,
            s.camera_launch(),
            s.env_launch(),
            s.arch_launch(),
            s.fold_launch(),
            s.render_launch(time, dw, dh),
        );
    }

    /// Ein kompletter Frame: State-Buffer schreiben + Kernel-Launch.
    /// Wird aus `SdfPrimitive::prepare` aufgerufen (gleiche Queue → Ordnung garantiert).
    pub fn run_frame(&mut self, s: &ApplicationState, w: u32, h: u32, time: f32) {
        self.write_state_buffers(s);
        self.launch_kernel(w, h, s, time);
    }

    /// Blit: Raymarch-Buffer → Surface-Back-Buffer (eigener Encoder/Submit).
    pub fn render(&self, target: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder) {
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("bind_group is built in SdfPipeline::new");

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Screen Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

impl iced::widget::shader::Pipeline for SdfPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        use wgpu::util::DeviceExt;
        // --- cubecl: eigener Vulkan-Adapter, aber Device+Queue von iced ---
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
        .expect("kein passender Vulkan-Adapter gefunden");

        let wgpu_setup = WgpuSetup {
            instance,
            adapter,
            device: device.clone(),
            queue: queue.clone(),
            backend: wgpu::Backend::Vulkan,
        };
        let cubecl_device_id = init_device(wgpu_setup, RuntimeOptions::default());
        let client =
            WgpuRuntime::<cubecl_wgpu::AutoCompiler>::client(&cubecl_device_id);

        // Initial resolution (wird in prepare() auf den echten Viewport gesetzt).
        let width = 960;
        let height = 540;

        let output_handle = client.empty(output_byte_size(width, height));
        let slots_handle = client.empty(SLOTS_BYTES);
        let materials_handle = client.empty(MATERIALS_BYTES);
        let rotations_handle = client.empty(ROTATIONS_BYTES);
        let sdf_objects_handle = client.empty(SDF_OBJECTS_BYTES);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Screen Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let resolution = ResolutionUniform {
            width,
            height,
            _padding: [0; 2],
        };
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

        let mut pipeline = Self {
            client,
            device: device.clone(),
            queue: queue.clone(),
            render_pipeline,
            bind_group_layout,
            resolution_buffer,
            resolution,
            output_handle,
            slots_handle,
            materials_handle,
            rotations_handle,
            sdf_objects_handle,
            bind_group: None,
        };
        pipeline.rebuild_bind_group();
        pipeline
    }
}