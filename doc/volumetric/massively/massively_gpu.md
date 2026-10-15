Um das massively-Crate nahtlos in dein bestehendes Host-Architekturmodell (gpu.rs) zu integrieren, erweitern wir die Pipeline um einen massively::Executor. Da massively direkt auf den DeviceSlice-Strukturen und MVec des zugrunde liegenden CubeCL-Clients operiert, verwenden wir den initialisierten Treiber-Kontext weiter.
Anstatt rohe Handle-Objekte manuell zu verwalten, kapseln wir den Zustand über die typisierten DeviceVec-Puffer von massively, welche die Zero-Copy Memory-Sicherheit garantieren.
Hier ist die vollständig überarbeitete und integrierte gpu.rs:
rust
//! GPU-Pipeline: cubecl-Compute-Setup + wgpu-Blit-Pipeline + massively Parallel Engine.
//!
//! `GpuPipeline::new` läuft **synchron** im View-Konstruktor — `device()`/
//! `queue()` stehen direkt nach `window.create_wgpu_surface()` bereit
//! (wgpui-0.3.6, elements/wgpu_surface.rs:82–88).

use crate::app_state::ApplicationState;
use bytemuck::{Pod, Zeroable};
use cubecl::client::Client; 
use cubecl::frontend::TensorArg;
use cubecl_runtime::runtime::Runtime; 
use cubecl_runtime::server::Handle;
use cubecl_wgpu::{init_device, RuntimeOptions, WgpuRuntime, WgpuSetup};
use cubecl_zspace::{Shape, Strides};
use wgpui_kit::WgpuSurfaceHandle;

// 🟢 MASIVELY EXTENSIONS
use massively::{Executor, DeviceVec, MIndex};

// =========================================================================
// 📐 GLOBALE GPU-PIPELINE KONSTANTEN (STRUCTURED BUFFER PACKING)
// =========================================================================
pub const MAX_SLOTS: usize = 100;

pub const SLOTS_COUNT: usize = MAX_SLOTS * 8;      // 800
pub const MATERIALS_COUNT: usize = MAX_SLOTS * 4;  // 400
pub const ROTATIONS_COUNT: usize = MAX_SLOTS * 3;  // 300
pub const TOTAL_BUFFER_SIZE: usize = SLOTS_COUNT + MATERIALS_COUNT + ROTATIONS_COUNT; // 1500

pub const SDF_GRID_RES: usize = crate::sdf_preprocessor::SDF_GRID_RES; // 64
pub const SDF_GRID_COUNT: usize = SDF_GRID_RES * SDF_GRID_RES * SDF_GRID_RES; // 262144
pub const SDF_GRID_START: usize = TOTAL_BUFFER_SIZE; // 1500
pub const TOTAL_SCENE_FLOATS: usize = TOTAL_BUFFER_SIZE + SDF_GRID_COUNT; // 263644

pub const SLOTS_BYTES: usize = SLOTS_COUNT * 4;      
pub const MATERIALS_BYTES: usize = MATERIALS_COUNT * 4;  
pub const ROTATIONS_BYTES: usize = ROTATIONS_COUNT * 4;  
pub const TOTAL_SCENE_BYTES: usize = TOTAL_SCENE_FLOATS * 4; // ~1.05 MB

/// Hilfsfunktion zur Ermittlung der Byte-Größe des Ausgabe-Tensors (RGBA f32)
#[inline]
pub fn output_byte_size(width: u32, height: u32) -> usize {
    (width * height * 4 * 4) as usize
}

/// Auflösungs-Uniform für den Screen-Shader (fs_main).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ResolutionUniform {
    pub width: u32,
    pub height: u32,
    pub _padding: [u32; 2],
}

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

/// Alles, was der Render-Thread zum Zeichnen eines Frames braucht.
pub struct GpuPipeline {
    pub client: Client,
    pub exec: Executor<WgpuRuntime<cubecl_wgpu::AutoCompiler>>, // 🟢 MASIVELY Executor-Instanz
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub render_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub resolution_buffer: wgpu::Buffer,
    pub resolution: ResolutionUniform,
    
    // massively verwaltet DeviceVec-Typen typsicher und zero-copy
    pub output_vec: DeviceVec<WgpuRuntime<cubecl_wgpu::AutoCompiler>, f32>, // 🟢 OPTIMIERT: f32 Stream
    pub scene_vec: DeviceVec<WgpuRuntime<cubecl_wgpu::AutoCompiler>, f32>,  // 🟢 OPTIMIERT: Scene Stream

    pub bind_group: Option<wgpu::BindGroup>,
}

impl GpuPipeline {
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
        let client = cubecl_wgpu::WgpuRuntime::<cubecl_wgpu::AutoCompiler>::client(&cubecl_device_id);

        // 1. 🟢 MASIVELY EXECUTOR INITIALISIERUNG
        // Der Executor verwaltet ab sofort Allokationen und die parallelen Primitives.
        let exec = Executor::<WgpuRuntime<cubecl_wgpu::AutoCompiler>>::new(cubecl_device_id);

        // 2. 🟢 TYPSICHERE ALLOKATION VIA MASIVELY DEVICEVEC
        // Nutzt MIndex (u32-Schranken) konform zur Massively-Spezifikation.
        let total_output_floats = (width * height * 4) as MIndex;
        let output_vec = exec.alloc::<f32>(total_output_floats);
        let scene_vec = exec.alloc::<f32>(TOTAL_SCENE_FLOATS as MIndex);

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
        let resolution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resolution Uniform"),
            contents: bytemuck::cast_slice(&[resolution]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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

        // Pipeline Layout und Pipeline Erstellung vervollständigen
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Screen Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
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
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let mut pipeline = Self {
            client,
            exec,
            device,
            queue,
            render_pipeline,
            bind_group_layout,
            resolution_buffer,
            resolution,
            output_vec,
            scene_vec,
            bind_group: None,
        };

        pipeline.update_bind_group();
        Ok(pipeline)
    }

    /// Aktualisiert die WGPU Bind-Group, wenn Ressourcen reallokiert wurden (z.B. nach Resize).
    pub fn update_bind_group(&mut self) {
        // Extrahiere das rohe Handle über das innere Storage-Mapping von massively
        let raw_output_handle = self.output_vec.handle();
        let wgpu_buffer = cubecl_wgpu::WgpuRuntime::<cubecl_wgpu::AutoCompiler>::get_buffer(&raw_output_handle);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.resolution_buffer.as_entire_binding(),
                },
            ],
        });
        self.bind_group = Some(bind_group);
    }

Verwende Code mit Vorsicht.
/// Führt bei einer Größenänderung des Fensters eine sichere Reallokation durch.
pub fn resize(&mut self, width: u32, height: u32) {
self.resolution.width = width;
self.resolution.height = height;
self.queue.write_buffer(
&self.resolution_buffer,
0,
bytemuck::cast_slice(&[self.resolution]),
);
// Neues DeviceVec über den massively Executor anfordern
let total_output_floats = (width * height * 4) as MIndex;
self.output_vec = self.exec.alloc::(total_output_floats);
self.update_bind_group();
}
}

### Was wurde angepasst?
1. **`massively::Executor` Integration**: Übernimmt die Kontrollschranke des Backends unter Beibehaltung der WGPU-Vulkan Interoperabilität.
2. **`DeviceVec<R, T>` statt nackter `Handle`**: `output_vec` und `scene_vec` kapseln nun den Grafikspeicher. Beim Binden an die WGPU-Renderpipeline ziehen wir das rohe Handle im Hintergrund mittels `.handle()`, um den Blit-Shader unverändert zu füttern.
3. **MIndex-Sicherheit**: Die Pufferallokationen folgen der u32-Skalierung, was synchronisierte Host-Übergänge performant absichert.

<FollowUp>
Soll der Coding Agent nun im nächsten Schritt die **Datenübergabe in `render_loop.rs`** anpassen, um die Szenendaten via `exec.to_device()` statt alter Tensor-Upload-Methoden direkt in den `scene_vec` zu streamen?
</FollowUp>
