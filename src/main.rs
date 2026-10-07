#![allow(warnings)]

mod kernel;
pub mod input; 
use input::InputManager; 

// --- CUBECL IMPORTS (Kompatibel mit v0.11.0-pre.2) ---
use cubecl::prelude::*;
use cubecl::client::Client; // 🟢 KORREKTUR: client:: hinzufügen
use cubecl::server::Handle; // 🟢 FIX: Direkt über das Haupt-Crate importieren
use cubecl::frontend::{TensorArg, TensorBinding};

// Alle wgpu-Backend-spezifischen Typen kommen nativ aus cubecl_wgpu
use cubecl_wgpu::{WgpuRuntime, AutoCompiler, WgpuResource, WgpuSetup, init_device, RuntimeOptions};
use cubecl_runtime::runtime::Runtime; // RICHTIG


// (Falls benötigt für dein Projekt - bleibt unverändert)
use cubecl_zspace::{Strides, Shape};

// --- CORE / UTILS IMPORTS ---
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use std::time::{Instant, Duration};

// --- MODERN WINIT 0.30 IMPORTS ---
use winit::dpi::LogicalSize;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

// --- NATIVE WGPU IMPORTS ---
use wgpu::util::DeviceExt;
// Füge hier sicherheitshalber noch die Basis-Typen von wgpu hinzu, falls dein Code sie nutzt:
use wgpu::{
    Backends, Device, Instance, InstanceDescriptor, Queue, Surface, SurfaceConfiguration, TextureUsages
};


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ResolutionUniform {
    width: u32,
    height: u32,
    _padding: [u32; 2], // 16-byte alignment for uniform buffer
}

// 1. Der Winit 0.30 Zustands-Manager kapselt deine Application und den InputManager
struct App {
    app_state: Option<Application>,
    input_manager: InputManager,
}

struct Application {
        pub window: Arc<Window>, // In Winit 0.30 ist das Window-Objekt direkt im winit-Root
    //pub window: Arc<winit::window::Window>,
    pub surface: wgpu::Surface<'static>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub config: wgpu::SurfaceConfiguration,
    
    // CubeCL resources
    pub client: Client,
    pub output_handle: Handle,
    
    // wgpu render resources
    pub render_pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub resolution_buffer: wgpu::Buffer,
    pub resolution_uniform: ResolutionUniform,
        // 🟢 NEU: Kamera-Zustand dauerhaft in der Struktur speichern!
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_z: f32,
    pub cam_yaw: f32,
    pub cam_pitch: f32,
    //
    // 🟢 NEU: Licht- und Rendering-Parameter in der Struktur speichern
    pub light_intensity: f32,
    pub ambient_strength: f32,
    pub enable_ao_mode: u32,
    pub current_shadow_mode: u32,
    pub enable_key: u32,
    pub enable_fill: u32,
    pub enable_rim: u32,
    pub dynamic_blend_factor: f32,
    // Frame timing
    frame_counter: u32,
    last_frame_time: Instant,
    total_time: f32, // 🟢 NEU hinzufügen
}

impl Application {
    async fn new(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let width = size.width;
        let height = size.height;
        let total_pixels = (width * height) as usize;
       // let byte_size = total_pixels * std::mem::size_of::<u32>();
          let byte_size = total_pixels * 3 * std::mem::size_of::<f32>(); 
        // 1. Create wgpu Instance, Adapter, Device, Queue
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        
        let surface = instance.create_surface(window.clone()).unwrap();
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .unwrap();

        let (wgpu_device, wgpu_queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("SDF Demo Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .unwrap();

        let device = Arc::new(wgpu_device);
        let queue = Arc::new(wgpu_queue);

        // 2. Configure Surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo, // V-SYNC Immediate (V-Sync AUS),Mailbox (Fast V-Sync / G-Sync)
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        surface.configure(&device, &config);

        // ====================================================================
        // ZERO-COPY KEY: Register the SAME device/queue with CubeCL
        // ====================================================================
        let wgpu_setup = WgpuSetup {
            instance: instance.clone(),
            adapter: adapter.clone(),
            device: device.as_ref().clone(),
            queue: queue.as_ref().clone(),
            backend: wgpu::Backend::Vulkan,
        };
        
        let cubecl_device_id = init_device(wgpu_setup, RuntimeOptions::default());
        
       // let client: Client<WgpuRuntime<AutoCompiler>> = 
       //     WgpuRuntime::client(&cubecl_device_id);
// WgpuRuntime::client erfordert nun das importierte Runtime-Trait
// Nutzt den AutoCompiler (empfohlen)
let client = cubecl_wgpu::WgpuRuntime::<cubecl_wgpu::AutoCompiler>::client(&cubecl_device_id);


        // 3. Create output buffer on the SHARED device (via CubeCL client)
        let output_handle = client.empty(byte_size);

        // 4. Create wgpu render pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Screen Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Resolution uniform buffer
        let resolution_uniform = ResolutionUniform {
            width,
            height,
            _padding: [0, 0],
        };
        
        let resolution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resolution Uniform"),
            contents: bytemuck::cast_slice(&[resolution_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Screen Bind Group Layout"),
            entries: &[
                // Binding 0: Storage buffer (CubeCL output)
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
                // Binding 1: Uniform buffer (Resolution)
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

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Screen Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Render pipeline
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
                    format: config.format,
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

        // Placeholder bind group (will be recreated each frame with actual CubeCL buffer)
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen Bind Group (placeholder)"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &resolution_buffer, // placeholder
                        offset: 0,
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

        Self {
            window,
            surface,
            device,
            queue,
            config,
            client,
            output_handle,
            render_pipeline,
            bind_group,
            bind_group_layout,
            resolution_buffer,
            resolution_uniform,
           
             // 🟢 NEU: Startwerte für die Kamera festlegen
            cam_x: 0.0,
            cam_y: 0.0,
            cam_z: -5.0,
            cam_yaw: 0.0,
            cam_pitch: 0.0,
            
            // 🟢 NEU: Licht-Startwerte setzen
            light_intensity: 1.0,
            ambient_strength: 0.1,
            enable_ao_mode: 1,
            current_shadow_mode: 1,
            enable_key: 1,
            enable_fill: 1,
            enable_rim: 1,
            dynamic_blend_factor: 0.0,
            
            frame_counter: 0,
            last_frame_time:     Instant::now(),
            total_time: 0.0, // 🟢 HIER REINSCHREIBEN!
        }
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == 0 || new_height == 0 {
            return;
        }
        
        self.config.width = new_width;
        self.config.height = new_height;
        self.surface.configure(&self.device, &self.config);
        
        // Update resolution uniform
        self.resolution_uniform.width = new_width;
        self.resolution_uniform.height = new_height;
        self.queue.write_buffer(
            &self.resolution_buffer,
            0,
            bytemuck::cast_slice(&[self.resolution_uniform]),
        );
        
        // Reallocate CubeCL output buffer for new size
        let total_pixels = (new_width * new_height) as usize;
        //let byte_size = total_pixels * std::mem::size_of::<u32>();
        let byte_size = total_pixels * 3 * std::mem::size_of::<f32>(); 
        self.output_handle = self.client.empty(byte_size);
    }

    fn update_compute(&mut self) {
        // 🟢 KORREKTUR: Wir nutzen self.total_time, die oben in render()      flüssig berechnet wurde!
        // 1. Echte Delta-Time berechnen (zu Beginn von update_compute)
        let now = Instant::now();
        let delta_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now; // Zeitstempel für das nächste Frame aktualisieren

        // 2. Gesamtzeit flüssig aufaddieren
        self.total_time += delta_time * 1.0;
        let time = self.total_time; 
        // Camera and rendering parameters (from original code)
        //let time = self.last_frame_time.elapsed().as_secs_f32();
        let width = self.config.width;
        let height = self.config.height;
        let total_pixels = (width * height) as usize;
        
       // 🟢 KORREKTUR: Werte aus self auslesen, statt sie fest auf 0.0 / -5.0 zu setzen!
    let cam_x = self.cam_x;
    let cam_y = self.cam_y;
    let cam_z = self.cam_z;
    let cam_yaw = self.cam_yaw;
    let cam_pitch = self.cam_pitch;
        
        /// 🟢 KORREKTUR: Jetzt dynamisch aus der Struktur lesen!
    let light_intensity = self.light_intensity;
    let ambient_strength = self.ambient_strength;
    let enable_ao_mode = self.enable_ao_mode;
    let current_shadow_mode = self.current_shadow_mode;
    let enable_key = self.enable_key;
    let enable_fill = self.enable_fill;
    let enable_rim = self.enable_rim;
    let dynamic_blend_factor = self.dynamic_blend_factor;

        // Launch the SDF kernel
        let cube_count = cubecl::CubeCount::Static(
            (width + 15) / 16, 
            (height + 3) / 4, 
            1
        );
        let cube_dim = cubecl::CubeDim::new_3d(16, 4, 1);

        // We need to create the meta, slots, and materials handles
        // For now, let's create minimal buffers for them
        let meta_handle = self.client.empty(std::mem::size_of::<f32>()); // 1 f32
        let slots_handle = self.client.empty(800 * std::mem::size_of::<f32>()); // TOTAL_SLOT_FLOATS
        let materials_handle = self.client.empty(100 * std::mem::size_of::<f32>()); // TOTAL_MATERIAL_FLOATS

        // Create TensorArgs from raw handles
        let output_arg = unsafe {
            TensorArg::from_raw_parts(
                self.output_handle.clone(),
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
                &self.client,
                cube_count,
                cube_dim,
                output_arg,
                meta_arg,
                slots_arg,
                materials_arg,
                time, 
                width, 
                height, 
                current_shadow_mode,
                cam_x, 
                cam_y, 
                cam_z, 
                dynamic_blend_factor, 
                enable_ao_mode,
                cam_yaw, 
                cam_pitch, 
                light_intensity, 
                ambient_strength,
                enable_key, 
                enable_fill, 
                enable_rim,
            );
        }
        
        // Synchronize to ensure compute is done before render
        let _ = self.client.sync();
    }

fn render(&mut self) {
//  Berechne die echte Delta-Time seit dem letzten Frame
   // let now = Instant::now();
   // let delta_time = now.duration_since(self.last_frame_time).as_secs_f32();
   // self.last_frame_time = now; // Zeitstempel sofort für das nächste Frame aktualisieren

    //  Erhöhe die Gesamtzeit flüssig um die vergangene Zeit
    // (1.0 ist die normale Geschwindigkeit. Erhöhe es, falls die Animation schneller sein soll)
   // self.total_time += delta_time * 1.0;

    // 1. Hole den aktuellen Status der Surface-Textur (WGPU v30 Syntax)
    let current_surface = self.surface.get_current_texture();
    
    // Extrahiere die eigentliche SurfaceTexture
    let output = match current_surface {
        wgpu::CurrentSurfaceTexture::Success(texture) => texture,
        wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
        other => {
            eprintln!("Failed to acquire surface texture: {:?}", other);
            return;
        }
    };
    
    // View für den RenderPass erstellen
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

    // 2. Extrahiere den nativen wgpu::Buffer aus dem CubeCL Handle
   // Ergänzen Sie die Turbofish-Syntax mit dem WgpuServer-Typen:
let managed_resource = self.client
    .get_resource::<cubecl_wgpu::WgpuServer<cubecl_wgpu::AutoCompiler>>(self.output_handle.clone())
    .unwrap();

let wgpu_resource: &cubecl_wgpu::WgpuResource = managed_resource.resource();
let src_wgpu_buffer = &wgpu_resource.buffer;

    let buffer_offset = wgpu_resource.offset;

    // 3. Erstelle die Bind-Group mit dem ECHTEN CubeCL Ausgabe-Buffer
    self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Screen Bind Group"),
        layout: &self.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: src_wgpu_buffer,
                    offset: buffer_offset,
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
    });

    // 4. Command Encoder und Render Pass instanziieren
    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Screen Render Pass"),
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
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Fullscreen Dreieck zeichnen
    }

    // 5. Befehle abschicken
    self.queue.submit(std::iter::once(encoder.finish()));
    
    // 🟢 JETZT KORREKT FÜR WGPU v30: Wir übergeben die Textur der Queue zur Präsentation!
    self.queue.present(output);
    
    self.frame_counter += 1;
    //self.last_frame_time = Instant::now(); // ❌ DIESE ZEILE BITTE LÖSCHEN! implemented delta time at top.
}

}

impl ApplicationHandler for App {
    // Wird vom OS aufgerufen, sobald die Grafik-Infrastruktur bereit ist
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app_state.is_none() {
            // Fenster-Attribute modern in v0.30 definieren
            let window_attributes = Window::default_attributes()
                .with_title("SDF Demo - Zero Copy PBR Raymarching")
                .with_inner_size(winit::dpi::LogicalSize::new(960, 540));
            
            // Winit 0.30 erzeugt das Fenster über das event_loop-Target
            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            // 🟢 WICHTIG: Maus im Fenster einsperren und verstecken
        // Das deaktiviert die OS-Palm-Rejection und aktiviert Raw Input!
//let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
        //let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None); 
        window.set_cursor_visible(false);
        // 🟢 NEU: Winit zwingen, Raw Input direkt vom Linux-Kernel zu lesen!
        // Das überspringt die Drosselung der Desktop-Umgebung komplett.
        event_loop.listen_device_events(winit::event_loop::DeviceEvents::Always);


            // Initialisiere deine bestehende Application::new via pollster
            let app = pollster::block_on(Application::new(window));
            self.app_state = Some(app);
        }
    }
      

    // Verarbeitet alle Ereignisse des Fensters
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Falls die App noch nicht initialisiert ist (vor resumed), ignorieren
        let app = match &mut self.app_state {
            Some(a) => a,
            None => return,
        };

       // Innerhalb von impl ApplicationHandler for App -> fn window_event:

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            // Tastatur-Events weiterleiten
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                self.input_manager.handle_key_event(&key_event, event_loop, app);
            }
         //   // 🟢 NEU & LÖSUNG FÜR WAYLAND: Mausbewegung synchron im Fenster verarbeiten
          //  WindowEvent::CursorMoved { position, .. } => {
           //     self.input_manager.handle_window_mouse_move(position.x, position.y, app);
          //  }

                         // 🟢 DER RENDER-LOOP: Verarbeitet Tastatur + Linux-Maus perfekt gleichzeitig
            // Der Render- und Rechen-Loop
            WindowEvent::RedrawRequested => {
                self.input_manager.update_camera_movement(app);
                app.update_compute();
                app.render();
                app.window.request_redraw();
            }

            WindowEvent::Resized(size) => {
                app.resize(size.width, size.height);
            }

            _ => {}
        }

    }
}

// Die neue, extrem aufgeräumte main-Funktion
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new().unwrap();
    
    // Instanziere unseren Handler
    let mut app = App {
        app_state: None,
        input_manager: InputManager::new(5.0f32, 0.002f32), // 🟢 0.002 als Mausempfindlichkeit
    };
    
    // run_app startet den deklarativen Lebenszyklus
    event_loop.run_app(&mut app)?;

    Ok(())
}