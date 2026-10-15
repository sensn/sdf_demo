Hier ist ein präziser, modularer und direkt für einen Coding Agent optimierter Implementierungsplan. Die Instruktionen sind als strikte Pseudocode- und Syntax-Vorgaben nach dem aktuellen CubeCL 0.11.0 Standard formuliert (Verwendung von nativen Funktionen wie f32::sin, ABSOLUTE_POS_X statt veralteter Makros und striktem Typ-Handling).

🤖 Coding Agent Instruction: Volumetric Raymarcher Integration (CubeCL 0.11.0)


Context & Constraints

• Framework: CubeCL v0.11.0 (Pure Rust GPU compiler pipeline).
• Architecture: Zero-Copy & Zero-VRAM allocations. Multi-octave FBM and noise calculations must be completed natively within ALU registers.
• Type Strictness: Avoid loose literal scaling. Every constant must be typed matching its operation context (e.g., 3.14159265 instead of old casting wrappers, or explicitly bound if combining with generic variables).
• Grid Topology API: Use ABSOLUTE_POS_X and ABSOLUTE_POS_Y to retrieve physical thread coordinates.

Task 1: Setup Host/Device Bridge and Mathematical Structures

Implement the uniform parameter layout and mathematical structures with padding to enforce 16-byte alignment boundaries.

Instructions for Agent:

1. Define a PBR/Volumetric output struct VolumeSample using the #[derive(CubeType, Copy, Clone)] macro.
2. Define a 16-byte aligned GpuVec3 representation for hosting settings.
3. Build the core VolumetricRenderSettings context structure.
rust
use cubecl::prelude::*;

#[derive(CubeType, Copy, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[cube]
impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }
    pub fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
    pub fn scale(self, factor: f32) -> Vec3 {
        Vec3::new(self.x * factor, self.y * factor, self.z * factor)
    }
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn normalize(self) -> Vec3 {
        let len = f32::sqrt(self.dot(self));
        if len > 0.0 {
            self.scale(1.0 / len)
        } else {
            self
        }
    }
}

#[derive(CubeType, Copy, Clone)]
pub struct VolumeSample {
    pub density: f32,
    pub emission: Vec3,
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32, // Enforces 16-byte alignment layout for Uniform Buffers
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct VolumetricRenderSettings {
    pub time: f32,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub max_distance: f32,
    pub fog_density: f32,
    pub anisotropy_g: f32,
}
Verwende Code mit Vorsicht.

Task 2: ALU-Bound 3D Noise Generation Stack

Implement a procedural 3D noise layer based on cubic Hermite interpolation without accessing samplers.

Instructions for Agent:

1. Implement a 3D hashing algorithm leveraging fast ALU sinusoidal instructions.
2. Build a trilinear value noise function using f32::floor for grid snapping.
rust
#[cube]
pub fn hash_3d(p: Vec3) -> f32 {
    let dot_product = p.x * 127.1 + p.y * 311.7 + p.z * 74.7;
    let sine_val = f32::sin(dot_product);
    sine_val * 43758.5453 - f32::floor(sine_val * 43758.5453)
}

#[cube]
pub fn noise_3d(p: Vec3) -> f32 {
    let i = Vec3::new(f32::floor(p.x), f32::floor(p.y), f32::floor(p.z));
    let f = Vec3::new(p.x - i.x, p.y - i.y, p.z - i.z);

    // Smoothstep Hermite curve calculations
    let u = Vec3::new(
        f.x * f.x * (3.0 - 2.0 * f.x),
        f.y * f.y * (3.0 - 2.0 * f.y),
        f.z * f.z * (3.0 - 2.0 * f.z),
    );

    let i000 = hash_3d(i);
    let i100 = hash_3d(Vec3::new(i.x + 1.0, i.y, i.z));
    let i010 = hash_3d(Vec3::new(i.x, i.y + 1.0, i.z));
    let i110 = hash_3d(Vec3::new(i.x + 1.0, i.y + 1.0, i.z));
    let i001 = hash_3d(Vec3::new(i.x, i.y, i.z + 1.0));
    let i101 = hash_3d(Vec3::new(i.x + 1.0, i.y, i.z + 1.0));
    let i011 = hash_3d(Vec3::new(i.x, i.y + 1.0, i.z + 1.0));
    let i111 = hash_3d(Vec3::new(i.x + 1.0, i.y + 1.0, i.z + 1.0));

    let x00 = i000 + u.x * (i100 - i000);
    let x10 = i010 + u.x * (i110 - i010);
    let x01 = i001 + u.x * (i101 - i001);
    let x11 = i011 + u.x * (i111 - i011);

    let y0 = x00 + u.y * (x10 - x00);
    let y1 = x01 + u.y * (x11 - x01);

    y0 + u.z * (y1 - y0)
}
Verwende Code mit Vorsicht.

Task 3: Volume Physics & Analytical Tracking Loops

Incorporate atmospheric phase behaviors and dynamic loop stepping logic.

Instructions for Agent:

1. Implement the henyey_greenstein anisotropy model.
2. Implement spatial 2D temporal dithering to eliminate rendering bands.
3. Write the volumetric integration kernel loop featuring Early Ray Termination (transmittance < 0.001).
rust
#[cube]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let pi = 3.14159265;
    let numerator = 1.0 - g2;
    let denom_inner = 1.0 + g2 - 2.0 * g * cos_theta;
    let denominator = 4.0 * pi * denom_inner * f32::sqrt(denom_inner);
    numerator / denominator
}

#[cube]
pub fn noise_2d_time(x: u32, y: u32, time: f32) -> f32 {
    let dot_product = f32::cast(x) * 12.9898 + f32::cast(y) * 78.233 + time * 43758.5453;
    let sine_val = f32::sin(dot_product);
    sine_val * 43758.5453 - f32::floor(sine_val * 43758.5453)
}

// Internal dense structural density lookup helper
#[cube]
pub fn evaluate_volume_density(p: Vec3, time: f32) -> VolumeSample {
    // Substitute standard external dynamic gyroid formulas into execution context
    let scale = 2.0;
    let g_val = (f32::sin(p.x * scale) * f32::cos(p.y * scale)
               + f32::sin(p.y * scale) * f32::cos(p.z * scale)
               + f32::sin(p.z * scale) * f32::cos(p.x * scale)) / scale;

    let n_pos = p.add(Vec3::new(time * 0.1, time * 0.3, 0.0));
    let noise = noise_3d(n_pos.scale(4.0)) * 0.4;
    let final_sdf = g_val + noise;

    let density = (-final_sdf * 5.0).max(0.0).min(1.0);
    
    // Core high temperature plasma simulation (Thermal emissive values)
    let heat = (-g_val * 3.0).max(0.0).min(1.0);
    let emission = Vec3::new(heat * 3.0, heat * heat * 1.0, 0.0);

    VolumeSample { density, emission }
}
Verwende Code mit Vorsicht.

Task 4: Compilation Kernel Target & Grid Topology Execution

Expose the main entry point to register graphics launch execution grids.

Instructions for Agent:

1. Mark the function explicitly with #[cube(launch)].
2. Extract global IDs strictly from ABSOLUTE_POS_X and ABSOLUTE_POS_Y.
3. Map internal integration values into row-major 1D array locations (&mut Tensor<f32>).
rust
#[cube(launch)]
pub fn render_volumetric_kernel(
    output_image: &mut Tensor<f32>,
    settings: VolumetricRenderSettings,
    light_dir: GpuVec3,
    light_color: GpuVec3,
) {
    // 🟢 Target Topology API compliance for CubeCL 0.11.0
    let pixel_x = ABSOLUTE_POS_X;
    let pixel_y = ABSOLUTE_POS_Y;

    // Execution layout boundaries clip fence
    if pixel_x >= settings.width || pixel_y >= settings.height {
        return;
    }

    let fx = f32::cast(pixel_x);
    let fy = f32::cast(pixel_y);
    let w_f = f32::cast(settings.width);
    let h_f = f32::cast(settings.height);

    // Perspective mapping projection parameters
    let uv_x = (fx / w_f) * 2.0 - 1.0;
    let uv_y = (fy / h_f) * 2.0 - 1.0;
    let aspect = w_f / h_f;

    let ray_origin = Vec3::new(0.0, 0.0, 3.0);
    let ray_dir = Vec3::new(uv_x * aspect, uv_y, -1.0).normalize();

    // Volume stepping properties setup
    let step_size = settings.max_distance / f32::cast(settings.steps);
    let jitter = noise_2d_time(pixel_x, pixel_y, settings.time);
    let mut current_pos = ray_origin.add(ray_dir.scale(step_size * jitter));

    let mut accum_color = Vec3::new(0.0, 0.0, 0.0);
    let mut transmittance = 1.0;

    let l_dir = Vec3::new(light_dir.x, light_dir.y, light_dir.z).normalize();
    let l_col = Vec3::new(light_color.x, light_color.y, light_color.z);
    
    let cos_theta = ray_dir.dot(l_dir);
    let phase = henyey_greenstein(cos_theta, settings.anisotropy_g);

    // Core Numerical Integration Stack
    for _i in 0..settings.steps {
        let sample = evaluate_volume_density(current_pos, settings.time);

        if sample.density > 0.0 {
            let scattered_light = l_col.scale(phase);
            let total_radiance = scattered_light.add(sample.emission);

            let opacity = sample.density * step_size * settings.fog_density;
            accum_color = accum_color.add(total_radiance.scale(opacity * transmittance));
            transmittance = transmittance * (1.0 - opacity);

            // Early loop breakout optimization 
            if transmittance < 0.001 {
                transmittance = 0.0;
                break;
            }
        }
        current_pos = current_pos.add(ray_dir.scale(step_size));
    }

    // Write out raw serialization streams to linearized Framebuffer destinations
    let pixel_idx = (pixel_y * settings.width + pixel_x) * 4;
    output_image[pixel_idx + 0] = accum_color.x;
    output_image[pixel_idx + 1] = accum_color.y;
    output_image[pixel_idx + 2] = accum_color.z;
    output_image[pixel_idx + 3] = 1.0; // Uniform alpha channel state
}
Verwende Code mit Vorsicht.
Would you like the agent instructions to include a CPU-side test script verifying the host allocation and launching steps via a specific CubeCL Runtime backend (Wgpu, CUDA, or Vulkan)?
