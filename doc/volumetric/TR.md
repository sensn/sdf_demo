
Technical Report: High-Performance Volumetric Raymarching via CubeCL

Architecture & Implementation Specification

1. Executive Summary

This report outlines the transformation of a hardware-accelerated Surface Raymarching (Sphere Tracing) engine into an un-allocated, zero-copy Volumetric Path/Ray Tracer utilizing CubeCL.
Traditional volumetric rendering pipelines heavily rely on offline 3D texture synthesis (Texture3D) or dense Voxel Grids, introducing extreme memory bandwidth constraints and cache-miss overheads on modern GPU architectures. This implementation strictly operates under a Zero-Copy, Zero-Cost philosophy: all spatial data generation, phase scattering functions, temporal dithering, and internal radiant emissions are computed procedurally inside the GPU’s Arithmetic Logic Unit (ALU) registers.
By evaluating mathematical primitives (such as Infinite Gyroid Fields) directly at the shading stage, the engine achieves scalable performance across multi-backend runtimes (WebGPU, Vulkan, CUDA, Metal) while maintaining strict host-device type safety through Rust’s type system.

2. Theoretical Background & Mathematical Foundations


2.1 From Distance Fields to Density Fields

Surface raymarching solves for a boundary interface where the Signed Distance Field \(f(\mathbf{p}) = 0\). In contrast, volumetric raymarching treats space as a continuous participating medium. We map a Signed Distance Field \(d = f(\mathbf{p})\) to a localized density field \(\rho(\mathbf{p})\) using a sharp clamped-linear inversion kernel:
\(\rho (\mathbf{p})=\text{clip}\left(-d\cdot \sigma _{\text{scale}},\,0.0,\,1.0\right)\)
Where negative distances (interior to the mathematical solid) yield high density, while positive distances yield rapid decay to empty space (ρ = 0.0).

2.2 Volumetric Transport & Beer-Lambert Law

As light travels along a view ray \(\mathbf{R}(t) = \mathbf{O} + t\mathbf{D}\), its radiant intensity attenuates due to absorption and out-scattering. The fraction of light reaching the camera is governed by Transmittance (T), defined continuously via the Beer-Lambert Law:
\(T(t)=\exp \left(-\int _{0}^{t}\rho (\mathbf{R}(s))\cdot \kappa _{\text{fog}}\,ds\right)\)
In our numerical Riemann-sum approximation along discrete steps Δ x, transmittance is tracked recursively:
\(T_{i+1}=T_{i}\cdot (1.0-\alpha _{i}),\quad \text{where\ }\alpha _{i}=\rho (\mathbf{p}_{i})\cdot \Delta x\cdot \kappa _{\text{fog}}\)

2.3 Single In-Scattering & Self-Shadowing (Beer-Lambert Shadowing)

To calculate internal shadows (Self-Shadowing), each step along the view ray casts a secondary, lower-resolution ray \(\mathbf{R}_{\text{light}}(s)\) directly toward the light source direction \(\mathbf{L}\):
\(T_{\text{light}}(\mathbf{p}_{i})=\exp \left(-\sum _{j=0}^{M}\rho (\mathbf{p}_{i}+j\cdot \Delta s\cdot \mathbf{L})\cdot \Delta s\cdot \kappa _{\text{absorb}}\right)\)

2.4 Phase Functions: Henyey-Greenstein Anisotropy

To simulate micro-physical scattering behaviors (like forward-scattering within dense vapor or smoke clouds), we substitute uniform (isotropic) scattering with the Henyey-Greenstein Phase Function. This models the probability of light scattering at an angle θ relative to the incoming light vector:
\(P(\cos \theta ,g)=\frac{1}{4\pi }\cdot \frac{1-g^{2}}{\left(1+g^{2}-2g\cos \theta \right)^{1.5}}\)
Where \(\cos\theta = \mathbf{D} \cdot \mathbf{L}\), and g ∈ (-1.0, 1.0) represents the asymmetry factor (g > 0 defines intense forward-glow under backlight conditions).
          [Light Source (L)]
                 \
                  \  (Shadow Ray Calculation)
                   \
 [Camera (O)] ----> (p_i) ----> [Forward View Ray (D)]
                     \
                      `--- θ (Angle for Henyey-Greenstein Scattering)

2.5 High-Frequency Variance Elimination: Temporal Dithering

Discrete raymarching exhibits concentric banding artifacts (slicing) if the step count is low. To eliminate this without scaling execution costs, we apply a high-frequency Temporal Jittering wrapper. The initial ray origin is shifted forward along the view vector by a pseudo-random scalar fraction of the step size:
\(\mathbf{O}_{\text{jittered}}=\mathbf{O}+\mathbf{D}\cdot (\Delta x\cdot R(\text{pixel}_{x,y},t_{\text{time}}))\)
This turns uniform banding artifacts into high-frequency blue/white-noise patterns that easily dissolve via hardware temporal anti-aliasing (TAA) or native sensory persistence at high framerates.

3. Detailed Conceptual Step-by-Step Implementation

The comprehensive implementation is structured into modular, strictly compiled GPU devices routines using the #[cube] and #[cube(launch)] meta-programming macro pipeline.

Step 1: The Zero-Copy 3D Noise and Core Density Field Generator

This module computes high-frequency fractional Brownian motion (FBM) entirely in registers via trilinear Hermite interpolation, avoiding all hardware texture unit bottlenecks.
rust
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

// Primitives and parameter structures inherited from host architecture
use crate::gpu::{CameraSettings, EnvSettings, ArchParams, FoldParams, RenderSettings, Vec3};

#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct VolumeSample {
    pub density: f32,
    pub emission: Vec3,
}

/// Computes a register-isolated 3D spatial hash vector.
#[cube]
pub fn hash_3d(p: Vec3) -> f32 {
    let dot_product = p.x * f32::new(127.1) 
                    + p.y * f32::new(311.7) 
                    + p.z * f32::new(74.7);
    let sine_val = dot_product.sin();
    sine_val * f32::new(43758.5453) - (sine_val * f32::new(43758.5453)).floor()
}

/// Generates a smooth trilinear value noise layer.
#[cube]
pub fn noise_3d(p: Vec3) -> f32 {
    let i = Vec3::new(p.x.floor(), p.y.floor(), p.z.floor());
    let f = Vec3::new(p.x - p.x.floor(), p.y - p.y.floor(), p.z - p.z.floor());

    // Cubic Hermite Smoothstep curves
    let u = Vec3::new(
        f.x * f.x * (f32::new(3.0) - f32::new(2.0) * f.x),
        f.y * f.y * (f32::new(3.0) - f32::new(2.0) * f.y),
        f.z * f.z * (f32::new(3.0) - f32::new(2.0) * f.z)
    );

    let i000 = hash_3d(i);
    let i100 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(0.0), f32::new(0.0))));
    let i010 = hash_3d(i.add(Vec3::new(f32::new(0.0), f32::new(1.0), f32::new(0.0))));
    let i110 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(1.0), f32::new(0.0))));
    let i001 = hash_3d(i.add(Vec3::new(f32::new(0.0), f32::new(0.0), f32::new(1.0))));
    let i101 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(0.0), f32::new(1.0))));
    let i011 = hash_3d(i.add(Vec3::new(f32::new(0.0), f32::new(1.0), f32::new(1.0))));
    let i111 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(1.0), f32::new(1.0))));

    let x00 = i000 + u.x * (i100 - i000);
    let x10 = i010 + u.x * (i110 - i010);
    let x01 = i001 + u.x * (i101 - i001);
    let x11 = i011 + u.x * (i111 - i011);

    let y0 = x00 + u.y * (x10 - x00);
    let y1 = x01 + u.y * (x11 - x01);

    y0 + u.z * (y1 - y0)
}

/// Evaluates core implicit gyroid equations modulated with multi-octave FBM fluid noise.
#[cube]
pub fn get_volumetric_data(p: Vec3, time: f32, size: f32) -> VolumeSample {
    let base_gyroid = evaluate_dynamic_gyroid(p, time, size);
    
    // Extrapolate structural wind-velocity vector field over runtime
    let wind = Vec3::new(time * f32::new(0.2), time * f32::new(0.5), time * f32::new(0.1));
    let noise_pos = p.add(wind);

    // Multi-octave fractional Brownian motion structure
    let noise_octave1 = noise_3d(noise_pos.scale(f32::new(3.0)));
    let noise_octave2 = noise_3d(noise_pos.scale(f32::new(6.12)));
    let combined_noise = noise_octave1 * f32::new(0.7) + noise_octave2 * f32::new(0.3);

    // Deform perfect geometry boundary fields into turbulent vapor clouds
    let modulated_sdf = base_gyroid + combined_noise * f32::new(0.15);
    
    let density_scale = f32::new(6.0);
    let density = (-modulated_sdf * density_scale).min(f32::new(1.0)).max(f32::new(0.0));

    // Internal combustion thermal evaluation (Emissions core)
    let heat = (-base_gyroid * f32::new(4.0) + combined_noise * f32::new(2.0))
        .min(f32::new(1.0))
        .max(f32::new(0.0));

    // High Dynamic Range (HDR) radiant thermal emission profile (Blackbody simulation)
    let glow_color = Vec3::new(
        heat * f32::new(3.5),                     // Over-saturated high-intensity radiant energy
        heat * heat * f32::new(1.4),              // Mid-range spectral green contribution 
        heat * heat * heat * heat * f32::new(0.3) // Core localized thermal blue spectrum
    );

    VolumeSample { density, emission: glow_color }
}
Verwende Code mit Vorsicht.

Step 2: The Phase Scattering and Shadowing Core

This module handles anisotropic forward-scattering via Henyey-Greenstein equations alongside Beer-Lambert single-scattering shadow-ray calculation.
rust
/// Formulates the probability distribution curve for forward/backward particle anisotropy.
#[cube]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let pi = f32::new(3.14159265);
    let numerator = f32::new(1.0) - g2;
    let denom_inner = f32::new(1.0) + g2 - f32::new(2.0) * g * cos_theta;
    let denominator = f32::new(4.0) * pi * denom_inner * denom_inner.sqrt();
    numerator / denominator
}

/// Secondary raymarching operation analyzing optical depth towards light nodes.
#[cube]
pub fn calculate_volume_shadow(
    sample_pos: Vec3, 
    light_dir: Vec3, 
    time: f32, 
    size: f32,
    shadow_steps: u32,
    shadow_step_size: f32
) -> f32 {
    let mut accumulated_shadow_density = f32::new(0.0);
    let mut current_pos = sample_pos;

    for _s in 0..shadow_steps {
        current_pos = current_pos.add(light_dir.scale(shadow_step_size));
        let sample = get_volumetric_data(current_pos, time, size);
        accumulated_shadow_density = accumulated_shadow_density + sample.density;
    }

    let absorption_coefficient = f32::new(0.8);
    f32::exp(-accumulated_shadow_density * shadow_step_size * absorption_coefficient)
}
Verwende Code mit Vorsicht.

Step 3: High-Frequency Spatial Dithering Operator

A fast, register-isolated sinus-fractional pseudo-random number generator designed to execute on unified shader core ALUs.
rust
/// Evaluates pixel-locked spatial coordinates mixed with global frame-time for high-frequency noise.
#[cube]
pub fn noise_2d_time(x: u32, y: u32, time: f32) -> f32 {
    let fx = f32::cast(x);
    let fy = f32::cast(y);
    let dot_product = fx * f32::new(12.9898) + fy * f32::new(78.233) + time * f32::new(43758.5453);
    let sine_val = dot_product.sin();
    sine_val * f32::new(43758.5453) - (sine_val * f32::new(43758.5453)).floor()
}
Verwende Code mit Vorsicht.

Step 4: The Volumetric Integration Core Loop with Early Termination

This function brings the entire pipeline together. It steps along the camera view vector, updates analytical transmittance tracking, processes phase scattering profiles, injects emmersive light sources, and implements early-loop branching optimizations.
rust
#[cube]
pub fn volumetric_raymarch_integrated(
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_distance: f32,
    steps: u32,
    pixel_x: u32,
    pixel_y: u32,
    settings: RenderSettings,
    env: EnvSettings,
) -> Vec3 {
    let step_size = max_distance / f32::cast(steps);
    
    // Initialize high-frequency spatial temporal dithering offset
    let jitter = noise_2d_time(pixel_x, pixel_y, settings.time);
    let mut current_pos = ray_origin.add(ray_dir.scale(step_size * jitter));
    
    let mut accumulated_color = Vec3::new(f32::new(0.0), f32::new(0.0), f32::new(0.0));
    let mut transmittance = f32::new(1.0);

    let light_direction = env.key_light_pos.position_to_direction_vector(); 
    let light_color = Vec3::new(env.key_color.x, env.key_color.y, env.key_color.z);

    // Precalculate Anisotropic Angular Scattering coefficient once per view ray
    let cos_theta = ray_dir.dot(light_direction);
    let anisotropy_g = f32::new(0.65); // Positive value for stylized front-lit halo blooming
    let phase_value = henyey_greenstein(cos_theta, anisotropy_g);

    for _i in 0..steps {
        let sample = get_volumetric_data(current_pos, settings.time, f32::new(1.5));

        if sample.density > f32::new(0.0) {
            // Compute shadow depth via secondary shadow steps
            let shadow_steps = u32::new(4); 
            let shadow_step_size = step_size * f32::new(1.5);
            let light_transmission = calculate_volume_shadow(
                current_pos, 
                light_direction, 
                settings.time, 
                f32::new(1.5),
                shadow_steps,
                shadow_step_size
            );

            // Compute global single in-scattering phase mechanics
            let scattered_light_intensity = env.light_intensity * phase_value;
            let external_light = light_color.scale(light_transmission * scattered_light_intensity);
            
            // Unify external in-scattering with interior self-illumination (Emission)
            let total_light = external_light.add(sample.emission);
            
            // Continuous opacity numerical integration
            let opacity = sample.density * step_size * env.fog_density;
            let factored_color = total_light.scale(opacity * transmittance);
            
            accumulated_color = accumulated_color.add(factored_color);
            transmittance = transmittance * (f32::new(1.0) - opacity);

            // 🟢 CRITICAL ARCHITECTURAL OPTIMIZATION: Early Ray Termination
            // Once transmittance drops below the visibility threshold, break execution
            if transmittance < f32::new(0.001) {
                transmittance = f32::new(0.0);
                break;
            }
        }

        current_pos = current_pos.add(ray_dir.scale(step_size));
    }

    let background = Vec3::new(env.bg_color.x, env.bg_color.y, env.bg_color.z);
    accumulated_color.add(background.scale(transmittance))
}
Verwende Code mit Vorsicht.

Step 5: Hardware-Linked Primary Compute Launch Kernel

This top-level device routine links hardware invocation IDs to physical screen fragments, sets up the camera vectors, evaluates the integration stack, and commits the output directly into linear memory storage.
rust
#[cube(launch)]
pub fn render_volumetric_kernel(
    output_image: &mut Tensor<f32>,
    camera: CameraSettings,
    env: EnvSettings,
    arch: ArchParams,
    fold: FoldParams,
    settings: RenderSettings,
) {
    // Read hardware-allocated global invocation identifiers directly via register indices
    let pixel_x = LAUNCH_ID_X;
    let pixel_y = LAUNCH_ID_Y;

    // Out-of-bounds safety fence for non-divisible subgroup allocation boundaries
    if pixel_x >= settings.width || pixel_y >= settings.height {
        return;
    }

    let fx = f32::cast(pixel_x);
    let fy = f32::cast(pixel_y);
    let width_f = f32::cast(settings.width);
    let height_f = f32::cast(settings.height);

    // Transform raw screenspace positions to zentrierte, projection-corrected UV spaces
    let uv_x = (fx / width_f) * f32::new(2.0) - f32::new(1.0);
    let uv_y = (fy / height_f) * f32::new(2.0) - f32::new(1.0);
    let aspect = width_f / height_f;
    let uv_x_adjusted = uv_x * aspect;

    let ray_origin = Vec3::new(camera.position.x, camera.position.y, camera.position.z);
    let ray_dir = Vec3::new(uv_x_adjusted, uv_y, f32::new(-1.0)).normalize();

    let max_dist = f32::new(25.0);
    let steps = u32::new(32); 
    
    // Evaluate the composite volumetric computation pipeline
    let final_hdr_color = volumetric_raymarch_integrated(
        ray_origin,
        ray_dir,
        max_dist,
        steps,
        pixel_x,
        pixel_y,
        settings,
        env,
    );

    // Memory Address Offset serialization: Row-major linear byte assignment
    let pixel_index = (pixel_y * settings.width + pixel_x) * u32::new(4);

    output_image[pixel_index + u32::new(0)] = final_hdr_color.x; 
    output_image[pixel_index + u32::new(1)] = final_hdr_color.y; 
    output_image[pixel_index + u32::new(2)] = final_hdr_color.b; 
    output_image[pixel_index + u32::new(3)] = f32::new(1.0);  
}
Verwende Code mit Vorsicht.

4. Architectural Analysis & Performance Optimization Profiling


4.1 Zero-Copy Execution Strategy

Traditional GPGPU frameworks read volumetric noise by binding a Texture3D sampler object. This stalls processing pipelines due to texture cache misses and high VRAM-to-register latency. By keeping noise synthesis entirely arithmetic inside individual core execution stacks, memory bandwidth drops to absolute zero for intermediate steps.

4.2 Early Ray Termination (Dynamic Warp/Wavefront Efficiency)

The dynamic branch if transmittance < 0.001 { break; } provides substantial speedups. In traditional raymarching pipelines, threads inside a warp must execute until the longest loop finishes. However, by short-circuiting heavy math execution immediately after a volume sample hits maximum opacity, memory store pipelines are unburdened, and execution resources are quickly freed for neighboring threads.

4.3 ALU Arithmetic Instruction Distribution

By moving structural parameters (such as EnvSettings, RenderSettings) directly into the specialized info uniform padding structures (GpuVec3 with explicit _pad bounds), host transfers strictly fulfill 16-byte alignment specifications, allowing modern architectures to populate local thread cache hierarchies seamlessly without unaligned memory access penalties.
Would you like to analyze how to implement a High-Dynamic Range (HDR) Tone-mapping Stage (such as ACES or Reinhard equations) within the compute kernel to cleanly map the emmissive over-saturated pixel colors back to the standard LDR screen color range?
