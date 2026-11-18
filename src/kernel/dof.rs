//! `kernel/dof.rs` — Depth of Field Pre-Pass (cubek-convolution Vorbereitung).
//!
//! 🟢 Feature: Echte Schärfentiefe über den Tiefenkanal des Raymarchers.
//!
//! Pipeline (doc/feature/DOF_CUBEK_CONVOLUTION.md):
//!   1. `raymarch_sdf_kernel` schreibt RGBA — A = `hit_dist`.
//!   2. `compute_coc_and_mask_kernel` (hier) berechnet pro Pixel den
//!      Circle of Confusion (CoC) und schreibt das CoC-gewichtete RGB
//!      in **CHW**-Layout — der Layout-Erwartung von cubek-convolution.
//!   3. `generate_disk_kernel` (hier) füllt den Bokeh-Disk-Kernel [1,1,K,K].
//!   4. `cubek_convolution::launch_ref` faltet per im2col + Tensor Cores.
//!
//! Formel CoC = |(TIEFE − FOKUS_DISTANZ) / TIEFE| × BLENDEN_ÖFFNUNG.
//! Pixel auf der Fokusebene bleiben scharf; Skybox wird maximal unscharf.
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;
use cubecl::ir::FastMath;

use super::types::Vec3;

/// 🟢 DoF: CoC-Vorbereitungskernel.
///
/// Liest den RGBA-Output des Raymarchers (interleaved, Stride 4) und schreibt
/// das CoC-gewichtete RGB ins **CHW**-Layout `[3, H, W]`, das die Faltung
/// erwartet (Kanal-planes, nicht interleaved).
///
/// * `depth <= SKYBOX_DEPTH` → maximales Bokeh (CoC = 1.0).
/// * `CoC = min(|(depth − focal) / depth| × aperture, 1.0)`.
#[cube(launch, fast_math = FastMath::all().difference(FastMath::AllowContraction.into()))]
pub fn compute_coc_and_mask_kernel<F: Float + LaunchArg>(
    rgba_input: &Tensor<F>,     // [H*W*4] RGBA interleaved
    rgb_output: &mut Tensor<F>, // [3, H, W] CHW planar
    width: u32,
    height: u32,
    focal_distance: F,
    aperture: F,
    max_depth: F,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x < width && y < height {
        let xu = usize::cast_from(x);
        let yu = usize::cast_from(y);
        let wu = usize::cast_from(width);

        // Einstieg in den RGBA-Interleaved-Output.
        let idx_in = (yu * wu + xu) * usize::new(4);

        let r = rgba_input[idx_in];
        let g = rgba_input[idx_in + usize::new(1)];
        let b = rgba_input[idx_in + usize::new(2)];
        let depth = rgba_input[idx_in + usize::new(3)];

        // Circle of Confusion: Abweichung von der Fokusebene, skaliert.
        // Skybox (depth >= max_depth) wird maximal unscharf.
        let mut coc = F::new(0.0);
        if depth < max_depth {
            let delta = depth - focal_distance;
            let ratio = delta / depth.max(F::new(0.0001));
            coc = ratio.abs() * aperture;
            coc = coc.min(F::new(1.0));
        } else {
            coc = F::new(1.0);
        }

        // 🟢 cubek-convolution erwartet **channels-last** (NHWC): Die 3
        // Farbkanäle liegen interleaved hintereinander, nicht in 3 Planes.
        let pixel_offset = (yu * wu + xu) * usize::new(3);

        // Farben mit dem Unschärfefaktor vor gewichten.
        rgb_output[pixel_offset] = r * coc;
        rgb_output[pixel_offset + usize::new(1)] = g * coc;
        rgb_output[pixel_offset + usize::new(2)] = b * coc;
    }
}

/// 🟢 DoF: Erzeugt UND normalisiert den Bokeh-Disk-Kernel `[K×K]` in EINEM
/// GPU-Pass — komplett ohne CPU↔GPU-Sync (Option A).
///
/// Da `k ≤ 15`, läuft das gesamte Grid (≤ 225 Elemente) in einem einzigen
/// Workgroup. Statt SharedMemory (in dieser CubeCL-Version nicht im Frontend
/// verfügbar) macht der Kernel zwei Schleifen über den **globalen** Speicher:
///
/// 1. Jeder Thread berechnet sein rohes Disk-Gewicht (Smoothstep-Abfall über
///    den Blendenrand) und schreibt es.
/// 2. Jeder Thread liest alle `k*k` Rohgewichte zurück und summiert sie.
/// 3. Jeder Thread teilt sein eigenes Gewicht durch die Summe.
///
/// Ergebnis: `sum(weights) == 1` — die Bildhelligkeit bleibt erhalten, und
/// es gibt keinen `map_async`-Readback auf einem CubeCL-Storage-Buffer mehr.
#[cube(launch, fast_math = FastMath::all().difference(FastMath::AllowContraction.into()))]
pub fn generate_disk_kernel<F: Float + LaunchArg>(
    kernel: &mut Tensor<F>, // [out_ch=3, K, K, in_ch=3] channels-last
    kernel_size: u32,       // K (ungerade, 3..=15)
) {
    let x = UNIT_POS_X;
    let y = UNIT_POS_Y;

    if x < kernel_size && y < kernel_size {
        let k_sz = usize::cast_from(kernel_size);
        let kt_sz = (kernel_size as usize) * (kernel_size as usize);
        let kx = usize::cast_from(x);
        let ky = usize::cast_from(y);
        // 🟢 Channels-last: [3, K, K, 3] — innerhalb eines Kanal-Paars (oc, ic)
        // ist die räumliche Anordnung [ky, kx], jeder Thread kümmert sich um
        // ALLE 3x3 Kanal-Kombinationen desselben Raumtaps.
        let ch = usize::new(3);
        let spatial = ky * k_sz + kx;

        // --- 1) Rohes Disk-Gewicht: glatter Abfall über den Blendenrand ---
        let size_f = F::cast_from(kernel_size);
        let half = (size_f - F::new(1.0)) * F::new(0.5);

        // Abstand vom Zentrum, normiert auf die halbe Kantenlänge.
        let dx = F::cast_from(x) - half;
        let dy = F::cast_from(y) - half;
        let dist = (dx * dx + dy * dy).sqrt() / half;

        let radius = F::new(1.0);
        let w = if dist > radius {
            F::new(0.0)
        } else {
            // Smoothstep-Abfall: hart am Rand, glatt im Zentrum.
            let t = radius - dist;
            t * t * (F::new(3.0) - F::new(2.0) * t)
        };
        kernel[spatial] = w;

        // --- 2) Summe über alle Rohgewichte des RAUM-Taps (nur globale Lesezugriffe) ---
        // Es reicht K*K Werte zu summieren, da alle 9 Kanal-Paare dasselbe
        // Raumgewicht tragen (der Kernel ist kanalisch identisch).
        let mut sum = F::new(0.0);
        let mut i = usize::new(0);
        loop {
            if i >= kt_sz {
                break;
            }
            sum += kernel[i];
            i += usize::new(1);
        }

        // --- 3) Normalisiertes Gewicht für alle 9 Kanal-Paare schreiben ---
        let norm = if sum > F::new(0.0) { w / sum } else { F::new(0.0) };
        let mut oc = usize::new(0);
        loop {
            if oc >= ch {
                break;
            }
            let mut ic = usize::new(0);
            loop {
                if ic >= ch {
                    break;
                }
                // [oc, ky, kx, ic]  (channels-last)
                let idx = ((oc * k_sz + ky) * k_sz + kx) * ch + ic;
                kernel[idx] = norm;
                ic += usize::new(1);
            }
            oc += usize::new(1);
        }
    }
}
