//! `kernel/volume/` — 🌌 Zero-Copy Volumetric Renderer per TR.md
//! (doc/volumetric/TR.md).
//!
//! Ersetzt Texture3D-/Voxel-basierte Pipelines durch rein arithmetische
//! Dichte-Synthese in Registern (Zero-Copy, Zero-Cost-Philosophie):
//! - `density`  : Gyroid-SDF → Dichtefeld (ρ = clip(-d·σ, 0, 1)) + FBM-Wind,
//!                HDR-Emissionskern (Blackbody-Profil), sin-hash Trilinear-Noise
//! - `phase`    : Henyey-Greenstein-Streufunktion + Beer-Lambert-Schatten-Ray
//! - `dither`   : zeitlicher Hochfrequenz-Jitter (Banding → Blaurauschen)
//! - `integrate`: Volumetrischer Raymarch (Transmittanz, Early-Ray-Termination)
//! - `volume_kernel` : Hardware-Linked Launch (Pixel → Ray → HDR-Framebuffer)

pub mod density;
pub mod phase;
pub mod dither;
pub mod integrate;
pub mod volume_kernel;
