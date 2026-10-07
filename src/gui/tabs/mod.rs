//! 🟢 Tab-Inhalte der Sidebar — ein File pro Tab.
//!
//! - `env.rs`   — Tensor 4 (Umwelt): Licht / Farben / Nebel
//! - `arch.rs`  — Tensor 5 (Architektur): Säulen & Raum / Bögen / Dekor
//! - `fold.rs`  — Tensor 6 (Faltung): Zellgröße + Tempo
//! - `mat.rs`   — PBR-Material-Tensor: Roughness/Metallic/Emissive/Specular pro Slot
//! - `ctrl.rs`  — Original-Steuerung: Kamera-Info, Schalter, Tasten-Hilfe

mod arch;
mod ctrl;
mod env;
mod fold;
mod mat;

pub(super) use arch::arch_tab;
pub(super) use ctrl::ctrl_tab;
pub(super) use env::env_tab;
pub(super) use fold::fold_tab;
pub(super) use mat::mat_tab;
