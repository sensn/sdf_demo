//! 🟢 GUI-Modul: Sidebar mit Tabs (Env / Arch / Fold / Ctrl) für alle Tensor-Parameter.
//!
//! Architektur:
//! - `GuiState` (diese Datei) hält alle Slider-Entities + die UI-Zustands-
//!   Entities (`TabState`, `AccordionUiState`). Lebt in `SurfaceExample`.
//! - Slider -> ApplicationState: `subscribe_slider` (widgets.rs, einmalig
//!   beim Bau). State -> Slider: `sync_sliders` (jeden Frame, Epsilon-Guard
//!   gegen Float-Rückkopplung; während eines Drags ist das Zurückschreiben
//!   des bereits geschriebenen Werts ein No-Op).
//! - Tabs + Accordions lösen das Platzproblem: nur der aktive Tab rendert,
//!   Accordions falten Gruppen zusammen, der Tab-Inhalt ist scrollbar.
//!   Tab-/Accordion-Klicks laufen über Entity-Updates (wgpui-Callbacks
//!   bekommen `&mut App`), SurfaceExample rendert eh jeden Frame neu.
//!
//! Aufteilung:
//! - `widgets.rs`  — Zeilen-Helfer + Slider-Verdrahtung
//! - `sidebar.rs`  — Layout, TabBar, Scroll-Container
//! - `tabs/`       — ein File pro Tab (env, arch, fold, ctrl)

mod sidebar;
mod tabs;
mod widgets;

pub use sidebar::sidebar;

use widgets::{subscribe_slider, sync_one};

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::slider::SliderState,
    App, AppContext, Context, Entity, Window,
};

use crate::app_state::ApplicationState;

// =========================================================================
// Farben (Sidebar-Theme)
// =========================================================================
pub(super) const PANEL_BG: u32 = 0x1a2333;
pub(super) const ITEM_BG: u32 = 0x161b28;
pub(super) const ACCENT: u32 = 0x00ffcc;
pub(super) const TEXT: u32 = 0xffffff;
pub(super) const MUTED: u32 = 0x8a92a6;

// =========================================================================
// UI-Zustand (als Entities, damit wgpui-Callbacks sie mutieren können)
// =========================================================================

/// Aktiver Sidebar-Tab.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GuiTab {
    Environment,
    Architecture,
    Fold,
    Controls,
}

/// Aktiver Tab — als Entity, damit TabBar::on_click ihn schreiben kann.
pub struct TabState {
    pub active: GuiTab,
}

/// Accordion-Öffnungs-Flags (Index-basiert, pro Tab).
pub struct AccordionUiState {
    /// Env-Tab: [0] = Licht, [1] = Farben, [2] = Nebel
    pub env_open: Vec<bool>,
    /// Arch-Tab: [0] = Säulen & Raum, [1] = Bögen, [2] = Dekor
    pub arch_open: Vec<bool>,
}

// =========================================================================
// GuiState
// =========================================================================

/// Gesamter GUI-Zustand: Slider-Entities + UI-Zustands-Entities.
pub struct GuiState {
    // TENSOR 4 (Umwelt):
    pub light_intensity_slider: Entity<SliderState>,
    pub ambient_strength_slider: Entity<SliderState>,
    pub key_light_x_slider: Entity<SliderState>,
    pub key_light_y_slider: Entity<SliderState>,
    pub key_light_z_slider: Entity<SliderState>,
    pub key_r_slider: Entity<SliderState>,
    pub key_g_slider: Entity<SliderState>,
    pub key_b_slider: Entity<SliderState>,
    pub bg_r_slider: Entity<SliderState>,
    pub bg_g_slider: Entity<SliderState>,
    pub bg_b_slider: Entity<SliderState>,
    pub fog_density_slider: Entity<SliderState>,
    // TENSOR 5 (Architektur):
    pub pillar_dist_slider: Entity<SliderState>,
    pub pillar_thick_slider: Entity<SliderState>,
    pub room_height_slider: Entity<SliderState>,
    pub ceiling_thick_slider: Entity<SliderState>,
    pub arch_radius_slider: Entity<SliderState>,
    pub arch_height_slider: Entity<SliderState>,
    pub decor_freq_slider: Entity<SliderState>,
    pub decor_depth_slider: Entity<SliderState>,
    pub decor_thick_slider: Entity<SliderState>,
    // TENSOR 6 (Faltung):
    pub cell_size_slider: Entity<SliderState>,
    pub fold_speed_slider: Entity<SliderState>,

    /// UI-Zustand
    pub tab_state: Entity<TabState>,
    pub accordion_state: Entity<AccordionUiState>,
}

impl GuiState {
    /// Baut alle Slider-Entities mit min/max/step/default (Original-Konstanten)
    /// und verdrahtet die Subscriptions (Slider -> ApplicationState).
    pub(crate) fn new(
        cx: &mut Context<crate::SurfaceExample>,
        state: &Arc<Mutex<ApplicationState>>,
    ) -> Self {
        // --- Slider-Entities (min/max/step/default aus den Original-Konstanten)
        let light_intensity_slider =
            cx.new(|_| SliderState::new().min(0.0).max(3.0).step(0.05).default_value(1.0));
        let ambient_strength_slider =
            cx.new(|_| SliderState::new().min(0.0).max(1.0).step(0.01).default_value(0.1));
        let key_light_x_slider =
            cx.new(|_| SliderState::new().min(-20.0).max(20.0).step(0.1).default_value(4.0));
        let key_light_y_slider =
            cx.new(|_| SliderState::new().min(0.0).max(20.0).step(0.1).default_value(7.0));
        let key_light_z_slider =
            cx.new(|_| SliderState::new().min(-20.0).max(20.0).step(0.1).default_value(-4.0));
        let key_r_slider =
            cx.new(|_| SliderState::new().min(0.0).max(1.0).step(0.01).default_value(1.0));
        let key_g_slider =
            cx.new(|_| SliderState::new().min(0.0).max(1.0).step(0.01).default_value(0.95));
        let key_b_slider =
            cx.new(|_| SliderState::new().min(0.0).max(1.0).step(0.01).default_value(0.85));
        let bg_r_slider =
            cx.new(|_| SliderState::new().min(0.0).max(1.0).step(0.01).default_value(0.35));
        let bg_g_slider =
            cx.new(|_| SliderState::new().min(0.0).max(1.0).step(0.01).default_value(0.45));
        let bg_b_slider =
            cx.new(|_| SliderState::new().min(0.0).max(1.0).step(0.01).default_value(0.60));
        let fog_density_slider =
            cx.new(|_| SliderState::new().min(0.0).max(2.0).step(0.01).default_value(0.5));
        let pillar_dist_slider =
            cx.new(|_| SliderState::new().min(2.0).max(12.0).step(0.1).default_value(5.0));
        let pillar_thick_slider =
            cx.new(|_| SliderState::new().min(0.1).max(2.0).step(0.05).default_value(0.6));
        let room_height_slider =
            cx.new(|_| SliderState::new().min(1.0).max(8.0).step(0.1).default_value(3.0));
        let ceiling_thick_slider =
            cx.new(|_| SliderState::new().min(0.01).max(1.0).step(0.01).default_value(0.1));
        let arch_radius_slider =
            cx.new(|_| SliderState::new().min(1.0).max(6.0).step(0.1).default_value(3.2));
        let arch_height_slider =
            cx.new(|_| SliderState::new().min(0.0).max(4.0).step(0.1).default_value(1.0));
        let decor_freq_slider =
            cx.new(|_| SliderState::new().min(0.0).max(8.0).step(0.1).default_value(2.0));
        let decor_depth_slider =
            cx.new(|_| SliderState::new().min(0.0).max(0.2).step(0.005).default_value(0.03));
        let decor_thick_slider =
            cx.new(|_| SliderState::new().min(0.0).max(0.1).step(0.002).default_value(0.01));
        let cell_size_slider =
            cx.new(|_| SliderState::new().min(2.0).max(40.0).step(0.5).default_value(10.0));
        let fold_speed_slider =
            cx.new(|_| SliderState::new().min(0.0).max(5.0).step(0.05).default_value(1.0));

        // --- Subscriptions (Slider -> ApplicationState)
        subscribe_slider(cx, &light_intensity_slider, state, |s, v| s.light_intensity = v);
        subscribe_slider(cx, &ambient_strength_slider, state, |s, v| s.ambient_strength = v);
        subscribe_slider(cx, &key_light_x_slider, state, |s, v| s.key_light_x = v);
        subscribe_slider(cx, &key_light_y_slider, state, |s, v| s.key_light_y = v);
        subscribe_slider(cx, &key_light_z_slider, state, |s, v| s.key_light_z = v);
        subscribe_slider(cx, &key_r_slider, state, |s, v| s.key_r = v);
        subscribe_slider(cx, &key_g_slider, state, |s, v| s.key_g = v);
        subscribe_slider(cx, &key_b_slider, state, |s, v| s.key_b = v);
        subscribe_slider(cx, &bg_r_slider, state, |s, v| s.bg_r = v);
        subscribe_slider(cx, &bg_g_slider, state, |s, v| s.bg_g = v);
        subscribe_slider(cx, &bg_b_slider, state, |s, v| s.bg_b = v);
        subscribe_slider(cx, &fog_density_slider, state, |s, v| s.fog_density = v);
        subscribe_slider(cx, &pillar_dist_slider, state, |s, v| s.pillar_dist = v);
        subscribe_slider(cx, &pillar_thick_slider, state, |s, v| s.pillar_thick = v);
        subscribe_slider(cx, &room_height_slider, state, |s, v| s.room_height = v);
        subscribe_slider(cx, &ceiling_thick_slider, state, |s, v| s.ceiling_thick = v);
        subscribe_slider(cx, &arch_radius_slider, state, |s, v| s.arch_radius = v);
        subscribe_slider(cx, &arch_height_slider, state, |s, v| s.arch_height = v);
        subscribe_slider(cx, &decor_freq_slider, state, |s, v| s.decor_freq = v);
        subscribe_slider(cx, &decor_depth_slider, state, |s, v| s.decor_depth = v);
        subscribe_slider(cx, &decor_thick_slider, state, |s, v| s.decor_thick = v);
        subscribe_slider(cx, &cell_size_slider, state, |s, v| s.cell_size = v);
        subscribe_slider(cx, &fold_speed_slider, state, |s, v| s.fold_speed = v);

        // --- UI-Zustand: Tab 0 (Env) aktiv, erste Accordion-Gruppe offen
        let tab_state = cx.new(|_| TabState {
            active: GuiTab::Environment,
        });
        let accordion_state = cx.new(|_| AccordionUiState {
            env_open: vec![true, false, false],
            arch_open: vec![true, false, false],
        });

        Self {
            light_intensity_slider,
            ambient_strength_slider,
            key_light_x_slider,
            key_light_y_slider,
            key_light_z_slider,
            key_r_slider,
            key_g_slider,
            key_b_slider,
            bg_r_slider,
            bg_g_slider,
            bg_b_slider,
            fog_density_slider,
            pillar_dist_slider,
            pillar_thick_slider,
            room_height_slider,
            ceiling_thick_slider,
            arch_radius_slider,
            arch_height_slider,
            decor_freq_slider,
            decor_depth_slider,
            decor_thick_slider,
            cell_size_slider,
            fold_speed_slider,
            tab_state,
            accordion_state,
        }
    }

    /// State -> Slider synchronisieren (jeden Frame, Epsilon-Guard).
    pub fn sync_sliders(&self, state: &ApplicationState, window: &mut Window, cx: &mut App) {
        sync_one(&self.light_intensity_slider, state.light_intensity, window, cx);
        sync_one(&self.ambient_strength_slider, state.ambient_strength, window, cx);
        sync_one(&self.key_light_x_slider, state.key_light_x, window, cx);
        sync_one(&self.key_light_y_slider, state.key_light_y, window, cx);
        sync_one(&self.key_light_z_slider, state.key_light_z, window, cx);
        sync_one(&self.key_r_slider, state.key_r, window, cx);
        sync_one(&self.key_g_slider, state.key_g, window, cx);
        sync_one(&self.key_b_slider, state.key_b, window, cx);
        sync_one(&self.bg_r_slider, state.bg_r, window, cx);
        sync_one(&self.bg_g_slider, state.bg_g, window, cx);
        sync_one(&self.bg_b_slider, state.bg_b, window, cx);
        sync_one(&self.fog_density_slider, state.fog_density, window, cx);
        sync_one(&self.pillar_dist_slider, state.pillar_dist, window, cx);
        sync_one(&self.pillar_thick_slider, state.pillar_thick, window, cx);
        sync_one(&self.room_height_slider, state.room_height, window, cx);
        sync_one(&self.ceiling_thick_slider, state.ceiling_thick, window, cx);
        sync_one(&self.arch_radius_slider, state.arch_radius, window, cx);
        sync_one(&self.arch_height_slider, state.arch_height, window, cx);
        sync_one(&self.decor_freq_slider, state.decor_freq, window, cx);
        sync_one(&self.decor_depth_slider, state.decor_depth, window, cx);
        sync_one(&self.decor_thick_slider, state.decor_thick, window, cx);
        sync_one(&self.cell_size_slider, state.cell_size, window, cx);
        sync_one(&self.fold_speed_slider, state.fold_speed, window, cx);
    }
}
