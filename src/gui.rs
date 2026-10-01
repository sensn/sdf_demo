//! 🟢 GUI-Modul: Sidebar mit Tabs (Env / Arch / Fold / Ctrl) für alle Tensor-Parameter.
//!
//! Architektur:
//! - `GuiState` hält alle Slider-Entities + die UI-Zustands-Entities
//!   (`TabState`, `AccordionUiState`). Lebt in `SurfaceExample`.
//! - Slider -> ApplicationState: `subscribe_slider` (einmalig beim Bau).
//! - State -> Slider: `sync_sliders` (jeden Frame, Epsilon-Guard gegen
//!   Float-Rückkopplung; während eines Drags ist das Zurückschreiben des
//!   bereits geschriebenen Werts ein No-Op).
//! - Tabs + Accordions lösen das Platzproblem: nur der aktive Tab rendert,
//!   Accordions falten Gruppen zusammen, der Tab-Inhalt ist scrollbar.
//!   Tab-/Accordion-Klicks laufen über Entity-Updates (wgpui-Callbacks
//!   bekommen `&mut App`), SurfaceExample rendert eh jeden Frame neu.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::{
        accordion::{Accordion, AccordionItem},
        slider::{Slider, SliderEvent, SliderState},
        switch::Switch,
        tab::{Tab, TabBar},
        Selectable,
    },
    div,
    prelude::*,
    px, rgb,
    AnyElement, App, Context, Entity, FontWeight, Styled, Window,
};

use crate::ApplicationState;

// =========================================================================
// Farben (Sidebar-Theme)
// =========================================================================
const PANEL_BG: u32 = 0x1a2333;
const ITEM_BG: u32 = 0x161b28;
const ACCENT: u32 = 0x00ffcc;
const TEXT: u32 = 0xffffff;
const MUTED: u32 = 0x8a92a6;

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
    pub(crate) fn new(cx: &mut Context<crate::SurfaceExample>, state: &Arc<Mutex<ApplicationState>>) -> Self {
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

// =========================================================================
// Slider-Verdrahtung
// =========================================================================

/// Slider -> ApplicationState: einmalige Subscription pro Slider.
/// Handler-Signatur von Context::subscribe: (&mut T, Entity<T2>, &Evt, &mut Context<T>).
fn subscribe_slider(
    cx: &mut Context<crate::SurfaceExample>,
    slider: &Entity<SliderState>,
    state: &Arc<Mutex<ApplicationState>>,
    write: fn(&mut ApplicationState, f32),
) {
    let state = state.clone();
    cx.subscribe(slider, move |_, _, event: &SliderEvent, _cx| {
        if let SliderEvent::Change(value) = event {
            if let Ok(mut s) = state.lock() {
                write(&mut s, value.start());
            }
        }
    })
    .detach();
}

/// State -> Slider: Wert nur schreiben, wenn er sich wirklich unterscheidet
/// (Epsilon-Guard gegen Float-Rückkopplung; Drag-No-Op siehe Modul-Doku).
fn sync_one(slider: &Entity<SliderState>, value: f32, window: &mut Window, cx: &mut App) {
    let current = slider.read(cx).value().start();
    if (current - value).abs() > f32::EPSILON {
        slider.update(cx, |state, cx| {
            state.set_value(value, window, cx);
        });
    }
}

// =========================================================================
// Bausteine
// =========================================================================

/// Eine Slider-Zeile: Label + Live-Wert + horizontaler Slider.
fn slider_row(label: &str, slider: &Entity<SliderState>, value: f32) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap(px(2.0))
        .child(
            div()
                .flex()
                .flex_row()
                .justify_between()
                .child(
                    div()
                        .text_color(rgb(MUTED))
                        .text_xs()
                        .child(label.to_string()),
                )
                .child(
                    div()
                        .text_color(rgb(ACCENT))
                        .text_xs()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(format!("{:.2}", value)),
                ),
        )
        .child(Slider::new(slider).horizontal())
}

/// Accordion-Item-Bauplan: Titel + offener Zustand + Slider-Zeilen.
/// Titel wird geowned (String) und die Zeilen sind als AnyElement typgelöscht
/// — die Closure ist damit 'static und fängt keine anonymen Lifetimes.
fn acc_item(
    title: &str,
    open: bool,
    rows: Vec<AnyElement>,
) -> impl FnOnce(AccordionItem) -> AccordionItem + 'static {
    let title = title.to_string();
    let mut content = div().flex().flex_col().gap(px(6.0)).p_2();
    for row in rows {
        content = content.child(row);
    }
    move |item: AccordionItem| {
        item.title(
                div()
                    .text_color(rgb(ACCENT))
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(title),
            )
            .open(open)
            .bg(rgb(ITEM_BG))
            .rounded_md()
            .child(content)
    }
}

// =========================================================================
// Sidebar
// =========================================================================

/// Komplette Sidebar: TabBar + scrollbarer Inhalt des aktiven Tabs.
pub fn sidebar(
    gui: &mut GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    window: &mut Window,
    cx: &mut App,
) -> impl IntoElement {
    // State -> Slider synchronisieren (vor dem Rendern, damit die Werte stimmen)
    {
        let s = state.lock().unwrap();
        gui.sync_sliders(&s, window, cx);
    }

    let active = gui.tab_state.read(cx).active;

    // TabBar: Klick schreibt in TabState-Entity
    let tab_state = gui.tab_state.clone();
    let tab_bar = TabBar::new("tensor_tabs")
        .child(
            Tab::new()
                .label("Env")
                .selected(active == GuiTab::Environment),
        )
        .child(
            Tab::new()
                .label("Arch")
                .selected(active == GuiTab::Architecture),
        )
        .child(Tab::new().label("Fold").selected(active == GuiTab::Fold))
        .child(
            Tab::new()
                .label("Ctrl")
                .selected(active == GuiTab::Controls),
        )
        .on_click(move |ix: &usize, _win, cx| {
            let tab = match ix {
                0 => GuiTab::Environment,
                1 => GuiTab::Architecture,
                2 => GuiTab::Fold,
                _ => GuiTab::Controls,
            };
            tab_state.update(cx, |ts, cx| {
                ts.active = tab;
                cx.notify();
            });
        });

    // Inhalt des aktiven Tabs
    let content = match active {
        GuiTab::Environment => env_tab(gui, state, cx).into_any_element(),
        GuiTab::Architecture => arch_tab(gui, state, cx).into_any_element(),
        GuiTab::Fold => fold_tab(gui, state, cx).into_any_element(),
        GuiTab::Controls => controls_tab(state).into_any_element(),
    };

    div()
        .id("sidebar")
        .w(px(300.0))
        .h_full()
        .flex()
        .flex_col()
        .gap(px(8.0))
        .p_2()
        .bg(rgb(0x10121a))
        .child(tab_bar)
        .child(
            div()
                .id("sidebar_scroll")
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .child(content),
        )
}

// =========================================================================
// Tab-Inhalte
// =========================================================================

/// Env-Tab: ein Accordion mit Licht / Farben / Nebel.
fn env_tab(gui: &GuiState, state: &Arc<Mutex<ApplicationState>>, cx: &App) -> impl IntoElement {
    let s = state.lock().unwrap();
    let open = gui.accordion_state.read(cx).env_open.clone();

    let acc_state = gui.accordion_state.clone();
    Accordion::new("env_acc")
        .bordered(true)
        .multiple(true)
        .on_toggle_click(move |open_ix: &[usize], _win, cx| {
            acc_state.update(cx, |st, cx| {
                st.env_open = (0..3).map(|i| open_ix.contains(&i)).collect();
                cx.notify();
            });
        })
        .item(acc_item(
            "Light",
            open[0],
            vec![
                slider_row("Intensity", &gui.light_intensity_slider, s.light_intensity).into_any_element(),
                slider_row("Ambient", &gui.ambient_strength_slider, s.ambient_strength).into_any_element(),
                slider_row("Key Light X", &gui.key_light_x_slider, s.key_light_x).into_any_element(),
                slider_row("Key Light Y", &gui.key_light_y_slider, s.key_light_y).into_any_element(),
                slider_row("Key Light Z", &gui.key_light_z_slider, s.key_light_z).into_any_element(),
            ],
        ))
        .item(acc_item(
            "Colors",
            open[1],
            vec![
                slider_row("Key R", &gui.key_r_slider, s.key_r).into_any_element(),
                slider_row("Key G", &gui.key_g_slider, s.key_g).into_any_element(),
                slider_row("Key B", &gui.key_b_slider, s.key_b).into_any_element(),
                slider_row("Background R", &gui.bg_r_slider, s.bg_r).into_any_element(),
                slider_row("Background G", &gui.bg_g_slider, s.bg_g).into_any_element(),
                slider_row("Background B", &gui.bg_b_slider, s.bg_b).into_any_element(),
            ],
        ))
        .item(acc_item(
            "Fog",
            open[2],
            vec![
                slider_row("Fog Density", &gui.fog_density_slider, s.fog_density).into_any_element(),
                fog_switch_row(state, s.fog_enabled == 1).into_any_element(),
            ],
        ))
}

/// Arch-Tab: ein Accordion mit Säulen & Raum / Bögen / Dekor.
fn arch_tab(gui: &GuiState, state: &Arc<Mutex<ApplicationState>>, cx: &App) -> impl IntoElement {
    let s = state.lock().unwrap();
    let open = gui.accordion_state.read(cx).arch_open.clone();

    let acc_state = gui.accordion_state.clone();
    Accordion::new("arch_acc")
        .bordered(true)
        .multiple(true)
        .on_toggle_click(move |open_ix: &[usize], _win, cx| {
            acc_state.update(cx, |st, cx| {
                st.arch_open = (0..3).map(|i| open_ix.contains(&i)).collect();
                cx.notify();
            });
        })
        .item(acc_item(
            "Pillars & Room",
            open[0],
            vec![
                slider_row("Pillar Distance", &gui.pillar_dist_slider, s.pillar_dist).into_any_element(),
                slider_row("Pillar Thickness", &gui.pillar_thick_slider, s.pillar_thick).into_any_element(),
                slider_row("Room Height", &gui.room_height_slider, s.room_height).into_any_element(),
                slider_row("Ceiling Thickness", &gui.ceiling_thick_slider, s.ceiling_thick).into_any_element(),
            ],
        ))
        .item(acc_item(
            "Arches",
            open[1],
            vec![
                slider_row("Arch Radius", &gui.arch_radius_slider, s.arch_radius).into_any_element(),
                slider_row("Arch Height", &gui.arch_height_slider, s.arch_height).into_any_element(),
            ],
        ))
        .item(acc_item(
            "Decor",
            open[2],
            vec![
                slider_row("Decor Frequency", &gui.decor_freq_slider, s.decor_freq).into_any_element(),
                slider_row("Decor Depth", &gui.decor_depth_slider, s.decor_depth).into_any_element(),
                slider_row("Decor Thickness", &gui.decor_thick_slider, s.decor_thick).into_any_element(),
            ],
        ))
}

/// Fold-Tab: schlichtes Panel mit Zellgröße + Faltungs-Tempo.
fn fold_tab(gui: &GuiState, state: &Arc<Mutex<ApplicationState>>, _cx: &App) -> impl IntoElement {
    let s = state.lock().unwrap();

    div()
        .flex()
        .flex_col()
        .gap(px(8.0))
        .child(
            div()
                .bg(rgb(PANEL_BG))
                .p_3()
                .rounded_md()
                .flex()
                .flex_col()
                .gap(px(4.0))
                .child(
                    div()
                        .text_color(rgb(ACCENT))
                        .text_sm()
                        .child("Infinite Fold (Tensor 6):"),
                )
                .child(slider_row("Cell Size", &gui.cell_size_slider, s.cell_size))
                .child(slider_row("Fold Speed", &gui.fold_speed_slider, s.fold_speed)),
        )
}

/// Nebel-Zeile: Label + Switch (synchron mit N-Taste via ApplicationState).
fn fog_switch_row(state: &Arc<Mutex<ApplicationState>>, checked: bool) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .justify_between()
        .items_center()
        .child(div().text_color(rgb(MUTED)).text_xs().child("Fog Enabled"))
        .child(
            Switch::new("fog_switch")
                .checked(checked)
                .on_click({
                    let state = state.clone();
                    move |checked: &bool, _win, _cx| {
                        if let Ok(mut s) = state.lock() {
                            s.fog_enabled = if *checked { 1 } else { 0 };
                        }
                    }
                }),
        )
}

// =========================================================================
// Ctrl-Tab: Original-Steuerung (Kamera-Info, Schalter, Tasten-Hilfe)
// =========================================================================

/// Schalter-Zeile: Label + Switch, schreibt 0/1 in ein ApplicationState-Feld.
fn toggle_row(
    id: &'static str,
    label: &str,
    state: &Arc<Mutex<ApplicationState>>,
    checked: bool,
    write: fn(&mut ApplicationState, bool),
) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .justify_between()
        .items_center()
        .child(div().text_color(rgb(TEXT)).text_sm().child(label.to_string()))
        .child(
            Switch::new(id)
                .checked(checked)
                .on_click({
                    let state = state.clone();
                    move |checked: &bool, _win, _cx| {
                        if let Ok(mut s) = state.lock() {
                            write(&mut s, *checked);
                        }
                    }
                }),
        )
}

/// Info-Zeile: Label links, Live-Wert rechts (z. B. Kameraposition).
fn info_row(label: &str, value: String) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .justify_between()
        .child(div().text_color(rgb(MUTED)).text_xs().child(label.to_string()))
        .child(
            div()
                .text_color(rgb(ACCENT))
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .child(value),
        )
}

/// Tasten-Hilfe: eine Zeile "• Taste : Beschreibung".
fn key_row(key: &str, desc: &str) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .gap(px(6.0))
        .child(div().text_color(rgb(ACCENT)).text_xs().child(format!("• {key}")))
        .child(div().text_color(rgb(MUTED)).text_xs().child(desc.to_string()))
}

/// Ctrl-Tab: Kamera-Info, Render-Schalter (AO/Lichter/Schatten), Tasten-Hilfe.
fn controls_tab(state: &Arc<Mutex<ApplicationState>>) -> impl IntoElement {
    let s = state.lock().unwrap();

    div()
        .flex()
        .flex_col()
        .gap(px(8.0))
        // --- Kamera ---
        .child(section("Camera"))
        .child(
            div()
                .bg(rgb(ITEM_BG))
                .rounded_md()
                .p_3()
                .flex()
                .flex_col()
                .gap(px(4.0))
                .child(info_row("Cam X", format!("{:.2}", s.cam_x)))
                .child(info_row("Cam Y", format!("{:.2}", s.cam_y)))
                .child(info_row("Cam Z", format!("{:.2}", s.cam_z)))
                .child(info_row("Yaw", format!("{:.2}", s.cam_yaw)))
                .child(info_row("Pitch", format!("{:.2}", s.cam_pitch))),
        )
        // --- Render-Schalter ---
        .child(section("Rendering"))
        .child(
            div()
                .bg(rgb(ITEM_BG))
                .rounded_md()
                .p_3()
                .flex()
                .flex_col()
                .gap(px(8.0))
                .child(toggle_row(
                    "ao_switch",
                    "Ambient Occlusion",
                    state,
                    s.enable_ao_mode == 1,
                    |s, on| s.enable_ao_mode = if on { 1 } else { 0 },
                ))
                .child(toggle_row(
                    "key_light_switch",
                    "Key Light",
                    state,
                    s.enable_key == 1,
                    |s, on| s.enable_key = if on { 1 } else { 0 },
                ))
                .child(toggle_row(
                    "fill_light_switch",
                    "Fill Light",
                    state,
                    s.enable_fill == 1,
                    |s, on| s.enable_fill = if on { 1 } else { 0 },
                ))
                .child(toggle_row(
                    "rim_light_switch",
                    "Rim Light",
                    state,
                    s.enable_rim == 1,
                    |s, on| s.enable_rim = if on { 1 } else { 0 },
                ))
                .child(toggle_row(
                    "shadow_switch",
                    "Hard Shadows",
                    state,
                    s.current_shadow_mode == 1,
                    |s, on| s.current_shadow_mode = if on { 1 } else { 0 },
                )),
        )
        // --- Tasten-Hilfe ---
        .child(section("Key Bindings"))
        .child(
            div()
                .bg(rgb(ITEM_BG))
                .rounded_md()
                .p_3()
                .flex()
                .flex_col()
                .gap(px(3.0))
                .child(key_row("WASD", "Fly through Scene"))
                .child(key_row("Mouse", "Look around"))
                .child(key_row("Key 1", "Toggle Key Light"))
                .child(key_row("Key 2", "Toggle Fill Light"))
                .child(key_row("Key 3", "Toggle Rim Light"))
                .child(key_row("Key 4", "Toggle Hard Shadows"))
                .child(key_row("Key 5", "Toggle AO"))
                .child(key_row("Q / E", "Light Intensity -/+"))
                .child(key_row("F / R", "Ambient Strength -/+"))
                .child(key_row("O / L", "Fog Density -/+"))
                .child(key_row("N", "Toggle Fog")),
        )
}

/// Sektions-Überschrift (wie "Camera", "Rendering", "Key Bindings").
fn section(title: &str) -> impl IntoElement {
    div()
        .text_color(rgb(ACCENT))
        .text_sm()
        .font_weight(FontWeight::SEMIBOLD)
        .child(title.to_string())
}
