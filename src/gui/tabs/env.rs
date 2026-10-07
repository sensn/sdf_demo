//! 🟢 Env-Tab (Tensor 4 — Umwelt): Accordion mit Licht / Farben / Nebel.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::{
        accordion::Accordion,
        switch::Switch,
    },
    div,
    prelude::*,
    rgb,
    App, IntoElement, Styled,
};

use crate::app_state::ApplicationState;

use super::super::widgets::{acc_item, slider_row};
use super::super::{GuiState, MUTED};

/// Env-Tab: ein Accordion mit Licht / Farben / Nebel.
pub(in crate::gui) fn env_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    cx: &App,
) -> impl IntoElement {
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

/// Nebel-Zeile: Label + Switch (synchron mit N-Taste via ApplicationState).
pub(super) fn fog_switch_row(
    state: &Arc<Mutex<ApplicationState>>,
    checked: bool,
) -> impl IntoElement {
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
