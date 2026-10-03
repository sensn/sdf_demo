//! 🟢 Arch-Tab (Tensor 5 — Architektur): Accordion mit Säulen & Raum /
//! Bögen / Dekor.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::accordion::Accordion,
    App, IntoElement,
};

use crate::app_state::ApplicationState;

use super::super::widgets::{acc_item, slider_row};
use super::super::GuiState;

/// Arch-Tab: ein Accordion mit Säulen & Raum / Bögen / Dekor.
pub(in crate::gui) fn arch_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    cx: &App,
) -> impl IntoElement {
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
