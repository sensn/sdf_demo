//! 🟢 Fold-Tab (Tensor 6 — Faltung): schlichtes Panel mit Zellgröße +
//! Faltungs-Tempo.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    div,
    prelude::*,
    px, rgb,
    App, IntoElement, Styled,
};

use crate::app_state::ApplicationState;

use super::super::widgets::slider_row;
use super::super::{GuiState, ACCENT, PANEL_BG};

/// Fold-Tab: schlichtes Panel mit Zellgröße + Faltungs-Tempo.
pub(in crate::gui) fn fold_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    _cx: &App,
) -> impl IntoElement {
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
