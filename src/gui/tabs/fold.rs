//! 🟢 Fold-Tab (Tensor 6 — Faltung): Accordion mit Zellgröße, Faltungs-Tempo
//! und unendlicher Pfeiler-Repetition.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::accordion::Accordion, // 🟢 FIX: Accordion-Import hinzugefügt
    div,
    prelude::*,
    px, rgb,
    App, IntoElement, Styled,
};

use crate::app_state::ApplicationState;

use super::super::widgets::{acc_item, slider_row}; // 🟢 FIX: acc_item hinzugefügt
use super::super::{GuiState, MUTED}; // 🟢 MUTED für Konsistenz mit Blaupause genutzt

/// Fold-Tab: Accordion für Infinite Fold und Infinite Repetition.
pub(in crate::gui) fn fold_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    cx: &App, // 🟢 FIX: Unterstrich entfernt, cx wird nun gelesen!
) -> impl IntoElement {
    let s = state.lock().unwrap();

    // Nutzt das dedizierte fold_open Array aus dem globalen Zustand
    let open = gui.accordion_state.read(cx).fold_open.clone();
    let acc_state = gui.accordion_state.clone();

    // 🟢 Blaupause angewendet: Äußerer div-Wrapper für nahtloses Scrolling in der Sidebar
    div()
        .flex()
        .flex_col()
        .gap(px(8.0))
        .child(
            div().child(
                Accordion::new("fold_acc")
                    .bordered(true)
                    .multiple(true)
                    .on_toggle_click(move |open_ix: &[usize], _win, cx| {
                       acc_state.update(cx, |st, cx| {
            // 🟢 FIX: Schreibt jetzt korrekt in fold_open statt env_open!
            st.fold_open = (0..2).map(|i| open_ix.contains(&i)).collect();
            cx.notify();
        });
                    })
                    .item(acc_item(
                        "Infinite Fold (Tensor 6)",
                        open[0], // Erster Eintrag
                        vec![
                            slider_row("Cell Size", &gui.cell_size_slider, s.cell_size).into_any_element(),
                            slider_row("Fold Speed", &gui.fold_speed_slider, s.fold_speed).into_any_element(),
                        ],
                    ))
                    // 🟢 PARAM-GUI Step 3: Infinite Repetition (Pfeiler-Faltung)
                    .item(acc_item(
                        "Infinite Repetition",
                        open[1], // Zweiter Eintrag
                        vec![
                            slider_row("Pillar Spacing", &gui.pillar_spacing_slider, s.pillar_spacing).into_any_element(),
                            slider_row("Pillar Height", &gui.pillar_height_slider, s.pillar_height).into_any_element(),
                        ],
                    ))
            )
        )
}
