//! 🟢 Mat-Tab (PBR-Material-Tensor): Slot-Auswahl + die 4 PBR-Parameter
//! (Roughness / Metallic / Emissive / Specular) für den gewählten Slot.
//!
//! Die Werte leben in `ApplicationState` als parallele Vektoren
//! (`slot_roughness` usw., Index = Slot). Der GUI-Tab editiert immer den
//! aktuell gewählten Slot (`current_selected_slot`), den die Tasten 1–6
//! setzen. `materials_data()` packt die Vektoren jeden Frame in den
//! VRAM-Buffer (Stride 4) — die Slider brauchen also nur in die Vektoren
//! zu schreiben.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::{
        button::Button,
        Selectable,
    },
    div,
    prelude::*,
    px, rgb,
    App, IntoElement, Styled,
};

use crate::app_state::ApplicationState;

use super::super::widgets::{section, slider_row};
use super::super::{GuiState, ITEM_BG, MUTED};

/// Mat-Tab: Slot-Auswahl (Buttons) + PBR-Slider für den gewählten Slot.
pub(in crate::gui) fn mat_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    _cx: &App,
) -> impl IntoElement {
    let s = state.lock().unwrap();
    let sel = s.current_selected_slot.min(s.slot_roughness.len().saturating_sub(1));

    // Aktuelle PBR-Werte des gewählten Slots (für Live-Anzeige + Slider-Sync)
    let rough = s.slot_roughness[sel];
    let metal = s.slot_metallic[sel];
    let emiss = s.slot_emissive[sel];
    let spec = s.slot_specular[sel];

    // Slot-Name + Typ für die Kopfzeile
    let slot_name = format!("Slot {} — {}", sel + 1, slot_type_name(s.slot_types[sel]));

    // Slot-Selector: ein Button pro Slot, der aktive ist selected.
    // Klick schreibt `current_selected_slot` (Tasten 1–6 machen dasselbe).
    let slot_count = s.slot_roughness.len();
    let mut selector_row = div().flex().flex_row().flex_wrap().gap(px(4.0));
    for i in 0..slot_count {
        let state_clone = state.clone();
        selector_row = selector_row.child(
            Button::new(("mat_slot", i))
                .label(format!("{}", i + 1))
                .compact()
                .selected(i == sel)
                .on_click(move |_ev, _win, _cx| {
                    if let Ok(mut s) = state_clone.lock() {
                        s.current_selected_slot = i;
                    }
                }),
        );
    }

    div()
        .flex()
        .flex_col()
        .gap(px(8.0))
        // --- Slot-Auswahl ---
        .child(section("Material Slot"))
        .child(
            div()
                .bg(rgb(ITEM_BG))
                .rounded_md()
                .p_3()
                .flex()
                .flex_col()
                .gap(px(6.0))
                .child(selector_row)
                .child(
                    div()
                        .text_color(rgb(MUTED))
                        .text_xs()
                        .child(slot_name),
                ),
        )
        // --- PBR-Parameter ---
        .child(section("PBR Parameters"))
        .child(
            div()
                .bg(rgb(ITEM_BG))
                .rounded_md()
                .p_3()
                .flex()
                .flex_col()
                .gap(px(8.0))
                .child(slider_row("Roughness", &gui.mat_roughness_slider, rough))
                .child(slider_row("Metallic", &gui.mat_metallic_slider, metal))
                .child(slider_row("Emissive", &gui.mat_emissive_slider, emiss))
                .child(slider_row("Specular", &gui.mat_specular_slider, spec)),
        )
}

/// Typ-Name eines Slots (1=Kristall, 2=Gyroid, 3=Torus, 0=inaktiv).
fn slot_type_name(t: f32) -> &'static str {
    match t as u32 {
        1 => "Crystal",
        2 => "Gyroid",
        3 => "Torus",
        _ => "Inactive",
    }
}
