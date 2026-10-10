//! 🟢 Obj-Tab (Objekt-Tensor): Slot-Auswahl + Position / Scale / Rotation
//! für den gewählten Slot.
//!
//! Folgt exakt dem Mat-Tab-Muster: Die Werte leben in `ApplicationState`
//! als parallele Vektoren (`slot_offsets_x/y/z`, `slot_sizes`,
//! `slot_rot_x/y/z`, Index = Slot). Der GUI-Tab editiert immer den
//! aktuell gewählten Slot (`current_selected_slot`). `slots_data()` und
//! `rotations_data()` packen die Vektoren jeden Frame in die VRAM-Buffer
//! — die Slider brauchen also nur in die Vektoren zu schreiben.

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

/// Obj-Tab: Slot-Auswahl (Buttons) + Position/Scale/Rotation-Slider
/// für den gewählten Slot.
pub(in crate::gui) fn obj_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    _cx: &App,
) -> impl IntoElement {
    let s = state.lock().unwrap();
    let sel = s.current_selected_slot.min(s.slot_offsets_x.len().saturating_sub(1));

    // Aktuelle Objekt-Werte des gewählten Slots (für Live-Anzeige + Slider-Sync)
    let pos_x = s.slot_offsets_x[sel];
    let pos_y = s.slot_offsets_y[sel];
    let pos_z = s.slot_offsets_z[sel];
    let scale = s.slot_sizes[sel];
    let rot_x = s.slot_rot_x[sel];
    let rot_y = s.slot_rot_y[sel];
    let rot_z = s.slot_rot_z[sel];

    // Slot-Name + Typ für die Kopfzeile
    let slot_name = format!("Slot {} — {}", sel + 1, slot_type_name(s.slot_types[sel]));

    // Slot-Selector: ein Button pro Slot, der aktive ist selected.
    // Klick schreibt `current_selected_slot` (Tasten 1–6 machen dasselbe).
    let slot_count = s.slot_offsets_x.len();
    let mut selector_row = div().flex().flex_row().flex_wrap().gap(px(4.0));
    for i in 0..slot_count {
        let state_clone = state.clone();
        selector_row = selector_row.child(
            Button::new(("obj_slot", i))
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
        .child(section("Object Slot"))
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
        // --- Position ---
        .child(section("Position"))
        .child(
            div()
                .bg(rgb(ITEM_BG))
                .rounded_md()
                .p_3()
                .flex()
                .flex_col()
                .gap(px(8.0))
                .child(slider_row("Pos X", &gui.obj_pos_x_slider, pos_x))
                .child(slider_row("Pos Y", &gui.obj_pos_y_slider, pos_y))
                .child(slider_row("Pos Z", &gui.obj_pos_z_slider, pos_z)),
        )
        // --- Scale ---
        .child(section("Scale"))
        .child(
            div()
                .bg(rgb(ITEM_BG))
                .rounded_md()
                .p_3()
                .flex()
                .flex_col()
                .gap(px(8.0))
                .child(slider_row("Scale", &gui.obj_scale_slider, scale)),
        )
        // --- Rotation ---
        .child(section("Rotation (Euler, rad)"))
        .child(
            div()
                .bg(rgb(ITEM_BG))
                .rounded_md()
                .p_3()
                .flex()
                .flex_col()
                .gap(px(8.0))
                .child(slider_row("Rot X", &gui.obj_rot_x_slider, rot_x))
                .child(slider_row("Rot Y", &gui.obj_rot_y_slider, rot_y))
                .child(slider_row("Rot Z", &gui.obj_rot_z_slider, rot_z)),
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
