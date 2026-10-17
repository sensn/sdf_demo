//! 🟢 Mat-Tab (PBR-Material-Tensor): Slot-Auswahl + die PBR-Parameter im Accordion.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::{
        accordion::Accordion, // 🟢 FIX: Import hinzugefügt
        button::Button,
        Selectable,
    },
    div,
    prelude::*,
    px, rgb,
    App, IntoElement, Styled,
};

use crate::app_state::ApplicationState;

use super::super::widgets::{acc_item, section, slider_row, slider_row_fmt};
use super::super::{GuiState, ITEM_BG, MUTED};

/// Mat-Tab: Slot-Auswahl (Buttons) + PBR-Slider im Accordion für den gewählten Slot.
pub(in crate::gui) fn mat_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    cx: &App, // 🟢 FIX: Unterstrich entfernt, cx wird nun gelesen!
) -> impl IntoElement {
    let s = state.lock().unwrap();
    let sel = s.current_selected_slot.min(s.slot_roughness.len().saturating_sub(1));

    // Nutzt das bewährte env_open Array als Blaupause für die Index-Zuweisung
let open = gui.accordion_state.read(cx).mat_open.clone(); // 🟢 Nutzt mat_open

    // Aktuelle PBR-Werte des gewählten Slots (für Live-Anzeige + Slider-Sync)
    let rough = s.slot_roughness[sel];
    let metal = s.slot_metallic[sel];
    let emiss = s.slot_emissive[sel];
    let spec = s.slot_specular[sel];

    // Slot-Name + Typ für die Kopfzeile
    let slot_name = format!("Slot {} — {}", sel + 1, slot_type_name(s.slot_types[sel]));

    // Slot-Selector: ein Button pro Slot, der aktive ist selected.
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

    let acc_state = gui.accordion_state.clone();

    // 🟢 Nach exakter Blaupause (env.rs): div-Umschlag für das reibungslose Scrolling
    div()
        .flex()
        .flex_col()
        .gap(px(8.0))
        // --- Slot-Auswahl (Bleibt für schnellen Zugriff fixiert über dem Accordion) ---
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
        // --- Accordion-Block ---
        .child(
            div().child(
                Accordion::new("mat_acc")
                    .bordered(true)
                    .multiple(true)
                    .on_toggle_click(move |open_ix: &[usize], _win, cx| {
                        acc_state.update(cx, |st, cx| {
            // 🟢 FIX: Schreibt jetzt korrekt in mat_open statt env_open!
            st.mat_open = (0..3).map(|i| open_ix.contains(&i)).collect();
            cx.notify();
        });
                    })
                    .item(acc_item(
                        "PBR Parameters",
                        open[0], // 🟢 FIX: Array-Indexierung analog zu env.rs
                        vec![
                            slider_row("Roughness", &gui.mat_roughness_slider, rough).into_any_element(),
                            slider_row("Metallic", &gui.mat_metallic_slider, metal).into_any_element(),
                            slider_row("Emissive", &gui.mat_emissive_slider, emiss).into_any_element(),
                            slider_row("Specular", &gui.mat_specular_slider, spec).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Global Reflections",
                        open[1],
                        vec![
                            slider_row_fmt("Reflection Max Steps", &gui.reflection_max_steps_slider, s.reflection_max_steps as f32, 0).into_any_element(),
                            slider_row("Roughness Gate", &gui.reflection_roughness_gate_slider, s.reflection_roughness_gate).into_any_element(),
                            slider_row("Fresnel Strength", &gui.fresnel_strength_factor_slider, s.fresnel_strength_factor).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Material Physics",
                        open[2],
                        vec![
                            div()
                                .text_color(rgb(MUTED))
                                .text_xs()
                                .child("Statue PBR (ID 2)")
                                .into_any_element(),
                            slider_row("Statue Metallic", &gui.statue_metallic_slider, s.statue_metallic).into_any_element(),
                            div()
                                .text_color(rgb(MUTED))
                                .text_xs()
                                .child("Pillar PBR (ID 1)")
                                .into_any_element(),
                            slider_row("Pillar Roughness", &gui.pillar_roughness_slider, s.pillar_roughness).into_any_element(),
                        ],
                    ))
            )
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
