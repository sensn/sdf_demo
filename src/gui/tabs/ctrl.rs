//! 🟢 Ctrl-Tab: Original-Steuerung — Kamera-Info, Render-Schalter
//! (AO / Lichter / Schatten), Tasten-Hilfe im Accordion.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::accordion::Accordion,
    div,
    prelude::*,
    px, rgb,
    App, IntoElement, Styled,
};

use crate::app_state::ApplicationState;

use super::super::widgets::{acc_item, info_row, key_row, section, toggle_row};
use super::super::{GuiState, ITEM_BG};

/// Ctrl-Tab: Kamera-Info, Render-Schalter (AO/Lichter/Schatten), Tasten-Hilfe im Accordion.
pub(in crate::gui) fn ctrl_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    cx: &App,
) -> impl IntoElement {
    let s = state.lock().unwrap();

let open = gui.accordion_state.read(cx).ctrl_open.clone(); // 🟢 Nutzt ctrl_open
    let acc_state = gui.accordion_state.clone();

    div()
        .flex()
        .flex_col()
        .gap(px(8.0))
        .child(
            div().child(
                Accordion::new("ctrl_acc")
                    .bordered(true)
                    .multiple(true)
                    .on_toggle_click(move |open_ix: &[usize], _win, cx| {
                       acc_state.update(cx, |st, cx| {
            // 🟢 FIX: Schreibt jetzt korrekt in ctrl_open statt env_open!
            st.ctrl_open = (0..3).map(|i| open_ix.contains(&i)).collect();
            cx.notify();
        });
                    })
                    .item(acc_item(
                        "Camera",
                        open[0], // 🟢 FIX: Array-Index hinzugefügt
                        vec![
                            info_row("Cam X", format!("{:.2}", s.cam_x)).into_any_element(),
                            info_row("Cam Y", format!("{:.2}", s.cam_y)).into_any_element(),
                            info_row("Cam Z", format!("{:.2}", s.cam_z)).into_any_element(),
                            info_row("Yaw", format!("{:.2}", s.cam_yaw)).into_any_element(),
                            info_row("Pitch", format!("{:.2}", s.cam_pitch)).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Rendering",
                        open[1], // 🟢 FIX: Array-Index hinzugefügt
                        vec![
                            toggle_row(
                                "ao_switch",
                                "Ambient Occlusion",
                                state,
                                s.enable_ao_mode == 1,

                                |s, on| s.enable_ao_mode = if on { 1 } else { 0 },
                            ).into_any_element(),
                            toggle_row(
                                "key_light_switch",
                                "Key Light",
                                state,
                                s.enable_key == 1,

                                |s, on| s.enable_key = if on { 1 } else { 0 },
                            ).into_any_element(),
                            toggle_row(
                                "fill_light_switch",
                                "Fill Light",
                                state,
                                s.enable_fill == 1,

                                |s, on| s.enable_fill = if on { 1 } else { 0 },
                            ).into_any_element(),
                            toggle_row(
                                "rim_light_switch",
                                "Rim Light",
                                state,
                                s.enable_rim == 1,

                                |s, on| s.enable_rim = if on { 1 } else { 0 },
                            ).into_any_element(),
                            toggle_row(
                                "shadow_switch",
                                "Hard Shadows",
                                state,
                                s.current_shadow_mode == 1,

                                |s, on| s.current_shadow_mode = if on { 1 } else { 0 },
                            ).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Key Bindings",
                        open[2], // 🟢 FIX: Array-Index hinzugefügt
                        vec![
                            key_row("WASD", "Fly through Scene").into_any_element(),
                            key_row("Q / E", "Camera Up / Down").into_any_element(),
                            key_row("Mouse", "Look around").into_any_element(),
                            key_row("Key 1", "Toggle Key Light").into_any_element(),
                            key_row("Key 2", "Toggle Fill Light").into_any_element(),
                            key_row("Key 3", "Toggle Rim Light").into_any_element(),
                            key_row("Key 4", "Toggle Hard Shadows").into_any_element(),
                            key_row("Key 5", "Toggle AO").into_any_element(),
                            key_row("G / T", "Light Intensity -/+").into_any_element(),
                            key_row("F / R", "Ambient Strength -/+").into_any_element(),
                            key_row("O / L", "Fog Density -/+").into_any_element(),
                            key_row("N", "Toggle Fog").into_any_element(),
                        ],
                    ))
            )
        )
}
