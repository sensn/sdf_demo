//! 🟢 Ctrl-Tab: Original-Steuerung — Kamera-Info, Render-Schalter
//! (AO / Lichter / Schatten), Tasten-Hilfe.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    div,
    prelude::*,
    px, rgb,
    IntoElement, Styled,
};

use crate::app_state::ApplicationState;

use super::super::widgets::{info_row, key_row, section, toggle_row};
use super::super::ITEM_BG;

/// Ctrl-Tab: Kamera-Info, Render-Schalter (AO/Lichter/Schatten), Tasten-Hilfe.
pub(in crate::gui) fn ctrl_tab(state: &Arc<Mutex<ApplicationState>>) -> impl IntoElement {
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
