//! 🟢 Obj-Tab (Objekt-Tensor): Slot-Auswahl + Position / Scale / Rotation
//! für den gewählten Slot.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::{
        accordion::Accordion,
        button::Button,
        Selectable,
    },
    div,
    prelude::*,
    px, rgb,
    App, IntoElement, Styled,
};

use crate::app_state::ApplicationState;

use super::super::widgets::{acc_item, section, slider_row};
use super::super::{GuiState, ITEM_BG, MUTED};

/// Obj-Tab: Slot-Auswahl (Buttons) + Accordion-Transformationen.
pub(in crate::gui) fn obj_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    cx: &App,
) -> impl IntoElement {
    let s = state.lock().unwrap();
    let sel = s.current_selected_slot.min(s.slot_offsets_x.len().saturating_sub(1));

    // Nutzt das bewährte env_open Array als Blaupause für die Index-Zuweisung
let open = gui.accordion_state.read(cx).obj_open.clone(); // 🟢 Nutzt obj_open

    // Aktuelle Objekt-Werte des gewählten Slots
    let pos_x = s.slot_offsets_x[sel];
    let pos_y = s.slot_offsets_y[sel];
    let pos_z = s.slot_offsets_z[sel];
    let scale = s.slot_sizes[sel];
    let rot_x = s.slot_rot_x[sel];
    let rot_y = s.slot_rot_y[sel];
    let rot_z = s.slot_rot_z[sel];

    // Slot-Name + Typ für die Kopfzeile
    let slot_name = format!("Slot {} — {}", sel + 1, slot_type_name(s.slot_types[sel]));

    // Slot-Selector
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

    let acc_state = gui.accordion_state.clone();

    // 🟢 Blaupause angewendet: div Wrapper -> Header fixiert -> Accordion umschlossen
    div()
        .flex()
        .flex_col()
        .gap(px(8.0))
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
        .child(
            div().child(
                Accordion::new("obj_acc")
                    .bordered(true)
                    .multiple(true)
                    .on_toggle_click(move |open_ix: &[usize], _win, cx| {
                       acc_state.update(cx, |st, cx| {
            // 🟢 FIX: Schreibt jetzt korrekt in obj_open statt env_open!
            st.obj_open = (0..5).map(|i| open_ix.contains(&i)).collect();
            cx.notify();
        });
                    })
                    .item(acc_item(
                        "Position",
                        open[0],
                        vec![
                            slider_row("Pos X", &gui.obj_pos_x_slider, pos_x).into_any_element(),
                            slider_row("Pos Y", &gui.obj_pos_y_slider, pos_y).into_any_element(),
                            slider_row("Pos Z", &gui.obj_pos_z_slider, pos_z).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Scale",
                        open[1],
                        vec![
                            slider_row("Scale", &gui.obj_scale_slider, scale).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Rotation (Euler, rad)",
                        open[2],
                        vec![
                            slider_row("Rot X", &gui.obj_rot_x_slider, rot_x).into_any_element(),
                            slider_row("Rot Y", &gui.obj_rot_y_slider, rot_y).into_any_element(),
                            slider_row("Rot Z", &gui.obj_rot_z_slider, rot_z).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "SDF Grid Space",
                        open[3],
                        vec![
                            slider_row("SDF Grid Size", &gui.sdf_grid_size_slider, s.sdf_grid_size).into_any_element(),
                            slider_row("Mesh Scale Factor", &gui.mesh_scale_factor_slider, s.mesh_scale_factor).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Transformations & Bounds",
                        open[4],
                        vec![
                            slider_row("Statue Center X", &gui.statue_center_x_slider, s.statue_center_x).into_any_element(),
                            slider_row("Statue Center Y", &gui.statue_center_y_slider, s.statue_center_y).into_any_element(),
                            slider_row("Statue Center Z", &gui.statue_center_z_slider, s.statue_center_z).into_any_element(),
                            slider_row("Bounding Sphere Offset", &gui.bounding_sphere_radius_offset_slider, s.bounding_sphere_radius_offset).into_any_element(),
                        ],
                    ))
            )
        )
}

fn slot_type_name(t: f32) -> &'static str {
    match t as u32 {
        1 => "Crystal",
        2 => "Gyroid",
        3 => "Torus",
        _ => "Inactive",
    }
}
