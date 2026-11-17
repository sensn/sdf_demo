//! 🟢 SDF-Tab (SDF-Objekt-Tensor): Slot-Auswahl + Load-Button + Position / Size / Rotation / Material
//! für den gewählten SDF-Objekt-Slot (wie Mat/Obj, aber pro Grid-Slot).

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::{
        accordion::Accordion,
        button::{Button, ButtonVariants},
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

/// SDF-Tab: Slot-Auswahl (Buttons) + Load/Clear-Buttons + Accordion-Transformationen.
pub(in crate::gui) fn sdf_tab(
    gui: &GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    cx: &App,
) -> impl IntoElement {
    let s = state.lock().unwrap();
    let sel = s.current_selected_sdf_object.min(s.sdf_center_x.len().saturating_sub(1));
    let slot_count = s.sdf_paths.len();

    // UI-Zustand für die Accordions
    let open = gui.accordion_state.read(cx).sdf_open.clone();

    // Aktuelle Werte des gewählten Slots
    let pos_x = s.sdf_center_x[sel];
    let pos_y = s.sdf_center_y[sel];
    let pos_z = s.sdf_center_z[sel];
    let size = s.sdf_sizes[sel];
    let rot_x = s.sdf_rot_x[sel];
    let rot_y = s.sdf_rot_y[sel];
    let rot_z = s.sdf_rot_z[sel];
    let r = s.sdf_r[sel];
    let g = s.sdf_g[sel];
    let b = s.sdf_b[sel];
    let rough = s.sdf_roughness[sel];
    let metal = s.sdf_metallic[sel];
    let emiss = s.sdf_emissive[sel];
    let spec = s.sdf_specular[sel];
    let active = s.sdf_active[sel];
    let path = s.sdf_paths.get(sel).cloned().unwrap_or_default();

    // Slot-Name für die Kopfzeile
    let slot_name = format!(
        "SDF-Objekt {} — {}",
        sel + 1,
        if active { "Active" } else { "Inactive" }
    );

    // Slot-Selector (Buttons 1..4)
    let mut selector_row = div().flex().flex_row().flex_wrap().gap(px(4.0));
    for i in 0..slot_count {
        let state_clone = state.clone();
        selector_row = selector_row.child(
            Button::new(("sdf_slot", i))
                .label(format!("{}", i + 1))
                .compact()
                .selected(i == sel)
                .on_click(move |_ev, _win, _cx| {
                    if let Ok(mut s) = state_clone.lock() {
                        s.current_selected_sdf_object = i;
                    }
                }),
        );
    }

    // Load-Button (öffnet nativen Datei-Dialog im Hintergrund-Thread)
    let load_state = state.clone();
    let load_slot = sel;
    let load_button = Button::new("sdf_load")
        .label("Load .obj…")
        .compact()
        .on_click(move |_ev, _win, cx| {
            let load_state = load_state.clone();
            let slot = load_slot;
            cx.spawn(async move |_cx| {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("OBJ Mesh", &["obj"])
                    .pick_file()
                {
                    let path_str = path.display().to_string();
                    if let Ok(mut s) = load_state.lock() {
                        if slot < s.sdf_paths.len() {
                            s.sdf_paths[slot] = path_str;
                            s.sdf_active[slot] = true;
                        }
                    }
                }
            }).detach();
        });

    // Clear-Button (deaktiviert den Slot)
    let clear_state = state.clone();
    let clear_slot = sel;
    let clear_button = Button::new("sdf_clear")
        .label("Clear")
        .compact()
        .with_variant(wgpui_kit::component::button::ButtonVariant::Ghost)
        .on_click(move |_ev, _win, _cx| {
            if let Ok(mut s) = clear_state.lock() {
                if clear_slot < s.sdf_active.len() {
                    s.sdf_active[clear_slot] = false;
                    s.sdf_paths[clear_slot].clear();
                }
            }
        });

    let acc_state = gui.accordion_state.clone();

    div()
        .flex()
        .flex_col()
        .gap(px(8.0))
        .child(section("SDF Object Slot"))
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
                        .flex()
                        .flex_row()
                        .gap(px(4.0))
                        .child(load_button)
                        .child(clear_button),
                )
                .child(
                    div()
                        .text_color(rgb(MUTED))
                        .text_xs()
                        .child(slot_name),
                )
                .child(
                    div()
                        .text_color(rgb(MUTED))
                        .text_xs()
                        .child(format!("Pfad: {}", if path.is_empty() { "(leer)" } else { &path })),
                ),
        )
        .child(
            div().child(
                Accordion::new("sdf_acc")
                    .bordered(true)
                    .multiple(true)
                    .on_toggle_click(move |open_ix: &[usize], _win, cx| {
                        acc_state.update(cx, |st, cx| {
                            st.sdf_open = (0..5).map(|i| open_ix.contains(&i)).collect();
                            cx.notify();
                        });
                    })
                    .item(acc_item(
                        "Position",
                        open[0],
                        vec![
                            slider_row("Pos X", &gui.sdf_center_x_slider, pos_x).into_any_element(),
                            slider_row("Pos Y", &gui.sdf_center_y_slider, pos_y).into_any_element(),
                            slider_row("Pos Z", &gui.sdf_center_z_slider, pos_z).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Rotation (Euler, rad)",
                        open[1],
                        vec![
                            slider_row("Rot X", &gui.sdf_rot_x_slider, rot_x).into_any_element(),
                            slider_row("Rot Y", &gui.sdf_rot_y_slider, rot_y).into_any_element(),
                            slider_row("Rot Z", &gui.sdf_rot_z_slider, rot_z).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Size & Grid",
                        open[2],
                        vec![
                            slider_row("Size", &gui.sdf_size_slider, size).into_any_element(),
                            slider_row_fmt("Grid Res", &gui.mesh_scale_factor_slider, 64.0, 0).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Material (Albedo + PBR)",
                        open[3],
                        vec![
                            slider_row("Albedo R", &gui.sdf_r_slider, r).into_any_element(),
                            slider_row("Albedo G", &gui.sdf_g_slider, g).into_any_element(),
                            slider_row("Albedo B", &gui.sdf_b_slider, b).into_any_element(),
                            slider_row("Roughness", &gui.sdf_roughness_slider, rough).into_any_element(),
                            slider_row("Metallic", &gui.sdf_metallic_slider, metal).into_any_element(),
                            slider_row("Emissive", &gui.sdf_emissive_slider, emiss).into_any_element(),
                            slider_row("Specular", &gui.sdf_specular_slider, spec).into_any_element(),
                        ],
                    ))
                    .item(acc_item(
                        "Global Settings",
                        open[4],
                        vec![
                            slider_row("Mesh Scale Factor", &gui.mesh_scale_factor_slider, s.mesh_scale_factor).into_any_element(),
                            slider_row("Bounding Sphere Offset", &gui.bounding_sphere_radius_offset_slider, s.bounding_sphere_radius_offset).into_any_element(),
                        ],
                    ))
            )
        )
}