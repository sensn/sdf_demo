//! 🟢 Sidebar-Layout: Panel, TabBar, Scroll-Container, Tab-Dispatch.
//!
//! Nur der aktive Tab rendert — das hält die Element-Baum-Kosten klein
//! und löst das Platzproblem (Inhalt scrollbar statt überlaufend).

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::{
        tab::{Tab, TabBar},
        Selectable,
    },
    div,
    prelude::*,
    px, rgb,
    App, IntoElement, Styled, Window,
};

use crate::app_state::ApplicationState;

use super::{tabs, GuiState, GuiTab};

/// Komplette Sidebar: TabBar + scrollbarer Inhalt des aktiven Tabs.
pub fn sidebar(
    gui: &mut GuiState,
    state: &Arc<Mutex<ApplicationState>>,
    window: &mut Window,
    cx: &mut App,
) -> impl IntoElement {
    // State -> Slider synchronisieren (vor dem Rendern, damit die Werte stimmen)
    {
        let s = state.lock().unwrap();
        gui.sync_sliders(&s, window, cx);
    }

    let active = gui.tab_state.read(cx).active;

    // TabBar: Klick schreibt in TabState-Entity
    let tab_state = gui.tab_state.clone();
    let tab_bar = TabBar::new("tensor_tabs")
        .child(
            Tab::new()
                .label("Env")
                .selected(active == GuiTab::Environment),
        )
        .child(
            Tab::new()
                .label("Arch")
                .selected(active == GuiTab::Architecture),
        )
        .child(Tab::new().label("Fold").selected(active == GuiTab::Fold))
        .child(Tab::new().label("Mat").selected(active == GuiTab::Material))
        .child(Tab::new().label("Obj").selected(active == GuiTab::Object))
        .child(
            Tab::new()
                .label("Ctrl")
                .selected(active == GuiTab::Controls),
        )
        .on_click(move |ix: &usize, _win, cx| {
            let tab = match ix {
                0 => GuiTab::Environment,
                1 => GuiTab::Architecture,
                2 => GuiTab::Fold,
                3 => GuiTab::Material,
                4 => GuiTab::Object, // 🟢 NEU: Obj-Tab
                _ => GuiTab::Controls,
            };
            tab_state.update(cx, |ts, cx| {
                ts.active = tab;
                cx.notify();
            });
        });

    // Inhalt des aktiven Tabs
    let content = match active {
        GuiTab::Environment => tabs::env_tab(gui, state, cx).into_any_element(),
        GuiTab::Architecture => tabs::arch_tab(gui, state, cx).into_any_element(),
        GuiTab::Fold => tabs::fold_tab(gui, state, cx).into_any_element(),
        GuiTab::Material => tabs::mat_tab(gui, state, cx).into_any_element(),
        GuiTab::Object => tabs::obj_tab(gui, state, cx).into_any_element(), // 🟢 NEU
        GuiTab::Controls => tabs::ctrl_tab(state).into_any_element(),
    };

    div()
        .id("sidebar")
        .w(px(300.0))
        .h_full()
        .flex()
        .flex_col()
        .gap(px(8.0))
        .p_2()
        .bg(rgb(0x10121a))
        .child(tab_bar)
        .child(
            div()
                .id("sidebar_scroll")
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .child(content),
        )
}
