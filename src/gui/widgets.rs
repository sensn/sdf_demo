//! 🟢 GUI-Widgets: wiederverwendbare Zeilen-Helfer für alle Tabs.
//!
//! Enthält die Layout-Bausteine (`slider_row`, `acc_item`, `section`,
//! `toggle_row`, `info_row`, `key_row`) sowie die Slider-Verdrahtung
//! (`subscribe_slider`, `sync_one`). Alles reine Funktionen ohne
//! `GuiState`-Abhängigkeit.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::{
        accordion::AccordionItem,
        slider::{Slider, SliderEvent, SliderState},
        switch::Switch,
    },
    div,
    prelude::*,
    px, rgb,
    AnyElement, App, Context, Entity, FontWeight, Styled, Window,
};

use crate::app_state::ApplicationState;

use super::{ACCENT, ITEM_BG, MUTED, TEXT};

// =========================================================================
// Slider-Verdrahtung
// =========================================================================

/// Slider -> ApplicationState: einmalige Subscription pro Slider.
/// Handler-Signatur von Context::subscribe: (&mut T, Entity<T2>, &Evt, &mut Context<T>).
pub(super) fn subscribe_slider(
    cx: &mut Context<crate::SurfaceExample>,
    slider: &Entity<SliderState>,
    state: &Arc<Mutex<ApplicationState>>,
    write: fn(&mut ApplicationState, f32),
) {
    let state = state.clone();
    cx.subscribe(slider, move |_, _, event: &SliderEvent, _cx| {
        if let SliderEvent::Change(value) = event {
            if let Ok(mut s) = state.lock() {
                write(&mut s, value.start());
            }
        }
    })
    .detach();
}

/// State -> Slider: Wert nur schreiben, wenn er sich wirklich unterscheidet
/// (Epsilon-Guard gegen Float-Rückkopplung; Drag-No-Op siehe Modul-Doku).
pub(super) fn sync_one(
    slider: &Entity<SliderState>,
    value: f32,
    window: &mut Window,
    cx: &mut App,
) {
    let current = slider.read(cx).value().start();
    if (current - value).abs() > f32::EPSILON {
        slider.update(cx, |state, cx| {
            state.set_value(value, window, cx);
        });
    }
}

// =========================================================================
// Layout-Bausteine
// =========================================================================

/// Eine Slider-Zeile: Label + Live-Wert + horizontaler Slider.
pub(super) fn slider_row(
    label: &str,
    slider: &Entity<SliderState>,
    value: f32,
) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap(px(2.0))
        .child(
            div()
                .flex()
                .flex_row()
                .justify_between()
                .child(
                    div()
                        .text_color(rgb(MUTED))
                        .text_xs()
                        .child(label.to_string()),
                )
                .child(
                    div()
                        .text_color(rgb(ACCENT))
                        .text_xs()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(format!("{value:.2}")),
                ),
        )
        .child(Slider::new(slider).horizontal())
}

/// Accordion-Item-Bauplan: Titel + offener Zustand + Slider-Zeilen.
/// Titel wird geowned (String) und die Zeilen sind als AnyElement typgelöscht
/// — die Closure ist damit 'static und fängt keine anonymen Lifetimes.
pub(super) fn acc_item(
    title: &str,
    open: bool,
    rows: Vec<AnyElement>,
) -> impl FnOnce(AccordionItem) -> AccordionItem + 'static {
    let title = title.to_string();
    let mut content = div().flex().flex_col().gap(px(6.0)).p_2();
    for row in rows {
        content = content.child(row);
    }
    move |item: AccordionItem| {
        item.title(
                div()
                    .text_color(rgb(ACCENT))
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(title),
            )
            .open(open)
            .bg(rgb(ITEM_BG))
            .rounded_md()
            .child(content)
    }
}

/// Sektions-Überschrift (wie "Camera", "Rendering", "Key Bindings").
pub(super) fn section(title: &str) -> impl IntoElement {
    div()
        .text_color(rgb(ACCENT))
        .text_sm()
        .font_weight(FontWeight::SEMIBOLD)
        .child(title.to_string())
}

/// Schalter-Zeile: Label + Switch, schreibt 0/1 in ein ApplicationState-Feld.
pub(super) fn toggle_row(
    id: &'static str,
    label: &str,
    state: &Arc<Mutex<ApplicationState>>,
    checked: bool,
    write: fn(&mut ApplicationState, bool),
) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .justify_between()
        .items_center()
        .child(div().text_color(rgb(TEXT)).text_sm().child(label.to_string()))
        .child(
            Switch::new(id)
                .checked(checked)
                .on_click({
                    let state = state.clone();
                    move |checked: &bool, _win, _cx| {
                        if let Ok(mut s) = state.lock() {
                            write(&mut s, *checked);
                        }
                    }
                }),
        )
}

/// Info-Zeile: Label links, Live-Wert rechts (z. B. Kameraposition).
pub(super) fn info_row(label: &str, value: String) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .justify_between()
        .child(div().text_color(rgb(MUTED)).text_xs().child(label.to_string()))
        .child(
            div()
                .text_color(rgb(ACCENT))
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .child(value),
        )
}

/// Tasten-Hilfe: eine Zeile "• Taste : Beschreibung".
pub(super) fn key_row(key: &str, desc: &str) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .gap(px(6.0))
        .child(div().text_color(rgb(ACCENT)).text_xs().child(format!("• {key}")))
        .child(div().text_color(rgb(MUTED)).text_xs().child(desc.to_string()))
}
