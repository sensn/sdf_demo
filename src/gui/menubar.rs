//! 🟢 Menubar-Komponente: Verschachtelte Kaskadenmenüs via nativem .submenu()-Builder.

//! 🟢 Menubar-Komponente: Verschachtelte Kaskadenmenüs via nativem .submenu()-Builder.

use std::sync::{Arc, Mutex};

use wgpui_kit::{
    component::{
        // Holt die Menü-Typen aus der Dokumentation
        menu::{PopupMenu, PopupMenuItem, DropdownMenu},
        // 🟢 FIX: Button und ButtonVariants direkt aus dem button-Modul importieren!
        button::{Button, ButtonVariants},
    },
    div,
    prelude::*,
    px, rgb,
    App, IntoElement, Styled, Window,Anchor, 
};

wgpui_kit::actions!(tensor_menu, [ResetScene, ExitApp, ToggleAO, ToggleKeyLight]);

use crate::app_state::ApplicationState;
use super::GuiState;

// ... (Rest der menubar-Funktion bleibt exakt identisch)


/// Rendert die obere horizontale Menüleiste mit echten, nativen Kaskaden-Untermenüs auf Hover.
pub fn menubar(
    _gui: &mut GuiState,                         
    state: &Arc<Mutex<ApplicationState>>,
    window: &mut Window,                        
    cx: &mut App,                                
) -> impl IntoElement {
    
    let state_reset = state.clone();
    let state_ao = state.clone();
    let state_key = state.clone();

    // 🟢 GLOBAL HANDLER: Fängt die Menü-Aktionen zentral und sicher ab
    cx.on_action(move |_: &ResetScene, _| {
        if let Ok(mut s) = state_reset.lock() {
            s.cam_x = 0.0;
            s.cam_y = 0.0;
            s.cam_z = -5.0;
        }
    });

    cx.on_action(move |_: &ToggleAO, _| {
        if let Ok(mut s) = state_ao.lock() {
            s.enable_ao_mode = if s.enable_ao_mode == 1 { 0 } else { 1 };
        }
    });

    cx.on_action(move |_: &ToggleKeyLight, _| {
        if let Ok(mut s) = state_key.lock() {
            s.enable_key = if s.enable_key == 1 { 0 } else { 1 };
        }
    });

    cx.on_action(|_: &ExitApp, _| {
        std::process::exit(0);
    });

    div()
        .id("menubar")
        .w_auto() 
        .h(px(32.0))
        .flex()
        .flex_row()
        .items_center()
        .px_2()
        .rounded_br_md()
        .bg(rgb(0x161923)) 
        .border_b(px(1.0))
        .border_r(px(1.0))
        .border_color(rgb(0x222634))
        // --- HAUPTMENÜ: DATEI ---
        .child(
            Button::new("file_menu_trigger")
                .ghost()
                .compact()
                .label("Datei")
                // 🟢 EXAKTE DOCS REZEPTUR: .dropdown_menu mit Fluent-Kaskaden-Builder
                // 🟢 FIX: Nutze .dropdown_menu_with_anchor() mit Anchor::TopRight aus deinem Handbuch.
                // Das zwingt die Untermenüs (Szene & Optionen), sich bündig nach rechts hin zu öffnen!
                .dropdown_menu_with_anchor(Anchor::TopRight, move |menu, window, cx| {
                    menu
                        .min_w(px(200.0))
                        // 1. VERSCHACHTELTES UNTERMENÜ: Szene
                        .submenu("Szene", window, cx, |submenu, _win, _cx| {
                            submenu.menu("Zurücksetzen", Box::new(ResetScene))
                        })
                        
                        // 2. VERSCHACHTELTES UNTERMENÜ: Optionen
                        .submenu("Optionen", window, cx, |submenu, _win, _cx| {
                            submenu
                                .menu("Ambient Occlusion", Box::new(ToggleAO))
                                .menu("Key Light toggeln", Box::new(ToggleKeyLight))
                        })
                        
                        // Trennlinie für visuelle Ordnung
                        .separator()
                        
                        // Regulärer, flacher Sammeleintrag im Hauptzweig
                        .menu("Beenden", Box::new(ExitApp))
                })
        )
}
