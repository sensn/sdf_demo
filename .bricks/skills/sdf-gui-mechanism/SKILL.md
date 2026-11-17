---
name: sdf-gui-mechanism
description: Describes the complete GUI architecture (wgpui + wgpui-kit) of the SDF demo, enabling agents to add new tabs, sliders, buttons, menus, and wire them to ApplicationState.
---

# SDF Demo GUI Mechanism — Agent Reference

This skill documents the entire GUI stack so a coding agent can **add new tabs, controls, and data bindings** without breaking the existing patterns.

---

## 1. High-Level Flow

```
SurfaceExample (wgpui root)
 ├─ menubar()        → top horizontal native menus (Datei, Optionen, …)
 ├─ sidebar()        → vertical panel with TabBar (Env, Arch, Fold, Mat, Obj, SDF, Ctrl)
 │    └─ active tab  → renders only that tab’s content (scrollable Accordion columns)
 └─ input handling   → keys 1–6 select slot, WASD/Q/E camera, H toggles panel
```

All mutable UI state lives in **Entities** owned by `GuiState` (in `SurfaceExample`).  
`ApplicationState` (CPU render state) is an `Arc<Mutex<…>>` shared with the render thread.  
GUI ↔ State sync happens **every frame** in `GuiState::sync_sliders()`.

---

## 2. Core Types (src/gui/mod.rs)

### GuiTab
```rust
pub enum GuiTab {
    Environment, Architecture, Fold, Material, Object, Sdf, Controls
}
```
Add a new variant here, then update `sidebar.rs` TabBar + match.

### AccordionUiState
One `Vec<bool>` per tab group (index = accordion item).
```rust
pub struct AccordionUiState {
    pub env_open: Vec<bool>,
    pub arch_open: Vec<bool>,
    pub fold_open: Vec<bool>,
    pub mat_open: Vec<bool>,
    pub obj_open: Vec<bool>,
    pub sdf_open: Vec<bool>,   // ← new tabs add a field here
    pub ctrl_open: Vec<bool>,
}
```

### GuiState — the single source of all slider/button entities
Every interactive control is an `Entity<SliderState>` or `Entity<ButtonState>`.
```rust
pub struct GuiState {
    // Slider entities (one per controllable parameter)
    pub light_intensity_slider: Entity<SliderState>,
    // … 60+ more …

    // UI state entities
    pub tab_state: Entity<TabState>,
    pub accordion_state: Entity<AccordionUiState>,
}
```
**Rule:** every new parameter needs
1. a field in `GuiState`
2. an entity created in `GuiState::new()`
3. a subscription (`subscribe_slider`) in `GuiState::new()`
4. a sync call in `GuiState::sync_sliders()`

---

## 3. Slider Wiring (src/gui/widgets.rs)

### subscribe_slider
```rust
pub(super) fn subscribe_slider(
    cx: &mut Context<SurfaceExample>,
    slider: &Entity<SliderState>,
    state: &Arc<Mutex<ApplicationState>>,
    write: fn(&mut ApplicationState, f32),
) { … }
```
- Called **once** during `GuiState::new()`.
- Closure receives the new `f32` value and writes into `ApplicationState`.
- Use **per-slot pattern** for parameters that belong to the currently selected slot:
```rust
subscribe_slider(cx, &obj_pos_x_slider, state, |s, v| {
    let i = s.current_selected_slot;
    if i < s.slot_offsets_x.len() { s.slot_offsets_x[i] = v; }
});
```

### sync_one (State → Slider)
```rust
pub(super) fn sync_one(
    slider: &Entity<SliderState>,
    value: f32,
    window: &mut Window,
    cx: &mut App,
) { … }
```
- Called **every frame** in `GuiState::sync_sliders()`.
- Epsilon guard prevents feedback loops while dragging.
- For per-slot values, read the current selection index first:
```rust
let sel = state.current_selected_slot.min(state.slot_offsets_x.len().saturating_sub(1));
if sel < state.slot_offsets_x.len() {
    sync_one(&self.obj_pos_x_slider, state.slot_offsets_x[sel], window, cx);
}
```

---

## 4. Adding a New Slider (Checklist)

1. **ApplicationState** — add the field (plain `f32` or `Vec<f32>` for per-slot).
2. **GuiState** — add `pub xxx_slider: Entity<SliderState>`.
3. **GuiState::new()** — create entity with `cx.new(|_| SliderState::new().min(…).max(…).step(…).default_value(…))`.
4. **GuiState::new()** — subscribe:
   - Global scalar: `|s, v| s.field = v`
   - Per-slot: closure capturing `current_selected_…` index.
5. **GuiState::sync_sliders()** — call `sync_one(&self.xxx_slider, value, window, cx)`.
6. **Tab file** — add `slider_row("Label", &gui.xxx_slider, current_value).into_any_element()` inside the appropriate `acc_item`.

---

## 5. Tab Structure (src/gui/tabs/*.rs)

Each tab is a function `pub(in crate::gui) fn xxx_tab(gui, state, cx) -> impl IntoElement`.

Pattern:
```rust
pub(in crate::gui) fn env_tab(gui: &GuiState, state: &Arc<Mutex<ApplicationState>>, cx: &App) -> impl IntoElement {
    let s = state.lock().unwrap();
    let open = gui.accordion_state.read(cx).env_open.clone(); // Vec<bool>

    div().flex().flex_col().gap(px(8.0))
        .child(section("Environment"))
        .child(
            div().bg(rgb(ITEM_BG)).rounded_md().p_3() // fixed header
                .flex().flex_col().gap(px(6.0))
                .child(/* optional selector row */)
                .child(/* header label */)
        )
        .child(
            div().child(
                Accordion::new("env_acc")
                    .bordered(true).multiple(true)
                    .on_toggle_click(move |open_ix, _win, cx| {
                        gui.accordion_state.update(cx, |st, cx| {
                            st.env_open = (0..5).map(|i| open_ix.contains(&i)).collect();
                            cx.notify();
                        });
                    })
                    .item(acc_item("Light", open[0], vec![
                        slider_row("Intensity", &gui.light_intensity_slider, s.light_intensity).into_any_element(),
                        // …
                    ]))
                    // more items…
            )
        )
}
```

**Key points:**
- Lock state **once** at the top (`let s = state.lock().unwrap();`).
- Read `accordion_state` for open flags (`open = gui.accordion_state.read(cx).xxx_open.clone()`).
- Use `acc_item(title, open[index], vec![…rows…])` helper.
- Each row: `slider_row("Label", &gui.xxx_slider, current_value).into_any_element()`.

---

## 6. Slot Selector Buttons (Mat / Obj / SDF tabs)

```rust
let mut selector_row = div().flex().flex_row().flex_wrap().gap(px(4.0));
for i in 0..slot_count {
    let state_clone = state.clone();
    selector_row = selector_row.child(
        Button::new(("mat_slot", i))   // unique ElementId per button
            .label(format!("{}", i + 1))
            .compact()
            .selected(i == sel)
            .on_click(move |_ev, _win, _cx| {
                if let Ok(mut s) = state_clone.lock() {
                    s.current_selected_slot = i; // or current_selected_sdf_object
                }
            }),
    );
}
```
- Button ID tuple `("prefix", index)` guarantees uniqueness.
- `.selected(i == sel)` highlights the active slot.
- Click writes the index into the shared `ApplicationState`.

---

## 7. Native File Dialog (Load .obj)

Used in **SDF tab** and **menubar** for loading OBJ files.

```rust
Button::new("sdf_load")
    .label("Load .obj…")
    .compact()
    .on_click(move |_ev, _win, cx| {
        let load_state = state.clone();
        let slot = current_selected_sdf_object;
        cx.spawn(async move |_cx| {          // ← background task!
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
    })
```
- **Always** `cx.spawn(async move …).detach()` — file dialog blocks.
- `rfd::FileDialog` is cross-platform (xdg-portal on Linux, native on macOS/Windows).
- Write result back into `ApplicationState` under the lock.

---

## 8. Menubar Actions (src/gui/menubar.rs)

```rust
wgpui_kit::actions!(tensor_menu, [ResetScene, ExitApp, ToggleAO, ToggleKeyLight, LoadSdfObject]);
```
Each action is a zero-sized struct. Register handlers in `menubar()`:
```rust
cx.on_action(move |_: &LoadSdfObject, cx| {
    let state = state_load.clone();
    cx.spawn(async move |_cx| { … }).detach();
});
```
Menu definition uses the **fluent builder**:
```rust
.dropdown_menu_with_anchor(Anchor::TopRight, move |menu, window, cx| {
    menu.min_w(px(200.0))
        .submenu("Szene", window, cx, |sub| sub.menu("Zurücksetzen", Box::new(ResetScene)))
        .menu("SDF-Objekt laden…", Box::new(LoadSdfObject))
        .separator()
        .menu("Beenden", Box::new(ExitApp))
})
```

---

## 9. Colors & Constants (src/gui/mod.rs)

```rust
pub(super) const ITEM_BG: u32 = 0x161b28;
pub(super) const ACCENT: u32 = 0x00ffcc;
pub(super) const TEXT: u32 = 0xffffff;
pub(super) const MUTED: u32 = 0x8a92a6;
```
Use `rgb(ITEM_BG)`, `rgb(ACCENT)`, etc. in tab files.

---

## 10. Common Pitfalls

| Problem | Fix |
|---|---|
| Slider jumps while dragging | `sync_one` epsilon guard is missing or `subscribe_slider` writes back into the same slider. |
| Per-slot slider shows wrong value after slot switch | `sync_sliders` reads the **current** selection index every frame. |
| File dialog freezes UI | Forgot `cx.spawn(async …).detach()`. |
| New tab not shown | Add variant to `GuiTab`, update `sidebar.rs` TabBar + match, add `xxx_open` to `AccordionUiState`, add tab function to `tabs/mod.rs`. |
| Compile error: `ghost()` not found | Use `.with_variant(ButtonVariant::Ghost)` (import `ButtonVariants`). |

---

## 11. Quick-Start: Add a New Tab “Foobar”

1. `GuiTab::Foobar` in `mod.rs`.
2. `foobar_open: Vec<bool>` in `AccordionUiState`, init in `GuiState::new()`.
3. `tabs/foobar.rs` with `pub(in crate::gui) fn foobar_tab(…)`.
4. `tabs/mod.rs`: `mod foobar; pub(super) use foobar::foobar_tab;`.
5. `sidebar.rs`: TabBar child + match arm `5 => GuiTab::Foobar` + content arm.
6. Add slider entities + subscriptions + sync for any new parameters.
7. Done.

---

## 12. Files Overview

| File | Purpose |
|---|---|
| `src/gui/mod.rs` | `GuiState`, `GuiTab`, `AccordionUiState`, all slider entities, `new()`, `sync_sliders()` |
| `src/gui/widgets.rs` | `subscribe_slider`, `sync_one`, `slider_row`, `acc_item`, `section`, `toggle_row`, `info_row`, `key_row` |
| `src/gui/sidebar.rs` | TabBar + scroll container, dispatches to active tab |
| `src/gui/tabs/*.rs` | One file per tab (`env.rs`, `arch.rs`, `fold.rs`, `mat.rs`, `obj.rs`, `sdf.rs`, `ctrl.rs`) |
| `src/gui/menubar.rs` | Native menubar with cascading submenus, action handlers |
| `src/gui/tabs/mod.rs` | Re-exports all tab functions |

---

**That’s it.** Follow the patterns above and the GUI will stay consistent, performant (only active tab renders), and type-safe.