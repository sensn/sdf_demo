# REFACTORING_PLAN.md — sdf_demo Refactoring (Phase 2)

**Zweck dieses Dokuments:** Phase 1 (main.rs-Split) ist strukturell
abgeschlossen — `main.rs` ist von 706 auf 107 Zeilen geschrumpft und die
Module `app_state.rs`, `gpu.rs`, `render_loop.rs`, `view.rs` existieren.
Der Code kompilierte zum Zeitpunkt der Analyse **nicht** (7 Fehler,
2 Warnungen — alles API-Mismatches gegen wgpui 0.3.6 / wgpu 30 /
cubecl 0.11.0-pre.2). **Phase 2a ist inzwischen umgesetzt:** alle
Fehler sind behoben, `cargo check`, `cargo clippy` und
`cargo build --release` sind grün. Dieses Dokument beschreibt die
verbleibenden Phasen (kernel.rs-Split, gui.rs-Split, Aufräumarbeiten,
Fehlerbehandlung).

Alle API-Angaben sind gegen die tatsächlich gelockten Crate-Versionen
und den Compiler verifiziert (`~/.cargo/registry/src/.../wgpui-0.3.6`,
`wgpu-30.0.0`, `cubecl-wgpu-0.11.0-pre.2`).

---

## 1. Ist-Zustand (Stand: 2026-10-03)

### 1.1 Dateien & Zeilen

| Datei | Zeilen | Rolle | Status |
|---|---|---|---|
| `src/main.rs` | 107 | Bootstrap, Window, View-Wiring | ✅ Struktur fertig, 1 Fehler |
| `src/app_state.rs` | 171 | `ApplicationState` + Tensor-Packing | ✅ fertig |
| `src/gpu.rs` | 418 | `GpuPipeline` (cubecl-Init, Blit, resize) | ⚠️ 3 Fehler, 1 Warnung |
| `src/render_loop.rs` | 78 | Frame-Loop (Thread-Body) | ✅ fertig |
| `src/view.rs` | 137 | `SurfaceExample`-View (Layout, Keys, FPS) | ⚠️ 3 Fehler, 1 Warnung |
| `src/gui.rs` | 726 | Sidebar (Tabs, Accordions, Slider) | ✅ kompiliert, aber zu groß |
| `src/kernel.rs` | 758 | cubecl-Raymarch-Kernel | ✅ kompiliert, aber zu groß |
| `src/input.rs` | 80 | WASD-Kamerabewegung | ✅ fertig |
| `csg.rs` | 92 | CSG-Helfer (smin/ssub/sinter) | ⚠️ **tot** (nicht im Modul-Baum) |

### 1.2 Compile-Fehler vor Phase 2a (alle behoben ✅)

```
src/view.rs:100:14   E0599  no method `id`     for `wgpui_kit::Div`        → InteractiveElement-Import
src/view.rs:120:22   E0599  no method `child`  for `wgpui_kit::Div`        → ParentElement-Import
src/view.rs:128:30   E0599  no method `child`  for `wgpui_kit::Div`        → ParentElement-Import
src/gpu.rs:104:10    E0599  no method `ok_or`  for `Result`                 → map_err statt ok_or
src/gpu.rs:113:14    E0599  no method `map_err` for `WgpuDevice`             → init_device ist infallibel
src/gpu.rs:224:42    E0599  `BindGroup::create_placeholder` fehlt in wgpu 30 → Option<BindGroup>-Pattern
src/main.rs:83:20    E0599  no method `new` for `&mut App`                 → AppContext-Import
```

Warnungen: `view.rs:11` unused import `App`; `gpu.rs:276` unnötiges
`mut` — beide ebenfalls behoben. Zusätzlich getilgt: ein E0382
(`surface` moved) in main.rs, der durch die Trait-Fixes sichtbar
wurde, und ein collapsible-if in render_loop.rs (clippy).

### 1.3 Wurzelursachen (verifiziert gegen Crate-Quellen + Compiler)

1. **`view.rs`**: `.id()`/`.child()` sind Trait-Methoden — `InteractiveElement`
   (div.rs:609) bzw. `ParentElement` (element.rs:172). Beide fehlten im Import.
   `wgpui_kit` re-exportiert `pub use ::gpui::*` (lib.rs:82), also sind sie
   über `wgpui_kit::{InteractiveElement, ParentElement}` erreichbar.
2. **`gpu.rs:104`**: `request_adapter` liefert in wgpu 30 ein
   `Result<Adapter, RequestAdapterError>` → `ok_or` (Option-Methode) war
   falsch, korrekt ist `map_err(|_| GpuInitError::NoAdapter)`. Das Feld
   `apply_limit_buckets` existiert in wgpu 30 und ist Pflicht.
3. **`gpu.rs:113`**: `init_device(setup, options) -> WgpuDevice` ist **infallibel**
   (runtime.rs:244) — kein `Result`, also kein `map_err`. Direkte Zuweisung.
4. **`gpu.rs:224`**: `wgpu::BindGroup::create_placeholder` existiert in wgpu 30
   nicht. Feld zu `Option<BindGroup>` geändert; `rebuild_bind_group()` setzt
   es via `Some(...)`, `render_frame()` holt es via `as_ref().expect(...)`.
5. **`main.rs:83`**: `cx.new(...)` ist Trait-Methode von `AppContext`
   (wgpui.rs:139, app.rs:2232). Fix: `use wgpui_kit::AppContext;`.
6. **`main.rs` E0382**: `surface` wurde in den Render-Thread gemoved und
   danach erneut für `SurfaceExample` benutzt. Fix: `let render_surface =
   surface.clone();` vor dem `thread::spawn`.

---

## 2. Phase 2a — Compile-Fixes (UMGESETZT ✅)

### 2.1 `src/view.rs`

**Import-Fix (behebt 3 Fehler + 1 Warnung):**

```rust
use wgpui_kit::{
    div, px, rgb, wgpu_surface, Context, FocusHandle, IntoElement, InteractiveElement,
    ParentElement, Render, Styled, Window, WgpuSurfaceHandle,
};
```

- `InteractiveElement` → `.id()`, `.track_focus()`, `.on_key_down()`, `.on_key_up()`
- `ParentElement` → `.child()`
- `App`-Import entfernt (unused). `AppContext` wird hier nicht gebraucht
  (nur in main.rs) — dort ergänzt.

### 2.2 `src/main.rs`

**Trait-Import ergänzen (behebt 1 Fehler):**

```rust
use wgpui_kit::{component::theme::ThemeMode, App, AppContext, WindowOptions};
```

`cx.new(|cx| ...)` in Zeile 83 benötigt den `AppContext`-Trait in Scope
(wgpui-0.3.6: `src/wgpui.rs:139` definiert `fn new<T>(&mut self, ...)`,
implementiert für `App` in `src/app.rs:2232`).

### 2.3 `src/gpu.rs`

**Fix 1 — Adapter-Request (Zeile 98–104):** `Result`, nicht `Option`:

```rust
let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
    power_preference: wgpu::PowerPreference::HighPerformance,
    compatible_surface: None,
    force_fallback_adapter: false,
    apply_limit_buckets: false,
}))
.map_err(|_| GpuInitError::NoAdapter)?;
```

**Fix 2 — init_device ist infallibel (Zeile 112–114):**

```rust
let cubecl_device_id = init_device(wgpu_setup, RuntimeOptions::default());
let client: ComputeClient<WgpuRuntime> = WgpuRuntime::client(&cubecl_device_id);
```

`GpuInitError::CubeclInit` wird damit obsolet — Variante entfernen, da
`WgpuRuntime::client` ebenfalls infallibel ist.

**Fix 3 — BindGroup-Placeholder (Zeile 224):** Feld auf
`Option<wgpu::BindGroup>` ändern:

```rust
// Struct:
bind_group: Option<wgpu::BindGroup>,

// new():
bind_group: None,

// rebuild_bind_group():
self.bind_group = Some(self.device.create_bind_group(...));

// render_frame() — Aufrufstelle:
let bind_group = self.bind_group.as_ref()
    .expect("bind_group built in GpuPipeline::new");
```

**Fix 4 — Warnung gpu.rs:276:** `let mut write = ...` → `let write = ...`
(closure captured nur `&self`, kein `mut` nötig).

### 2.4 Akzeptanzkriterien Phase 2a — ERFÜLLT ✅

- [x] `cargo check` — 0 Fehler, 0 Warnungen
- [x] `cargo clippy` — 0 Warnungen
- [x] `cargo build --release` — erfolgreich (cubecl-Codegen läuft durch)
- [ ] Manueller Smoke-Test: Fenster öffnet, Raymarch rendert, WASD,
      Slider, Resize, FPS-Counter, Tasten 1–5/Q/E/F/R/O/L/N
      (**ausstehend — nur vom User am Display prüfbar**)

---

## 3. Phase 2b — Aufräumarbeiten (nach grünem Build)

### 3.1 Tote Datei `csg.rs`

`csg.rs` (92 Zeilen, im Repo-Root!) ist **nicht** im Modul-Baum
(`main.rs` deklariert kein `mod csg;`). Sein Inhalt (`smin`, `ssub`,
`sinter`) ist bereits 1:1 in `kernel.rs` vorhanden. Aktionen:

1. `git rm csg.rs` — **vorher mit User abklären** (Datei-Löschung).
2. `src/unused/` (13 Kernel-Varianten, ~4.500 Zeilen): behalten als
   Referenz-Implementierungen (bereits nicht im Build), in README
   erwähnen — oder löschen (User-Entscheidung).

### 3.2 `GpuInitError` vereinfachen

Nach Fix 2 bleibt nur `NoAdapter` übrig → tote `CubeclInit`-Variante
streichen. Keine API-Änderung.

### 3.3 Magische Zahlen konsolidieren

`gpu.rs` definiert bereits Konstanten (`SLOTS_BYTES` etc.). Ergänzen:

```rust
/// Raymarch-Output: RGB f32 → 3 × 4 Bytes pro Pixel.
const RGB_F32_BYTES: usize = 3 * 4;
fn output_byte_size(w: u32, h: u32) -> usize {
    (w as usize) * (h as usize) * RGB_F32_BYTES
}
```

(prüfen, ob `3*4` noch literal vorkommt — teils schon als Funktion da).

### 3.4 `render_loop.rs` — Shutdown-Signal dokumentieren

Die Loop bricht bei `fps_tx.send()`-Fehler ab (Receiver im View stirbt
mit dem Fenster). Modul-Header um einen Satz zum Shutdown-Pattern
ergänzen.

---

## 4. Phase 3 — kernel.rs-Split (758 → ~4 Dateien)

`kernel.rs` mischt vier Ebenen: Vec3-Mathematik, SDF-Primitiven,
Lighting (AO/Shadows), Raymarch-Loop. Zielstruktur:

```
src/kernel/
├── mod.rs          (~60)   Re-Exports, CubeDim/CubeCount-Doku
├── math.rs         (~120)  Vec3, smin, smin_material, dot/normalize/length
├── sdf.rs          (~280)  scene_sdf, scene_sdf_normal,
│                            evaluate_dynamic_{torus,gyroid,crystal}
└── lighting.rs     (~180)  calculate_ao, calculate_soft_shadow
```

**Wichtig:** cubecl-`#[cube]`-Funktionen sind monomorphisiert — ein Split
in Submodule ist rein organisatorisch (keine Generik über Modulgrenzen,
`use crate::kernel::math::*` funktioniert wie heute `crate::kernel::*`).
Der `#[cube(launch)]`-Kernel bleibt in `mod.rs`; die Launch-Parameter-
Reihenfolge **darf nicht geändert werden** (gpu.rs ruft mit fester
Reihenfolge auf).

### 4.1 Vorgehen

1. `mkdir src/kernel && git mv src/kernel.rs src/kernel/mod.rs`
2. `math.rs` extrahieren: `Vec3`, `Vec3Expand`, `smin`, `smin_material`
   (Zeilen 9–105 im heutigen kernel.rs)
3. `sdf.rs`: `evaluate_dynamic_*` (107–160), `scene_sdf` (162–396),
   `scene_sdf_normal` (398–419)
4. `lighting.rs`: `calculate_soft_shadow` (421–464), `calculate_ao` (466–505)
5. `mod.rs`: `raymarch_sdf_kernel` (507–758) + `pub use` der Submodule
6. `gpu.rs`/`render_loop.rs`-Imports prüfen (via `crate::kernel::`
   unverändert, wenn mod.rs re-exportiert)

### 4.2 Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| cubecl-Makro braucht alle `#[cube]`-Fns in einem Scope | Verifikation per `cargo build --release` (Codegen läuft dort) |
| Launch-Parameter-Reihenfolge bricht | gpu.rs-Aufruf nicht anfassen; nur Moves, keine Signatur-Änderungen |
| `Vec3`-Trait-Impls (`CubeType`) über Modulgrenzen | `impl Vec3` bleibt in math.rs; `pub use math::*` in mod.rs |

---

## 5. Phase 4 — gui.rs-Split (UMGESETZT ✅)

`gui.rs` (726 Zeilen) enthielt `GuiState` (Slider-Entities, Tab-/
Accordion-State), Sidebar-Layout und **vier** Tab-Renderer. Ergebnis:

```
src/gui/
├── mod.rs        (254)  GuiState, new(), sync_sliders, Farben, UI-Entities
├── sidebar.rs     (96)  sidebar() — Layout, TabBar, Scroll-Container, Dispatch
├── widgets.rs    (190)  slider_row, acc_item, section, toggle_row,
│                        info_row, key_row, subscribe_slider, sync_one
└── tabs/
    ├── mod.rs      (16)  Re-Exports
    ├── env.rs      (96)  env_tab() + fog_switch_row (Tensor 4)
    ├── arch.rs     (62)  arch_tab() (Tensor 5)
    ├── fold.rs     (47)  fold_tab() (Tensor 6)
    └── ctrl.rs    (110)  ctrl_tab() — Kamera-Info, Schalter, Tasten-Hilfe
```

Hinweis: der Plan sah 3 Tabs vor, der Code hat 4 (Env/Arch/Fold/**Ctrl**)
— `tabs/ctrl.rs` wurde ergänzt. `view.rs` ruft weiterhin
`gui::sidebar(...)` mit unveränderter Signatur.

### 5.1 Vorgehen (wie umgesetzt)

1. `widgets.rs` extrahiert (`slider_row`, `acc_item`, `section`,
   `toggle_row`, `info_row`, `key_row`, `subscribe_slider`, `sync_one` —
   reine Funktionen ohne GuiState-Abhängigkeit)
2. `GuiState` + `new()` + `sync_sliders()` in `mod.rs` behalten
   (Entity-Erstellung braucht den `AppContext`-Trait — Import ergänzt)
3. Tab-Renderer als freie Funktionen in eigene Dateien; Signaturen
   unverändert übernommen (`env_tab(&GuiState, &Arc<Mutex<...>>, &App)`
   usw., `ctrl_tab(state)` ohne GuiState)
4. `view.rs` ruft `gui::sidebar(...)` — Signatur unverändert

Sichtbarkeits-Setup, das sich bewährt hat: Widgets als `pub(super)` in
`gui::widgets`, Tabs als `pub(in crate::gui)` mit `pub(super) use` in
`tabs/mod.rs` — so bleibt alles Gui-intern.

### 5.2 Risiken (alle eingetreten & behoben)

| Risiko | Gegenmaßnahme | Ergebnis |
|---|---|---|
| Slider-Subscriptions halten Entity-Handles | `GuiState`-Felder nicht umbenennen; nur Moves | ✅ 1:1 übernommen |
| `Context<SurfaceExample>`-Typ in Submodulen | `crate::SurfaceExample` via `crate::`-Pfad | ✅ nicht nötig — Tabs brauchen nur `&App` |
| Kit-APIs abweichend (`AccordionItem::new(0 args)`, `TabBar::new(id)`, `acc_item` ist Closure) | Original aus `git show HEAD:src/gui.rs` rekonstruiert statt geraten | ✅ behoben |
| `cx.new()` in Submodulen | `AppContext`-Trait-Import in `gui/mod.rs` | ✅ behoben |

**Verifikation:** `cargo check` ✅ 0 Fehler/Warnungen · `cargo clippy` ✅ 0 ·
`cargo build --release` ✅ (43 s). Manueller Smoke-Test (Tabs klicken,
Slider ziehen, Schalter togglen) steht beim User aus.

---

## 6. Phase 5 — Fehlerbehandlung & Robustheit (optional)

1. **`get_resource().unwrap()` in gpu.rs** (rebuild_bind_group,
   write_state_buffers): Handles sind Pipeline-owned und immer valide —
   `unwrap` ist ok, aber `.expect("cubecl handle")` für Diagnose.
2. **`std::process::exit(1)` in main.rs** bei GPU-Init-Fehler. Besser:
   Fenster trotzdem öffnen mit Fehler-Overlay („kein Vulkan-Adapter")
   statt hartem Exit — vertagt, da UX-Entscheidung.
3. **`pollster`-Abhängigkeit:** nur für `block_on(request_adapter)`.
   Mini-Dep, ok; alternativ `futures::executor::block_on`, falls
   `futures` schon im Tree ist (prüfen).
4. **`input.rs`-Reste:** unbenutzte `crossbeam_channel`-Imports prüfen
   und entfernen, falls vorhanden.

---

## 7. Implementierungs-Reihenfolge (Gesamtübersicht)

| Schritt | Inhalt | Abhängigkeit |
|---|---|---|
| 1 | **Phase 2a:** Compile-Fixes (view.rs, main.rs, gpu.rs) | keine |
| 2 | Smoke-Test (Fenster, WASD, Slider, Resize, FPS) | 1 |
| 3 | **Phase 2b:** csg.rs-Entscheidung, GpuInitError, Konstanten | 2 |
| 4 | **Phase 3:** kernel.rs-Split | 2 |
| 5 | **Phase 5:** Fehlerbehandlung (optional) | 2 |
| 6 | **Phase 4:** gui.rs-Split | ✅ erledigt |
| 7 | Finale Verifikation: clippy, release-build, Smoke-Test | 3–6 |

---

## 8. Akzeptanzkriterien (final)

- [ ] `cargo check` + `cargo clippy` ohne Fehler und ohne neue Warnungen
- [ ] `cargo build --release` erfolgreich (cubecl-Codegen + Shader)
- [ ] Visuell identisches Rendering (gleicher Kernel, gleiche Parameter)
- [ ] WASD + alle Slider + Tasten 1–5, Q/E, F/R, O/L, N funktionieren
- [ ] Resize re-allokiert output (kein Stretching/Artefakte)
- [ ] FPS-Counter ~60
- [ ] Sauberes Beenden ohne Panic beim Fenster-Schließen
- [ ] Jede Datei < 300 Zeilen (Ausnahme: kernel/sdf.rs ≤ 300)
- [ ] `csg.rs` entfernt oder bewusst als Referenz dokumentiert

---

## 9. Verifizierte API-Referenz (für zukünftige Sessions)

| API | Version | Signatur / Quelle |
|---|---|---|
| `AppContext::new` | wgpui 0.3.6 | `fn new<T>(&mut self, build: impl FnOnce(&mut Context<T>) -> T) -> Entity<T>` — wgpui.rs:139, app.rs:2232 |
| `InteractiveElement::id` | wgpui 0.3.6 | `fn id(self, id: impl Into<ElementId>) -> Stateful<Self>` — div.rs:609 |
| `ParentElement::child` | wgpui 0.3.6 | `fn child(self, child: impl IntoElement) -> Self` — element.rs:172 |
| `wgpui_kit` re-exports | 0.6.1 | `pub use ::gpui::*` — lib.rs:82 (AppContext, InteractiveElement, ParentElement erreichbar) |
| `init_device` | cubecl-wgpu 0.11.0-pre.2 | `pub fn init_device(setup: WgpuSetup, options: RuntimeOptions) -> WgpuDevice` — runtime.rs:244 (**infallibel**) |
| `request_adapter` | wgpu 30.0.0 | liefert `Result<Adapter, RequestAdapterError>`; `RequestAdapterOptions`-Felder: `power_preference`, `compatible_surface`, `force_fallback_adapter`, `apply_limit_buckets` |
| `BindGroup::create_placeholder` | wgpu 30.0.0 | **existiert nicht** → `Option<BindGroup>`-Pattern nutzen |
| `Handle` | cubecl-runtime | `cubecl_runtime::server::Handle` (nicht über `cubecl::client::ComputeClient` re-exportiert) |

---

*Phase 1 (Ist-Analyse des 706-Zeilen-Monolithen, Soll-Architektur,
Startup-Sequenz, F1–F10-Fixes) ist historisch abgeschlossen — Details
im Git-Verlauf dieser Datei.*
