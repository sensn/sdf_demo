# FEATURE PLAN: SDF-Objekt-Tensor (Statue aus dem Mega-Scene-Tensor ausgekoppelt)

**Status:** In Arbeit · **Konvention:** Schritt für Schritt, nach jedem Schritt `cargo check`,
Fortschritt in `IMPLEMENTATION_SUMMARY.md` (Root).

---

## 1. Ziel

Die Statue (gebackenes 64³-SDF-Grid) lebt heute **im** Mega-Scene-Tensor
(`[slots|materials|rotations|sdf_grid]`, an Index 1500 angehängt) und ist im Kernel
hart verdrahtet: genau EIN Grid, festes Zentrum/Größe/PBR (`statue_center`,
`sdf_grid_size`, `statue_metallic` in `RenderSettings`).

Ziel: Ein **eigener SDF-Objekt-Tensor** mit **N Grid-Slots** (Analogie zum
Slot-Mechanismus der dynamischen Objekte Crystal/Gyroid/Torus):

- Mehrere gebackene SDF-Grids gleichzeitig in der Szene (Slot 1..N).
- Pro Slot: Aktiv-Flag, Position (Zentrum), Größe, Rotation, Albedo, PBR.
- GUI: neue Tabs-Gruppe **„SDF"** mit Slot-Selector-Buttons (wie Mat/Obj),
  **„Load .obj…"**-Button (Datei-Dialog, Backing in 64³-Grid, Cache),
  pro-Slot-Slider, Clear-Button.
- Menü **Datei → SDF-Objekt laden…** ruft denselben Lade-Pfad auf.
- Verhalten/Defaults identisch zum heutigen Stand: Slot 1 = `assets/statue.obj`,
  Zentrum (0, −1.4, 0), Größe 4.0, Albedo (0.85, 0.7, 0.3), metallic 0.8,
  roughness 0.35 — die Szene sieht nach dem Refactor genauso aus.

## 2. Neues Tensor-Layout („SDF-Objekt-Tensor")

Separater Storage-Buffer neben dem Mega-Scene-Buffer (der schrumpft zurück auf
1500 f32 = Slots+Materials+Rotations):

```
SDF-Objekt-Tensor (f32):
[ Header Slot 0 (Stride 24) ][ Header Slot 1 ]…[ Header Slot N-1 ]
[ Grid 0 (R³ = 64³ = 262144) ][ Grid 1 ]…[ Grid N-1 ]

Header (Stride 24, alle Werte f32):
  0  active        (0.0/1.0)
  1  res           (64.0)
  2  size          (Welt-Ausdehnung, Grid spannt [-size/2, size/2]³ um center)
  3  _pad
  4  center_x   5  center_y   6  center_z   7  _pad
  8  rot_x      9  rot_y     10  rot_z      11  _pad
  12 albedo_r  13  albedo_g  14  albedo_b   15  roughness
  16 metallic  17  emissive  18  specular   19..23 _pad
```

Konstanten (gpu.rs): `MAX_SDF_OBJECTS = 4`, `SDF_HEADER_STRIDE = 24`,
`SDF_OBJECTS_FLOATS = 4·24 + 4·64³ = 1_048_672` (~4.2 MB).
mat_id-Konvention bleibt: **2.0 + Slot-Index** für Grid-Treffer
(`> 1.5`-Test im Kernel bleibt gültig), 0.0 = Bounding-Shell.

## 3. Datenfluss

```
sdf_paths[slot] ──► sdf_preprocessor::get_or_bake_grid(path, mesh_scale_factor)
                    (Cache: <pfad>.sdfcache + In-Memory-Map)
                          │
ApplicationState::sdf_objects_data()  → Vec<GpuFloat> (Header + Grids)
                          │
gpu.rs: sdf_objects_handle (client.empty) ← queue.write_buffer pro Frame
                          │
launch_kernel(…, sdf_objects_arg, …) → raymarch_sdf_kernel(…, objects: &Tensor<F>, …)
```

Laden über GUI: `Button „Load…"` → Hintergrund-Thread → `rfd::FileDialog`
(natives Portal-Dialog-Fenster) → Backen (64³) → `state.lock()` → `sdf_paths[slot]`
setzen + Slot aktivieren. Der UI-Thread blockiert nicht (Backen kann >1 s dauern).

## 4. Kernel-Änderungen (ABI-Switch, ein Schritt)

- `main_kernel`: neuer Tensor-Parameter `objects: &Tensor<F>`;
  `statue_center`/`grid_size`/`statue_metallic` aus `RenderSettings` entfernt;
  Material von Grid-Treffern kommt jetzt aus dem Slot-Header (`sdf_object_pbr`).
- `statue_grid.rs` (wird faktisch zum „Grid-Objekt-Modul"):
  - `sample_sdf_grid(p, objects, grid_start, res, size)` — grid_start ist jetzt
    Parameter (Slot-Offset im Tensor, nicht mehr `SLOTS_COUNT+…`-Konstante).
  - `scene_map_with_material(p, objects, res, bound_offset, use_sphere_bound)`
    — loopt über alle N Slots (Sphere-Bound je Slot, inverse Euler-Rotation
    wie bei den dynamischen Slots), liefert (d, mat_id = 2+i).
    Die toten Parameter `spacing`/`pillar_height` fallen weg.
  - NEU `sdf_object_pbr(objects, mat_id) -> SdfResult` — PBR/Albedo aus Header.
  - `calculate_normal(p, objects, res, bound_offset)`.
- `shadows.rs` / `ao.rs` / `reflection.rs`: Signatur auf
  `(objects, grid_res, bound_offset)` umgestellt; `trace_reflection` holt das
  Material des Grid-Treffers aus dem Header statt hardcodierter Statuen-Farben.
- `types.rs`: `RenderSettings` schlank (−`sdf_grid_size`, −`statue_center`,
  −`statue_metallic`).

## 5. CPU-State-Änderungen (`app_state.rs`)

- NEU (parallele Vektoren, N = 4): `sdf_paths: Vec<String>` (leer = inaktiv),
  `sdf_active`, `sdf_sizes`, `sdf_center_x/y/z`, `sdf_rot_x/y/z`,
  `sdf_r/g/b`, `sdf_roughness/metallic/emissive/specular`,
  `current_selected_sdf_object: usize`.
- ENTFERNT: `sdf_grid_size`, `statue_center_x/y/z`, `statue_metallic`
  (wandern per-Slot). BLEIBT: `pillar_roughness` (Pfeiler), `mesh_scale_factor`
  (globaler Back-Parameter), `bounding_sphere_radius_offset` (global).
- `scene_data()`: hängt **kein** Grid mehr an (wieder 1500 f32).
- NEU `sdf_objects_data()`: packt Header + Grids (aus dem Preprocessor-Cache).
- `render_launch()`: schlanke `RenderSettings`.

## 6. GPU-Änderungen (`gpu.rs`)

- NEU `sdf_objects_handle` (+ `SDF_OBJECTS_FLOATS`); `scene_handle` schrumpft
  auf `TOTAL_BUFFER_SIZE` (1500 f32).
- `write_state_buffers`: schreibt beide Buffer.
- `launch_kernel`: übergibt den zweiten Tensor-Arg.

## 7. GUI-Änderungen

- **Neuer Tab „SDF"** (`gui/tabs/sdf.rs`): Slot-Selector 1..4, Load/Clear-Buttons,
  Accordion mit Position / Rotation / Größe & Grid / Material (Albedo+PBR).
- `GuiState`: Statue-Slider werden zu SDF-Slot-Slidern
  (`sdf_center_*`, `sdf_size`, `sdf_rot_*`, `sdf_r/g/b`, `sdf_roughness/metallic/
  emissive/specular`); Subscriptions schreiben per-Slot am Index
  `current_selected_sdf_object` (exakt wie das PBR-Slot-Muster in mat.rs).
- `Obj`-Tab: „SDF Grid Space" + „Transformations & Bounds" entfallen (→ SDF-Tab).
- `Mat`-Tab: „Statue Metallic" entfällt (→ pro Slot im SDF-Tab).
- `menubar.rs`: Aktion `LoadSdfObject` + Menüpunkt „SDF-Objekt laden…".

## 8. Implementierungsschritte

| Schritt | Inhalt | Verify |
|---|---|---|
| S1 | `sdf_preprocessor.rs`: In-Memory-Cache `get_or_bake_grid(path, scale) -> Arc<Vec<f32>>` (nutzt bestehenden `.sdfcache`) | `cargo check` |
| S2 | Kernel-ABI-Switch: types.rs, statue_grid.rs, scene.rs (nur Signatur-Weiterreichung), shadows.rs, ao.rs, reflection.rs, main_kernel.rs | `cargo check` (bricht bis S3/S4, daher S2–S4 als eine Check-Einheit) |
| S3 | `app_state.rs`: SDF-Slot-Felder, `sdf_objects_data()`, schlanke `scene_data()`/`render_launch()` | ↑ |
| S4 | `gpu.rs`: Konstanten, zweiter Handle, write/launch | `cargo check` + `cargo run --bin test_kernel_launch` (headless Codegen-Test) |
| S5 | GUI: `tabs/sdf.rs`, GuiState-Umbau, Obj/Mat-Tab-Bereinigung, Menü-Aktion | `cargo check` |
| S6 | Doku: `IMPLEMENTATION_SUMMARY.md`, Abschluss-Verifikation (Headless-Test, Pixel-Statistik) | Test + Doku |

## 9. Risiken / Grenzen

- **Res fest 64³ pro Slot** (alle Grids gleiche Auflösung — der Back-Cache wird
  auf `SDF_GRID_RES` normiert). Abweichende Auflösungen = Re-Back.
- **4 Slots** fest (MAX_SDF_OBJECTS); Erhöhen = Konstante + Neustart.
- Grid-Daten werden pro Frame mitgeschrieben (~4 MB) — auf Desktop-GPUs
  irrelevant; Dirty-Tracking wäre spätere Optimierung.
- `rfd`-Dialog läuft im Hintergrund-Thread (xdg-portal); falls kein Portal
  verfügbar ist, schlägt der Dialog fehl → Status in der Konsole, State bleibt
  unverändert (kein Crash).
