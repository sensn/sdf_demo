# IMPLEMENTATION_SUMMARY.md

Implementierung aller Features aus `doc/feature/Agent_Impl_Plan.md`
(CubeCL 0.11.0 Raymarcher) — abgeschlossen am 2026-10-12.

## 🔧 Nachbesserung (Bugfix-Runde 3)

**Gemeldetes Problem**: weiterhin schwarzer (ovaler) Bereich um die
Statuen-SDF.

**Ursachen & Fixes:**
1. **Statue füllte das Grid bis zum Rand**: Das Mesh war auf 100% der
   Einheits-Box normalisiert — am Grid-Rand liefert das geclampte
   Sampling ~0 als Distanz, was im Schatten-March als Phantom-Fläche
   wirkte → schwarzer Oval-Schatten auf dem Boden.
   **Fix**: Normalisierung auf 80% (`scale = 1.6/extent`) — zwischen
   Mesh und Grid-Grenze bleibt echter Abstand. `STATUE_CENTER` auf
   y = -1.4 angepasst (Halbhöhe jetzt 1.6, steht weiter auf dem Boden).
   Cache-Magic `SDF2` → `SDF3`.
2. **`hit_is_plan_obj` blieb hängen**: Sobald der unsichtbare
   Sphere-Bound mal einen Schritt gewann, blieb das Flag `true` —
   traf der Strahl danach den Boden, bekam er Statuen-Normalen
   (≈0-Gradient) → schwarzer Fleck.
   **Fix**: Flag wird jeden Schritt zurückgesetzt und nur gesetzt,
   wenn die Statue selbst (mat_id 2) die nächste Fläche ist.
3. **Sphere-Bound trug mat_id 2**: Auch außerhalb der Kugel wurde
   mat_id = 2 (Statue) gemeldet — der Marcher hielt die Kugel für
   die Statue.
   **Fix**: Sphere-Bound setzt mat_id = 0 (kein Objekt, nur
   Schritt-Schranke — kein Material, keine Normalen).

**Verifikation nach Fix:**
```
test_sdf_bake:      5.4% Innen-Voxel (vorher 10.2%), Range [-0.34, 2.25] ✅
test_kernel_launch: 99.7% nicht-schwarz, Ø-Lum 0.18 (vorher 0.06), Max 0.68 ✅
```

## 🔧 Nachbesserung (Bugfix-Runde 2)

**Gemeldete Probleme**: schwarzer Phantom-Schatten-Ring um die Statue;
Mirror-Effekt (Roughness < 0.25) verschwunden.

**Ursachen & Fixes:**
1. **Phantom-Schatten-Ring**: Die Bounding-Sphere-Distanz (sichere
   Schritt-Schranke für den Primär-March) wurde auch im Schatten-/AO-
   March als Szenen-Distanz gewertet → die unsichtbare Kugelgrenze
   dunkelte ab (`res.min(8·h/t)` über den Kugel-Abstand).
   **Fix**: `scene_map_with_material` hat jetzt `use_sphere_bound`:
   Primär-March `true` (sicheres Stepping), Schatten/AO/Normalen/
   Reflexionen `false` (nur reale Statuen-Distanz). Der Ring ist weg.
2. **Mirror-Effekt verschwunden**: `trace_reflection` marschierte nur
   das Statuen-Map — glatte Arch-/Slot-Flächen spiegelten leeren Himmel.
   **Fix**: `trace_reflection` marschiert jetzt die **komplette Szene**
   (Slots + Arch via `scene_sdf` + Statue), mit korrektem Albedo-
   Lookup des getroffenen Objekts und Normalen-Selektion per
   Distanz-Vergleich. Der Roughness < 0.25-Spiegel ist zurück und
   zeigt die echte Umgebung.

**Verifikation nach Fix:**
```
test_kernel_launch: 97.9% nicht-schwarz, Ø-Lum 0.06, Max 0.79 ✅
```

## 🔧 Nachbesserung (Bugfix-Runde 1)

**Gemeldete Probleme**: Phantom-Pfeiler in Nachbarräumen nach Kamera-
Aufwärtsbewegung, kaputte Arch-Schatten (Statuen-Schatten korrekt),
Statue zu groß.

**Ursachen & Fixes:**
1. **Phantom-Pfeiler**: Die Plan-Pfeiler (`op_repetition_xz` mit
   `round()`-Folding, Spacing 6.0, Höhe 4.0) kollidierten mit der
   bestehenden Arch-Zellen-Faltung (`floor()` + `cell_size` 10.0,
   Raum y∈[-3,3]). Andere Zellgrenzen + zu hohe Pfeiler → Bruchstücke
   ragten durch Boden/Decke in Nachbarräume.
   **Fix**: Plan-Pfeiler-System komplett entfernt. Die Architektur
   generiert seine eigenen, korrekt gefalteten Pfeiler (ArchParams).
2. **Kaputte Arch-Schatten**: Der Schatten/AO-March testete das
   Pfeiler-Feld mit — dessen Folding-Seams korruptierten die
   Schattenstrahlen der Architektur.
   **Fix**: Mit den Pfeilern verschwand auch die Korruption —
   Schatten/AO testen jetzt nur noch Slots+Arch (`scene_sdf`) und die
   Statue (`scene_map_with_material`), beide sauber.
3. **Statue zu groß**: Grid spannte 8 Welteinheiten in einem 6-Einheiten-
   Raum → ragte durch Boden und Decke.
   **Fix**: `SDF_GRID_SIZE` 8.0 → 4.0 (halbe Größe), `STATUE_CENTER`
   neu bei [0, -1, 0] (steht auf dem Raum-Boden bei y=-3.0, Halbhöhe 2.0).
   Cache-Magic `SDF1` → `SDF2` (invalidiert stale Caches mit alten
   Welt-Distanzen automatisch).
4. **March-Sicherheit**: Bounding-Sphere-Culling liefert jetzt
   außerhalb der Kugel die exakte Kugel-Distanz als untere Schranke
   (Lipschitz-Bedingung) — der Marcher kann nicht durch die Statue
   tunneln, und das teure Grid-Sampling läuft nur nahe der Statue.

**Verifikation nach Fix:**
```
test_sdf_bake:      Distanz-Range [-0.500, 1.969] (exakt halbiert) ✅
test_kernel_launch: 99.8% nicht-schwarz, Ø-Lum 0.07, Max 0.79 ✅
```

## Überblick

Alle 7 Plan-Schritte sind umgesetzt und über einen Headless-GPU-Test
verifiziert (Kernel-Compile via WGSL-Codegen + echter Launch + Readback).

## Schritt-für-Schritt

### Step 1 — Host-PBR-Konfigurationen (UBO-aligned)
**Status: bereits vorhanden** (`kernel.rs`: `GpuVec3`, `CameraSettings`,
`EnvSettings`, `ArchParams`, `FoldParams`, `RenderSettings` als
`CubeLaunch`-Structs). Keine Änderungen nötig.

### Step 2 — OBJ→SDF-Präprozessor + Binär-Cache
**Neu: `src/sdf_preprocessor.rs`**
- `load_obj_to_sdf(path, res)`: `tobj` lädt das OBJ (trianguliert),
  normalisiert es auf `[-1,1]³`, sampelt ein `64³`-Voxelgitter über
  `[-4,4]³` via `mesh_to_sdf::generate_sdf` (RtreeBvh-Beschleunigung).
  Grid in ZYX-Reihenfolge geflattet (`index = x + res*(y + res*z)`).
- `save_sdf_cache` / `load_sdf_cache`: Roh-Binärformat mit Magic-Header
  (`SDF1`) + Längen-Check; `bytemuck` für Zero-Copy-I/O.
- `load_or_bake_sdf`: Cache-first — Bake läuft nur einmal pro Mesh.
- **Neu: `src/bin/gen_statue.rs`**: erzeugt `assets/statue.obj`
  (prozedurale Low-Poly-Statue: Sockel, Torso, Arme, Oktaeder-Kopf),
  da kein OBJ-Asset existierte.
- **Neu: `src/bin/test_sdf_bake.rs`**: verifiziert Bake + Cache-Roundtrip
  + Vorzeichen (10.2% Innen-Voxel, Range [-1.0, 3.94]).

### Step 3 — Trilineares SDF-Grid-Sampling
**Neu in `kernel.rs`: `sample_sdf_grid(p, grid, res, size)`**
- Welt→Grid-Mapping auf `[0, res-2]` geklemmt (valide Nachbar-Voxel).
- 8 Voxel-Ecken via ZYX-Flattening, trilineare Mischung X→Y→Z.
- Grid reist im Mega-Scene-Buffer: `gpu.rs` hängt `64³ = 262.144`
  Floats an (`TOTAL_SCENE_FLOATS = 263.644`, ~1.05 MB);
  `app_state.scene_data()` ruft `load_or_bake_sdf` und appended.

### Step 4 — Space-Folding + fraktaler Pfeiler
**Neu in `kernel.rs`**
- `op_repetition_xz(p, spacing)`: Domain-Folding für unendliche
  XZ-Wiederholung (`p - spacing*round(p/spacing)`).
- `sdf_fractal_pillar(p, height)`: Basis-Box (0.35 halbe Breite) +
  4 fraktale Iterationen (Abs-Kreuz-Subtraktion, `scale *= 2`).
  Comptime-Range `range(usize::new(0), usize::new(4))` wird vom
  Compiler geunrollt (CubeCL-Äquivalent zum geplanten `loop_unrolled`).

### Step 5 — Scene-Map mit Material + Culling + Normalen
**Neu in `kernel.rs`**
- `Vec2 { d, mat_id }`-Struct (CubeType, mat_id: 1=Pfeiler, 2=Statue).
- `scene_map_with_material(...)`: kombiniert Pfeiler (via Folding) und
  Statuen-Grid mit **Bounding-Sphere-Distance-Culling** — das teure
  Grid-Sampling läuft nur, wenn `|p - statue_center| - r < 0.1`.
- `calculate_normal(...)`: zentrale Differenzen (ε=0.001) über die
  kombinierte Map.
- Primär-Raycast mischt `scene_map_with_material` in die bestehende
  Slot-Map (`hit_is_plan_obj`-Flag entscheidet über die Normalen-Quelle);
  Pfeiler/Statue bekommen eigene PBR-Materialien (Statue: gold-bronze,
  metallic 0.8; Pfeiler: Stein, roughness 0.8).

### Step 6 — PBR-Material-Lookup + Glow
- Material-Lookup war vorhanden (Cook-Torrance, `materials`-Slice).
- **Neu: Glow-Akkumulator** im March-Loop:
  `glow += 0.01/(d + 0.02) × emissive × albedo` pro Schritt —
  Leuchten sichtbar auch bei Strahl-Verfehlung (addiert nach dem
  Treffer-Block, unabhängig vom Hit).

### Step 7 — AO, Soft Shadows, Volumetric Fog, Reflexionen
- AO (5 Iterationen) + Soft Shadows (`min(res, k·h/t)`) waren vorhanden —
  **erweitert**: beide testen jetzt zusätzlich die Plan-Map, damit
  Pfeiler/Statue Okklusion/Schatten werfen (neue Parameter:
  `pillar_spacing/height, sdf_grid, grid_res, grid_size`).
- **Neu: `integrate_volumetric_fog(...)`**: Beer-Lambert-Integration
  (16 Schritte) mit Höhen-Profil (Boden-Nebel dichter, Falloff 0.08),
  Early-Out bei Transmittanz < 0.01. Ersetzt die alte lineare
  Distanz-Kompression.
- **Neu: `trace_reflection(...)`**: 1-Bounce-Sekundärstrahl (max. 32
  Schritte) mit eigener Schattierung (Normale + N·L, Material-Farben
  via `select()` — if-else-Expressions sind für Structs in CubeCL nicht
  unterstützt). Aktivierung im Haupt-Kernel über Roughness-Gate
  (< 0.25), Fresnel-artige Stärke `(1-rough)*0.35`.

## Infrastruktur

- **`gpu.rs`**: `launch_kernel()` von `render_frame()` getrennt (Blit
  separat) → Headless-Tests möglich. `new_headless()`-Konstruktor.
  Scene-Buffer-Größe auf `TOTAL_SCENE_FLOATS` erweitert.
- **`src/bin/test_kernel_launch.rs`**: Headless-GPU-Test — kompiliert
  den Kernel (WGSL-Codegen), launcht 64×64 px, liest den Output zurück
  und prüft Pixel-Statistik. **Ergebnis: 99.8% nicht-schwarz,
  Ø-Luminanz 0.18, Max 0.99.**

## CubeCL-0.11-API-Abweichungen vom Plan

| Plan (veraltet) | Verwendet (0.11 real) |
|---|---|
| `Line<f32, 2>` | eigenes `Vec2`-CubeType |
| `Comptime::loop_unrolled` | Comptime-Range (auto-unroll) |
| `#[unroll]`-Attribut | nicht nötig — Ranges unrollen automatisch |

## Verifikation

```
$ ./target/debug/test_sdf_bake
Bake statue.obj @ 64³ = 262144 Voxel...
Innen-Voxel: 26691/262144 (10.2%)
Distanz-Range: [-1.000, 3.938]
✅ SDF-Präprozessor OK (Bake + Cache + Vorzeichen)

$ ./target/debug/test_kernel_launch
Pixel: 4096, nicht-schwarz: 4086 (99.8%), Ø-Luminanz: 0.1845, Max: 0.986
✅ Headless-Launch OK — Kernel kompilierte fehlerfrei (WGSL-Codegen)
```

`cargo check`: 0 Fehler (nur 4 Pre-existing-Warnings).
