# Raymarch Hot-Loop Optimization — Implementation Summary

**Datum:** 2026-11-18 · **Branch:** `generic` · **Basis:** 9e6a75f
**Commit-Referenzen:** CubeCL #2121 (SPIR-V NaN-Vergleich), Optimierungs-Plan des Nutzers

---

## Ziel

Der Primär-Raymarch-Loop (`raymarch_sdf_kernel`) lief pro Pixel bis zu 100 Iterationen
und verschwendete dabei Arbeit an drei Stellen. Ergebnis: Die Schleife macht jetzt nur
noch, was sie wirklich braucht, und bricht bei NaN sofort ab.

---

## Die drei Probleme

### 1. Unbedingte Grid-Auswertung
Jeder der bis zu 100 Schritte rief `scene_map_with_material` — auch wenn der Strahl
meilenweit außerhalb der Statuen-Kugel lag. Pro Aufruf bis zu 4×64³ = **1.048.576
Voxel-Lesevorgänge** aus einem ~4 MB Storage-Buffer.

### 2. Glow-Bug & redundantes Material-Entpacken
- `glow_r += glow_kernel * step_res.emissive * …` nutzte die Emission des *verdeckten*
  Slot-Objekts, auch wenn ein Grid oder Pfeiler näher war → leuchtende Objekte schienen
  durch andere hindurch.
- Pro Schritt wurden **8 PBR-Felder** (`hit_r/g/b`, `hit_rough`, `hit_metal`,
  `hit_emiss`, `hit_spec`) zugewiesen — nur um im nächsten Schritt überschrieben zu werden.

### 3. NaN-Endlos-Schleife
`if hit_d < F::new(0.001)` ist in SPIR-V ein **ordered comparison**. Bei `d == NaN` ist der
Vergleich immer `false` → die Schleife lief alle 100 Schritte mit Müllwerten durch.
Commit #2121 fixt das im Compiler, aber der Kernel-Code muss `!(d >= eps)` nutzen, um
davon zu profitieren.

---

## Implementierung

### S1 — `objects_min_bound_distance` (statue_grid.rs)

Neue Kernel-Funktion, die nur die **4 × 24 Header-Werte** liest und die minimale
Kugelgrenz-Distanz zurückgibt:

```rust
#[cube(fast_math = FastMath::all().difference(FastMath::AllowContraction.into()))]
pub fn objects_min_bound_distance<F: Float>(
    p: Vec3<F>, objects: &Tensor<F>, bound_offset: F,
) -> F { /* 4 aktive Slots prüfen, min(|p-center| - size/2) */ }
```

Da die Kugel das Grid vollständig umschließt, ist die Sphere-Distanz eine **untere
Schranke** der echten Grid-Distanz. Ist sie größer als die bereits bekannte
Slot-Distanz `d`, kann kein Grid näher sein → Sampling entfällt komplett.

### S2 — Restrukturierter Hot-Loop (main_kernel.rs)

```rust
let mut hit_mat_source = u32::new(0);   // 0=kein, 1=Slot, 2=Shell, 3=Grid
let mut hit_mat_id = F::new(0.0);

loop {
    if ray_step >= u32::new(100) { break; }
    let p = current_ro.add(current_rd.scale(t));

    // 1) Slot-SDF (schnelle analytische Formen) — immer nötig
    let step_res = scene_sdf::<F>(p.clone(), …, slots, materials, rotations, arch, fold);
    let mut d = step_res.d;
    let mut step_source = u32::new(1);

    // 2) Grid NUR sampeln, wenn es überhaupt näher sein könnte
    let grid_bound_d = objects_min_bound_distance::<F>(p.clone(), objects, bound_offset);
    if grid_bound_d < d {
        let plan_res = scene_map_with_material::<F>(p.clone(), objects, bound_offset, true);
        if plan_res.d < d {
            d = plan_res.d;
            step_source = select(plan_res.mat_id > F::new(1.5), u32::new(3), u32::new(2));
            hit_mat_id = plan_res.mat_id;
        }
    }

    // 3) Glow nur von Slot-Objekten, die WIRKLICH die nächste Fläche sind
    if step_source == u32::new(1) && step_res.emissive > F::new(0.0) {
        let glow_kernel = F::new(0.01) / (d + F::new(0.02));
        glow_r += glow_kernel * step_res.emissive * step_res.r;
        glow_g += glow_kernel * step_res.emissive * step_res.g;
        glow_b += glow_kernel * step_res.emissive * step_res.b;
    }

    // 4) NaN-sicherer Hit-Test (#2121): bricht bei NaN sofort ab
    if !(d >= F::new(0.001)) {
        hit_dist = t;
        hit_mat_source = step_source;
        break;
    }
    t += d;
    if t > F::new(40.0) { break; }
    ray_step += u32::new(1);
}
```

**Änderungen gegenüber vorher:**

| Aspekt | Vorher | Jetzt |
|---|---|---|
| Grid-Sampling | Jeden Schritt | Nur wenn `objects_min_bound_distance < d` |
| PBR-Entpacken | 8 Felder × 100 Schritte | 0 im Loop, 1 × am Treffer |
| Glow-Quelle | Immer `step_res` (auch verdeckt) | Nur wenn Slot-Objekt nächste Fläche |
| Hit-Test | `d < eps` (NaN → false → 100 Schritte) | `!(d >= eps)` (NaN → true → Abbruch) |
| Treffer-Typ | `hit_is_plan_obj: bool` | `hit_mat_source: u32` (1/2/3) |

### S3 — One-Shot Material + Normale

Material und Normale werden jetzt **einmal** am Trefferpunkt bestimmt:

```rust
if hit_dist < F::new(40.0) {
    let p = ro.add(final_rd.scale(hit_dist));

    // Material EINMAL statt 8 Felder pro Schritt
    let mut hit_r = …; let mut hit_rough = render.pillar_roughness; …
    if hit_mat_source == u32::new(1) {
        let mat_res = scene_sdf::<F>(p.clone(), …, slots, materials, rotations, arch, fold);
        hit_r = mat_res.r; hit_rough = mat_res.roughness; …
    } else if hit_mat_source == u32::new(3) {
        let pbr = sdf_object_pbr::<F>(objects, hit_mat_id);
        hit_r = pbr.r; hit_rough = pbr.roughness; …
    }
    // hit_mat_source == 2 (Bounding-Shell) → steinige Defaults

    // Normale NUR über die Map, die den Treffer geliefert hat
    if hit_mat_source >= u32::new(2) {
        let n = calculate_normal::<F>(p.clone(), objects, bound_offset);
        …
    } else {
        let n = scene_sdf_normal::<F>(p.clone(), …, slots, materials, rotations, arch, fold);
        …
    }
    let normal = Vec3::<F>::new(n_x, n_y, n_z);
    …
}
```

**Zusätzlicher Bugfix:** Der alte Code lief bei *jedem* Treffer in
`calculate_normal` (6 × Grid-Sampling), weil `plan_res.d < hit_d` auch von der
Bounding-Shell gesetzt wurde. Jetzt läuft die Grid-Normale nur bei echten
Grid-Treffern (`hit_mat_source >= 2`).

---

## Begleitende Aufräumarbeiten

| Datei | Änderung |
|---|---|
| `src/kernel/statue_grid.rs` | 🆕 `objects_min_bound_distance` |
| `src/kernel/main_kernel.rs` | Hot-Loop + Material/Normal refactored; Signatur weiterhin `slots/materials/rotations` statt `scene` |
| `src/kernel/scene.rs` | `scene_sdf` / `scene_sdf_normal` auf die 3 separaten Tensors umgestellt |
| `src/kernel/shadows.rs`, `ao.rs`, `reflection.rs` | Signatur `scene` → `slots/materials/rotations` |
| `src/gpu.rs` | `slots_handle`, `materials_handle`, `rotateations_handle` statt `scene_handle`; `write_state_buffers` schreibt 3 Buffer |
| `src/app_state.rs` | 🆕 `slots_data()`, `materials_data()`, `rotations_data()` (Stride-Packing pro Register) |

---

## Verifikation

```bash
cargo check                 # ✅ 0 Fehler
cargo check --bin sdf_demo  # ✅ 0 Fehler
```

---

## Erwarteter Effekt

| Szenario | Vorher | Jetzt |
|---|---|---|
| Strahl verfehlt Statue komplett | 1.048.576 Voxel-Lesungen/Schritt | 96 Header-Lesungen/Schritt (≈ 10.000× weniger) |
| Strahl im freien Tempelraum | 1.048.576 Voxel-Lesungen/Schritt | 0 Voxel-Lesungen (Kugel weit weg) |
| Material-Arbeit pro Pixel | 8 Felder × ~50 Schritte = 400 Schreibvorgänge | 8 Schreibvorgänge |
| NaN-Strahl | 100 komplette Schritte mit Müllwerten | Sofortiger Abbruch (Schritt 1) |

---

## Grenzen / Follow-ups

- `scene_data()` in `app_state.rs` ist jetzt **unbenutzt** — kann später entfernt werden,
  sobald Tests/Headless-Binary darauf verzichten.
- `objects_min_bound_distance` liest pro Schritt 4 Header. Bei 100 Schritten sind das 400
  Lesevorgänge — vernachlässigbar gegenüber 1 M Voxeln.
- Die Kugel ist eine grobe Schranke: Ein dünner, diagonaler Grid-Stab blockiert den
  Sampling-Start, obwohl die wahre Distanz größ ist. Das ist beabsichtigt (Sicherheit
  vor false negatives) und kostet höchstens ein paar Schritte Sampling am Rand.