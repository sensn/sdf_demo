# PBR_Optimisation_Plan — sdf_demo (Stand: 2026-10-03)

**Zweck:** Analyse der aktuellen PBR-Material-Implementierung in
`src/kernel.rs`, Diagnose der vier beobachteten Slider-Probleme und
detaillierter Optimierungs-Plan mit konkreten Formeln und Code-Skizzen.

---

## 1. Ist-Zustand: Wie PBR heute funktioniert

### 1.1 Datenfluss (korrekt implementiert ✅)

```
GUI-Slider (Mat-Tab)
   │  schreibt slot_roughness[i] / slot_metallic[i] /
   │  slot_emissive[i] / slot_specular[i]  (i = current_selected_slot)
   ▼
ApplicationState::materials_data()          [app_state.rs:182]
   │  packt pro Slot 4 f32 (Stride 4) in einen Vec
   ▼
GpuPipeline::write_state_buffers()           [gpu.rs]
   │  write(&self.materials_handle, cast_slice(&s.materials_data()))
   ▼
materials: &Tensor<f32>                      [kernel.rs:509]
   │  scene_sdf liest mat_idx = i*4:
   │    [0]=roughness [1]=metallic [2]=emissive [3]=specular
   ▼
SdfResult { d, r, g, b, roughness, metallic, emissive, specular }
   │  gewichtete Akkumulation über aktive Slots (w = 1/max(d,0.001)²)
   ▼
raymarch_sdf_kernel: hit_rough / hit_metal / hit_emiss / hit_spec
   ▼
Shading (kernel.rs:645–722)
```

Der Transport der Materialwerte bis in den Kernel ist vollständig und
fehlerfrei. **Alle vier beobachteten Probleme liegen im Shading.**

### 1.2 Das heutige Shading-Modell (kernel.rs:645–722)

```rust
// Diffuse Anteile (Lambert, 3 Lichter)
diff_key  = max(0, N·L_key)
diff_fill = max(0, N·L_fill)
diff_rim  = max(0, N·L_rim)

// Specular: Blinn-Phong mit roughness als Kehrwert-Exponent
spec_power = 1 / max(hit_rough, 0.01)          // 1.0 … 100.0
specular_highlight = pow(max(0, N·H), spec_power * 15.0) * hit_spec

// Kombination
lighting = fill*diff_fill + rim*diff_rim + key*diff_key * l_intensity
final = (albedo * (lighting + ambient) + specular_highlight) * ao + emissive
```

### 1.3 Ausgabe-Pfad

- Kernel schreibt lineares RGB als rohe f32 in den Storage-Buffer.
- `shader.wgsl` liest die f32 und gibt sie **unverändert** an
  `Rgba8UnormSrgb` weiter (fs_main, Zeile 41).
- **Kein Tone-Mapping, keine HDR-Verarbeitung.** Werte > 1.0 werden von
  der Hardware beim 8-Bit-Write hart geclampt.

---

## 2. Diagnose der vier Slider-Probleme

### 2.1 Emissive: „unter 1.0 heller, über 1.0 nur noch weiß"

**Beobachtung korrekt — zwei Ursachen:**

1. **Kein Tone-Mapping:** `final = ... + hit_emiss` kann beliebig groß
   werden. Der Framebuffer (Rgba8UnormSrgb) clampt bei 1.0 → alles
   ≥ 1.0 ist totes Weiß. Der Slider wirkt daher nur im Bereich
   [0, 1] sichtbar.
2. **Emissive ist farblos:** `hit_emiss` ist ein Skalar, der auf **alle**
   drei Kanäle addiert wird → Glühen ist immer weiß, nie farbig.

**Fix:** ACES-Tone-Mapping + optionale farbige Emission (siehe 3.1/3.3).

### 2.2 Metallic: „ändert fast nichts, sieht nicht metallisch aus"

**Wurzelursache: `hit_metal` wird im Shading NICHT VERWENDET.**

Beweis (grep über kernel.rs): `hit_metal` wird nur zugewiesen
(kernel.rs:599, 620) und nie gelesen. Der Metallic-Slider ändert also
buchstäblich nichts am Bild — was der User sieht, ist vermutlich der
gleichzeitige Einfluss des Akkumulator-Gewichts oder Zufall.

**Physikalisch korrekt wäre:**

- Metall: Albedo wird zur Reflexionsfarbe (F0), diffuse Reflexion → 0
- Dielektrikum: F0 ≈ 0.04, Albedo bleibt diffuse Farbe

**Fix:** Cook-Torrance-Specular mit F0-Interpolation (siehe 3.2).

### 2.3 Roughness: „wirkt wie Specular"

**Heute:** `spec_power = 1/roughness` → je **glatter** (roughness↓), desto
**größer** der Exponent → schärferer Glanz. Der Slider invertiert also
das visuelle Verhalten: roughness=0.01 gibt einen extrem scharfen
Spiegel-Glanz, roughness=1.0 gibt einen breiten, schwachen Glanz.

Da `specular_highlight` zusätzlich mit `hit_spec` skaliert wird, wirken
beide Slider wie „Glanz-Intensität" — daher die Verwechslung.

**Fix:** GGX-Verteilung mit `α = roughness²` (siehe 3.2) — dann steuert
roughness die **Größe/Weichheit** des Glanzlichts und specular die
**Reflektivität** (F0-Skalierung), sauber getrennt.

### 2.4 Specular: „scheint zu funktionieren"

Funktioniert als Glanz-Intensität — aber physikalisch falsch
interpretiert: In PBR ist „specular" kein Intensitäts-Faktor, sondern
die **Reflektivität F0** (Dielektrikum ≈ 0.04, Metall = Albedo).
Nach der Cook-Torrance-Migration wird der Slider zur F0-Skalierung.

---

## 3. Optimierungs-Plan

### Phase A — Korrektes Cook-Torrance-PBR (Kernstück)

**Ziel:** Die vier Slider bekommen ihre physikalisch definierte Bedeutung,
das Bild sieht „echt" aus (Metall glänzt farbig, Glanz folgt Roughness).

#### A1. Fresnel (Schlick-Approximation)

```rust
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0)
}
```

#### A2. GGX-Normalverteilung

```rust
fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}
```

#### A3. Smith-Geometrie-Schattung

```rust
fn g_smith(n_dot_v: n_dot_l: f32, alpha: f32) -> f32 {
    let k = (alpha + 1.0)² / 8.0;   // für analytische Lichter
    (n_dot_l / (n_dot_l * (1-k) + k)) * (n_dot_v / (n_dot_v * (1-k) + k))
}
```

#### A4. F0 aus Metallic + Specular

```rust
// specular-Slider skaliert F0 des Dielektrikums (Basis 0.04)
let f0_dielectric = 0.04 * (1.0 + hit_spec * 3.0);   // 0.04 … 0.16
let f0 = mix(f0_dielectric, albedo, hit_metal);      // Metall: Albedo als F0
```

#### A5. Vollständige Cook-Torrance-Kombination pro Licht

```rust
// pro Lichtquelle (key/fill/rim):
let h = normalize(l_dir + v_dir);
let n_dot_h = max(0, N·H);
let n_dot_l = max(0, N·L);
let n_dot_v = max(0, N·V);
let l_dot_h = max(0, L·H);

let alpha = hit_rough * hit_rough;
let d = d_ggx(n_dot_h, alpha);
let g = g_smith(n_dot_v, n_dot_l, alpha);
let f = fresnel_schlick(l_dot_h, f0);

let specular_brdf = (d * g * f) / (4 * n_dot_v * n_dot_l + eps);
let kd = (1.0 - f) * (1.0 - hit_metal);   // diffuse Restenergie
let diffuse = kd * albedo / PI * n_dot_l;

color += (diffuse + specular_brdf) * light_color * intensity;
```

#### A6. Energie-Konsistenz

- `kd = (1 - F) * (1 - metallic)`: Metall hat **keine** diffuse Farbe.
- Diffuse wird durch `/PI` normiert (Lambert korrekt).
- Fill/Rim-Lichter behalten ihre fixen Farben, Key-Licht die dynamische.

### Phase B — HDR & Tone-Mapping (behebt Emissive-Weiß-Clipping)

#### B1. ACES-Narkowicz-Tone-Mapping im Kernel

```rust
fn aces_tonemap(x: f32) -> f32 {
    let a = 2.51; let b = 0.03; let c = 2.43;
    let d = 0.59; let e = 0.14;
    (x * (a*x + b)) / (x * (c*x + d) + e)
}
```

Wird **vor** dem Schreiben in den Output-Buffer angewandt, auf alle drei
Kanäle. Damit sind Werte > 1.0 nicht mehr „totes Weiß", sondern
natürlich überstrahlt (z. B. Emissive 10.0 → sichtbares, farbiges Glühen).

#### B2. Optional: Bloom (vertagt)

Echtes Bloom bräuchte einen zweiten Render-Pass (downsample → blur →
composite). Mit dem aktuellen Single-Pass-Design nicht abbildbar;
als „Phase D" notiert, wenn die Performance es erlaubt.

### Phase C — Emissive als echte Lichtquelle

**Ziel:** Emissive-Objekte beleuchten ihre Umgebung („emit real light").

#### C1. Emissive-Beitrag im Shading sammeln

In `scene_sdf` wird `emissive` bereits pro Sample akkumliert — aber nur
für den **Hit-Punkt**. Für Umgebungsbeleuchtung müsste der Emissive-Wert
der **Umgebung** in Richtung Lichtquelle gesampelt werden.

**Minimal-Invasiver Ansatz (empfohlen):** Emissive-Objekte werden als
**zusätzliche Punktlichter** behandelt:

1. CPU-seitig: `ApplicationState` berechnet pro aktivem Slot mit
   `emissive > 0` eine Lichtposition (Slot-Offset) + Farbe (Slot-Farbe ×
   Emissive-Intensität).
2. Diese werden in den **env_settings-Tensor** gepackt (neue Blöcke) oder
   in einen eigenen kleinen Tensor.
3. Der Kernel looped über diese „Emissive-Lichter" und addiert pro Licht
   ein Cook-Torrance-Beleuchtungsterm (mit Distanz-Attenuation 1/d²).

**Kosten:** ~N zusätzliche Lichtauswertungen pro Pixel (N = Anzahl
emissiver Slots, meist 1–3). Kein zweiter Pass nötig.

#### C2. Emissive-Farbe

`hit_emiss` wird mit der Slot-Farbe multipliziert:
`emissive_rgb = slot_color * emissive_scalar` → farbiges Glühen statt
weißem Addieren.

### Phase D — Optional / Vertagt

- **Bloom-Post-Process** (siehe B2)
- **IBL (Image-Based Lighting)**: Umgebungshelligkeit aus einer Probe-
  Richtung statt konstantem Ambient — würde das Bild deutlich natürlicher
  machen, braucht aber Sampling-Infrastruktur.
- **Roughness-basierte Reflexionsumgebung** (reflektierter Himmel):
  `env_reflection = mix(bg_color, reflect(rd, N), fresnel)` als billige
  Umgebungsreflexion für Metalle — **empfohlen als Phase D1**, da billig
  (kein Extra-Pass) und großer visueller Gewinn für Metallic.

---

## 4. Implementierungs-Reihenfolge

| Schritt | Inhalt | Datei | Aufwand |
|---|---|---|---|
| 1 | ACES-Tone-Mapping | kernel.rs (Shading-Ende) | ~15 Zeilen |
| 2 | Cook-Torrance (D_GGX, G_Smith, Fresnel, F0-Mix) | kernel.rs (neue #[cube]-Fns) | ~80 Zeilen |
| 3 | Shading auf Cook-Torrance umstellen | kernel.rs:645–722 | ~60 Zeilen Umschreiben |
| 4 | Emissive farbig machen (× Slot-Farbe) | kernel.rs | ~5 Zeilen |
| 4b | Emissive-Lichter: CPU sammelt emissive Slots → Tensor | app_state.rs + gpu.rs + kernel.rs | ~60 Zeilen |
| 5 | Verifikation: check + clippy + release + Smoke-Test | — | — |

**Risiken:**

| Risiko | Gegenmaßnahme |
|---|---|
| cubecl unterstützt kein `powf` in allen Backends | `powf` wird bereits genutzt (kernel.rs:666) — sicher |
| Performance: Cook-Torrance × 3 Lichter pro Pixel | GGX/Smith/Fresnel sind ~20 ALU-Ops; bei 100 Raymarch-Steps vernachlässigbar |
| Visueller Bruch (Bild sieht komplett anders aus) | Beabsichtigt — das ist der Punkt der Optimierung. Alte Optik via `metallic=0, roughness=0.5, specular=0.5`-Defaults annähern |
| Emissive-Lichter brauchen Tensor-Erweiterung | env_settings hat Platz (Stride 13 → erweitern auf 16+) |

---

## 5. Akzeptanzkriterien

- [ ] Metallic-Slider: 0→1 wechselt sichtbar von matter Farbe zu farbigem Spiegel-Glanz (Albedo wird Reflexionsfarbe)
- [ ] Roughness-Slider: steuert die **Weichheit** des Glanzlichts (0.01 = Nadelglanz, 1.0 = breit/matt), nicht die Helligkeit
- [ ] Specular-Slider: steuert die Grund-Reflektivität (F0) unabhängig von Roughness
- criteria Emissive-Slider: 0→10 zeigt **farbiges** Glühen, das über 1.0 hinaus sichtbar zunimmt (kein Weiß-Clipping)
- [ ] Emissive-Objekte beleuchten umliegende Geometrie (Phase C)
- [ ] `cargo check` + `cargo clippy` + `cargo build --release` grün
- [ ] FPS bleibt im Rahmen (Ziel: ≥ 50% der aktuellen Performance)

---

*Analyse-Stand: kernel.rs (758 Zeilen), app_state.rs (250), gpu.rs (418),
shader.wgsl (42). Alle Zeilenangaben beziehen sich auf diese Versionen.*
