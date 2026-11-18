//! Geteilter Anwendungs-Zustand (CPU-seitig) + Packing in die GPU-Register.
//!
//! Die Struktur ist bewusst flach und `pub`: GUI (gui.rs), Input (input.rs)
//! und Render-Loop (render_loop.rs) greifen über `Arc<Mutex<ApplicationState>>`
//! auf dieselben Felder zu. Die `*_data()`-Methoden packen den State 1:1 in
//! die Tensor-4/5/6-Register-Arrays, die jede Frame per `queue.write_buffer`
//! in die cubecl-Buffer geschrieben werden.

/// Kompilierte Kernel-Register (Tensor 4/5/6) — Reihenfolge ist ABI mit
/// src/kernel.rs (env/arch/fold-Blöcke). NICHT umsortieren!
pub struct ApplicationState {
    // 🟢 DYNAMISCHES SLOT-SYSTEM STATE
    pub current_selected_slot: usize,
    pub active_slots_count: f32,
    pub slot_types: Vec<f32>,
    pub slot_sizes: Vec<f32>,
    pub slot_offsets_x: Vec<f32>,
    pub slot_offsets_y: Vec<f32>, // 🟢 NEU: Jetzt als echter State-Vektor
    pub slot_offsets_z: Vec<f32>,
    pub slot_r: Vec<f32>,
    pub slot_g: Vec<f32>,
    pub slot_b: Vec<f32>,
    // 🟢 NEU: Die 4 parallelen Vektoren für das PBR-Modell
    pub slot_roughness: Vec<f32>,
    pub slot_metallic: Vec<f32>,
    pub slot_emissive: Vec<f32>,
    pub slot_specular: Vec<f32>,
    // 🟢 NEU: Rotations-Vektoren (Euler-Winkel in Radianten) pro Slot
    pub slot_rot_x: Vec<f32>,
    pub slot_rot_y: Vec<f32>,
    pub slot_rot_z: Vec<f32>,
    // Kamera
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_z: f32,
    pub cam_yaw: f32,
    pub cam_pitch: f32,

    // Bewegungs-Flags (WASD)
    pub w_pressed: bool,
    pub a_pressed: bool,
    pub s_pressed: bool,
    pub d_pressed: bool,
    // 🟢 NEU: Kamera hoch/runter (Q/E)
    pub q_pressed: bool,
    pub e_pressed: bool,

    // Licht & Schatten
    pub light_intensity: f32,
    pub ambient_strength: f32,
    pub enable_ao_mode: u32,
    pub current_shadow_mode: u32,
    pub enable_key: u32,
    pub enable_fill: u32,
    pub enable_rim: u32,
    pub dynamic_blend_factor: f32,

    // 🟢 TENSOR 4: Umwelt-Register (Environmental Control)
    pub key_light_x: f32,
    pub key_light_y: f32,
    pub key_light_z: f32,
    pub key_r: f32,
    pub key_g: f32,
    pub key_b: f32,
    pub bg_r: f32,
    pub bg_g: f32,
    pub bg_b: f32,
    // 🟢 NEU: Dedizierte Nebelfarbe für das App-State-Management
    pub fog_r: f32,
    pub fog_g: f32,
    pub fog_b: f32,

    pub fog_density: f32,
    pub fog_enabled: u32,
    
    

    // 🟢 TENSOR 5: Tempel-Architektur (data-driven)
    pub pillar_dist: f32,
    pub pillar_thick: f32,
    pub room_height: f32,
    pub ceiling_thick: f32,
    pub arch_radius: f32,
    pub arch_height: f32,
    pub decor_freq: f32,
    pub decor_depth: f32,
    pub decor_thick: f32,

    // 🟢 TENSOR 6: Unendliche Raumfaltung (modulo grid)
    pub cell_size: f32,
    pub fold_speed: f32,

    // 🟢 PARAM-GUI: Atmosphäre & Lighting (Env-Tab)
 // 🟢 NEU: Parameter-GUI Steuerung für den volumetrischen Nebel
    pub volumetric_fog_steps: u32,  // Nebel-Integrationsschritte (z.B. Default: 16)
    pub fog_density_falloff: f32,   // Höhenprofil-Falloff (z.B. Default: 0.08)
    pub min_height_falloff: f32,    // Minimale Dichte in der Höhe (z.B. Default: 0.1)
    pub max_optical_depth: f32,     // Grenzwert für Opazitäts-Abbruch (z.B. Default: 4.605)
    pub fog_sample_offset: f32,     // Sampling-Position im Segment (z.B. Default: 0.5)
    
    pub ao_iterations: u32,        // AO-Sampling-Iterationen (1..10)
    pub shadow_softness_k: f32,     // Weichheits-Konstante (1.0..64.0)

    // 🟢 PARAM-GUI: Infinite Repetition (Fold-Tab)
    pub pillar_spacing: f32,       // XZ-Wiederholungs-Abstand (2.0..20.0)
    pub pillar_height: f32,        // Pfeilerhöhe (1.0..10.0)

    // 🟢 PARAM-GUI: Materialien & Reflexionen (Mat-Tab)
    pub reflection_max_steps: u32,     // Sekundär-Schritte (8..64)
    pub reflection_roughness_gate: f32, // Aktivierungsschwelle (0.05..1.0)
    pub fresnel_strength_factor: f32,  // Reflexions-Stärke (0.0..1.0)
    pub pillar_roughness: f32,         // Pfeiler PBR (ID 1)

    // 🟢 SDF-OBJEKT-TENSOR: N Grid-Objekt-Slots (Analogie zum dynamischen
    // Slot-Mechanismus). Jeder Slot: Pfad zum gebackenen Grid (leer =
    // inaktiv) + Transform + Albedo/PBR. Parallele Vektoren, Index =
    // Slot-Index (0..MAX_SDF_OBJECTS).
    pub sdf_paths: Vec<String>,
    pub sdf_active: Vec<bool>,
    pub sdf_sizes: Vec<f32>,
    pub sdf_center_x: Vec<f32>,
    pub sdf_center_y: Vec<f32>,
    pub sdf_center_z: Vec<f32>,
    pub sdf_rot_x: Vec<f32>,
    pub sdf_rot_y: Vec<f32>,
    pub sdf_rot_z: Vec<f32>,
    pub sdf_r: Vec<f32>,
    pub sdf_g: Vec<f32>,
    pub sdf_b: Vec<f32>,
    pub sdf_roughness: Vec<f32>,
    pub sdf_metallic: Vec<f32>,
    pub sdf_emissive: Vec<f32>,
    pub sdf_specular: Vec<f32>,
    pub current_selected_sdf_object: usize,

    // 🟢 PARAM-GUI: Objekt / Statue-Konfiguration (Obj-Tab)
    pub bounding_sphere_radius_offset: f32, // Proximity-Schranke (default 0.1)
    pub mesh_scale_factor: f32,             // Mesh-Normalisierung (0.1..1.0, default 0.8)

    // 🟢 DoF (Depth of Field) — Post-Processing via cubek-convolution.
    // Gesteuert vom Ctrl-Tab; `dof_enabled = 0` bedeutet Bypass (Direkt-Render).
    pub dof_enabled: u32,       // 0/1-Schalter
    pub dof_focal_distance: f32, // Schärfeentfernung (0.5..40.0, default 8.0)
    pub dof_aperture: f32,       // Blendenöffnung / Unschärfe-Stärke (0.0..5.0, default 0.6)
    pub dof_kernel_size: u32,   // Bokeh-Disk-Seitenlänge K (3..15, ungerade, default 7)
}

impl Default for ApplicationState {
    fn default() -> Self {
        Self {
        // 
            current_selected_slot: 0,
            active_slots_count: 5.0,
            slot_types: vec![1.0, 2.0, 3.0, 0.0, 0.0],
            slot_sizes: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            slot_offsets_x: vec![0.0, -1.8, 1.8, 0.0, 0.0],
            slot_offsets_y: vec![0.0,  0.0, 0.0, 0.0, 0.0], // 🟢 Initialisiert auf der Nullebene
            slot_offsets_z: vec![0.0, 0.0, 0.0, 1.8, -1.8],
            // Farb-Zuordnung (Rot, Grün, Blau, Weiß, Weiß)
            slot_r:         vec![1.0, 0.0, 0.0, 1.0, 1.0],
            slot_g:         vec![0.0, 1.0, 0.0, 1.0, 1.0],
            slot_b:         vec![0.0, 0.0, 1.0, 1.0, 1.0],
        
            // 🟢 NEU: PBR-Startwerte für die 5 Slots aus der Doc
            slot_roughness: vec![0.2, 0.5, 0.1, 0.5, 0.5], 
            slot_metallic:  vec![1.0, 0.0, 0.8, 0.0, 0.0], 
            slot_emissive:  vec![0.0, 0.0, 0.0, 0.0, 0.0],
            slot_specular:  vec![1.0, 0.5, 1.0, 0.5, 0.5],
            // 🟢 NEU: Rotation (Euler-Winkel in Radianten) — initial 0
            slot_rot_x: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            slot_rot_y: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            slot_rot_z: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            // ... restliche Felder ...
            cam_x: 0.0,
            cam_y: 0.0,
            cam_z: -5.0,
            cam_yaw: 0.0,
            cam_pitch: 0.0,
            w_pressed: false,
            a_pressed: false,
            s_pressed: false,
            d_pressed: false,
            q_pressed: false,
            e_pressed: false,
            light_intensity: 1.0,
            ambient_strength: 0.1,
            enable_ao_mode: 1,
            current_shadow_mode: 1,
            enable_key: 1,
            enable_fill: 1,
            enable_rim: 1,
            dynamic_blend_factor: 0.5,

            // 🟢 TENSOR 4: Umwelt-Defaults (Blueprint: Environmental Control Tensor)
            key_light_x: 4.0,
            key_light_y: 7.0,
            key_light_z: -4.0,
            key_r: 1.00,
            key_g: 0.95,
            key_b: 0.85,
            bg_r: 0.35,
            bg_g: 0.45,
            bg_b: 0.60,
            fog_density: 0.5,
            // Original-Kernel hatte keinen Nebel-Block → default AUS,
            // damit die Szene exakt wie das Original aussieht.
            fog_enabled: 0,

            // 🟢 TENSOR 5: Tempel-Architektur-Defaults
            pillar_dist: 5.0,
            pillar_thick: 0.6,
            room_height: 3.0,
            ceiling_thick: 0.1,
            arch_radius: 3.2,
            arch_height: 1.0,
            decor_freq: 2.0,
            decor_depth: 0.03,
            decor_thick: 0.01,

            // 🟢 TENSOR 6: Raumfaltungs-Defaults
            cell_size: 10.0,
            fold_speed: 1.0,

          
            // 🟢 NEU: Standard-Nebelfarbe (z.B. ein neutrales Grauweiß/Nebelgrau)
            fog_r: 0.7,
            fog_g: 0.7,
            fog_b: 0.7,

            // 🟢 PARAM-GUI: Atmosphäre & Lighting-Defaults
            volumetric_fog_steps: 16,
            fog_density_falloff: 0.08,
            
            // 🟢 NEU: Die neuen erweiterten Nebel-Parameter
            min_height_falloff: 0.1,      // Verhindert das komplette Verschwinden in der Höhe
            max_optical_depth: 4.605,     // Entspricht ~1% Restlicht (optimaler Abbruchwert)
            fog_sample_offset: 0.5,       // Standard-Raymarching (Mitte des Segments)

            ao_iterations: 5,
            shadow_softness_k: 8.0,

            // 🟢 PARAM-GUI: Infinite Repetition-Defaults
            pillar_spacing: 6.0,
            pillar_height: 4.0,

            // 🟢 PARAM-GUI: Materialien & Reflexionen-Defaults
            reflection_max_steps: 32,
            reflection_roughness_gate: 0.25,
            fresnel_strength_factor: 0.35,
            pillar_roughness: 0.8,

            // 🟢 SDF-OBJEKT-TENSOR-Defaults: Slot 1 = Statue (Ex-Verhalten
            // 1:1 erhalten: Zentrum (0,-1.4,0), Größe 4.0, Albedo
            // (0.85,0.7,0.3), roughness 0.35, metallic 0.8). Slots 2..4 inaktiv.
            sdf_paths: vec![
                "assets/statue.obj".to_string(),
                String::new(),
                String::new(),
                String::new(),
            ],
            sdf_active: vec![true, false, false, false],
            sdf_sizes: vec![4.0, 4.0, 4.0, 4.0],
            sdf_center_x: vec![0.0, 3.0, -3.0, 0.0],
            sdf_center_y: vec![-1.4, -1.4, -1.4, -1.4],
            sdf_center_z: vec![0.0, 0.0, 0.0, 3.0],
            sdf_rot_x: vec![0.0, 0.0, 0.0, 0.0],
            sdf_rot_y: vec![0.0, 0.0, 0.0, 0.0],
            sdf_rot_z: vec![0.0, 0.0, 0.0, 0.0],
            sdf_r: vec![0.85, 0.7, 0.6, 0.6],
            sdf_g: vec![0.7, 0.8, 0.7, 0.7],
            sdf_b: vec![0.3, 0.85, 0.75, 0.75],
            sdf_roughness: vec![0.35, 0.4, 0.4, 0.4],
            sdf_metallic: vec![0.8, 0.3, 0.3, 0.3],
            sdf_emissive: vec![0.0, 0.0, 0.0, 0.0],
            sdf_specular: vec![0.5, 0.5, 0.5, 0.5],
            current_selected_sdf_object: 0,

            // 🟢 PARAM-GUI: Objekt / Statue-Konfiguration-Defaults
            bounding_sphere_radius_offset: 0.1,
            mesh_scale_factor: 0.8,

            // 🟢 DoF-Defaults: aktiv, Fokus 8.0, moderate Blende, 7×7 Bokeh-Disk
            dof_enabled: 1,
            dof_focal_distance: 8.0,
            dof_aperture: 0.6,
            dof_kernel_size: 7,
        }
    }
}

use crate::kernel::{
    ArchParamsLaunch, CameraSettingsLaunch, EnvSettingsLaunch, FoldParamsLaunch,
    GpuVec3Launch, RenderSettingsLaunch,
};
// 🟢 Bringe den zentralen Typ-Alias in den Scope (Passe den Pfad ggf. an deine Struktur an)
 use crate::gpu::GpuFloat;
use std::sync::Arc;


impl ApplicationState {
    /// Kamera-Struct (params5): Position + Yaw/Pitch als 16-Byte-Alignment-Struct.
  pub fn camera_launch(&self) -> CameraSettingsLaunch<GpuFloat> {
        CameraSettingsLaunch::<GpuFloat> {
            position: GpuVec3Launch {
                // Wenn deine State-Variablen (cam_x, etc.) normale f32 sind,
                // mappen sie direkt ohne Cast, solange GpuFloat = f32 ist.
                x: self.cam_x as GpuFloat,
                y: self.cam_y as GpuFloat,
                z: self.cam_z as GpuFloat,
                _pad: 0.0_f32 as GpuFloat, // Verhindert future-incompatibility Warnungen
            },
            yaw: self.cam_yaw as GpuFloat,
            pitch: self.cam_pitch as GpuFloat,
        }
    }
    /// Slots-Tensor: Packt alle parallelen Slot-Vektoren in das vom 
    /// Kernel erwartete Interleaved-Format [Typ, Größe, X, Z, ...].
     /// Slots-Tensor: Packt alle parallelen Slot-Vektoren in das vom 
    /// Kernel erwartete 8er-Stride Interleaved-Format [Typ, Größe, X, Y, Z, R, G, B, ...].
    /// Slots-Tensor: Packt alle 8 CPU-Kanäle dynamisch und ohne Abstraktionsverlust
    /// in das vom Kernel geforderte Stride-8 Interleaved Format.
  /// Packt Slots, Materials und Rotations in einen einzigen, zusammenhängenden f32-Buffer (Größe: 1500 f32s)
pub fn scene_data(&self) -> Vec<GpuFloat> {
    const MAX_SLOTS: usize = 100;
    
    // Gesamtgröße: (100 * 8) + (100 * 4) + (100 * 3) = 1500 Elemente
    const TOTAL_SIZE: usize = (MAX_SLOTS * 8) + (MAX_SLOTS * 4) + (MAX_SLOTS * 3);
    
    // Mit typisierten Nullen vorinitialisieren (GpuFloat)
    let mut packed_data = vec![0.0_f32 as GpuFloat; TOTAL_SIZE];
    
    // Exakte Start-Offsets im Ziel-Array definieren
    let slots_offset = 0;
    let materials_offset = MAX_SLOTS * 8;                  // Index 800
    let rotations_offset = materials_offset + (MAX_SLOTS * 4); // Index 1200

    // Schleife über alle aktuell existierenden CPU-Einträge (maximal bis MAX_SLOTS)
    for i in 0..self.slot_types.len() {
        if i >= MAX_SLOTS { break; }

        // 1. SKALARE & POSITIONS-SLOTS PACKEN (Stride 8)
        let s_base = slots_offset + (i * 8);
        packed_data[s_base]     = self.slot_types[i] as GpuFloat;      // Typ
        packed_data[s_base + 1] = self.slot_sizes[i] as GpuFloat;      // Größe (Scale)
        packed_data[s_base + 2] = self.slot_offsets_x[i] as GpuFloat;  // Position X
        packed_data[s_base + 3] = self.slot_offsets_y[i] as GpuFloat;  // Position Y
        packed_data[s_base + 4] = self.slot_offsets_z[i] as GpuFloat;  // Position Z
        packed_data[s_base + 5] = self.slot_r[i] as GpuFloat;          // Farbe R
        packed_data[s_base + 6] = self.slot_g[i] as GpuFloat;          // Farbe G
        packed_data[s_base + 7] = self.slot_b[i] as GpuFloat;          // Farbe B

        // 2. MATERIALIEN PACKEN (Stride 4)
        let m_base = materials_offset + (i * 4);
        packed_data[m_base]     = self.slot_roughness[i] as GpuFloat;
        packed_data[m_base + 1] = self.slot_metallic[i] as GpuFloat;
        packed_data[m_base + 2] = self.slot_emissive[i] as GpuFloat;
        packed_data[m_base + 3] = self.slot_specular[i] as GpuFloat;

        // 3. ROTATIONEN PACKEN (Stride 3)
        // 🟢 FIX: Deklaration von r_base wieder hinzugefügt!
        let r_base = rotations_offset + (i * 3);
        packed_data[r_base]     = *self.slot_rot_x.get(i).unwrap_or(&0.0_f32) as GpuFloat;
        packed_data[r_base + 1] = *self.slot_rot_y.get(i).unwrap_or(&0.0_f32) as GpuFloat;
        packed_data[r_base + 2] = *self.slot_rot_z.get(i).unwrap_or(&0.0_f32) as GpuFloat;
    }

    // 🟢 SDF-OBJEKT-TENSOR: Das gebackene Grid hängt NICHT mehr am
    // Mega-Scene-Buffer — es lebt im separaten Objekt-Tensor
    // (siehe sdf_objects_data()). Die Größe ist wieder 1500 f32.
    packed_data
}

/// 🟢 OPTIMIZATION 1: Slots separat packen (Stride 8 → Flat-F32-Buffer).
/// Ersetzt `scene_data()` + Offsets zugunsten eines eigenen GPU-Buffers.
pub fn slots_data(&self) -> Vec<GpuFloat> {
    const MAX_SLOTS: usize = 100;
    let mut packed = vec![0.0_f32 as GpuFloat; MAX_SLOTS * 8];
    for i in 0..self.slot_types.len() {
        if i >= MAX_SLOTS { break; }
        let base = i * 8;
        packed[base]     = self.slot_types[i] as GpuFloat;
        packed[base + 1] = self.slot_sizes[i] as GpuFloat;
        packed[base + 2] = self.slot_offsets_x[i] as GpuFloat;
        packed[base + 3] = self.slot_offsets_y[i] as GpuFloat;
        packed[base + 4] = self.slot_offsets_z[i] as GpuFloat;
        packed[base + 5] = self.slot_r[i] as GpuFloat;
        packed[base + 6] = self.slot_g[i] as GpuFloat;
        packed[base + 7] = self.slot_b[i] as GpuFloat;
    }
    packed
}

/// 🟢 OPTIMIZATION 1: Materialien separat packen (Stride 4 → Flat-F32-Buffer).
pub fn materials_data(&self) -> Vec<GpuFloat> {
    const MAX_SLOTS: usize = 100;
    let mut packed = vec![0.0_f32 as GpuFloat; MAX_SLOTS * 4];
    for i in 0..self.slot_types.len() {
        if i >= MAX_SLOTS { break; }
        let base = i * 4;
        packed[base]     = self.slot_roughness[i] as GpuFloat;
        packed[base + 1] = self.slot_metallic[i] as GpuFloat;
        packed[base + 2] = self.slot_emissive[i] as GpuFloat;
        packed[base + 3] = self.slot_specular[i] as GpuFloat;
    }
    packed
}

/// 🟢 OPTIMIZATION 1: Rotationen separat packen (Stride 3 → Flat-F32-Buffer).
pub fn rotations_data(&self) -> Vec<GpuFloat> {
    const MAX_SLOTS: usize = 100;
    let mut packed = vec![0.0_f32 as GpuFloat; MAX_SLOTS * 3];
    for i in 0..self.slot_types.len() {
        if i >= MAX_SLOTS { break; }
        let base = i * 3;
        packed[base]     = *self.slot_rot_x.get(i).unwrap_or(&0.0_f32) as GpuFloat;
        packed[base + 1] = *self.slot_rot_y.get(i).unwrap_or(&0.0_f32) as GpuFloat;
        packed[base + 2] = *self.slot_rot_z.get(i).unwrap_or(&0.0_f32) as GpuFloat;
    }
    packed
}

/// 🟢 SDF-OBJEKT-TENSOR: Packt die N Grid-Objekt-Slots in den separaten
/// Objekt-Tensor. Layout (doc/feature/SDF_OBJECT_TENSOR.md):
/// `[ Header 0 (Stride 24) ]…[ Header N-1 ][ Grid 0 (64³) ]…[ Grid N-1 ]`
///
/// Header (f32): active, res, size, pad, center(3)+pad, rot(3)+pad,
/// albedo(3)+roughness, metallic+emissive+specular+pad.
/// Grids kommen aus dem Prozess-Cache (`get_or_bake_grid`) — leere/
/// inaktive Slots schreiben ein "weit weg"-Grid (d = 1e6).
pub fn sdf_objects_data(&self) -> Vec<GpuFloat> {
    use crate::gpu::{MAX_SDF_OBJECTS, SDF_HEADER_STRIDE, SDF_GRID_COUNT};

    const N: usize = MAX_SDF_OBJECTS;
    let grid_count = SDF_GRID_COUNT;
    let total = N * SDF_HEADER_STRIDE + N * grid_count;
    let mut packed = vec![0.0_f32 as GpuFloat; total];

    // Leeres "weit weg"-Grid für inaktive Slots (prozessweit geteilt).
    static EMPTY_GRID: std::sync::OnceLock<Arc<Vec<f32>>> = std::sync::OnceLock::new();
    let empty = EMPTY_GRID.get_or_init(|| Arc::new(vec![1.0e6_f32; grid_count]));

    let mut grids: Vec<Arc<Vec<f32>>> = Vec::with_capacity(N);
    for slot in 0..N {
        let path = self.sdf_paths.get(slot).map(|s| s.as_str()).unwrap_or("");
        let scale = self.mesh_scale_factor;
        if !path.is_empty() && self.sdf_active.get(slot).copied().unwrap_or(false) {
            grids.push(crate::sdf_preprocessor::get_or_bake_grid(path, scale));
        } else {
            grids.push(empty.clone());
        }
    }

    for slot in 0..N {
        // --- Header (Stride 24) ---
        let hdr = slot * SDF_HEADER_STRIDE;
        let active = self.sdf_active.get(slot).copied().unwrap_or(false)
            && !self.sdf_paths.get(slot).map(|s| s.is_empty()).unwrap_or(true);
        packed[hdr]     = if active { 1.0 as GpuFloat } else { 0.0 as GpuFloat };
        packed[hdr + 1] = crate::gpu::SDF_GRID_RES as f32 as GpuFloat; // res
        packed[hdr + 2] = *self.sdf_sizes.get(slot).unwrap_or(&4.0) as GpuFloat;
        packed[hdr + 3] = 0.0 as GpuFloat; // pad
        packed[hdr + 4] = *self.sdf_center_x.get(slot).unwrap_or(&0.0) as GpuFloat;
        packed[hdr + 5] = *self.sdf_center_y.get(slot).unwrap_or(&0.0) as GpuFloat;
        packed[hdr + 6] = *self.sdf_center_z.get(slot).unwrap_or(&0.0) as GpuFloat;
        packed[hdr + 7] = 0.0 as GpuFloat; // pad
        packed[hdr + 8]  = *self.sdf_rot_x.get(slot).unwrap_or(&0.0) as GpuFloat;
        packed[hdr + 9]  = *self.sdf_rot_y.get(slot).unwrap_or(&0.0) as GpuFloat;
        packed[hdr + 10] = *self.sdf_rot_z.get(slot).unwrap_or(&0.0) as GpuFloat;
        packed[hdr + 11] = 0.0 as GpuFloat; // pad
        packed[hdr + 12] = *self.sdf_r.get(slot).unwrap_or(&0.85) as GpuFloat;
        packed[hdr + 13] = *self.sdf_g.get(slot).unwrap_or(&0.7) as GpuFloat;
        packed[hdr + 14] = *self.sdf_b.get(slot).unwrap_or(&0.3) as GpuFloat;
        packed[hdr + 15] = *self.sdf_roughness.get(slot).unwrap_or(&0.35) as GpuFloat;
        packed[hdr + 16] = *self.sdf_metallic.get(slot).unwrap_or(&0.8) as GpuFloat;
        packed[hdr + 17] = *self.sdf_emissive.get(slot).unwrap_or(&0.0) as GpuFloat;
        packed[hdr + 18] = *self.sdf_specular.get(slot).unwrap_or(&0.5) as GpuFloat;
        // 19..23 pad — bleiben 0 (vec-initialisiert)
    }

    // --- Grids hinter den Headern ---
    for slot in 0..N {
        let g_base = N * SDF_HEADER_STRIDE + slot * grid_count;
        let grid = &grids[slot];
        for (k, val) in grid.iter().enumerate() {
            packed[g_base + k] = *val as GpuFloat;
        }
    }

    packed
}
    /// Umwelt-Struct (ersetzt Tensor 4 / env_data()[16]): Licht, Farben, Nebel.
    /// fog_enabled als u32-Flag 0/1 (params3) statt f32.
   pub fn env_launch(&self) -> EnvSettingsLaunch<GpuFloat> {
        EnvSettingsLaunch::<GpuFloat> {
            key_light_pos: GpuVec3Launch {
                x: self.key_light_x as GpuFloat,
                y: self.key_light_y as GpuFloat,
                z: self.key_light_z as GpuFloat,
                _pad: 0.0_f32 as GpuFloat,
            },
            light_intensity: self.light_intensity as GpuFloat,
            key_color: GpuVec3Launch {
                x: self.key_r as GpuFloat,
                y: self.key_g as GpuFloat,
                z: self.key_b as GpuFloat,
                _pad: 0.0_f32 as GpuFloat,
            },
            ambient_strength: self.ambient_strength as GpuFloat,
            bg_color: GpuVec3Launch {
                x: self.bg_r as GpuFloat,
                y: self.bg_g as GpuFloat,
                z: self.bg_b as GpuFloat,
                _pad: 0.0_f32 as GpuFloat,
            },
            // 🟢 KORREKTUR: Typparameter auf GpuFloat umgestellt
            fog_color: GpuVec3Launch::<GpuFloat> {
                x: self.fog_r as GpuFloat,
                y: self.fog_g as GpuFloat,
                z: self.fog_b as GpuFloat,
                _pad: 0.0_f32 as GpuFloat,
            },
            fog_density: self.fog_density as GpuFloat,
            // Ein Boolean bleibt im Host-Struct erhalten, da CubeLaunch ihn für den Verzweigungs-Zustand mappt
            fog_enabled: self.fog_enabled, 
            
            // 🟢 PARAM-GUI: Atmosphäre & Lighting
            volumetric_fog_steps: self.volumetric_fog_steps, // bleibt Ganzzahl (u32/i32)
            fog_density_falloff: self.fog_density_falloff as GpuFloat,
            
            // 🟢 NEU: Die erweiterten Nebelsteuerungsparameter übergeben
            min_height_falloff: self.min_height_falloff as GpuFloat,
            max_optical_depth: self.max_optical_depth as GpuFloat,
            fog_sample_offset: self.fog_sample_offset as GpuFloat,
            
            ao_iterations: self.ao_iterations, // bleibt Ganzzahl
            shadow_softness_k: self.shadow_softness_k as GpuFloat,
        }
    }

    /// Architektur-Struct (ersetzt Tensor 5 / arch_data()[12]).
    pub fn arch_launch(&self) -> ArchParamsLaunch<GpuFloat> {
        ArchParamsLaunch::<GpuFloat> {
            pillar_dist: self.pillar_dist as GpuFloat,
            pillar_thick: self.pillar_thick as GpuFloat,
            room_height: self.room_height as GpuFloat,
            ceiling_thick: self.ceiling_thick as GpuFloat,
            arch_radius: self.arch_radius as GpuFloat,
            arch_height: self.arch_height as GpuFloat,
            decor_freq: self.decor_freq as GpuFloat,
            decor_depth: self.decor_depth as GpuFloat,
            decor_thick: self.decor_thick as GpuFloat,
        }
    }

    /// Faltungs-Struct (ersetzt Tensor 6 / fold_data()[4]).
    pub fn fold_launch(&self) -> FoldParamsLaunch<GpuFloat> {
        FoldParamsLaunch::<GpuFloat> {
            cell_size: self.cell_size as GpuFloat,
            // 🟢 KORREKTUR: Multiplikation mit typisiertem Suffix, um Fallback-Warnungen zu umgehen
            half_cell: (self.cell_size * 0.5_f32) as GpuFloat,
            fold_speed: self.fold_speed as GpuFloat,
            // 🟢 PARAM-GUI: Infinite Repetition
            pillar_spacing: self.pillar_spacing as GpuFloat,
            pillar_height: self.pillar_height as GpuFloat,
        }
    }


    /// Render-Struct (ersetzt Meta-Tensor + 12 lose Launch-Skalare).
    /// active_slots als u32-Count (params3).
    pub fn render_launch(&self, time: GpuFloat, width: u32, height: u32) -> RenderSettingsLaunch<GpuFloat> {
        RenderSettingsLaunch::<GpuFloat> {
            time,
            width,
            height,
            blend_factor: self.dynamic_blend_factor as GpuFloat,
            active_slots: self.active_slots_count as u32,
            shadow_mode: self.current_shadow_mode,
            enable_ao_mode: self.enable_ao_mode,
            enable_key: self.enable_key,
            enable_fill: self.enable_fill,
            enable_rim: self.enable_rim,
            
            // 🟢 PARAM-GUI: Materialien & Reflexionen
            reflection_max_steps: self.reflection_max_steps, // bleibt u32 / Ganzzahl
            reflection_roughness_gate: self.reflection_roughness_gate as GpuFloat,
            fresnel_strength_factor: self.fresnel_strength_factor as GpuFloat,
            pillar_roughness: self.pillar_roughness as GpuFloat,

            // 🟢 SDF-OBJEKT-TENSOR: Zentrum/Größe/PBR pro Slot im Objekt-Tensor.
            bounding_sphere_radius_offset: self.bounding_sphere_radius_offset as GpuFloat,
        }
    }

    /// 🟢 DoF: Fokus-Distanz und Blendenöffnung als GPU-Skalare
    /// (direkte Kernel-Parameter für `compute_coc_and_mask_kernel`).
    pub fn dof_focal(&self) -> GpuFloat {
        self.dof_focal_distance as GpuFloat
    }

    pub fn dof_aperture_scalar(&self) -> GpuFloat {
        self.dof_aperture as GpuFloat
    }

    /// 🟢 DoF: Ist der Post-Pass aktiv?
    pub fn dof_is_enabled(&self) -> bool {
        self.dof_enabled != 0
    }
}
