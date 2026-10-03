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
    pub slot_offsets_z: Vec<f32>,
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
            slot_offsets_z: vec![0.0, 0.0, 0.0, 1.8, -1.8],
        //
        
/*         // 🟢 NEU: Echte Material-Daten im CPU-State (verhindert Shader-Glitches)
            slot_roughness: vec![0.5, 0.5, 0.5, 0.5, 0.5],
            slot_metallic: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            slot_emissive: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            slot_specular: vec![0.5, 0.5, 0.5, 0.5, 0.5],
*/
            cam_x: 0.0,
            cam_y: 0.0,
            cam_z: -5.0,
            cam_yaw: 0.0,
            cam_pitch: 0.0,
            w_pressed: false,
            a_pressed: false,
            s_pressed: false,
            d_pressed: false,
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
        }
    }
}

impl ApplicationState {
    /// Meta-Tensor: Enthält die aktuelle Anzahl aktiver Slots für die GPU.
    pub fn meta_data(&self) -> [f32; 1] {
        [self.active_slots_count]
    }

    /// Slots-Tensor: Packt alle parallelen Slot-Vektoren in das vom 
    /// Kernel erwartete Interleaved-Format [Typ, Größe, X, Z, ...].
     /// Slots-Tensor: Packt alle parallelen Slot-Vektoren in das vom 
    /// Kernel erwartete 8er-Stride Interleaved-Format [Typ, Größe, X, Y, Z, R, G, B, ...].
    pub fn slots_data(&self) -> Vec<f32> {
        // Kapazität auf 8 Werte pro Slot erhöhen
        let mut raw_data = Vec::with_capacity(self.slot_types.len() * 8);
        for i in 0..self.slot_types.len() {
            raw_data.push(self.slot_types[i]);      // [0] Typ (1=Kristall, 2=Gyroid, 3=Torus)
            raw_data.push(self.slot_sizes[i]);      // [1] Größe
            raw_data.push(self.slot_offsets_x[i]);  // [2] Offset X
            raw_data.push(0.0f32);                  // [3] Offset Y (Standard 0.0, da nicht in CPU-State)
            raw_data.push(self.slot_offsets_z[i]);  // [4] Offset Z
            raw_data.push(1.0f32);                  // [5] R (Farbe Weiß als Default)
            raw_data.push(1.0f32);                  // [6] G
            raw_data.push(1.0f32);                  // [7] B
        }
        raw_data
    }
    
    /// Materials-Tensor: Definiert Standard-Materialien für die Slots (Stride 4).
    pub fn materials_data(&self) -> Vec<f32> {
        let mut raw_data = Vec::with_capacity(self.slot_types.len() * 4);
        for _ in 0..self.slot_types.len() {
            raw_data.push(0.5f32);  // roughness (0.5 = matt/normal)
            raw_data.push(0.0f32);  // metallic
            raw_data.push(0.0f32);  // emissive
            raw_data.push(0.5f32);  // specular (0.5 = Standard-Glanz)
        }
        raw_data
    }


    /// Tensor 4 (Umwelt): Licht-Pos, Intensität, Key-RGB, Ambient, BG-RGB,
    /// Nebel-Dichte, Nebel-Schalter, Padding. 16 f32 = 64 Bytes.
    pub fn env_data(&self) -> [f32; 16] {
        [
            self.key_light_x,
            self.key_light_y,
            self.key_light_z,
            self.light_intensity,
            self.key_r,
            self.key_g,
            self.key_b,
            self.ambient_strength,
            self.bg_r,
            self.bg_g,
            self.bg_b,
            self.fog_density,
            self.fog_enabled as f32,
            0.0,
            0.0,
            0.0,
        ]
    }

    /// Tensor 5 (Architektur): Säulen, Raum, Bögen, Dekor. 12 f32 = 48 Bytes.
    pub fn arch_data(&self) -> [f32; 12] {
        [
            self.pillar_dist,
            self.pillar_thick,
            self.room_height,
            self.ceiling_thick,
            self.arch_radius,
            self.arch_height,
            self.decor_freq,
            self.decor_depth,
            self.decor_thick,
            0.0,
            0.0,
            0.0,
        ]
    }

    /// Tensor 6 (Faltung): Zellgröße, halbe Zelle, Faltungs-Tempo. 4 f32 = 16 Bytes.
    pub fn fold_data(&self) -> [f32; 4] {
        [self.cell_size, self.cell_size * 0.5, self.fold_speed, 0.0]
    }

    // Meta: active_slots = 0 → nur statische Architektur rendert
    // (deterministisch, verlässt sich nicht auf zeroed VRAM).
    // meta_handle = client.empty(4) = 4 Bytes = 1 f32!
   // pub fn meta_data(&self) -> [f32; 1] {
     //   [0.0]
  //  }
}
