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
        }
    }
}

use crate::kernel::{
    ArchParamsLaunch, CameraSettingsLaunch, EnvSettingsLaunch, FoldParamsLaunch,
    GpuVec3Launch, RenderSettingsLaunch,
};

impl ApplicationState {
    /// Kamera-Struct (params5): Position + Yaw/Pitch als 16-Byte-Alignment-Struct.
    pub fn camera_launch(&self) -> CameraSettingsLaunch {
        CameraSettingsLaunch {
            position: GpuVec3Launch {
                x: self.cam_x,
                y: self.cam_y,
                z: self.cam_z,
                _pad: 0.0,
            },
            yaw: self.cam_yaw,
            pitch: self.cam_pitch,
        }
    }

    /// Slots-Tensor: Packt alle parallelen Slot-Vektoren in das vom 
    /// Kernel erwartete Interleaved-Format [Typ, Größe, X, Z, ...].
     /// Slots-Tensor: Packt alle parallelen Slot-Vektoren in das vom 
    /// Kernel erwartete 8er-Stride Interleaved-Format [Typ, Größe, X, Y, Z, R, G, B, ...].
    /// Slots-Tensor: Packt alle 8 CPU-Kanäle dynamisch und ohne Abstraktionsverlust
    /// in das vom Kernel geforderte Stride-8 Interleaved Format.
    pub fn slots_data(&self) -> Vec<f32> {
        let mut raw_data = Vec::with_capacity(self.slot_types.len() * 8);
        for i in 0..self.slot_types.len() {
            raw_data.push(self.slot_types[i]);      // [0] Typ
            raw_data.push(self.slot_sizes[i]);      // [1] Größe (Scale)
            raw_data.push(self.slot_offsets_x[i]);  // [2] Position X
            raw_data.push(self.slot_offsets_y[i]);  // [3] Position Y 🟢 (Jetzt dynamisch!)
            raw_data.push(self.slot_offsets_z[i]);  // [4] Position Z
            raw_data.push(self.slot_r[i]);          // [5] Farbe R 🟢
            raw_data.push(self.slot_g[i]);          // [6] Farbe G 🟢
            raw_data.push(self.slot_b[i]);          // [7] Farbe B 🟢
        }
        raw_data
    }
    
    /// Materials-Tensor: Erzeugt das finale 400-f32 flache PBR-Register-Array
    /// (100 Slots max * Stride 4). Verhindert unaligned cross-boundary reads.
    pub fn materials_data(&self) -> Vec<f32> {
        const MAX_SLOTS: usize = 100;
        let mut dynamic_materials = vec![0.0f32; MAX_SLOTS * 4]; // Flat 400 capacity
        
        for i in 0..self.slot_types.len() {
            if i >= MAX_SLOTS { break; }
            let base = i * 4;
            dynamic_materials[base]     = self.slot_roughness[i];
            dynamic_materials[base + 1] = self.slot_metallic[i];
            dynamic_materials[base + 2] = self.slot_emissive[i];
            dynamic_materials[base + 3] = self.slot_specular[i];
        }
        dynamic_materials
    }


    /// Rotations-Tensor: Erzeugt das flache Rotations-Register-Array
    /// (100 Slots max * Stride 3 = 300 f32). Reihenfolge [rotX, rotY, rotZ] pro Slot.
      /// Rotations-Tensor: Erzeugt das flache Rotations-Register-Array
    /// (100 Slots max * Stride 3 = 300 f32). Reihenfolge [rotX, rotY, rotZ] pro Slot.
    pub fn rotations_data(&self) -> Vec<f32> {
        const MAX_SLOTS: usize = 100;
        let mut dynamic_rotations = vec![0.0f32; MAX_SLOTS * 3];
        for i in 0..self.slot_types.len() {
            if i >= MAX_SLOTS { break; }
            let base = i * 3;
            
            // Nutze .get(), um IndexOutOfBounds zu verhindern, falls die Arrays noch zu kurz sind
            dynamic_rotations[base]     = *self.slot_rot_x.get(i).unwrap_or(&0.0f32);
            dynamic_rotations[base + 1] = *self.slot_rot_y.get(i).unwrap_or(&0.0f32);
            dynamic_rotations[base + 2] = *self.slot_rot_z.get(i).unwrap_or(&0.0f32);
        }
        dynamic_rotations
    }

    /// Umwelt-Struct (ersetzt Tensor 4 / env_data()[16]): Licht, Farben, Nebel.
    /// fog_enabled als u32-Flag 0/1 (params3) statt f32.
    pub fn env_launch(&self) -> EnvSettingsLaunch {
        EnvSettingsLaunch {
            key_light_pos: GpuVec3Launch {
                x: self.key_light_x,
                y: self.key_light_y,
                z: self.key_light_z,
                _pad: 0.0,
            },
            light_intensity: self.light_intensity,
            key_color: GpuVec3Launch {
                x: self.key_r,
                y: self.key_g,
                z: self.key_b,
                _pad: 0.0,
            },
            ambient_strength: self.ambient_strength,
            bg_color: GpuVec3Launch {
                x: self.bg_r,
                y: self.bg_g,
                z: self.bg_b,
                _pad: 0.0,
            },
            fog_density: self.fog_density,
            fog_enabled: self.fog_enabled,
        }
    }

    /// Architektur-Struct (ersetzt Tensor 5 / arch_data()[12]).
    pub fn arch_launch(&self) -> ArchParamsLaunch {
        ArchParamsLaunch {
            pillar_dist: self.pillar_dist,
            pillar_thick: self.pillar_thick,
            room_height: self.room_height,
            ceiling_thick: self.ceiling_thick,
            arch_radius: self.arch_radius,
            arch_height: self.arch_height,
            decor_freq: self.decor_freq,
            decor_depth: self.decor_depth,
            decor_thick: self.decor_thick,
        }
    }

    /// Faltungs-Struct (ersetzt Tensor 6 / fold_data()[4]).
    pub fn fold_launch(&self) -> FoldParamsLaunch {
        FoldParamsLaunch {
            cell_size: self.cell_size,
            half_cell: self.cell_size * 0.5,
            fold_speed: self.fold_speed,
        }
    }

    /// Render-Struct (ersetzt Meta-Tensor + 12 lose Launch-Skalare).
    /// active_slots als u32-Count (params3).
    pub fn render_launch(&self, time: f32, width: u32, height: u32) -> RenderSettingsLaunch {
        RenderSettingsLaunch {
            time,
            width,
            height,
            blend_factor: self.dynamic_blend_factor,
            active_slots: self.active_slots_count as u32,
            shadow_mode: self.current_shadow_mode,
            enable_ao_mode: self.enable_ao_mode,
            enable_key: self.enable_key,
            enable_fill: self.enable_fill,
            enable_rim: self.enable_rim,
        }
    }

}
