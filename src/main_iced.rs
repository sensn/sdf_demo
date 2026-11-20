//! sdf_demo - iced 0.15 port.
//!
//! Architecture:
//! - `SdfScene` holds the application state and implements `shader::Program`.
//! - `SdfPrimitive` carries per-frame data, implements `shader::Primitive`.
//! - `SdfPipeline` owns the wgpu pipeline + CubeCL client, implements
//!   `shader::Pipeline`.
//! - The GUI is built inline with iced widgets (blueprint: custom_shader,
//!   events and editor examples of iced 0.15-dev).
//!
//! Layout:
//! ```text
//! ┌──────────────────────────────────────────────┬───────────────┐
//! │ menubar (Datei + popover)          [FPS]     │  Tab buttons  │
//! ├──────────────────────────────────────────────┤  ┌───────────┐│
//! │            shader (SdfScene)                 │  │scrollable ││
//! │      (mouse move = look, wheel = look)       │  │ accordions││
//! │                                              │  │  sliders  ││
//! └──────────────────────────────────────────────┴──┴───────────┴┘
//! ```

mod app_state;
mod gpu_iced;
mod kernel;
mod sdf_preprocessor;
mod shader_program;

use shader_program::SdfScene;
use app_state::ApplicationState;

use std::path::PathBuf;

use iced::time::Instant;
use iced::widget::{button, center, column, container, popover, row, scrollable, shader, slider, space, stack, text, toggler};
use iced::{Alignment, Color, Fill, Padding, Subscription, Task, Widget};
use iced::{event, keyboard, mouse, window};
use iced::widget::{mouse_area, opaque};

fn main() -> iced::Result {
    iced::application(IcedSdf::default, IcedSdf::update, IcedSdf::view)
        .subscription(IcedSdf::subscription)
        .theme(|_: &IcedSdf| iced::Theme::Dark)
        .run()
}

// =========================================================================
// Theme colors (ported 1:1 from src/gui/mod.rs)
// =========================================================================
const PANEL_BG: Color = Color::from_rgba8(0x10, 0x12, 0x1a,0.3);
const MENUBAR_BG: Color = Color::from_rgba8(0x16, 0x19, 0x23,0.5);
const BORDER: Color = Color::from_rgb8(0x1c, 0x1e, 0x26);
const MENU_BORDER: Color = Color::from_rgb8(0x22, 0x26, 0x34);
const ACCENT: Color = Color::from_rgb8(0x00, 0xff, 0xcc);
const TEXT: Color = Color::WHITE;
const MUTED: Color = Color::from_rgb8(0x8a, 0x92, 0xa6);
const SIDEBAR_W: f32 = 320.0;

// =========================================================================
// Tabs
// =========================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Tab {
    #[default]
    Env,
    Arch,
    Fold,
    Mat,
    Obj,
    Sdf,
    Ctrl,
}

impl Tab {
    const ALL: [Tab; 7] = [
        Tab::Env,
        Tab::Arch,
        Tab::Fold,
        Tab::Mat,
        Tab::Obj,
        Tab::Sdf,
        Tab::Ctrl,
    ];

    fn label(self) -> &'static str {
        match self {
            Tab::Env => "Env",
            Tab::Arch => "Arch",
            Tab::Fold => "Fold",
            Tab::Mat => "Mat",
            Tab::Obj => "Obj",
            Tab::Sdf => "SDF",
            Tab::Ctrl => "Ctrl",
        }
    }
}

// =========================================================================
// Menu actions
// =========================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuAction {
    ResetScene,
    ToggleAo,
    ToggleKeyLight,
    LoadSdfObject,
    Exit,
}

// =========================================================================
// Messages — one variant per control (see ICED_PORT_PLAN.md table)
// =========================================================================
#[derive(Debug, Clone)]
enum Message {
    Tick(Instant),
    Event(event::Event),

    // -- GUI chrome ------------------------------------------------------
    TabSelected(Tab),
    AccordionToggled { tab: Tab, index: usize },
    MenuToggled(bool),
    MenuAction(MenuAction),
    SdfFilePicked(Option<PathBuf>),
    GuiHovered(bool),
    // -- Scalar sliders --------------------------------------------------
    LightIntensity(f32),
    AmbientStrength(f32),
    KeyLightX(f32),
    KeyLightY(f32),
    KeyLightZ(f32),
    KeyR(f32),
    KeyG(f32),
    KeyB(f32),
    BgR(f32),
    BgG(f32),
    BgB(f32),
    FogR(f32),
    FogG(f32),
    FogB(f32),
    FogDensity(f32),
    VolumetricFogSteps(f32),
    FogDensityFalloff(f32),
    MinHeightFalloff(f32),
    MaxOpticalDepth(f32),
    FogSampleOffset(f32),
    AoIterations(f32),
    ShadowSoftnessK(f32),
    PillarDist(f32),
    PillarThick(f32),
    RoomHeight(f32),
    CeilingThick(f32),
    ArchRadius(f32),
    ArchHeight(f32),
    DecorFreq(f32),
    DecorDepth(f32),
    DecorThick(f32),
    CellSize(f32),
    FoldSpeed(f32),
    PillarSpacing(f32),
    PillarHeight(f32),
    ReflectionMaxSteps(f32),
    ReflectionRoughnessGate(f32),
    FresnelStrengthFactor(f32),
    PillarRoughness(f32),
    MeshScaleFactor(f32),
    BoundingSphereOffset(f32),

    // -- Indexed sliders (selected slot / sdf object) ---------------------
    SlotRoughness(f32),
    SlotMetallic(f32),
    SlotEmissive(f32),
    SlotSpecular(f32),
    SlotPosX(f32),
    SlotPosY(f32),
    SlotPosZ(f32),
    SlotScale(f32),
    SlotRotX(f32),
    SlotRotY(f32),
    SlotRotZ(f32),
    SdfPosX(f32),
    SdfPosY(f32),
    SdfPosZ(f32),
    SdfSize(f32),
    SdfRotX(f32),
    SdfRotY(f32),
    SdfRotZ(f32),
    SdfR(f32),
    SdfG(f32),
    SdfB(f32),
    SdfRoughness(f32),
    SdfMetallic(f32),
    SdfEmissive(f32),
    SdfSpecular(f32),

    // -- Toggles ----------------------------------------------------------
    FogEnabled(bool),
    EnableAo(bool),
    EnableKey(bool),
    EnableFill(bool),
    EnableRim(bool),
    ShadowHard(bool),

    // -- Selectors / buttons ----------------------------------------------
    SlotSelected(usize),
    SdfSlotSelected(usize),
    SdfLoadClicked,
    SdfClearClicked,
}

// =========================================================================
// App state
// =========================================================================
struct IcedSdf {
    start: Instant,
    last_frame: Instant,
    fps: f32,
    scene: SdfScene,
    show_gui: bool,
    active_tab: Tab,
    /// Open/closed state per tab (same indices as the wgpui Accordion).
    accordion: [Vec<bool>; 7],
    menu_open: bool,
    last_mouse_x: f32,
    last_mouse_y: f32,
    pointer_over_gui: bool,
}

impl Default for IcedSdf {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            last_frame: Instant::now(),
            fps: 0.0,
            scene: SdfScene::new(),
            show_gui: true,
            pointer_over_gui: false,
            active_tab: Tab::default(),
            accordion: [
                vec![true, false, false, false, false], // Env
                vec![true, false, false, false],        // Arch
                vec![true, false],                      // Fold
                vec![true, true, true],                 // Mat
                vec![true, true, true, true],           // Obj
                vec![true, false, false, true, false],  // SDF
                vec![true, true, true],                 // Ctrl
            ],
            menu_open: false,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
        }
    }
}

// =========================================================================
// Update
// =========================================================================
impl IcedSdf {
    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Tick(now) => {
                self.scene.elapsed = (now - self.start).as_secs_f32();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                if dt > 0.0 {
                    self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1;
                }
                // Camera movement from WASD/QE flags (ported from input.rs).
                update_camera_movement(&mut self.scene.state, dt);
                Task::none()
            }
           
             Message::GuiHovered(hovered) => {
             self.pointer_over_gui = hovered;
             Task::none()
             }
            
            Message::Event(event) => self.handle_event(event),

            // -- GUI chrome ------------------------------------------------
            Message::TabSelected(tab) => {
                self.active_tab = tab;
                Task::none()
            }
            Message::AccordionToggled { tab, index } => {
                let list = &mut self.accordion[tab as usize];
                if let Some(slot) = list.get_mut(index) {
                    *slot = !*slot;
                }
                Task::none()
            }
            Message::MenuToggled(open) => {
                self.menu_open = open;
                Task::none()
            }
            Message::MenuAction(action) => self.menu_action(action),
            Message::SdfFilePicked(path) => {
                if let Some(path) = path {
                    let s = &mut self.scene.state;
                    let slot = s.current_selected_sdf_object;
                    if slot < s.sdf_paths.len() {
                        s.sdf_paths[slot] = path.display().to_string();
                        s.sdf_active[slot] = true;
                    }
                }
                Task::none()
            }

            // -- Scalar sliders ---------------------------------------------
            Message::LightIntensity(v) => {
                self.scene.state.light_intensity = v;
                Task::none()
            }
            Message::AmbientStrength(v) => {
                self.scene.state.ambient_strength = v;
                Task::none()
            }
            Message::KeyLightX(v) => {
                self.scene.state.key_light_x = v;
                Task::none()
            }
            Message::KeyLightY(v) => {
                self.scene.state.key_light_y = v;
                Task::none()
            }
            Message::KeyLightZ(v) => {
                self.scene.state.key_light_z = v;
                Task::none()
            }
            Message::KeyR(v) => {
                self.scene.state.key_r = v;
                Task::none()
            }
            Message::KeyG(v) => {
                self.scene.state.key_g = v;
                Task::none()
            }
            Message::KeyB(v) => {
                self.scene.state.key_b = v;
                Task::none()
            }
            Message::BgR(v) => {
                self.scene.state.bg_r = v;
                Task::none()
            }
            Message::BgG(v) => {
                self.scene.state.bg_g = v;
                Task::none()
            }
            Message::BgB(v) => {
                self.scene.state.bg_b = v;
                Task::none()
            }
            Message::FogR(v) => {
                self.scene.state.fog_r = v;
                Task::none()
            }
            Message::FogG(v) => {
                self.scene.state.fog_g = v;
                Task::none()
            }
            Message::FogB(v) => {
                self.scene.state.fog_b = v;
                Task::none()
            }
            Message::FogDensity(v) => {
                self.scene.state.fog_density = v;
                Task::none()
            }
            Message::VolumetricFogSteps(v) => {
                self.scene.state.volumetric_fog_steps = v as u32;
                Task::none()
            }
            Message::FogDensityFalloff(v) => {
                self.scene.state.fog_density_falloff = v;
                Task::none()
            }
            Message::MinHeightFalloff(v) => {
                self.scene.state.min_height_falloff = v;
                Task::none()
            }
            Message::MaxOpticalDepth(v) => {
                self.scene.state.max_optical_depth = v;
                Task::none()
            }
            Message::FogSampleOffset(v) => {
                self.scene.state.fog_sample_offset = v;
                Task::none()
            }
            Message::AoIterations(v) => {
                self.scene.state.ao_iterations = v as u32;
                Task::none()
            }
            Message::ShadowSoftnessK(v) => {
                self.scene.state.shadow_softness_k = v;
                Task::none()
            }
            Message::PillarDist(v) => {
                self.scene.state.pillar_dist = v;
                Task::none()
            }
            Message::PillarThick(v) => {
                self.scene.state.pillar_thick = v;
                Task::none()
            }
            Message::RoomHeight(v) => {
                self.scene.state.room_height = v;
                Task::none()
            }
            Message::CeilingThick(v) => {
                self.scene.state.ceiling_thick = v;
                Task::none()
            }
            Message::ArchRadius(v) => {
                self.scene.state.arch_radius = v;
                Task::none()
            }
            Message::ArchHeight(v) => {
                self.scene.state.arch_height = v;
                Task::none()
            }
            Message::DecorFreq(v) => {
                self.scene.state.decor_freq = v;
                Task::none()
            }
            Message::DecorDepth(v) => {
                self.scene.state.decor_depth = v;
                Task::none()
            }
            Message::DecorThick(v) => {
                self.scene.state.decor_thick = v;
                Task::none()
            }
            Message::CellSize(v) => {
                self.scene.state.cell_size = v;
                Task::none()
            }
            Message::FoldSpeed(v) => {
                self.scene.state.fold_speed = v;
                Task::none()
            }
            Message::PillarSpacing(v) => {
                self.scene.state.pillar_spacing = v;
                Task::none()
            }
            Message::PillarHeight(v) => {
                self.scene.state.pillar_height = v;
                Task::none()
            }
            Message::ReflectionMaxSteps(v) => {
                self.scene.state.reflection_max_steps = v as u32;
                Task::none()
            }
            Message::ReflectionRoughnessGate(v) => {
                self.scene.state.reflection_roughness_gate = v;
                Task::none()
            }
            Message::FresnelStrengthFactor(v) => {
                self.scene.state.fresnel_strength_factor = v;
                Task::none()
            }
            Message::PillarRoughness(v) => {
                self.scene.state.pillar_roughness = v;
                Task::none()
            }
            Message::MeshScaleFactor(v) => {
                self.scene.state.mesh_scale_factor = v;
                Task::none()
            }
            Message::BoundingSphereOffset(v) => {
                self.scene.state.bounding_sphere_radius_offset = v;
                Task::none()
            }

            // -- Indexed sliders ---------------------------------------------
            Message::SlotRoughness(v) => {
                self.set_slot(|s, i| s.slot_roughness[i] = v);
                Task::none()
            }
            Message::SlotMetallic(v) => {
                self.set_slot(|s, i| s.slot_metallic[i] = v);
                Task::none()
            }
            Message::SlotEmissive(v) => {
                self.set_slot(|s, i| s.slot_emissive[i] = v);
                Task::none()
            }
            Message::SlotSpecular(v) => {
                self.set_slot(|s, i| s.slot_specular[i] = v);
                Task::none()
            }
            Message::SlotPosX(v) => {
                self.set_slot(|s, i| s.slot_offsets_x[i] = v);
                Task::none()
            }
            Message::SlotPosY(v) => {
                self.set_slot(|s, i| s.slot_offsets_y[i] = v);
                Task::none()
            }
            Message::SlotPosZ(v) => {
                self.set_slot(|s, i| s.slot_offsets_z[i] = v);
                Task::none()
            }
            Message::SlotScale(v) => {
                self.set_slot(|s, i| s.slot_sizes[i] = v);
                Task::none()
            }
            Message::SlotRotX(v) => {
                self.set_slot(|s, i| s.slot_rot_x[i] = v);
                Task::none()
            }
            Message::SlotRotY(v) => {
                self.set_slot(|s, i| s.slot_rot_y[i] = v);
                Task::none()
            }
            Message::SlotRotZ(v) => {
                self.set_slot(|s, i| s.slot_rot_z[i] = v);
                Task::none()
            }
            Message::SdfPosX(v) => {
                self.set_sdf(|s, i| s.sdf_center_x[i] = v);
                Task::none()
            }
            Message::SdfPosY(v) => {
                self.set_sdf(|s, i| s.sdf_center_y[i] = v);
                Task::none()
            }
            Message::SdfPosZ(v) => {
                self.set_sdf(|s, i| s.sdf_center_z[i] = v);
                Task::none()
            }
            Message::SdfSize(v) => {
                self.set_sdf(|s, i| s.sdf_sizes[i] = v);
                Task::none()
            }
            Message::SdfRotX(v) => {
                self.set_sdf(|s, i| s.sdf_rot_x[i] = v);
                Task::none()
            }
            Message::SdfRotY(v) => {
                self.set_sdf(|s, i| s.sdf_rot_y[i] = v);
                Task::none()
            }
            Message::SdfRotZ(v) => {
                self.set_sdf(|s, i| s.sdf_rot_z[i] = v);
                Task::none()
            }
            Message::SdfR(v) => {
                self.set_sdf(|s, i| s.sdf_r[i] = v);
                Task::none()
            }
            Message::SdfG(v) => {
                self.set_sdf(|s, i| s.sdf_g[i] = v);
                Task::none()
            }
            Message::SdfB(v) => {
                self.set_sdf(|s, i| s.sdf_b[i] = v);
                Task::none()
            }
            Message::SdfRoughness(v) => {
                self.set_sdf(|s, i| s.sdf_roughness[i] = v);
                Task::none()
            }
            Message::SdfMetallic(v) => {
                self.set_sdf(|s, i| s.sdf_metallic[i] = v);
                Task::none()
            }
            Message::SdfEmissive(v) => {
                self.set_sdf(|s, i| s.sdf_emissive[i] = v);
                Task::none()
            }
            Message::SdfSpecular(v) => {
                self.set_sdf(|s, i| s.sdf_specular[i] = v);
                Task::none()
            }

            // -- Toggles ------------------------------------------------------
            Message::FogEnabled(on) => {
                self.scene.state.fog_enabled = if on { 1 } else { 0 };
                Task::none()
            }
            Message::EnableAo(on) => {
                self.scene.state.enable_ao_mode = if on { 1 } else { 0 };
                Task::none()
            }
            Message::EnableKey(on) => {
                self.scene.state.enable_key = if on { 1 } else { 0 };
                Task::none()
            }
            Message::EnableFill(on) => {
                self.scene.state.enable_fill = if on { 1 } else { 0 };
                Task::none()
            }
            Message::EnableRim(on) => {
                self.scene.state.enable_rim = if on { 1 } else { 0 };
                Task::none()
            }
            Message::ShadowHard(on) => {
                self.scene.state.current_shadow_mode = if on { 1 } else { 0 };
                Task::none()
            }

            // -- Selectors / buttons --------------------------------------------
            Message::SlotSelected(i) => {
                self.scene.state.current_selected_slot = i;
                Task::none()
            }
            Message::SdfSlotSelected(i) => {
                self.scene.state.current_selected_sdf_object = i;
                Task::none()
            }
            Message::SdfLoadClicked => self.menu_action(MenuAction::LoadSdfObject),
            Message::SdfClearClicked => {
                let s = &mut self.scene.state;
                let slot = s.current_selected_sdf_object;
                if slot < s.sdf_active.len() {
                    s.sdf_active[slot] = false;
                    s.sdf_paths[slot].clear();
                }
                Task::none()
            }
        }
    }

    fn set_slot(&mut self, f: impl FnOnce(&mut ApplicationState, usize)) {
        let s = &mut self.scene.state;
        let i = s.current_selected_slot.min(s.slot_roughness.len().saturating_sub(1));
        f(s, i);
    }

    fn set_sdf(&mut self, f: impl FnOnce(&mut ApplicationState, usize)) {
        let s = &mut self.scene.state;
        let i = s
            .current_selected_sdf_object
            .min(s.sdf_center_x.len().saturating_sub(1));
        f(s, i);
    }

    fn menu_action(&mut self, action: MenuAction) -> Task<Message> {
        match action {
            MenuAction::ResetScene => {
                let s = &mut self.scene.state;
                s.cam_x = 0.0;
                s.cam_y = 0.0;
                s.cam_z = -5.0;
                Task::none()
            }
            MenuAction::ToggleAo => {
                let s = &mut self.scene.state;
                s.enable_ao_mode = if s.enable_ao_mode == 1 { 0 } else { 1 };
                Task::none()
            }
            MenuAction::ToggleKeyLight => {
                let s = &mut self.scene.state;
                s.enable_key = if s.enable_key == 1 { 0 } else { 1 };
                Task::none()
            }
            MenuAction::LoadSdfObject => Task::perform(
                async {
                    rfd::AsyncFileDialog::new()
                        .add_filter("OBJ Mesh", &["obj"])
                        .pick_file()
                        .await
                        .map(|handle| handle.path().to_path_buf())
                },
                Message::SdfFilePicked,
            ),
            MenuAction::Exit => window::latest().and_then(window::close),
        }
    }

    // -- Keyboard + mouse (1:1 from src/view.rs) ----------------------------
    fn handle_event(&mut self, event: event::Event) -> Task<Message> {
        match event {
            event::Event::Keyboard(keyboard::Event::KeyPressed { key, repeat, .. }) => {
                let key = key.as_ref();
                match key {
                    keyboard::Key::Character("w" | "W") => self.scene.state.w_pressed = true,
                    keyboard::Key::Character("a" | "A") => self.scene.state.a_pressed = true,
                    keyboard::Key::Character("s" | "S") => self.scene.state.s_pressed = true,
                    keyboard::Key::Character("d" | "D") => self.scene.state.d_pressed = true,
                    keyboard::Key::Character("q" | "Q") => self.scene.state.q_pressed = true,
                    keyboard::Key::Character("e" | "E") => self.scene.state.e_pressed = true,

                    keyboard::Key::Character("h" | "H") if !repeat => {
                        self.show_gui = !self.show_gui;
                    }
                    keyboard::Key::Character("1") if !repeat => self.cycle_slot(0),
                    keyboard::Key::Character("2") if !repeat => self.cycle_slot(1),
                    keyboard::Key::Character("3") if !repeat => self.cycle_slot(2),
                    keyboard::Key::Character("4") if !repeat => self.cycle_slot(3),
                    keyboard::Key::Character("5") if !repeat => self.cycle_slot(4),
                    keyboard::Key::Character("6") if !repeat => self.spawn_slot6(),

                    keyboard::Key::Character("g" | "G") if !repeat => {
                        self.scene.state.light_intensity = (self.scene.state.light_intensity - 0.1).max(0.0)
                    }
                    keyboard::Key::Character("t" | "T") if !repeat => {
                        self.scene.state.light_intensity = (self.scene.state.light_intensity + 0.1).min(3.0)
                    }
                    keyboard::Key::Character("f" | "F") if !repeat => {
                        self.scene.state.ambient_strength = (self.scene.state.ambient_strength - 0.05).max(0.0)
                    }
                    keyboard::Key::Character("r" | "R") if !repeat => {
                        self.scene.state.ambient_strength = (self.scene.state.ambient_strength + 0.05).min(1.0)
                    }
                    keyboard::Key::Character("o" | "O") if !repeat => {
                        self.scene.state.fog_density = (self.scene.state.fog_density - 0.1).max(0.0)
                    }
                    keyboard::Key::Character("l" | "L") if !repeat => {
                        self.scene.state.fog_density = (self.scene.state.fog_density + 0.1).min(2.0)
                    }
                    keyboard::Key::Character("n" | "N") if !repeat => {
                        self.scene.state.fog_enabled = if self.scene.state.fog_enabled == 1 { 0 } else { 1 }
                    }
                    _ => {}
                }
                Task::none()
            }
            event::Event::Keyboard(keyboard::Event::KeyReleased { key, .. }) => {
                let key = key.as_ref();
                match key {
                    keyboard::Key::Character("w" | "W") => self.scene.state.w_pressed = false,
                    keyboard::Key::Character("a" | "A") => self.scene.state.a_pressed = false,
                    keyboard::Key::Character("s" | "S") => self.scene.state.s_pressed = false,
                    keyboard::Key::Character("d" | "D") => self.scene.state.d_pressed = false,
                    keyboard::Key::Character("q" | "Q") => self.scene.state.q_pressed = false,
                    keyboard::Key::Character("e" | "E") => self.scene.state.e_pressed = false,
                    _ => {}
                }
                Task::none()
            }
          event::Event::Mouse(mouse::Event::CursorMoved { position }) => {
    if !(self.show_gui && (self.pointer_over_gui || self.menu_open)) {
        let dx = position.x - self.last_mouse_x;
        let dy = position.y - self.last_mouse_y;
        if dx != 0.0 || dy != 0.0 {
            let s = &mut self.scene.state;
            let max_pitch = 89.0f32.to_radians();
            s.cam_yaw += dx * MOUSE_SENSITIVITY;
            s.cam_pitch -= dy * MOUSE_SENSITIVITY;
            s.cam_pitch = s.cam_pitch.clamp(-max_pitch, max_pitch);
        }
    }
    self.last_mouse_x = position.x;
    self.last_mouse_y = position.y;
    Task::none()
}
event::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
    if !(self.show_gui && self.pointer_over_gui) {
        let (dx, dy) = match delta {
            mouse::ScrollDelta::Pixels { x, y } => (-x, -y),
            mouse::ScrollDelta::Lines { x, y } => (-x * 20.0, -y * 20.0),
        };
        if dx != 0.0 || dy != 0.0 {
            let s = &mut self.scene.state;
            let max_pitch = 89.0f32.to_radians();
            let sens = MOUSE_SENSITIVITY * 1.5;
            s.cam_yaw += dx * sens;
            s.cam_pitch -= dy * sens;
            s.cam_pitch = s.cam_pitch.clamp(-max_pitch, max_pitch);
        }
    }
    Task::none()

            }
            _ => Task::none(),
        }
    }

    fn cycle_slot(&mut self, i: usize) {
        let s = &mut self.scene.state;
        if i < s.slot_types.len() {
            s.current_selected_slot = i;
            s.slot_types[i] = if s.slot_types[i] == 3.0 { 0.0 } else { s.slot_types[i] + 1.0 };
            update_slot_color(s, i);
        }
    }

    fn spawn_slot6(&mut self) {
        let s = &mut self.scene.state;
        if s.slot_types.len() <= 5 {
            s.current_selected_slot = 5;
            s.slot_types.push(1.0);
            s.slot_sizes.push(1.0);
            s.slot_offsets_x.push(0.0);
            s.slot_offsets_y.push(0.0);
            s.slot_offsets_z.push(0.0);
            s.slot_r.push(1.0);
            s.slot_g.push(0.0);
            s.slot_b.push(0.0);
            s.slot_roughness.push(0.3);
            s.slot_metallic.push(0.0);
            s.slot_emissive.push(0.0);
            s.slot_specular.push(0.5);
            s.slot_rot_x.push(0.0);
            s.slot_rot_y.push(0.0);
            s.slot_rot_z.push(0.0);
            s.active_slots_count = s.slot_types.len() as f32;
        } else {
            s.current_selected_slot = 5;
            s.slot_types[5] = if s.slot_types[5] == 3.0 { 0.0 } else { s.slot_types[5] + 1.0 };
            update_slot_color(s, 5);
        }
    }
}

const MOUSE_SENSITIVITY: f32 = 0.002;

// =========================================================================
// Camera movement (ported from src/input.rs::update_camera_movement)
// =========================================================================
fn update_camera_movement(s: &mut ApplicationState, dt: f32) {
    const CAMERA_SPEED: f32 = 3.0; // matches input.rs default
    let move_dist = CAMERA_SPEED * dt;
    let cos_yaw = s.cam_yaw.cos();
    let sin_yaw = s.cam_yaw.sin();

    if s.w_pressed {
        s.cam_x += sin_yaw * move_dist;
        s.cam_z += cos_yaw * move_dist;
    }
    if s.s_pressed {
        s.cam_x -= sin_yaw * move_dist;
        s.cam_z -= cos_yaw * move_dist;
    }
    if s.a_pressed {
        s.cam_x -= cos_yaw * move_dist;
        s.cam_z += sin_yaw * move_dist;
    }
    if s.d_pressed {
        s.cam_x += cos_yaw * move_dist;
        s.cam_z -= sin_yaw * move_dist;
    }
    if s.q_pressed {
        s.cam_y -= move_dist;
    }
    if s.e_pressed {
        s.cam_y += move_dist;
    }
}

fn update_slot_color(s: &mut ApplicationState, idx: usize) {
    match s.slot_types[idx] as u32 {
        1 => {
            s.slot_r[idx] = 1.0;
            s.slot_g[idx] = 0.0;
            s.slot_b[idx] = 0.0;
        }
        2 => {
            s.slot_r[idx] = 0.0;
            s.slot_g[idx] = 1.0;
            s.slot_b[idx] = 0.0;
        }
        3 => {
            s.slot_r[idx] = 0.0;
            s.slot_g[idx] = 0.0;
            s.slot_b[idx] = 1.0;
        }
        _ => {
            s.slot_r[idx] = 1.0;
            s.slot_g[idx] = 1.0;
            s.slot_b[idx] = 1.0;
        }
    }
}

// =========================================================================
// View
// =========================================================================
impl IcedSdf {
    fn view(&self) -> impl Widget<Message> {
        let shader_widget = shader(&self.scene).width(Fill).height(Fill);

        let base: iced::Element<'_, Message> = if self.show_gui {
            let sidebar = opaque(
                mouse_area(self.sidebar())
                    .on_enter(Message::GuiHovered(true))
                    .on_exit(Message::GuiHovered(false)),
            );

            let menubar = opaque(
                mouse_area(container(self.menubar()).width(Fill).height(32))
                    .on_enter(Message::GuiHovered(true))
                    .on_exit(Message::GuiHovered(false)),
            );

            let gui_overlay = row![
    // Left side: menubar sits at top-left, open space below
    column![
        menubar,
        iced::widget::Space::new().height(Fill),
    ]
    .width(Fill)
    .height(Fill),
    // Right side: sidebar starts at the very top (y = 0) and spans full height
    sidebar,
]
.width(Fill)
.height(Fill);
            
            let fps_overlay = container(
    text(format!("FPS: {:.1}", self.fps))
        .color(ACCENT)
        .size(20),
)
.padding(iced::Padding {
    top: 40.0,  // Clears the 32px menubar + 8px margin
    left: 12.0,
    right: 0.0,
    bottom: 0.0,
});

            // Shader at Layer 0, GUI overlay at Layer 1, FPS text at Layer 2
            container(
                stack![
                    shader_widget,
                    gui_overlay,
                    fps_overlay,
                ]
                .width(Fill)
                .height(Fill),
            )
            .boxed()
        } else {
            center(shader_widget).boxed()
        };

        base
    }

    fn menubar(&self) -> impl Widget<Message> {
        let trigger = button(text("Datei").color(TEXT).size(12))
            .padding([4, 12])
            .on_press(Message::MenuToggled(true));

        popover(
            trigger,
            self.menu_open.then(|| self.menu_popup()),
        )
        .on_close(Message::MenuToggled(false))
    }

    fn menu_popup(&self) -> impl Widget<Message> {
    let item =

        |label: &'static str, msg: Message| -> iced::Element<Message> {
            button(
                container(text(label).color(TEXT).size(11))
                    .width(200)
                    .align_x(iced::alignment::Horizontal::Left),
            )
            .padding([6, 12])
            .style(button::text)
            .on_press(msg)
            .boxed()
        };
    let separator = || space().height(1).width(Fill);

    let popup_box = container(
        column![
            item("Zurücksetzen", Message::MenuAction(MenuAction::ResetScene)),
            separator(),
            item("Ambient Occlusion", Message::MenuAction(MenuAction::ToggleAo)),
            item("Key Light toggeln", Message::MenuAction(MenuAction::ToggleKeyLight)),

            separator(),
            item("SDF-Objekt laden…", Message::MenuAction(MenuAction::LoadSdfObject)),
            separator(),
            item("Beenden", Message::MenuAction(MenuAction::Exit)),
        ]
        .spacing(2)
        .padding(4),
    )
    .width(210)
    .padding(2)
    .style(|_: &iced::Theme| container::Style {
        background: Some(iced::Background::Color(MENUBAR_BG)),
        border: iced::Border {
            color: MENU_BORDER,
            width: 1.0,
            radius: 4.0.into(),
        },
        ..Default::default()
    });

    opaque(
        mouse_area(popup_box)
            .on_enter(Message::GuiHovered(true))
            .on_exit(Message::GuiHovered(false)),
    )
}


    // -- Sidebar -------------------------------------------------------------
    fn sidebar(&self) -> impl Widget<Message> {
        let tabs = Tab::ALL.iter().copied().fold(
            row![].spacing(4),
            |r, tab| {
                let mut b = button(text(tab.label()).size(10))
                    .padding([4, 10])
                    .style(if tab == self.active_tab {
                        button::primary
                    } else {
                        button::secondary
                    });
                b = b.on_press(Message::TabSelected(tab));
                r.push(b)
            },
        );

        let content = scrollable(self.tab_content()).width(Fill).height(Fill);

        container(column![tabs, content].spacing(8).padding(8))
            .width(SIDEBAR_W)
            .height(Fill)
            .style(|_: &iced::Theme| container::Style {
        background: Some(iced::Background::Color(PANEL_BG)),
        border: iced::Border::default(),
        shadow: iced::Shadow {
            color: Color::from_rgba8(0x00, 0x00, 0x00, 0.45),
            offset: iced::Vector::new(-4.0, 0.0), // Shifts shadow slightly left toward the viewport
            blur_radius: 16.0,
        },
        snap: true,
        ..Default::default()
    })
    }

    fn tab_content(&self) -> iced::Element<'_, Message> {
        let open = &self.accordion[self.active_tab as usize];
        let s = &self.scene.state;
        match self.active_tab {
            Tab::Env => self.env_tab(open, s),
            Tab::Arch => self.arch_tab(open, s),
            Tab::Fold => self.fold_tab(open, s),
            Tab::Mat => self.mat_tab(open, s),
            Tab::Obj => self.obj_tab(open, s),
            Tab::Sdf => self.sdf_tab(open, s),
            Tab::Ctrl => self.ctrl_tab(open, s),
        }
    }

    // -- Reusable helpers ------------------------------------------------------
    fn section<'a>(&self, title: &'a str) -> iced::Element<'a, Message> {
        text(title)
            .color(ACCENT)
            .size(12)
            .font(iced::Font::MONOSPACE)
            .boxed()
    }

    fn slider_row<'a>(
        &self,
        label: &'a str,
        value: f32,
        range: std::ops::RangeInclusive<f32>,
        step: f32,
        decimals: usize,
        on_change: impl Fn(f32) -> Message + 'a,
    ) -> iced::Element<'a, Message> {
        let value_text = format!("{:.dec$}", value, dec = decimals);
        row![
            text(label).color(MUTED).size(10).width(110),
            text(value_text).color(ACCENT).size(10).width(48),
            slider(range, value, on_change)
                .step(step)
                .width(Fill),
        ]
        .spacing(6)
        .align_y(Alignment::Center)
        .boxed()
    }

    fn toggle_row<'a>(
        &self,
        label: &'a str,
        checked: bool,
        on_toggle: impl Fn(bool) -> Message + 'a,
    ) -> iced::Element<'a, Message> {
        row![
            text(label).color(TEXT).size(10).width(110),
            toggler(checked).size(18).spacing(4).on_toggle(on_toggle),
        ]
        .spacing(6)
        .align_y(Alignment::Center)
        .boxed()
    }

    fn info_row(&self, label: &'static str, value: String) -> iced::Element<'_, Message> {
        row![
            text(label).color(MUTED).size(10).width(110),
            text(value).color(ACCENT).size(10).font(iced::Font::MONOSPACE),
        ]
        .spacing(6)
        .boxed()
    }

    fn key_row(&self, key: &str, desc: &str) -> iced::Element<'_, Message> {
        text(format!("{}  {}", key, desc)).color(MUTED).size(10).boxed()
    }

    fn acc<'a>(
        &self,
        title: &'a str,
        idx: usize,
        open: bool,
        tab: Tab,
        children: Vec<iced::Element<'a, Message>>,
    ) -> iced::Element<'a, Message> {
        let header = button(
            row![
                text(if open { "▾" } else { "▸" }).color(ACCENT).size(11),
                text(title).color(TEXT).size(11),
            ]
            .spacing(6),
        )
        .padding([4, 0])
        .style(button::text)
        .on_press(Message::AccordionToggled { tab, index: idx });

        let body: iced::Element<'a, Message> = if open {
            container(column(children).spacing(6).padding(Padding::new(0.0).bottom(4.0).left(12.0))).boxed()
        } else {
            container(column(Vec::<iced::Element<'a, Message>>::new()).spacing(0)).boxed()
        };

        column![header, body].spacing(2).boxed()
    }
}

// =========================================================================
// Tab views (1:1 port of gui/tabs/*.rs)
// =========================================================================
impl IcedSdf {
    fn env_tab(&self, open: &[bool], s: &ApplicationState) -> iced::Element<'_, Message> {
        column![
            self.acc("Light", 0, open[0], Tab::Env, vec![
                self.slider_row("Intensity", s.light_intensity, 0.0..=3.0, 0.05, 2, Message::LightIntensity),
                self.slider_row("Ambient", s.ambient_strength, 0.0..=1.0, 0.01, 2, Message::AmbientStrength),
                self.slider_row("Key Light X", s.key_light_x, -20.0..=20.0, 0.1, 2, Message::KeyLightX),
                self.slider_row("Key Light Y", s.key_light_y, 0.0..=20.0, 0.1, 2, Message::KeyLightY),
                self.slider_row("Key Light Z", s.key_light_z, -20.0..=20.0, 0.1, 2, Message::KeyLightZ),
            ]),
            self.acc("Colors", 1, open[1], Tab::Env, vec![
                self.slider_row("Key R", s.key_r, 0.0..=1.0, 0.01, 2, Message::KeyR),
                self.slider_row("Key G", s.key_g, 0.0..=1.0, 0.01, 2, Message::KeyG),
                self.slider_row("Key B", s.key_b, 0.0..=1.0, 0.01, 2, Message::KeyB),
                self.slider_row("Background R", s.bg_r, 0.0..=1.0, 0.01, 2, Message::BgR),
                self.slider_row("Background G", s.bg_g, 0.0..=1.0, 0.01, 2, Message::BgG),
                self.slider_row("Background B", s.bg_b, 0.0..=1.0, 0.01, 2, Message::BgB),
                self.slider_row("Fog R", s.fog_r, 0.0..=1.0, 0.01, 2, Message::FogR),
                self.slider_row("Fog G", s.fog_g, 0.0..=1.0, 0.01, 2, Message::FogG),
                self.slider_row("Fog B", s.fog_b, 0.0..=1.0, 0.01, 2, Message::FogB),
            ]),
            self.acc("Fog", 2, open[2], Tab::Env, vec![
                self.slider_row("Fog Density", s.fog_density, 0.0..=2.0, 0.01, 2, Message::FogDensity),
                self.toggle_row("Fog Enabled", s.fog_enabled == 1, Message::FogEnabled),
            ]),
            self.acc("Atmosphere", 3, open[3], Tab::Env, vec![
                self.slider_row("Fog Steps", s.volumetric_fog_steps as f32, 4.0..=64.0, 1.0, 0, Message::VolumetricFogSteps),
                self.slider_row("Fog Falloff", s.fog_density_falloff, 0.01..=0.5, 0.01, 2, Message::FogDensityFalloff),
                self.slider_row("Min Height Falloff", s.min_height_falloff, 0.0..=1.0, 0.01, 2, Message::MinHeightFalloff),
                self.slider_row("Max Optical Depth", s.max_optical_depth, 1.0..=10.0, 0.05, 2, Message::MaxOpticalDepth),
                self.slider_row("Fog Sample Offset", s.fog_sample_offset, 0.0..=1.0, 0.05, 2, Message::FogSampleOffset),
            ]),
            self.acc("Lighting & AO", 4, open[4], Tab::Env, vec![
                self.slider_row("AO Iterations", s.ao_iterations as f32, 1.0..=10.0, 1.0, 0, Message::AoIterations),
                self.slider_row("Shadow Softness K", s.shadow_softness_k, 1.0..=64.0, 0.5, 2, Message::ShadowSoftnessK),
            ]),
        ]
        .spacing(8)
        .boxed()
    }

    fn arch_tab(&self, open: &[bool], s: &ApplicationState) -> iced::Element<'_, Message> {
        column![
            self.acc("Pillars & Room", 0, open[0], Tab::Arch, vec![
                self.slider_row("Pillar Distance", s.pillar_dist, 2.0..=12.0, 0.1, 2, Message::PillarDist),
                self.slider_row("Pillar Thickness", s.pillar_thick, 0.1..=2.0, 0.05, 2, Message::PillarThick),
                self.slider_row("Room Height", s.room_height, 1.0..=8.0, 0.1, 2, Message::RoomHeight),
                self.slider_row("Ceiling Thickness", s.ceiling_thick, 0.01..=1.0, 0.01, 2, Message::CeilingThick),
            ]),
            self.acc("Arches", 1, open[1], Tab::Arch, vec![
                self.slider_row("Arch Radius", s.arch_radius, 1.0..=6.0, 0.1, 2, Message::ArchRadius),
                self.slider_row("Arch Height", s.arch_height, 0.0..=4.0, 0.1, 2, Message::ArchHeight),
            ]),
            self.acc("Decor", 2, open[2], Tab::Arch, vec![
                self.slider_row("Decor Frequency", s.decor_freq, 0.0..=8.0, 0.1, 2, Message::DecorFreq),
                self.slider_row("Decor Depth", s.decor_depth, 0.0..=0.2, 0.005, 3, Message::DecorDepth),
                self.slider_row("Decor Thickness", s.decor_thick, 0.0..=0.1, 0.002, 3, Message::DecorThick),
            ]),
            self.acc("Global Grid", 3, open[3], Tab::Arch, vec![
                self.slider_row("Cell Size", s.cell_size, 2.0..=40.0, 0.5, 2, Message::CellSize),
            ]),
        ]
        .spacing(8)
        .boxed()
    }

    fn fold_tab(&self, open: &[bool], s: &ApplicationState) -> iced::Element<'_, Message> {
        column![
            self.acc("Infinite Fold (Tensor 6)", 0, open[0], Tab::Fold, vec![
                self.slider_row("Cell Size", s.cell_size, 2.0..=40.0, 0.5, 2, Message::CellSize),
                self.slider_row("Fold Speed", s.fold_speed, 0.0..=5.0, 0.05, 2, Message::FoldSpeed),
            ]),
            self.acc("Infinite Repetition", 1, open[1], Tab::Fold, vec![
                self.slider_row("Pillar Spacing", s.pillar_spacing, 2.0..=20.0, 0.1, 2, Message::PillarSpacing),
                self.slider_row("Pillar Height", s.pillar_height, 1.0..=10.0, 0.1, 2, Message::PillarHeight),
            ]),
        ]
        .spacing(8)
        .boxed()
    }

    fn mat_tab(&self, open: &[bool], s: &ApplicationState) -> iced::Element<'_, Message> {
        let sel = s.current_selected_slot.min(s.slot_roughness.len().saturating_sub(1));
        column![
            self.section("Material Slot"),
            self.slot_selector(sel, s.slot_roughness.len()),
            text(format!(
                "Slot {} — {}",
                sel + 1,
                slot_type_name(s.slot_types[sel])
            ))
            .color(MUTED)
            .size(10),
            self.acc("PBR Parameters", 0, open[0], Tab::Mat, vec![
                self.slider_row("Roughness", s.slot_roughness[sel], 0.01..=1.0, 0.01, 2, Message::SlotRoughness),
                self.slider_row("Metallic", s.slot_metallic[sel], 0.0..=1.0, 0.01, 2, Message::SlotMetallic),
                self.slider_row("Emissive", s.slot_emissive[sel], 0.0..=10.0, 0.1, 2, Message::SlotEmissive),
                self.slider_row("Specular", s.slot_specular[sel], 0.0..=1.0, 0.01, 2, Message::SlotSpecular),
            ]),
            self.acc("Global Reflections", 1, open[1], Tab::Mat, vec![
                self.slider_row("Reflection Max Steps", s.reflection_max_steps as f32, 8.0..=64.0, 1.0, 0, Message::ReflectionMaxSteps),
                self.slider_row("Roughness Gate", s.reflection_roughness_gate, 0.05..=1.0, 0.01, 2, Message::ReflectionRoughnessGate),
                self.slider_row("Fresnel Strength", s.fresnel_strength_factor, 0.0..=1.0, 0.01, 2, Message::FresnelStrengthFactor),
            ]),
            self.acc("Material Physics", 2, open[2], Tab::Mat, vec![
                text("Pillar PBR (ID 1)").color(MUTED).size(10).boxed(),
                self.slider_row("Pillar Roughness", s.pillar_roughness, 0.0..=1.0, 0.01, 2, Message::PillarRoughness),
            ]),
        ]
        .spacing(8)
        .boxed()
    }

    fn obj_tab(&self, open: &[bool], s: &ApplicationState) -> iced::Element<'_, Message> {
        let sel = s.current_selected_slot.min(s.slot_offsets_x.len().saturating_sub(1));
        column![
            self.section("Object Slot"),
            self.slot_selector(sel, s.slot_offsets_x.len()),
            text(format!(
                "Slot {} — {}",
                sel + 1,
                slot_type_name(s.slot_types[sel])
            ))
            .color(MUTED)
            .size(10),
            self.acc("Position", 0, open[0], Tab::Obj, vec![
                self.slider_row("Pos X", s.slot_offsets_x[sel], -10.0..=10.0, 0.1, 2, Message::SlotPosX),
                self.slider_row("Pos Y", s.slot_offsets_y[sel], -10.0..=10.0, 0.1, 2, Message::SlotPosY),
                self.slider_row("Pos Z", s.slot_offsets_z[sel], -10.0..=10.0, 0.1, 2, Message::SlotPosZ),
            ]),
            self.acc("Scale", 1, open[1], Tab::Obj, vec![
                self.slider_row("Scale", s.slot_sizes[sel], 0.1..=4.0, 0.05, 2, Message::SlotScale),
            ]),
            self.acc("Rotation (Euler, rad)", 2, open[2], Tab::Obj, vec![
                self.slider_row("Rot X", s.slot_rot_x[sel], 0.0..=6.283, 0.01, 2, Message::SlotRotX),
                self.slider_row("Rot Y", s.slot_rot_y[sel], 0.0..=6.283, 0.01, 2, Message::SlotRotY),
                self.slider_row("Rot Z", s.slot_rot_z[sel], 0.0..=6.283, 0.01, 2, Message::SlotRotZ),
            ]),
            self.acc("Mesh Backing", 3, open[3], Tab::Obj, vec![
                self.slider_row("Mesh Scale Factor", s.mesh_scale_factor, 0.1..=1.0, 0.01, 2, Message::MeshScaleFactor),
            ]),
        ]
        .spacing(8)
        .boxed()
    }

    fn sdf_tab(&self, open: &[bool], s: &ApplicationState) -> iced::Element<'_, Message> {
        let sel = s
            .current_selected_sdf_object
            .min(s.sdf_center_x.len().saturating_sub(1));
        let path = s.sdf_paths.get(sel).cloned().unwrap_or_default();
        column![
            self.section("SDF Object Slot"),
            self.slot_selector_sdf(sel, s.sdf_paths.len()),
            row![
                button(text("Load .obj…").color(TEXT).size(10))
                    .padding([4, 8])
                    .style(button::secondary)
                    .on_press(Message::SdfLoadClicked),
                button(text("Clear").color(TEXT).size(10))
                    .padding([4, 8])
                    .style(button::text)
                    .on_press(Message::SdfClearClicked),
            ]
            .spacing(4),
            text(format!("Slot {} — {}", sel + 1, if s.sdf_active[sel] { "Active" } else { "Inactive" }))
                .color(MUTED)
                .size(10),
            text(format!("Pfad: {}", if path.is_empty() { "(leer)" } else { &path }))
                .color(MUTED)
                .size(10),
            self.acc("Position", 0, open[0], Tab::Sdf, vec![
                self.slider_row("Pos X", s.sdf_center_x[sel], -10.0..=10.0, 0.1, 2, Message::SdfPosX),
                self.slider_row("Pos Y", s.sdf_center_y[sel], -5.0..=5.0, 0.1, 2, Message::SdfPosY),
                self.slider_row("Pos Z", s.sdf_center_z[sel], -10.0..=10.0, 0.1, 2, Message::SdfPosZ),
            ]),
            self.acc("Rotation (Euler, rad)", 1, open[1], Tab::Sdf, vec![
                self.slider_row("Rot X", s.sdf_rot_x[sel], 0.0..=6.283, 0.01, 2, Message::SdfRotX),
                self.slider_row("Rot Y", s.sdf_rot_y[sel], 0.0..=6.283, 0.01, 2, Message::SdfRotY),
                self.slider_row("Rot Z", s.sdf_rot_z[sel], 0.0..=6.283, 0.01, 2, Message::SdfRotZ),
            ]),
            self.acc("Size & Grid", 2, open[2], Tab::Sdf, vec![
                self.slider_row("Size", s.sdf_sizes[sel], 1.0..=16.0, 0.1, 2, Message::SdfSize),
                self.info_row("Grid Res", "64".into()),
            ]),
            self.acc("Material (Albedo + PBR)", 3, open[3], Tab::Sdf, vec![
                self.slider_row("Albedo R", s.sdf_r[sel], 0.0..=1.0, 0.01, 2, Message::SdfR),
                self.slider_row("Albedo G", s.sdf_g[sel], 0.0..=1.0, 0.01, 2, Message::SdfG),
                self.slider_row("Albedo B", s.sdf_b[sel], 0.0..=1.0, 0.01, 2, Message::SdfB),
                self.slider_row("Roughness", s.sdf_roughness[sel], 0.01..=1.0, 0.01, 2, Message::SdfRoughness),
                self.slider_row("Metallic", s.sdf_metallic[sel], 0.0..=1.0, 0.01, 2, Message::SdfMetallic),
                self.slider_row("Emissive", s.sdf_emissive[sel], 0.0..=10.0, 0.1, 2, Message::SdfEmissive),
                self.slider_row("Specular", s.sdf_specular[sel], 0.1..=1.0, 0.01, 2, Message::SdfSpecular),
            ]),
            self.acc("Global Settings", 4, open[4], Tab::Sdf, vec![
                self.slider_row("Mesh Scale Factor", s.mesh_scale_factor, 0.1..=1.0, 0.01, 2, Message::MeshScaleFactor),
                self.slider_row("Bounding Sphere Offset", s.bounding_sphere_radius_offset, 0.0..=1.0, 0.01, 2, Message::BoundingSphereOffset),
            ]),
        ]
        .spacing(8)
        .boxed()
    }

    fn ctrl_tab(&self, open: &[bool], s: &ApplicationState) -> iced::Element<'_, Message> {
        column![
            self.acc("Camera", 0, open[0], Tab::Ctrl, vec![
                self.info_row("Cam X", format!("{:.2}", s.cam_x)),
                self.info_row("Cam Y", format!("{:.2}", s.cam_y)),
                self.info_row("Cam Z", format!("{:.2}", s.cam_z)),
                self.info_row("Yaw", format!("{:.2}", s.cam_yaw)),
                self.info_row("Pitch", format!("{:.2}", s.cam_pitch)),
            ]),
            self.acc("Rendering", 1, open[1], Tab::Ctrl, vec![
                self.toggle_row("Ambient Occlusion", s.enable_ao_mode == 1, Message::EnableAo),
                self.toggle_row("Key Light", s.enable_key == 1, Message::EnableKey),
                self.toggle_row("Fill Light", s.enable_fill == 1, Message::EnableFill),
                self.toggle_row("Rim Light", s.enable_rim == 1, Message::EnableRim),
                self.toggle_row("Hard Shadows", s.current_shadow_mode == 1, Message::ShadowHard),
            ]),
            self.acc("Key Bindings", 2, open[2], Tab::Ctrl, vec![
                self.key_row("WASD", "Fly through Scene"),
                self.key_row("Q / E", "Camera Up / Down"),
                self.key_row("Mouse", "Look around"),
                self.key_row("Wheel", "Look (fast)"),
                self.key_row("1-5", "Cycle slot type"),
                self.key_row("6", "Spawn slot 6"),
                self.key_row("G / T", "Light Intensity -/+"),
                self.key_row("F / R", "Ambient Strength -/+"),
                self.key_row("O / L", "Fog Density -/+"),
                self.key_row("N", "Toggle Fog"),
                self.key_row("H", "Toggle GUI"),
            ]),
        ]
        .spacing(8)
        .boxed()
    }

    fn slot_selector(&self, selected: usize, count: usize) -> iced::Element<'_, Message> {
        let mut r = row![].spacing(4);
        for i in 0..count {
            let b = button(text(format!("{}", i + 1)).size(10))
                .padding([4, 8])
                .style(if i == selected {
                    button::primary
                } else {
                    button::secondary
                })
                .on_press(Message::SlotSelected(i));
            r = r.push(b);
        }
        r.boxed()
    }

    fn slot_selector_sdf(&self, selected: usize, count: usize) -> iced::Element<'_, Message> {
        let mut r = row![].spacing(4);
        for i in 0..count {
            let b = button(text(format!("{}", i + 1)).size(10))
                .padding([4, 8])
                .style(if i == selected {
                    button::primary
                } else {
                    button::secondary
                })
                .on_press(Message::SdfSlotSelected(i));
            r = r.push(b);
        }
        r.boxed()
    }
}

fn slot_type_name(t: f32) -> &'static str {
    match t as u32 {
        1 => "Crystal",
        2 => "Gyroid",
        3 => "Torus",
        _ => "Inactive",
    }
}

// =========================================================================
// Subscription: frame ticks + all events (blueprint: examples/events)
// =========================================================================
impl IcedSdf {
    fn subscription(&self) -> Subscription<Message> {
        Subscription::batch([
            window::frames().map(Message::Tick),
            event::listen().map(Message::Event),
        ])
    }
}