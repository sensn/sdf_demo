//! `kernel/objects.rs` — Dynamische SDF-Objekte: Crystal, Gyroid, Torus.
//! (Auto-extrahiert aus kernel.rs — Funktionalität unverändert.)
#![allow(warnings)]

use cubecl::frontend::CubeType;
use cubecl::prelude::*;

use super::types::Vec3;
#[cube]
pub fn evaluate_dynamic_crystal(p: Vec3, time: f32, size: f32) -> f32 {
    let rot_speed = time * f32::new(0.4);
    let cos_r = rot_speed.cos();
    let sin_r = rot_speed.sin();
    let crystal_x = p.x * cos_r - p.z * sin_r;
    let crystal_z = p.x * sin_r + p.z * cos_r;

    let base_crystal =
        (crystal_x.abs() + p.y.abs() + crystal_z.abs() - size) * f32::new(0.57735027);
    let waves = (crystal_x * f32::new(6.0)).sin()
        * (p.y * f32::new(6.0)).sin()
        * (crystal_z * f32::new(6.0)).sin()
        * f32::new(0.04);
    base_crystal + waves
}

#[cube]
pub fn evaluate_dynamic_gyroid(p: Vec3, time: f32, size: f32) -> f32 {
    // Fix: Variable auslagern vor der Verwendung, um CubeType-Fehler zu vermeiden
    let center_y = (time * f32::new(3.5)).sin() * f32::new(0.2);
    let sphere_center = Vec3::new(f32::new(0.0), center_y, f32::new(0.0));
    let base_sphere = p.sub(sphere_center).length() - size;

    let scale = f32::new(6.0);
    let gyroid = ((p.x * scale).sin() * (p.y * scale).cos()
        + (p.x * scale).cos() * (p.z * scale).sin()
        + (p.y * scale).sin() * (p.z * scale).cos())
        / scale;

    base_sphere.max(gyroid * f32::new(0.5))
}

#[cube]
pub fn evaluate_dynamic_torus(p: Vec3, time: f32, size: f32) -> f32 {
    let rot_speed_x = time * f32::new(0.6);
    let rot_speed_y = time * f32::new(0.3);

    let cos_x = rot_speed_x.cos();
    let sin_x = rot_speed_x.sin();
    let cos_y = rot_speed_y.cos();
    let sin_y = rot_speed_y.sin();

    let ry_x = p.x * cos_y - p.z * sin_y;
    let ry_z = p.x * sin_y + p.z * cos_y;
    let rx_y = p.y * cos_x - ry_z * sin_x;
    let rx_z = p.y * sin_x + ry_z * cos_x;

    let r_major = size;
    let r_minor = size * f32::new(0.15);

    let q_x = (ry_x * ry_x + rx_z * rx_z).sqrt() - r_major;
    (q_x * q_x + rx_y * rx_y).sqrt() - r_minor
}

