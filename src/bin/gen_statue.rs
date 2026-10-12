//! Erzeugt `assets/statue.obj` — eine stilisierte Statue (Torso + Kopf +
//! Sockel) als prozedurales Low-Poly-Mesh, falls noch keine OBJ-Datei
//! existiert. Wird vom SDF-Präprozessor (Step 2) konsumiert.
//!
//! Aufbau (Y-Achse nach oben, Einheiten ~[-1,1]):
//!   - Sockel: Box 1.6×0.3×1.6
//!   - Torso:  gestapelte, verjüngte Quader (Schulter → Hüfte)
//!   - Kopf:   Ikosaeder-artiger Oktaeder
//!   - Arme:   zwei schräge Quader
//!
//! Das Mesh ist bewusst grob (Low-Poly): Das 64³-SDF-Grid glättet die
//! Kanten ohnehin, und der Bake kostet nur Sekunden.

use std::fs;
use std::path::Path;

fn main() {
    let out = Path::new("assets/statue.obj");
    if out.exists() {
        println!("assets/statue.obj existiert bereits — übersprungen.");
        return;
    }
    fs::create_dir_all("assets").expect("assets/ anlegen fehlgeschlagen");

    let mut obj = String::from("# Prozedurale Statue (sdf_demo, Step 2)\n");
    let mut v_count = 0usize;

    // Helper: Quader (achsenparallel) als 12 Dreiecke schreiben.
    let mut push_box = |obj: &mut String,
                        cx: f32,
                        cy: f32,
                        cz: f32,
                        hx: f32,
                        hy: f32,
                        hz: f32,
                        v_count: &mut usize| {
        let (x0, x1) = (cx - hx, cx + hx);
        let (y0, y1) = (cy - hy, cy + hy);
        let (z0, z1) = (cz - hz, cz + hz);
        let corners = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], // -Z
            [x1, y0, z1], [x0, y0, z1], [x0, y1, z1], [x1, y1, z1], // +Z
        ];
        for c in corners {
            obj.push_str(&format!("v {} {} {}\n", c[0], c[1], c[2]));
        }
        let b = *v_count;
        // 6 Flächen à 2 Dreiecke (1-basierte OBJ-Indizes).
        let faces: [[usize; 4]; 6] = [
            [1, 3, 4, 2], // -Z
            [5, 7, 8, 6], // +Z
            [1, 2, 6, 5], // -Y (unten)
            [3, 4, 8, 7], // +Y (oben)
            [1, 5, 7, 3], // -X
            [2, 4, 8, 6], // +X
        ];
        for f in faces {
            obj.push_str(&format!(
                "f {} {} {}\n",
                b + f[0], b + f[1], b + f[2]
            ));
            obj.push_str(&format!(
                "f {} {} {}\n",
                b + f[1], b + f[3], b + f[2]
            ));
        }
        *v_count += 8;
    };

    // Sockel (unten).
    push_box(&mut obj, 0.0, -0.85, 0.0, 0.8, 0.15, 0.8, &mut v_count);
    // Beine/Torso-Stumpf.
    push_box(&mut obj, 0.0, -0.55, 0.0, 0.35, 0.2, 0.3, &mut v_count);
    // Torso: 3 gestapelte, verjüngte Segmente.
    push_box(&mut obj, 0.0, -0.2, 0.0, 0.3, 0.2, 0.22, &mut v_count);
    push_box(&mut obj, 0.0, 0.1, 0.0, 0.33, 0.2, 0.24, &mut v_count);
    push_box(&mut obj, 0.0, 0.4, 0.0, 0.38, 0.15, 0.26, &mut v_count);
    // Arme (leicht angewinkelt).
    push_box(&mut obj, -0.5, 0.25, 0.0, 0.08, 0.3, 0.08, &mut v_count);
    push_box(&mut obj, 0.5, 0.25, 0.0, 0.08, 0.3, 0.08, &mut v_count);
    // Kopf: Oktaeder (8 Dreiecke).
    let head_c = [
        [0.0, 0.85, 0.0],  // oben
        [0.0, 0.55, 0.0],  // unten
        [0.18, 0.7, 0.0],  // +X
        [-0.18, 0.7, 0.0], // -X
        [0.0, 0.7, 0.18],  // +Z
        [0.0, 0.7, -0.18], // -Z
    ];
    for c in head_c {
        obj.push_str(&format!("v {} {} {}\n", c[0], c[1], c[2]));
    }
    let h = v_count;
    let head_faces = [
        [1, 3, 5], [1, 5, 4], [1, 4, 6], [1, 6, 3], // oberer Kranz
        [2, 5, 3], [2, 4, 5], [2, 6, 4], [2, 3, 6], // unterer Kranz
    ];
    for f in head_faces {
        obj.push_str(&format!("f {} {} {}\n", h + f[0], h + f[1], h + f[2]));
    }

    fs::write(out, obj).expect("statue.obj schreiben fehlgeschlagen");
    println!("assets/statue.obj erzeugt ({} Vertices).", v_count + 6);
}
