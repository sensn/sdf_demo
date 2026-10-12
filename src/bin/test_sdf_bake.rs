//! Test für den SDF-Präprozessor (Step 2): bäckt die Statue und prüft
//! Grid-Eigenschaften (Größe, Vorzeichen, Distanz-Range).

#[path = "../sdf_preprocessor.rs"]
mod sdf_preprocessor;

fn main() {
    let res = sdf_preprocessor::SDF_GRID_RES;
    let total = res * res * res;
    println!("Bake statue.obj @ {res}³ = {total} Voxel...");
    let grid = sdf_preprocessor::load_or_bake_sdf("assets/statue.obj");
    assert_eq!(grid.len(), total, "Grid-Größe falsch");

    // Cache-Roundtrip prüfen.
    let cache_path = "assets/statue.obj.sdfcache";
    assert!(std::path::Path::new(cache_path).is_file(), "Cache fehlt");
    let cached = sdf_preprocessor::load_sdf_cache(cache_path, total).unwrap();
    assert_eq!(cached.len(), total);

    // Statistik: Innen-Voxel (negativ) vs. Außen-Voxel.
    let inside = grid.iter().filter(|d| **d < 0.0).count();
    let min = grid.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = grid.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Innen-Voxel: {inside}/{total} ({:.1}%)", 100.0 * inside as f32 / total as f32);
    println!("Distanz-Range: [{min:.3}, {max:.3}]");
    assert!(inside > 0, "Keine Innen-Voxel — Mesh evtl. invertiert?");
    assert!(min < 0.0, "Keine negativen Distanzen — kein geschlossenes Volumen");
    println!("✅ SDF-Präprozessor OK (Bake + Cache + Vorzeichen)");
}
