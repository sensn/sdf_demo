//! CPU-seitiger SDF-Präprozessor (Agent_Impl_Plan.md, Step 2):
//! OBJ-Mesh → 3D-SDF-Grid (f32-Voxel) + Roh-Binär-Cache.
//!
//! - `tobj` lädt das OBJ (Vertices + Triangle-Indices).
//! - `mesh_to_sdf` befüllt ein reguläres Voxelgitter mit vorzeichen-
//!   behafteten Distanzen (Außen positiv, Innen negativ).
//! - Das Grid wird als rohe f32-Slice (`.bin`) gecacht, damit der
//!   teure Bake nur einmal pro Mesh/Resolution läuft.

use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Auflösung des SDF-Grids (Kantenlänge in Voxeln). Muss mit dem
/// Kernel (`sample_sdf_grid`) und dem Cache-Header übereinstimmen.
pub const SDF_GRID_RES: usize = 64;
/// Welt-Ausdehnung des Grids (Würfel-Kantenlänge, zentriert am Ursprung).
pub const SDF_GRID_SIZE: f32 = 8.0;

/// Lädt ein OBJ-File und bäckt es in ein 3D-SDF-Grid.
///
/// Das Mesh wird auf die Einheits-Box normalisiert und in einen
/// `[SDF_GRID_RES]^3`-Voxel-Buffer um `[0,0,0]` gesampelt.
/// Rückgabe: flaches Grid in ZYX-Reihenfolge (z am langsamsten).
pub fn load_obj_to_sdf(file_path: &str, resolution: usize) -> Vec<f32> {
    let (models, _materials) = tobj::load_obj(
        file_path,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("OBJ load failed for {file_path}: {e}"));

    // Vertices + Triangle-List-Indices aller Modelle zusammenführen.
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for model in &models {
        let mesh = &model.mesh;
        let base = vertices.len() as u32;
        for v in 0..mesh.positions.len() / 3 {
            vertices.push([
                mesh.positions[3 * v],
                mesh.positions[3 * v + 1],
                mesh.positions[3 * v + 2],
            ]);
        }
        for idx in &mesh.indices {
            indices.push(base + *idx as u32);
        }
    }

    if vertices.is_empty() {
        // Leeres Mesh → neutrales Grid (alles "weit weg").
        return vec![1e6; resolution * resolution * resolution];
    }

    // --- Mesh auf Einheits-Box normalisieren ([-1,1]^3) ---
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for v in &vertices {
        for c in 0..3 {
            min[c] = min[c].min(v[c]);
            max[c] = max[c].max(v[c]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let extent = (max[0] - min[0])
        .max(max[1] - min[1])
        .max(max[2] - min[2])
        .max(1e-6);
    let scale = 2.0 / extent;
    for v in vertices.iter_mut() {
        for c in 0..3 {
            v[c] = (v[c] - center[c]) * scale;
        }
    }

    // --- Query-Punkte: reguläres Grid über [-size/2, size/2]^3 ---
    // Voxel-Zentren; Abstand in Welt-Einheiten (Grid-Space).
    let half = SDF_GRID_SIZE * 0.5;
    let step = SDF_GRID_SIZE / (resolution as f32 - 1.0);
    let mut query_points: Vec<[f32; 3]> =
        Vec::with_capacity(resolution * resolution * resolution);
    for z in 0..resolution {
        for y in 0..resolution {
            for x in 0..resolution {
                query_points.push([
                    -half + x as f32 * step,
                    -half + y as f32 * step,
                    -half + z as f32 * step,
                ]);
            }
        }
    }
    // Mesh liegt in [-1,1]^3; Grid spannt [-4,4]^3 → Faktor 4.
    // Query-Punkte in Mesh-Raum umrechnen, SDF-Werte zurückskalieren.
    let mesh_scale = SDF_GRID_SIZE / 2.0; // 4.0
    let mesh_queries: Vec<[f32; 3]> = query_points
        .iter()
        .map(|p| [p[0] / mesh_scale, p[1] / mesh_scale, p[2] / mesh_scale])
        .collect();

    let sdf = mesh_to_sdf::generate_sdf(
        &vertices,
        mesh_to_sdf::Topology::TriangleList(Some(&indices)),
        &mesh_queries,
        mesh_to_sdf::AccelerationMethod::RtreeBvh,
    );

    // Zurück in Welt-Einheiten; Grid in ZYX-Reihenfolge (passt zu
    // sample_sdf_grid: index = x + res*(y + res*z)).
    sdf.into_iter().map(|d| d * mesh_scale).collect()
}

/// Speichert das SDF-Grid als rohe f32-Binärdatei (Header: magic + res).
pub fn save_sdf_cache(path: &str, data: &[f32]) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    use std::io::Write;
    w.write_all(b"SDF1")?;
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

/// Lädt das SDF-Grid aus der Roh-Binärdatei. `size` = erwartete
/// Elementzahl (res^3). Fällt bei Header/Größen-Mismatch hart aus.
pub fn load_sdf_cache(path: &str, size: usize) -> std::io::Result<Vec<f32>> {
    let file = std::fs::File::open(path)?;
    let mut r = BufReader::new(file);
    use std::io::Read;
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"SDF1" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "SDF cache: bad magic",
        ));
    }
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len != size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("SDF cache: expected {size} floats, found {len}"),
        ));
    }
    let mut data = vec![0.0f32; size];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    r.read_exact(bytes)?;
    Ok(data)
}

/// Lädt das SDF-Grid für `obj_path` — aus dem Cache, oder bäckt es
/// frisch und schreibt den Cache. Cache-Key: `<obj>.sdfcache`.
pub fn load_or_bake_sdf(obj_path: &str) -> Vec<f32> {
    let res = SDF_GRID_RES;
    let total = res * res * res;
    let cache_path = format!("{obj_path}.sdfcache");
    if Path::new(&cache_path).is_file() {
        if let Ok(data) = load_sdf_cache(&cache_path, total) {
            return data;
        }
    }
    let data = load_obj_to_sdf(obj_path, res);
    let _ = save_sdf_cache(&cache_path, &data);
    data
}
