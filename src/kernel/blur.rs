use cubecl::prelude::*;
use cubecl::frontend::{SharedMemory, sync_notification};
// Importiere Walk sowie die funktionalen Verben für den Datenzugriff
use cubek::tile::{TileArg, Walk};
use cubek::tile::ops::{read, write};

const TILE_WIDTH: u32 = 16;
const TILE_HEIGHT: u32 = 16;

const SHARED_STRIDE: u32 = TILE_WIDTH + 2;
const SHARED_MEM_SIZE: usize = ((TILE_WIDTH + 2) * (TILE_HEIGHT + 2)) as usize;

#[cube]
pub fn fast_tiled_blur_kernel<F: Float, V: Size>(
    input_image: &TileArg<F, V>,
    output_image: &TileArg<F, V>,
    // 🟢 KORREKTUR: Der Walk (Kilometerzähler) wird als dritter Kernel-Parameter 
    // vom Launcher injiziert und steuert die relative Gitter-Position.
    walk: &Walk,
) {
    // 🟢 KORREKTUR: Verwende die vom Launcher übergebene walk-Referenz
    let input_view = input_image.tile_as(walk); 
    let mut output_view = output_image.tile_as(walk);

    let mut shared_mem = SharedMemory::<F>::new(SHARED_MEM_SIZE);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;

    // Index innerhalb des Shared-Memory-Blocks
    let s_idx = (local_y + 1) * SHARED_STRIDE + (local_x + 1);
    
    // Datenzugriff über die funktionalen Verben der DSL
    shared_mem[s_idx] = read(&input_view, local_x, local_y);

    // Kooperatives Laden des Halo-Rands
    if local_x == 0 {
        shared_mem[s_idx - 1] = read(&input_view, local_x - 1, local_y);
    }
    if local_x == TILE_WIDTH - 1 {
        shared_mem[s_idx + 1] = read(&input_view, local_x + 1, local_y);
    }
    if local_y == 0 {
        shared_mem[s_idx - SHARED_STRIDE] = read(&input_view, local_x, local_y - 1);
    }
    if local_y == TILE_HEIGHT - 1 {
        shared_mem[s_idx + SHARED_STRIDE] = read(&input_view, local_x, local_y + 1);
    }

    // Synchronisation der Threads in der Workgroup
    sync_notification();

    let mut color_sum = F::new(0.0_f32);
    
    // 3x3 Faltungs-Filter
    for dy in 0..3 {
        for dx in 0..3 {
            let neighbor_idx = (local_y + dy) * SHARED_STRIDE + (local_x + dx);
            color_sum += shared_mem[neighbor_idx];
        }
    }

    let blurred_color = color_sum / F::new(9.0_f32);
    
    // In das Ausgabe-Tile schreiben
    write(&mut output_view, local_x, local_y, blurred_color);
}
