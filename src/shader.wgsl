// 1. KORREKTUR: Der Buffer ist jetzt ein f32-Array (nicht mehr u32)
@group(0) @binding(0) var<storage, read> cubecl_output: array<f32>;

struct Resolution {
    width: u32,
    height: u32,
}
@group(0) @binding(1) var<uniform> res: Resolution;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Das optimierte Fullscreen-Dreieck
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32((vertex_index >> 1u) & 1u) * 4 - 1);
    
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let pixel_x = u32(in.uv.x * f32(res.width));
    let pixel_y = u32(in.uv.y * f32(res.height));
    
    // 2. KORREKTUR: Multiplikation mit 3u, da R, G, B nacheinander liegen
    let base_index = (pixel_y * res.width + pixel_x) * 3u;

    // 3. KORREKTUR: Kanäle direkt als f32 auslesen (Kein Bit-Shifting mehr!)
    let r = cubecl_output[base_index];
    let g = cubecl_output[base_index + 1u];
    let b = cubecl_output[base_index + 2u];
    
    return vec4f(r, g, b, 1.0);
}
