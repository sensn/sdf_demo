// 1. KORREKTUR: Der Buffer ist jetzt ein f32-Array (nicht mehr u32)
@group(0) @binding(0) var<storage, read> cubecl_output: array<f32>;

struct Resolution {
    width: u32,
    height: u32,
    // 🟢 DoF: 0 = RGBA-interleaved (Raymarch-Output), 1 = CHW planar (DoF-Convolution-Output)
    layout_mode: u32,
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
    // Clamping verhindert OOB-Lesezugriffe an der Nahtstelle
    let px = min(u32(in.uv.x * f32(res.width)), res.width - 1u);
    let py = min(u32(in.uv.y * f32(res.height)), res.height - 1u);

    if res.layout_mode == 0u {
        // Modus 0: RGBA-interleaved (direkter Raymarch-Output, DoF aus)
        let base_index = (py * res.width + px) * 4u;
        let r = cubecl_output[base_index];
        let g = cubecl_output[base_index + 1u];
        let b = cubecl_output[base_index + 2u];
        // Kanal 3 (A) trägt die Tiefe — hier nicht anzeigbar.
        return vec4f(r, g, b, 1.0);
    } else {
        // Modus 1: CHW planar ([3, H, W]) — Output der DoF-Convolution
        let plane = res.width * res.height;
        let offset = py * res.width + px;
        let r = cubecl_output[offset];
        let g = cubecl_output[offset + plane];
        let b = cubecl_output[offset + plane * 2u];
        return vec4f(r, g, b, 1.0);
    }
}
