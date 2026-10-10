==Yes, there is a limit on the number of tensors you can pass, but it depends directly on your target hardware and backend API constraints rather than a fixed limit in [CubeCL](https://github.com/tracel-ai/cubecl) itself==. [1]

Your code snippet passes 7 tensors (plus your commented-out 8th one), which is perfectly fine on almost any modern hardware. However, if you plan to keep expanding this kernel, you should keep the following limitations in mind:

## 1. WebGPU / WGPU Binding Limits (The Most Restrictive)

If your CubeCL code targets WebGPU/WGPU, you will run into the strict uniform/storage buffer binding limits enforced by the browser or WebGPU API standards:

- Max Storage Buffers per Shader Stage: The default WebGPU tier guarantees a minimum of 8 storage buffers per shader stage (though many modern desktop GPUs support more, up to 16 or 24).
- Since every `&Tensor` or `&mut Tensor` maps to a storage buffer behind the scenes, crossing 8 tensors means your code might fail to compile or run on a system or web target that adheres strictly to baseline WebGPU profiles.

## 2. CUDA / HIP Parameter Memory Limits

If you are compiling natively via the `cubecl-cuda` or `cubecl-hip` backends, your arguments are bound by the GPU's kernel parameter memory space: [2]

- On older CUDA architectures, you have a limit of 4,096 bytes for kernel parameters.
- Since each CubeCL `Tensor` reference passes a structure containing pointer/handle information, strides, and metadata (shapes), passing dozens of tensors can consume this parameter space. (Note: Modern GPUs using CUDA 12.1+ expanded this limit significantly to 32,764 bytes). [2]

---

## How to optimize your kernel signature 💡

Your `raymarch_sdf_kernel` is reaching a high parameter count. Instead of passing flat parameter lists (like `env_settings`, `arch_params`, `fold_params`), you can optimize using these two strategies:

1. Pack parameters into a single Metadata Tensor:  
    Combine your loose float arrays (`env_settings`, `arch_params`, `fold_params`) into a single larger `&Tensor<f32>` (e.g., `params_buffer`). Inside the kernel, read the values by using hardcoded or comptime offsets:
    
    ```rust
    let light_intensity = params_buffer[0];
    let ambient_fog     = params_buffer[1];
    let fold_factor     = params_buffer[2];
    ```
    
2. Use Custom Launch Structs (If supported by your CubeCL version):  
    You can define a custom Rust struct annotated with `#[derive(CubeType)]` or similar trait implementations to bundle flat scalar parameters together, reducing the total amount of raw `Tensor` bindings.

Are you targeting WebGPU (Web/WGPU) or native CUDA/Metal for this raymarcher? Let me know if you run into any specific compilation errors, and I can help you pack your parameters!

  

[1] [https://github.com](https://github.com/tracel-ai/cubecl)

[2] [https://developer.nvidia.com](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/)