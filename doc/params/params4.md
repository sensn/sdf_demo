Der Performance-Vergleich zwischen `TensorArg::from_raw_parts` und Custom Launch Structs (wie `RaymarchParams`) ist ein Vergleich von Äpfeln mit Birnen, da sie auf der GPU in völlig unterschiedlichen Speicherbereichen landen. [1]

Grundsätzlich gilt: Custom Structs gewinnen beim Lese-Zugriff (Runtime Performance) auf fast allen Backends, während `from_raw_parts` lediglich flexibler bei dynamischen Speicherstrukturen ist. [2]

Ein detaillierter Vergleich schlüsselt die Unterschiede nach Hardware-Architektur und Backends auf:

---

## Direct Comparison Overview

|Kriterium|`TensorArg::from_raw_parts`|Custom Launch Structs (`#[derive(CubeLaunch)]`)|Gewinner|
|---|---|---|---|
|GPU-Speichertyp|Global Memory (Storage Buffer / VRAM)|Constant Memory / Uniforms (UBOs)|Custom Structs 🏆|
|Lese-Latenz (GPU)|Hoch (~200–800 Zyklen, falls nicht im L2/L1 Cache)|Extrem niedrig (~1–10 Zyklen, da gecached)|Custom Structs 🏆|
|Host-Overhead (CPU)|Extrem gering (nur Pointer- & Metadaten-Übergabe)|Minimal (Kopieren von wenigen Bytes in Staging-Buffer)|Unentschieden|
|WebGPU Bindings|Verbraucht wertvolle Storage-Buffer-Slots (Limit: meist 8)|Verbraucht Uniform-Buffer-Slots (separates Limit)|Custom Structs 🏆|
|Einsatzzweck|Große, dynamische Datenmengen (SDF-Grids, Texturen)|Statische Konfigurationsparameter (Licht, Faltung)|_Kontextabhängig_|

---

## Backend-Spezifische Analyse

## 1. WebGPU / WGPU Backend (Browser & Native WGPU)

- Custom Structs: Werden als Uniform Buffer (UBO) gebunden. In WebGPU ist der Lesezugriff auf Uniforms massiv optimiert, da die Hardware davon ausgeht, dass alle Threads im Grid zeitgleich dieselben Parameter lesen (_Uniform Broadcast_). Zudem belasten sie nicht dein Limit von standardmäßig maximal 8 Storage Buffers. [3, 4]
- TensorArg (`from_raw_parts`): Erzeugt im WGSL-Shader einen `binding(X, storage)`. Jeder Lesezugriff im Raymarcher muss den Umweg über den globalen Buffer gehen. Wenn du 7–8 Tensoren so übergibst, stößt du an das harte WebGPU-Limit, und die Performance sinkt, da kein automatischer Uniform-Cache greift. [2]

## 2. Vulkan / SPIR-V Backend (Nativ)

- Custom Structs: Werden entweder in Push Constants (wenn die Struktur sehr klein ist, meist <128/256 Bytes) oder in ein `VkDescriptorSet` als Uniform Buffer übersetzt. Push Constants sind das absolut Schnellste, was Vulkan zu bieten hat, da die Werte direkt in den Command-Buffer injiziert werden und quasi registerschnell abrufbar sind.
- TensorArg (`from_raw_parts`): Mappt auf `VkBuffer` mit `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`. Der Zugriff erfordert volles Speicher-Pipelining. Für mathematische Konstanten (wie deine SDF-Faktoren) ist das im Vergleich zu Push Constants oder UBOs messbar langsamer.

## 3. CUDA & HIP Backends (Nvidia / AMD)

- Custom Structs: Werden direkt in den Constant Memory (`__constant__`) oder als direkte Kernel-Parameter auf dem Stack abgelegt. CUDA-Parameter-Speicher (bis zu 4KB bei älteren, bis zu ~32KB bei modernen GPUs) ist extrem nah an den Rechenkernen und wird hardwareseitig in speziellen Constant-Caches gesichert.
- TensorArg (`from_raw_parts`): Übergibt lediglich die Pointer-Adresse und die Shapes (Strides/Dims) via Parameter-Stack. Der eigentliche Inhalt liegt im `__global__` Speicher. Ein `params[0]`-Aufruf erfordert eine globale Speicheranforderung. [2]

---

## Warum sind Custom Structs im SDF-Raymarcher schneller?

In einer Raymarching-Schleife liest dein Kernel pro Pixel hunderte Male dieselben Werte (z. B. `fold_factor` oder `light_intensity`), während sich der Strahl durch den Raum tastet.

- Nutzt du `from_raw_parts`, triggert jeder Schleifendurchlauf theoretisch globale Leseoperationen (die zwar im L1/L2 Cache landen können, aber dennoch die Cache-Validierung durchlaufen müssen).
- Nutzt du ein Custom Struct, liegen diese Werte im Constant- / Uniform-Cache direkt neben den ALUs der GPU. Die Latenz sinkt fast auf Null.

## Fazit & Empfehlung

Verwende `TensorArg::from_raw_parts` ausschließlich für deine echten, großen Datenstrukturen (z. B. den `output`-Bildpuffer, deine Material-Tabellen oder dynamische Voxel-Grids). [2]

Für alle architektonischen Parameter, mathematischen Gewichte, Licht-Vektoren und Iterations-Zähler deines SDF-Tempels sind Custom Structs auf jedem Backend (insbesondere WebGPU und Vulkan) die performantere und architektonisch sauberere Wahl.

Möchtest du sehen, wie man die Performance deines Kernels weiter optimiert, indem wir Comptime-Variablen in CubeCL nutzen, um z. B. Schleifen-Iterationen (`max_steps`) schon zur Kompilierzeit komplett aufzulösen?

  

[1] [https://crates.io](https://crates.io/crates/cubecl)

[2] [https://docs.rs](https://docs.rs/cubecl/latest/cubecl/frontend/struct.TensorBinding.html)

[3] [https://rustify.rs](https://rustify.rs/articles/rust-gpu-computing-wgpu-2026)

[4] [https://github.com](https://github.com/gfx-rs/wgpu)