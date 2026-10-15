## Summary of changes
  ### Step 1 — GUI typings & global sync (`gui/mod.rs`, `app_state.rs`)
    • 17 new `Entity<SliderState>` handles in `GuiState` with correct ranges/steps/defaults (e.g. `volumetr    ic_fog_steps`: 4–64 step 1 default 16; `statue_center_y`: default -1.4; `mesh_scale_factor`: 0.1–1.0 defa
  ult 0.8)
    • Every slider wired via `subscribe_slider(...).detach()` in the setup phase
    • `sync_sliders` extended with `sync_one` for all 17 sliders (State → Slider, feedback-loop safe)
  ### Step 2 — `env_tab` (`tabs/env.rs`)                                                                         • New accordion sections "Atmosphere" (Fog Steps, Fog Falloff) and "Lighting & AO" (AO Iterations, Shad
  ow Softness K); `env_open` vec extended to 5 entries
  ### Step 3 — `arch_tab` & `fold_tab`                                                                           • `arch.rs`: new "Global Grid" accordion section with `cell_size`                                            • `fold.rs`: new "Infinite Repetition" panel with `pillar_spacing` (2–20) and `pillar_height` (1–10)
  ### Step 4 — `mat_tab` (`tabs/mat.rs`)
    • "Global Reflections": `reflection_max_steps` (8–64), `reflection_roughness_gate` (0.05–1.0), `fresnel    _strength_factor` (0.0–1.0)
    • "Material Physics": Statue PBR (ID 2) `statue_metallic`, Pillar PBR (ID 1) `pillar_roughness`
  ### Step 5 — `obj_tab` (`tabs/obj.rs`)
    • "SDF Grid Space": `sdf_grid_size` (1–16), `mesh_scale_factor`
    • "Transformations": `statue_center_x/y/z`                                                                   • "Bounding Controls": `bounding_sphere_radius_offset`
  ### Backend wiring (parameters are actually functional, not just UI)                                           • `kernel.rs`: `EnvSettings` += `volumetric_fog_steps`, `fog_density_falloff`, `ao_iterations`, `shadow
  _softness_k`; `FoldParams` += `pillar_spacing`, `pillar_height`; `RenderSettings` += reflection/PBR/statu
  e fields. All hardcoded values (fog steps 16, falloff 0.08, AO 5, shadow 8.0, refl steps 32, gate 0.25, f
  resnel 0.35, metallic/roughness 0.8, `STATUE_CENTER`, bounding 0.1) replaced by struct fields; `scene_map    _with_material`, `calculate_normal`, `trace_reflection`, `calculate_soft_shadow`, `calculate_ao`, `integr
  ate_volumetric_fog` signatures extended accordingly
    • `sdf_preprocessor.rs`: `load_obj_to_sdf`/`load_or_bake_sdf` now take `mesh_scale_factor` (replaces ha    rdcoded `1.6/extent` = 80%); obsolete `STATUE_CENTER` const removed
    • `app_state.rs`: new fields + defaults + launch-struct builders (`env_launch`, `fold_launch`, `render_
  launch`) wired                                                                                               ### Guard rails
    • New `slider_row_fmt` helper: iteration/step sliders render as integers (`.0`), floats keep `.2` preci
  sion                                                                                                           • `cargo check --all-targets` clean; `test_kernel_launch` (99.1% non-black pixels, WGSL codegen OK) and
   `test_sdf_bake` pass
  Note: `mesh_scale_factor` affects the SDF bake — since the bake result is cached (`*.sdfcache`), changing     it requires deleting the cache file to take effect.