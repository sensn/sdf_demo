Der legendäre Blogeintrag "Creating a Volumetric Ray Marcher" von Ryan Brucks (Principal Technical Artist bei Epic Games) auf Shader Bits beschreibt fundamentale Techniken zur volumetrischen Lichtstreuung, die unter anderem in Unreal Engine zum Einsatz kommen. [1] (https://shaderbits.com/blog/tag/volumetrics), [2] (https://dev.epicgames.com/community/learning/tutorials/7B0b/unreal-engine-making-a-volume-raymarching-shader-from-scratch-in-unreal-5-3)

Der fundamentale konzeptionelle Unterschied

• Dein aktueller Code (Surface Raymarching): Nutzt Sphere Tracing. Der Strahl "springt" via SDF-Distanzen sicher an die Oberfläche eines Objekts heran und stoppt dort komplett. Es gibt ein klares "Drinnen" und "Draußen" mit harten oder weich ausgeblendeten PBR-Oberflächeneigenschaften. [1] (https://www.diva-portal.org/smash/get/diva2:1647354/FULLTEXT02.pdf)
• Shader Bits (Volumetric Raymarching): Stoppt nie an einer Oberfläche. Stattdessen durchdringt der Strahl ein vordefiniertes Volumen (z. B. eine Bounding Box) in konstanten Schrittweiten (fixed step size). An jedem Punkt misst er die lokale Dichte (density), berechnet die Lichtabsorption (Beer-Lambert-Gesetz) und führt eine sekundäre Strahlen-Märschung in Richtung der Lichtquelle aus, um Self-Shadowing (Eigenschatten innerhalb von Wolken/Rauch) zu simulieren. [1] (https://www.youtube.com/watch?v=SIlKTPnV4R8&t=1), [2] (https://www.reddit.com/r/GraphicsProgramming/comments/1oxlwrb/a_guide_to_volumetric_raymarching/), [3] (https://medium.com/@sushilgangaraju/real-time-volumetric-clouds-with-ray-marching-c6b46d1edeb0), [4] (https://wallisc.github.io/rendering/2020/05/02/Volumetric-Rendering-Part-2.html), [5] (https://shaderbits.com/blog/creating-volumetric-ray-marcher)

Anpassungsstrategie für CubeCL: "Zero-Cost, Zero-Copy"

In CubeCL bedeutet "Zero-Cost, Zero-Copy", dass wir keine neuen Speicherallokationen (Buffers) auf der GPU anlegen, sondern die bestehenden Datenstrukturen rein über mathematische Transformationen und Register-Instanzen auswerten.
1. Zero-Copy SDF-Dichte-Konvertierung: Wir kopieren keine Geometrie in ein Voxel-Grid. Wir nutzen deine mathematischen SDFs (evaluate_dynamic_gyroid etc.) und konvertieren sie direkt im Register von einer harten Oberfläche in ein Dichtevolumen (Density = clamp(-sdf_distance * density_scale, 0.0, 1.0)).
2. No-Copy Parameter passing: Durch die Verwendung deiner CubeType- und CubeLaunch-Strukturen (RenderSettings, EnvSettings) übergeben wir die volumetrischen Parameter (wie fog_density oder step_size) direkt im optimierten Uniform-Speicher ohne zusätzlichen Treiber-Overhead. [1] (https://www.youtube.com/watch?v=eDYyBc3cRmw&t=412)

Code-Adaption: Erweiterung deines Shaders um volumetrische Features

Hier ist die Implementierung des volumetrischen Kernalgorithmus basierend auf dem Shader Bits-Konzept, vollständig übersetzt in deine CubeCL/Rust-Struktur:
rust
#![allow(warnings)]
use cubecl::frontend::CubeType;
use cubecl::prelude::*;

// ... (Deine bestehenden Vec3, SdfResult und parametrisierten Structs bleiben exakt gleich)

/// Hilfsfunktion: Konvertiert ein signed SDF-Ergebnis in ein volumetrisches Dichtesignal.
/// Zero-Copy: Rechnet rein in GPU-Registern.
#[cube]
pub fn get_volumetric_density(p: Vec3, time: f32, size: f32) -> f32 {
    // Wir nutzen deine bestehende Gyroid-SDF als volumetrische Dichtebasis
    let d = evaluate_dynamic_gyroid(p, time, size);
    
    // Invertieren: Innerhalb des Objekts (negatives d) ist die Dichte hoch.
    // Shader Bits Prinzip: Sanftes Ausblenden an der Nebelkante.
    let density_scale = f32::new(5.0);
    (-d * density_scale).min(f32::new(1.0)).max(f32::new(0.0))
}

/// Integriert das Self-Shadowing (Eigenschatten) nach dem Shader Bits Konzept.
/// Erzeugt sekundäre Teststrahlen Richtung Lichtquelle ohne Speicherduplikation.
#[cube]
pub fn calculate_volume_shadow(
    sample_pos: Vec3, 
    light_dir: Vec3, 
    time: f32, 
    size: f32,
    shadow_steps: u32,
    shadow_step_size: f32
) -> f32 {
    let mut accumulated_shadow_density = f32::new(0.0);
    let mut current_pos = sample_pos;

    // Sekundärer Raymarch-Loop hin zur Lichtquelle
    for _s in 0..shadow_steps {
        current_pos = current_pos.add(light_dir.scale(shadow_step_size));
        let local_density = get_volumetric_density(current_pos, time, size);
        accumulated_shadow_density = accumulated_shadow_density + local_density;
    }

    // Beer-Lambert Gesetz zur Lichtabsorption: Transmission = exp(-Dichte * Absorptionskoeffizient)
    let absorption_coefficient = f32::new(0.8);
    f32::exp(-accumulated_shadow_density * shadow_step_size * absorption_coefficient)
}

/// Der Haupt-Volumetric-Raymarch-Loop im CubeCL Shader.
/// Ersetzt das klassische harte Stoppen an der Oberfläche.
#[cube]
pub fn volumetric_raymarch(
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_distance: f32,
    steps: u32,
    settings: RenderSettings,
    env: EnvSettings,
) -> Vec3 {
    let step_size = max_distance / f32::cast(steps);
    let mut current_pos = ray_origin;
    
    let mut accumulated_color = Vec3::new(f32::new(0.0), f32::new(0.0), f32::new(0.0));
    let mut transmittance = f32::new(1.0); // 1.0 bedeutet vollkommen transparent

    let light_direction = env.key_light_pos.position_to_direction_vector(); // Angenommener Richtungsvektor aus Position
    let light_color = Vec3::new(env.key_color.x, env.key_color.y, env.key_color.z);

    // Haupt-Loop entlang des Kamerasichtstrahls
    for _i in 0..steps {
        // Vorwärtsschritt
        current_pos = current_pos.add(ray_dir.scale(step_size));

        // 1. Lokale Dichte abfragen (Zero-Copy)
        let density = get_volumetric_density(current_pos, settings.time, f32::new(1.5));

        if density > f32::new(0.0) {
            // 2. Self-Shadowing via Shadow-Bits Methode (Sekundärer Strahl)
            let shadow_steps = u32::new(4); // Niedrig halten für Performance
            let shadow_step_size = step_size * f32::new(1.5);
            let light_transmission = calculate_volume_shadow(
                current_pos, 
                light_direction, 
                settings.time, 
                f32::new(1.5),
                shadow_steps,
                shadow_step_size
            );

            // 3. Lichtstreuung berechnen (In-Scattering)
            let local_light = light_color.scale(light_transmission * env.light_intensity);
            
            // 4. Akkumulation basierend auf der volumetrischen Opazität
            let opacity = density * step_size * env.fog_density;
            let factored_color = local_light.scale(opacity * transmittance);
            
            accumulated_color = accumulated_color.add(factored_color);
            
            // Transmissionsverlust berechnen
            transmittance = transmittance * (f32::new(1.0) - opacity);

            // 🟢 OPTIMIERUNG (Early Ray Termination):
            // Wenn das Volumen mathematisch blickdicht ist, brechen wir performant ab!
            if transmittance < f32::new(0.001) {
                transmittance = f32::new(0.0);
                break;
            }
        }
    }

    // Hintergrundfarbe einmischen basierend auf verbleibender Transparenz
    let background = Vec3::new(env.bg_color.x, env.bg_color.y, env.bg_color.z);
    accumulated_color.add(background.scale(transmittance))
}
Verwende Code mit Vorsicht.

Die wichtigsten Optimierungen für CubeCL im Detail:

1. Early Ray Termination (Loop-Break): Da Raymarching auf verschachtelten Schleifen basiert, ist der if transmittance < 0.001 { break; }-Check kritisch. Sobald der Nebel oder der Gyroid mathematisch "opaque" (blickdicht) ist, stoppt die GPU die Berechnungen für diesen Pixel sofort. Das spart massiv Rechenzeit bei dichten Clustern. [1] (https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers//ray-marching-get-it-right.html)
2. Geringe Shadow-Steps: Der Schatten-Loop tastet die SDF nur 4-mal ab. Da CubeCL dies direkt in mathematische Instruktionen auf den Shadereinheiten übersetzt (ohne Textursampling-Latenzen), bleibt der Shader trotz verschachtelter Schleifen extrem performant.
Möchtest du als Nächstes einen Temporal Jitter (Dither-Rauschen) einbauen, wie von Ryan Brucks empfohlen, um das sichtbare "Slicing" (Streifenbildung) bei niedrigen Schrittzahlen effektiv zu unterdrücken? [1] (https://shaderbits.com/blog/creating-volumetric-ray-marcher)
