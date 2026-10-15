Um eine glühende Explosion, Plasma oder magmatische Adern im Inneren der Rauchwolke zu simulieren, koppeln wir die Dichte nun mit einem prozeduralen Emissions-Wert (Self-Illumination).
Das mathematische Prinzip basiert darauf, dass die heißesten und am tiefsten im Volumen liegenden Bereiche am stärksten leuchten. Wir nutzen dafür das berechnete 3D-Rauschen und deine PBR-Materialparameter (emissive), um ohne zusätzliche Speicherzugriffe (Zero-Copy) ein dynamisches Glühen zu erzeugen.

Schritt 1: Dichte und Emission gekoppelt berechnen

Wir erweitern die Volumenabfrage so, dass sie sowohl die Dichte als auch die emittierte Lichtfarbe (basierend auf einer simulierten Hitze-Verteilung) zurückgibt. Dazu definieren wir eine kleine Hilfsstruktur.
rust
#[derive(Copy, Clone, CubeType)]
#[cube(derive(Copy, Clone))]
pub struct VolumeSample {
    pub density: f32,
    pub emission: Vec3,
}

#[cube]
pub fn get_volumetric_data(p: Vec3, time: f32, size: f32) -> VolumeSample {
    let base_gyroid = evaluate_dynamic_gyroid(p, time, size);
    
    let wind = Vec3::new(time * f32::new(0.2), time * f32::new(0.5), time * f32::new(0.1));
    let noise_pos = p.add(wind);

    let noise_octave1 = noise_3d(noise_pos.scale(f32::new(3.0)));
    let noise_octave2 = noise_3d(noise_pos.scale(f32::new(6.12)));
    let combined_noise = noise_octave1 * f32::new(0.7) + noise_octave2 * f32::new(0.3);

    // Die modulierte SDF bestimmt die Form des Rauchs
    let modulated_sdf = base_gyroid + combined_noise * f32::new(0.15);
    
    let density_scale = f32::new(6.0);
    let density = (-modulated_sdf * density_scale).min(f32::new(1.0)).max(f32::new(0.0));

    // 1. 🟢 HITZE-BERECHNUNG (Emission):
    // Je tiefer wir im Inneren des Objekts sind (negativere base_gyroid-Werte) 
    // und je turbulenter das Rauschen ist, desto heißer ist der Kern.
    let heat = (-base_gyroid * f32::new(4.0) + combined_noise * f32::new(2.0))
        .min(f32::new(1.0))
        .max(f32::new(0.0));

    // 2. GLÜH-FARBE (Blackbody-Approximation für Feuer/Magma)
    // Hohe Hitze = Gelb/Weiß, Mittlere Hitze = Orange, Niedrige Hitze = Dunkelrot
    let glow_color = Vec3::new(
        heat * f32::new(2.5),                           // Hoher Rot-Anteil (Intensität > 1 für HDR/Bloom)
        heat * heat * f32::new(1.2),                    // Moderater Grün-Anteil für Orange/Gelb
        heat * heat * heat * heat * f32::new(0.4)       // Geringer Blau-Anteil bei extremem Fokus
    );

    VolumeSample {
        density,
        emission: glow_color,
    }
}
Verwende Code mit Vorsicht.

Schritt 2: Integration des Glühens in den Hauptloop

Wir passen nun den volumetrischen Raymarcher an, um die emittierte Energie entlang des Strahls zu akkumulieren. Das emittierte Licht benötigt im Gegensatz zum Sonnenlicht keinen teuren Schattenstrahl, da es von innen heraus leuchtet. Das macht diese Ergänzung extrem performant (Zero-Cost).
rust
#[cube]
pub fn volumetric_raymarch_jittered_emissive(
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_distance: f32,
    steps: u32,
    pixel_x: u32,
    pixel_y: u32,
    settings: RenderSettings,
    env: EnvSettings,
) -> Vec3 {
    let step_size = max_distance / f32::cast(steps);
    let jitter = noise_2d_time(pixel_x, pixel_y, settings.time);
    let mut current_pos = ray_origin.add(ray_dir.scale(step_size * jitter));
    
    let mut accumulated_color = Vec3::new(f32::new(0.0), f32::new(0.0), f32::new(0.0));
    let mut transmittance = f32::new(1.0);

    let light_direction = env.key_light_pos.position_to_direction_vector(); 
    let light_color = Vec3::new(env.key_color.x, env.key_color.y, env.key_color.z);

    let cos_theta = ray_dir.dot(light_direction);
    let anisotropy_g = f32::new(0.6); 
    let phase_value = henyey_greenstein(cos_theta, anisotropy_g);

    for _i in 0..steps {
        // 🟢 Jetzt Dichte UND Emission gleichzeitig abfragen
        let sample = get_volumetric_data(current_pos, settings.time, f32::new(1.5));

        if sample.density > f32::new(0.0) {
            // Eigenschatten (Self-Shadowing) für das externe Sonnenlicht
            let shadow_steps = u32::new(4); 
            let shadow_step_size = step_size * f32::new(1.5);
            let light_transmission = calculate_volume_shadow(
                current_pos, 
                light_direction, 
                settings.time, 
                f32::new(1.5),
                shadow_steps,
                shadow_step_size
            );

            // Externes gestreutes Licht berechnen
            let scattered_light_intensity = env.light_intensity * phase_value;
            let external_light = light_color.scale(light_transmission * scattered_light_intensity);
            
            // 🟢 NEU: Kombiniere gestreutes Licht mit der internen Emission (Glühen)
            // Die Emission wird direkt addiert, da sie unabhängig von der externen Sonne existiert.
            let total_light = external_light.add(sample.emission);
            
            // Opazität bestimmen und akkumulieren
            let opacity = sample.density * step_size * env.fog_density;
            let factored_color = total_light.scale(opacity * transmittance);
            
            accumulated_color = accumulated_color.add(factored_color);
            transmittance = transmittance * (f32::new(1.0) - opacity);

            if transmittance < f32::new(0.001) {
                transmittance = f32::new(0.0);
                break;
            }
        }

        current_pos = current_pos.add(ray_dir.scale(step_size));
    }

    let background = Vec3::new(env.bg_color.x, env.bg_color.y, env.bg_color.z);
    accumulated_color.add(background.scale(transmittance))
}
Verwende Code mit Vorsicht.

Der visuelle Effekt

Wenn der Nebel dünner wird (z. B. an den äußeren Fransen durch den turbulenten Wind), siehst du dunklen, vom Sonnenlicht beschienenen Rauch. Schaut man jedoch tiefer in die Struktur des Gyroiden hinein, bricht ein intensives, orange-gelb pulsierendes Plasma-Licht hervor. Da die Rot-Werte in der Berechnung 1.0 überschreiten (f32::new(2.5)), ist dieses Signal perfekt für Post-Processing-Effekte wie HDR-Tonemapping und einen saftigen Bloom-Filter (Überstrahl-Effekt) vorbereitet.
Möchtest du sehen, wie du einen Tonemapper (wie ACES oder Reinhard) direkt am Ende deines Kernels in CubeCL implementierst, damit die intensiven HDR-Glühwerte auf dem Bildschirm nicht einfach hässlich weiß ausbrennen?
