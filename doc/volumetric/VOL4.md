Die Henyey-Greenstein-Phasenfunktion ist das mathematische Herzstück für kinoreife Nebel- und Wolkeneffekte. Ryan Brucks beschreibt sie im Shader Bits-Blog als essenziell, um Anisotropie (richtungsabhängige Lichtstreuung) zu simulieren.
In der Realität streuen mikroskopisch kleine Wassertröpfchen das Licht nicht gleichmäßig in alle Richtungen (isotrop), sondern stark nach vorne – also direkt in Blickrichtung der Lichtquelle (Forward Scattering). Wenn du direkt in die Sonne blickst, leuchtet der Nebel um sie herum extrem intensiv.
Wir implementieren diese physikalische Funktion nun vollständig in CubeCL als Zero-Cost-Registerberechnung.

Schritt 1: Die Henyey-Greenstein-Phasenfunktion

Füge diese mathematische Funktion zu deinem Shader hinzu. Sie benötigt den Kosinus des Winkels zwischen Blickrichtung und Lichtrichtung sowie einen Asymmetriefaktor g (zwischen -1.0 und 1.0).
• g > 0: Vorwärtsstreuung (Nebel leuchtet im Gegenlicht stark auf).
• g = 0: Isotrope, gleichmäßige Streuung.
rust
#[cube]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let pi = f32::new(3.14159265);
    
    // Die mathematische Formel: (1 - g^2) / (4 * pi * (1 + g^2 - 2 * g * cos_theta)^(1.5))
    let numerator = f32::new(1.0) - g2;
    
    let denom_inner = f32::new(1.0) + g2 - f32::new(2.0) * g * cos_theta;
    let denominator = f32::new(4.0) * pi * denom_inner * denom_inner.sqrt();
    
    numerator / denominator
}
Verwende Code mit Vorsicht.

Schritt 2: Integration in den volumetrischen Hauptloop

Wir berechnen nun einmalig vor der Schleife den Richtungs-Skalar cos_theta (über das Skalarprodukt dot aus Kamerastrahl und Lichtstrahl). Innerhalb der Schleife multiplizieren wir diesen Phasenwert auf das eintreffende Licht auf.
Hier ist die aktualisierte volumetric_raymarch_jittered-Funktion:
rust
#[cube]
pub fn volumetric_raymarch_jittered(
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
    
    // DITHER JITTER (aus dem vorherigen Schritt)
    let jitter = noise_2d_time(pixel_x, pixel_y, settings.time);
    let mut current_pos = ray_origin.add(ray_dir.scale(step_size * jitter));
    
    let mut accumulated_color = Vec3::new(f32::new(0.0), f32::new(0.0), f32::new(0.0));
    let mut transmittance = f32::new(1.0);

    let light_direction = env.key_light_pos.position_to_direction_vector(); 
    let light_color = Vec3::new(env.key_color.x, env.key_color.y, env.key_color.z);

    // 🟢 NEU: Berechne den Winkel zwischen Kamerasichtstrahl und Lichtrichtung
    // Da beide Vektoren normalisiert sind, entspricht das Skalarprodukt dem Kosinus des Winkels.
    let cos_theta = ray_dir.dot(light_direction);
    
    // Anisotropie-Faktor (0.5 bis 0.8 sind exzellente Werte für realistischen Dunst/Rauch)
    // Kann später auch sauber in dein EnvSettings-Struktur als Crate-Parameter ausgelagert werden.
    let anisotropy_g = f32::new(0.6); 
    let phase_value = henyey_greenstein(cos_theta, anisotropy_g);

    for _i in 0..steps {
        let density = get_volumetric_density(current_pos, settings.time, f32::new(1.5));

        if density > f32::new(0.0) {
            // Eigenschatten (Self-Shadowing)
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

            // 🟢 NEU: Modifiziere die Lichtintensität basierend auf der Phasenfunktion
            // Wenn man "in die Lichtquelle" blickt, verstärkt phase_value das Licht massiv.
            let scattered_light_intensity = env.light_intensity * phase_value;
            let local_light = light_color.scale(light_transmission * scattered_light_intensity);
            
            // Opazität und Farbakkumulation
            let opacity = density * step_size * env.fog_density;
            let factored_color = local_light.scale(opacity * transmittance);
            
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

Der visuelle Unterschied

• Ohne Phasenfunktion: Der Nebel wirkt flach, fast wie eine solide, matt eingefärbte Masse. Wenn sich die Kamera um das Objekt herumbewegt, bleibt die Helligkeit des Volumens starr.
• Mit Henyey-Greenstein: Wenn deine Kamera so steht, dass sich der Gyroid/Nebel direkt zwischen der Kamera und der Lichtquelle befindet, bricht ein gleißender, physikalisch korrekter Lichthof (Glow) an den Rändern aus. Befindet sich die Lichtquelle im Rücken der Kamera, wirkt der Nebel realistisch dunkler.
Da die Berechnung von phase_value nur ein einziges Mal pro Pixel vor der Schleife auf der CPU/GPU-Grenze stattfindet, ist diese Erweiterung absolut Zero-Cost innerhalb des teuren Raymarch-Loops.
Möchtest du als Nächstes sehen, wie wir 3D-Rauschstrukturen (wie progressives Pseudo-Perlin oder Simplex-Rauschen) komplett ohne Texturen (Zero-Copy) integrieren, um den Gyroiden in eine sich organisch auflösende Rauchwolke zu verwandeln?
