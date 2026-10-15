Um das typische volumetrische "Slicing" (Streifenbildung durch diskrete Schritte) ohne Erhöhung der Schrittanzahl zu eliminieren, beschreibt Ryan Brucks im Shader Bits-Blog eine der wichtigsten Optimierungen: Temporal Jittering via Dithering.
Dabei wird der Startpunkt des Strahls pro Pixel um einen pseudozufälligen Bruchteil der Schrittweite verschoben. In Kombination mit einem zeitlichen Rauschen (verändert per Frame) und einem nachträglichen TAA (Temporal Anti-Aliasing) oder einer leichten Bewegungsunschärfe verschmilzt das Muster zu einem perfekt glatten Volumen.
Hier ist die Anpassung für deinen CubeCL-Shader. Wir implementieren einen Zero-Cost-Pseudo-Zufallszahlengenerator (PRNG) direkt in den Registern, basierend auf den Pixelkoordinaten und der Systemzeit, sodass wir keinerlei Rauschtexturen von der CPU streamen müssen (Zero-Copy).

Schritt 1: Ein registerbasierter Zufallsgenerator (Hash-Funktion)

Füge diese mathematische Hilfsfunktion hinzu. Sie nimmt die zweidimensionalen Pixelkoordinaten sowie die aktuelle Zeit und liefert einen deterministischen Wert zwischen 0.0 und 1.0.
rust
#[cube]
pub fn noise_2d_time(x: u32, y: u32, time: f32) -> f32 {
    // Wandle die Pixel-IDs in mathematische Skalare um
    let fx = f32::cast(x);
    let fy = f32::cast(y);
    
    // Ein klassischer, schneller GPU-Hash (Sinus-Fraktal-Rauschen)
    // Verhindert Textur-Lookups und läuft komplett im Rechenwerk der GPU
    let dot_product = fx * f32::new(12.9898) + fy * f32::new(78.233) + time * f32::new(43758.5453);
    let sine_val = dot_product.sin();
    
    // Extrahiere den Nachkommateil (fract)
    sine_val * f32::new(43758.5453) - (sine_val * f32::new(43758.5453)).floor()
}
Verwende Code mit Vorsicht.

Schritt 2: Integration des Jitters in den Raymarch-Loop

Wir passen nun die volumetric_raymarch-Funktion an. Der Strahl springt vor dem eigentlichen Loop um ein zufälliges Stück nach vorne. Zudem übergeben wir die aktuellen Pixelkoordinaten (pixel_x, pixel_y), welche du aus den CubeCL Built-ins (cubecl::prelude::absolute_pos) im Hauptkernel erhältst.
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
    
    // 🟢 DITHER JITTER: Erzeuge einen Wert zwischen 0.0 und 1.0 pro Pixel und Frame
    let jitter = noise_2d_time(pixel_x, pixel_y, settings.time);
    
    // Verschiebe den Startpunkt leicht nach vorne (maximal um eine ganze Schrittweite)
    // Dadurch tastet jeder Pixel das Volumen auf einer leicht versetzten Ebene ab
    let mut current_pos = ray_origin.add(ray_dir.scale(step_size * jitter));
    
    let mut accumulated_color = Vec3::new(f32::new(0.0), f32::new(0.0), f32::new(0.0));
    let mut transmittance = f32::new(1.0);

    let light_direction = env.key_light_pos.position_to_direction_vector(); 
    let light_color = Vec3::new(env.key_color.x, env.key_color.y, env.key_color.z);

    for _i in 0..steps {
        // Lokale Dichte abfragen
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

            // Lichtstreuung & Opazität
            let local_light = light_color.scale(light_transmission * env.light_intensity);
            let opacity = density * step_size * env.fog_density;
            let factored_color = local_light.scale(opacity * transmittance);
            
            accumulated_color = accumulated_color.add(factored_color);
            transmittance = transmittance * (f32::new(1.0) - opacity);

            // Optimierung: Early Ray Termination
            if transmittance < f32::new(0.001) {
                transmittance = f32::new(0.0);
                break;
            }
        }

        // Vorwärtsschritt für die nächste Iteration
        current_pos = current_pos.add(ray_dir.scale(step_size));
    }

    let background = Vec3::new(env.bg_color.x, env.bg_color.y, env.bg_color.z);
    accumulated_color.add(background.scale(transmittance))
}
Verwende Code mit Vorsicht.

Der Effekt im Vergleich

• Ohne Jitter: Bei z.B. steps = 32 siehst du in konzentrischen Ringen harte Kanten an den Nebelgrenzen, da alle Pixel exakt auf denselben Distanz-Ebenen (step_size * 1, step_size * 2, ...) prüfen.
• Mit Jitter: Die harten Kanten verwandeln sich augenblicklich in ein feines, hochfrequentes Rauschen (Filmkorn-Effekt). Das Auge nimmt dieses Rauschen bei 60 FPS als homogenen, weichen Nebel wahr, anstatt als mathematische "Scheiben".
Da diese mathematische Operation direkt in den schnellen ALU-Rechenwerken der Shadereinheiten ausgeführt wird und keine Speicherbandbreite beansprucht, kostet dich diese dramatische visuelle Verbesserung nahezu 0 % Performance.
Möchtest du nun sehen, wie du die globale absolute Pixel-ID aus den CubeCL-Subplots ausliest, um sie korrekt an diesen Raymarcher zu übergeben?
