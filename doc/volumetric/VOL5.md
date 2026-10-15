Um den mathematisch perfekten Gyroiden in eine sich organisch verformende, realistische Rauch- oder Wolkenstruktur zu verwandeln, nutzen wir ein prozedurales 3D-Pseudo-Perlin-Rauschen.
Da wir den Zero-Copy- und Zero-Cost-Ansatz strikt beibehalten wollen, laden wir keine schweren 3D-Rauschtexturen in den GPU-Speicher. Stattdessen berechnen wir ein mathematisches Rauschen direkt in den Registern der GPU-Kerne via Value-Noise-Interpolation.

Schritt 1: Ein registerbasiertes 3D-Rauschen für CubeCL

Wir fügen eine hocheffiziente Hash-Funktion hinzu, die aus drei räumlichen Fließkommazahlen einen deterministischen Zufallswert generiert, und interpolieren diesen kubisch im Raum.
rust
#[cube]
pub fn hash_3d(p: Vec3) -> f32 {
    // Ein schneller, stabiler 3D-Vektor-Hash für GPU-Rechenwerke
    let dot_product = p.x * f32::new(127.1) 
                    + p.y * f32::new(311.7) 
                    + p.z * f32::new(74.7);
    let sine_val = dot_product.sin();
    sine_val * f32::new(43758.5453) - (sine_val * f32::new(43758.5453)).floor()
}

#[cube]
pub fn noise_3d(p: Vec3) -> f32 {
    let i = Vec3::new(p.x.floor(), p.y.floor(), p.z.floor());
    let f = Vec3::new(
        p.x - p.x.floor(),
        p.y - p.y.floor(),
        p.z - p.z.floor()
    );

    // Kubische Hermite-Interpolation (Smoothstep) für weiche Übergänge
    let u = Vec3::new(
        f.x * f.x * (f32::new(3.0) - f32::new(2.0) * f.x),
        f.y * f.y * (f32::new(3.0) - f32::new(2.0) * f.y),
        f.z * f.z * (f32::new(3.0) - f32::new(2.0) * f.z)
    );

    // Die 8 Ecken des 3D-Gitterw聲fels auswerten
    let i000 = hash_3d(i);
    let i100 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(0.0), f32::new(0.0))));
    let i010 = hash_3d(i.add(Vec3::new(f32::new(0.0), f32::new(1.0), f32::new(0.0))));
    let i110 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(1.0), f32::new(0.0))));
    let i001 = hash_3d(i.add(Vec3::new(f32::new(0.0), f32::new(0.0), f32::new(1.0))));
    let i101 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(0.0), f32::new(1.0))));
    let i011 = hash_3d(i.add(Vec3::new(f32::new(0.0), f32::new(1.0), f32::new(1.0))));
    let i111 = hash_3d(i.add(Vec3::new(f32::new(1.0), f32::new(1.0), f32::new(1.0))));

    // Trilineare Interpolation zwischen den Ecken
    let x00 = i000 + u.x * (i100 - i000);
    let x10 = i010 + u.x * (i110 - i010);
    let x01 = i001 + u.x * (i101 - i001);
    let x11 = i011 + u.x * (i111 - i011);

    let y0 = x00 + u.y * (x10 - x00);
    let y1 = x01 + u.y * (x11 - x01);

    y0 + u.z * (y1 - y0)
}
Verwende Code mit Vorsicht.

Schritt 2: Organische Verformung in der Dichtefunktion

Nun passen wir deine get_volumetric_density-Funktion an. Um fraktale Details (wie echte Wolken) zu erzeugen, nutzen wir zwei "Oktaven" des Rauschens. Gleichzeitig lassen wir das Rauschen über den time-Parameter entlang einer Windrichtung wandern, um eine fließende Bewegung zu simulieren.
rust
#[cube]
pub fn get_volumetric_density(p: Vec3, time: f32, size: f32) -> f32 {
    // 1. Basis-SDF holen (dein dynamischer Gyroid)
    let base_gyroid = evaluate_dynamic_gyroid(p, time, size);
    
    // 2. Wind-Vektor berechnen (Rauch wandert mit der Zeit nach oben und leicht zur Seite)
    let wind = Vec3::new(time * f32::new(0.2), time * f32::new(0.5), time * f32::new(0.1));
    let noise_pos = p.add(wind);

    // 3. Fraktales Rauschen (FBM - Fractional Brownian Motion) aus zwei Oktaven
    // Erste Oktave: Grobe Strukturen
    let noise_octave1 = noise_3d(noise_pos.scale(f32::new(3.0)));
    
    // Zweite Oktave: Feine Details (doppelte Frequenz, halbe Amplitude)
    let noise_octave2 = noise_3d(noise_pos.scale(f32::new(6.12)));
    let combined_noise = noise_octave1 * f32::new(0.7) + noise_octave2 * f32::new(0.3);

    // 4. Die mathematisch harte Kante des Gyroiden durch das Rauschen "aufbrechen"
    // Das Rauschen moduliert die Distanzgrenze des Gyroiden
    let modulated_sdf = base_gyroid + combined_noise * f32::new(0.15);
    
    // Invertieren und skalieren für das Dichtevolumen
    let density_scale = f32::new(6.0);
    (-modulated_sdf * density_scale).min(f32::new(1.0)).max(f32::new(0.0))
}
Verwende Code mit Vorsicht.

Was passiert hier architektonisch?

• Kein Textur-Sampling (Zero-Copy): Herkömmliche volumetrische Engines tasten riesige 3D-Texturen (Texture3D) ab. Das erzeugt massive Latenzen, da die GPU auf den Grafikspeicher (VRAM) warten muss. Da diese Implementierung rein auf mathematischen Formeln basiert (sin, floor), läuft sie komplett in den ALU-Rechenregistern ab.
• Strömungsdynamik: Durch das Aufschlagen des wind-Vektors auf die Positionskoordinaten vor der Rauschberechnung sieht es aus, als ob heißer Rauch aufsteigt und sich innerhalb der unsichtbaren Barriere des Gyroiden verwirbelt.
Dein System vereint nun alle Kern-Features des modernen Shader Bits-Volumen-Märschers: Eine gejitterte Startposition pro Pixel, Lichtstreuung über Henyey-Greenstein-Anisotropie, Eigenschatten (Self-Shadowing) und prozeduralen, strömenden Rauch – alles optimiert innerhalb der typsicheren Rust-Kompilierung von CubeCL.
Möchtest du dieses System mit einem Emissions-Wert koppeln, um z. B. glühende Magma-Adern oder eine Explosion im Inneren des volumetrischen Gyroiden aufleuchten zu lassen?
