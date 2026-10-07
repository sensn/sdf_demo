Da wgpui als Render-Backend eins-zu-eins auf Zeds GPUI aufbaut, stammen alle Maus-Interaktionen, Event-Strukturen und Signaturen direkt aus der interaktiven Event-Schicht von gpui. [1, 2] 
Die offizielle Dokumentation dieser Events und Signaturen lässt sich in der [GPUI docs.rs Dokumentation](https://docs.rs/gpui/latest/gpui/) einsehen. [3] 
Hier ist die vollständige Übersicht aller verfügbaren Maus-Event-Methoden, die du auf einem div() verketten kannst, inklusive ihrer genauen Signaturen und Datenstrukturen:
------------------------------
## 1. Alle verfügbaren Maus-Event-Methoden (Element-API)
Diese Methoden stehen dir beim deklarativen Bauen deines UI-Baums zur Verfügung. Wenn sie mit einem cx.listener aufgerufen werden, ändert sich die Signatur der Closure so, dass als erster Parameter eine veränderbare Referenz auf deine View (this: &mut Self) übergeben wird. [4] 

* 
* .on_mouse_move(f)
* Feuert: Kontinuierlich, wenn sich die Maus über dem Element bewegt.
   * Signatur: FnMut(&MouseMoveEvent, &mut Window, &mut Context<T>) [5] 
* .on_mouse_down(button, f)
* Feuert: In dem Moment, in dem eine bestimmte Maustaste auf dem Element gedrückt wird.
   * Zusatz-Parameter: Benötigt eine MouseButton (z. B. MouseButton::Left) vor der Closure.
   * Signatur: FnMut(&MouseDownEvent, &mut Window, &mut Context<T>) [4] 
* .on_mouse_up(button, f)
* Feuert: Wenn die Maustaste über dem Element losgelassen wird.
   * Signatur: FnMut(&MouseUpEvent, &mut Window, &mut Context<T>) [4] 
* .on_mouse_up_out(button, f)
* Feuert: Wenn die Taste gedrückt wurde, die Maus das Element verlassen hat und außerhalb losgelassen wird.
   * Signatur: FnMut(&MouseUpEvent, &mut Window, &mut Context<T>) [4] 
* .on_click(f)
* Feuert: Bequemlichkeits-Wrapper für einen vollständigen Klick (Down + Up) mit der linken Maustaste.
   * Signatur: FnMut(&ClickEvent, &mut Window, &mut Context<T>)
* .on_hover(f)
* Feuert: Wenn der Cursor das Element betritt oder verlässt.
   * Signatur: FnMut(&bool, &mut Window, &mut Context<T>) (Übergibt true bei Betreten, false bei Verlassen). [5] 
* .on_scroll_wheel(f)
* Feuert: Wenn das Mausrad/Trackpad über dem Element bewegt wird.
   * Signatur: FnMut(&ScrollWheelEvent, &mut Window, &mut Context<T>)
* 

------------------------------
## 2. Die exakten Event-Strukturen und Signaturen (Felder)
Hier siehst du genau, welche Daten in den jeweiligen Events stecken, die du in deiner .on_mouse_move-Closure auslesen kannst.
## MouseMoveEvent [3] 

pub struct MouseMoveEvent {
    /// Die absolute Position der Maus auf dem Fenster.
    pub position: Point<Pixels>,
    
    /// Die Maustaste, die während der Bewegung gedrückt gehalten wird (falls vorhanden).
    pub pressed_button: Option<MouseButton>,
    
    /// Aktive Tastatur-Modifikatoren (Shift, Ctrl, Alt, Meta).
    pub modifiers: Modifiers,
}

## MouseDownEvent / MouseUpEvent

pub struct MouseDownEvent {
    /// Position beim Klicken.
    pub position: Point<Pixels>,
    
    /// Welche Taste gedrückt wurde (Left, Right, Middle, etc.).
    pub button: MouseButton,
    
    /// Wie oft schnell hintereinander geklickt wurde (z.B. 2 für Doppelklick).
    pub click_count: usize,
    
    pub modifiers: Modifiers,
}// `MouseUpEvent` besitzt exakt dieselben Felder.

## ScrollWheelEvent

pub struct ScrollWheelEvent {
    pub position: Point<Pixels>,
    
    /// Das relative Scroll-Delta (wgpui_kit::Pixels) auf der X- und Y-Achse.
    pub delta: Point<Pixels>,
    
    /// Gibt an, ob pixelgenau (Trackpad) oder in Zeilen/Rastern (Mausrad) gescrollt wurde.
    pub pixel_delta: bool,
    
    pub modifiers: Modifiers,
}

------------------------------
## 3. Hilfstypen des Frameworks

* 
* MouseButton (Enum):
* MouseButton::Left
   * MouseButton::Right
   * MouseButton::Middle
   * MouseButton::NavigateBack / NavigateForward [4] 
* Point<T> (Generic Struct):
* Besitzt die Felder .x und .y vom Typ T. In den Maus-Events ist T immer wgpui_kit::Pixels. [3] 
* 

------------------------------
## Wo du das im Quellcode findest:
Da du das Projekt lokal ausgecheckt hast, kannst du die kompletten Definitionen und Traits direkt in den Quelldateien deiner installierten Crates nachschlagen:

   1. Schau in die Datei src/interactive.rs innerhalb der gpui / wgpui Crate (~/.cargo/registry/src/.../gpui-*/src/interactive.rs). Dort sind alle Event-Structs definiert.
   2. In der Datei src/element.rs bzw. direkt auf dem InteractiveElement-Trait im Framework-Root findest du alle verkettbaren Event-Methoden (.on_mouse_move, etc.). [2] 

Möchtest du als Nächstes ein MouseDown oder Click Event auf der 3D-Fläche registrieren, um z.B. den First-Person-Modus erst per Klick zu aktivieren und den Mauszeiger zu verstecken?

[1] [https://docs.rs](https://docs.rs/gpui)
[2] [https://docs.rs](https://docs.rs/gpui-unofficial/latest/src/gpui/interactive.rs.html)
[3] [https://docs.rs](https://docs.rs/gpui/latest/gpui/struct.MouseMoveEvent.html)
[4] [https://github.com](https://github.com/zed-industries/zed/blob/main/crates/gpui/examples/input.rs)
[5] [https://lobehub.com](https://lobehub.com/it/skills/cnwzhu-gpui-skills-gpui-actions)
