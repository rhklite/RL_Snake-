# Snake

A clean, modular Snake game built with vanilla JavaScript and HTML Canvas.  
No build tools required — open `src/index.html` in any modern browser.

## Quick Start

```bash
# Option A: open directly
open src/index.html          # macOS
xdg-open src/index.html      # Linux

# Option B: any static file server
npx serve src
python3 -m http.server -d src
```

## Controls

| Action        | Keys                       |
|---------------|----------------------------|
| Move          | Arrow keys / WASD          |
| Start / Retry | Space / Enter              |
| Pause         | Esc / P (or Space while playing) |
| Mobile        | Swipe to move, tap to start|

## Project Structure

```
src/
├── index.html          Entry point
├── css/
│   └── style.css       UI styling (dark theme, responsive)
└── js/
    ├── main.js         Bootstrap & game loop
    ├── game.js         State machine, scoring, speed ramp
    ├── snake.js        Snake body, direction queue, collision
    ├── food.js         Food spawning on unoccupied cells
    ├── renderer.js     Canvas drawing (gradient body, pulse food)
    ├── input.js        Keyboard & touch/swipe input
    ├── particles.js    Particle effects (eat burst)
    ├── events.js       Lightweight pub/sub EventEmitter
    ├── vector.js       Immutable Vec2 helper
    └── constants.js    Shared configuration & colours
```

## Architecture

Each module has a single responsibility and communicates through a minimal interface:

- **Game** — owns `Snake` and `Food`, manages state transitions (`idle → running ↔ paused → gameover`), and emits events (`score`, `state`, `eat`).
- **Renderer** — stateless per frame; given data each frame, draws it on a canvas. Features gradient body colouring and pulsing food glow.
- **InputHandler** — translates raw keyboard/touch events into direction, action, and pause callbacks. Fully swappable.
- **ParticleSystem** — fire-and-forget particle bursts for eat feedback.
- **Snake** — body segments, direction queue (up to 3 buffered turns), growth, and self-collision detection.
- **Food** — spawns on a random free cell; returns `false` when the board is full (win condition).
- **Vec2** — immutable 2D vector with `add`, `equals`, `wrap`.
- **EventEmitter** — lightweight pub/sub so `Game` stays decoupled from rendering and UI.

## Features

- Wrap-around movement (edges connect)
- Progressive speed increase as score rises
- Direction queue so rapid key presses between ticks are not lost
- Gradient snake body (head → tail colour fade)
- Pulsing food glow animation
- Particle burst on eating
- Persistent high score (localStorage)
- Responsive canvas (scales on small screens)
- Mobile touch/swipe support

## License

MIT
