# Snake

A clean, modular Snake game built with vanilla JavaScript and HTML Canvas.  
No build tools required — open `src/index.html` in any modern browser.

## Quick Start

```bash
# Option A: just open the file
open src/index.html          # macOS
xdg-open src/index.html      # Linux

# Option B: use any static file server
npx serve src
```

## Controls

| Action        | Keys                       |
|---------------|----------------------------|
| Move          | Arrow keys / WASD          |
| Start / Retry | Space / Enter              |
| Pause         | Space (while playing)      |
| Mobile        | Swipe to move, tap to start|

## Project Structure

```
src/
├── index.html          Entry point
├── css/
│   └── style.css       UI styling
└── js/
    ├── main.js         Bootstrap & game loop
    ├── game.js         State machine & scoring
    ├── snake.js        Snake body & movement
    ├── food.js         Food spawning
    ├── renderer.js     Canvas drawing
    ├── input.js        Keyboard & touch input
    ├── vector.js       Immutable Vec2 helper
    └── constants.js    Shared configuration
```

## Architecture

Each module has a single responsibility and communicates through a minimal interface:

- **Game** — owns `Snake` and `Food`, manages state transitions and scoring.
- **Renderer** — stateless; given data each frame, draws it on a canvas.
- **InputHandler** — translates raw events into direction / action callbacks.
- **Snake / Food / Vec2** — pure data-and-logic classes with no DOM dependency.

## License

MIT
