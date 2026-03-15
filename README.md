# Snake

A clean, modular Snake game built with vanilla JavaScript and HTML Canvas.  
Supports both **human play** and **AI inference** via ONNX Runtime Web.  
No build tools required — open `src/index.html` in any modern browser.

## Quick Start

```bash
# Option A: open directly
open src/index.html          # macOS
xdg-open src/index.html      # Linux

# Option B: any static file server (needed for ONNX loading)
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

## AI Mode

1. Switch the **Mode** dropdown to **AI (ONNX)**
2. Click **Load ONNX model** and select your `.onnx` file
3. Press **Space** to start — the AI drives the snake
4. Adjust the **Speed** slider to control tick rate (1–60 tps)

### Model Contract

The AI controller expects an ONNX model with this interface:

| | Shape | Type | Description |
|---|---|---|---|
| **Input** | `[1, 11]` | `float32` | Feature vector (see below) |
| **Output** | `[1, 3]` | `float32` | Logits for `[straight, right, left]` |

**11 input features** (all binary 0/1):

| Index | Feature |
|-------|---------|
| 0 | Danger straight (collision one step ahead) |
| 1 | Danger right |
| 2 | Danger left |
| 3 | Direction is left |
| 4 | Direction is right |
| 5 | Direction is up |
| 6 | Direction is down |
| 7 | Food is to the left of head |
| 8 | Food is to the right of head |
| 9 | Food is above head |
| 10 | Food is below head |

**3 output actions** (relative to current heading):

| Index | Action |
|-------|--------|
| 0 | Go straight |
| 1 | Turn right |
| 2 | Turn left |

### Exporting from PyTorch

```python
import torch

# After training, export with matching dimensions:
dummy = torch.randn(1, 11)
torch.onnx.export(
    model,
    dummy,
    "models/snake_dqn.onnx",
    input_names=["state"],
    output_names=["action_logits"],
    dynamic_axes={"state": {0: "batch"}, "action_logits": {0: "batch"}},
    opset_version=17,
)
```

The Python training environment (on its own branch) must use the **same 11-feature
encoding and 3-action space** defined in `src/js/state-encoder.js`.

### Custom State Encoders

If your model uses a different state representation (e.g., grid-based for a CNN),
subclass or replace `StateEncoder` in `src/js/state-encoder.js`. The AI controller
only cares about two methods:

- `encode(snake, food) → Float32Array`
- `actionToDirection(actionIdx, currentDir) → {x, y}`

## Project Structure

```
src/
├── index.html              Entry point
├── css/
│   └── style.css           UI styling (dark theme, responsive)
└── js/
    ├── main.js             Bootstrap, game loop, mode switching
    ├── game.js             State machine, scoring, speed ramp
    ├── snake.js            Snake body, direction queue, collision
    ├── food.js             Food spawning on unoccupied cells
    ├── renderer.js         Canvas drawing (gradient body, pulse food)
    ├── input.js            Keyboard & touch/swipe input
    ├── ai-controller.js    ONNX inference controller
    ├── state-encoder.js    Game state → feature vector encoding
    ├── particles.js        Particle effects (eat burst)
    ├── events.js           Lightweight pub/sub EventEmitter
    ├── vector.js           Immutable Vec2 helper
    └── constants.js        Shared configuration & colours
models/                     Place exported .onnx files here
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    main.js                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │
│  │  Input    │  │  AI      │  │  Renderer     │ │
│  │  Handler  │  │  Control │  │               │ │
│  └────┬─────┘  └────┬─────┘  └───────────────┘ │
│       │              │                           │
│       ▼              ▼                           │
│  ┌──────────────────────┐   ┌──────────────┐    │
│  │       Game            │──▶│  Particles   │    │
│  │  ┌───────┐ ┌───────┐ │   └──────────────┘    │
│  │  │ Snake │ │ Food  │ │                        │
│  │  └───────┘ └───────┘ │                        │
│  └──────────────────────┘                        │
└─────────────────────────────────────────────────┘
         ▲
         │  encode()
  ┌──────┴──────┐
  │   State     │
  │   Encoder   │
  └─────────────┘
```

- **Game** — owns `Snake` and `Food`, manages state transitions, emits events.
- **Renderer** — stateless per frame; draws canvas. Gradient body, pulsing food.
- **InputHandler** — keyboard/touch → direction/action/pause callbacks (human mode).
- **AIController** — loads ONNX, encodes state, runs inference, sets direction (AI mode).
- **StateEncoder** — converts live game state to the feature vector the model expects. Swappable for different model architectures.
- **ParticleSystem** — visual feedback on eat events.
- **Snake / Food / Vec2** — pure data-and-logic, no DOM dependency.

## License

MIT
