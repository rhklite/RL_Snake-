import { Game, STATE } from './game.js';
import { Renderer } from './renderer.js';
import { InputHandler } from './input.js';
import { AIController } from './ai-controller.js';
import { ParticleSystem } from './particles.js';
import { COLORS } from './constants.js';

/* ---- DOM refs ---- */
const canvas      = document.getElementById('game-canvas');
const scoreEl     = document.getElementById('score');
const highScoreEl = document.getElementById('high-score');
const modeToggle  = document.getElementById('mode-toggle');
const modelPicker = document.getElementById('model-picker');
const modelStatus = document.getElementById('model-status');
const speedSlider = document.getElementById('ai-speed');
const speedLabel  = document.getElementById('ai-speed-label');
const aiPanel     = document.getElementById('ai-panel');

/* ---- Core objects ---- */
const game      = new Game();
const renderer  = new Renderer(canvas);
const input     = new InputHandler();
const particles = new ParticleSystem();
const ai        = new AIController();

let mode = 'human';           // 'human' | 'ai'
let aiTickRate = 10;           // ticks/sec when AI is playing
let aiPending = false;         // guards against overlapping async inferences

/* ---- Score events ---- */
game.on('score', (s, hs) => {
  scoreEl.textContent = s;
  highScoreEl.textContent = hs;
});

game.on('eat', (pos) => {
  particles.burst(pos.x, pos.y, COLORS.FOOD, 10);
});

scoreEl.textContent = game.score;
highScoreEl.textContent = game.highScore;

/* ---- Human input ---- */
input.bind({
  onDirection(dir) {
    if (mode === 'human' && game.state === STATE.RUNNING) {
      game.snake.setDirection(dir);
    }
  },
  onAction() {
    if (game.state === STATE.IDLE || game.state === STATE.GAME_OVER) {
      game.start();
    } else if (mode === 'human' &&
               (game.state === STATE.RUNNING || game.state === STATE.PAUSED)) {
      game.togglePause();
    }
  },
  onPause() {
    if (mode === 'human' &&
        (game.state === STATE.RUNNING || game.state === STATE.PAUSED)) {
      game.togglePause();
    }
  },
});

input.bindTouch(canvas);

/* ---- Mode toggle ---- */
modeToggle.addEventListener('change', () => {
  setMode(modeToggle.value);
});

function setMode(newMode) {
  mode = newMode;
  modeToggle.value = mode;
  aiPanel.hidden = (mode !== 'ai');

  if (game.state === STATE.RUNNING || game.state === STATE.PAUSED) {
    game.state = STATE.GAME_OVER;
  }
}

/* ---- Model file picker ---- */
modelPicker.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  modelStatus.textContent = 'Loading\u2026';
  try {
    ai.dispose();
    await ai.loadModel(file);
    modelStatus.textContent = `Loaded: ${file.name}`;
  } catch (err) {
    modelStatus.textContent = `Error: ${err.message}`;
    console.error(err);
  }
});

/* ---- AI speed slider ---- */
speedSlider?.addEventListener('input', () => {
  aiTickRate = Number(speedSlider.value);
  speedLabel.textContent = `${aiTickRate} tps`;
});

/* ---- Game loop ---- */
let lastTick  = 0;
let lastFrame = 0;

function loop(timestamp) {
  requestAnimationFrame(loop);

  const dt = Math.min((timestamp - lastFrame) / 1000, 0.1);
  lastFrame = timestamp;

  if (game.state === STATE.RUNNING) {
    const interval = mode === 'ai' ? 1000 / aiTickRate : game.tickInterval;

    if (timestamp - lastTick >= interval) {
      if (mode === 'ai' && ai.ready) {
        if (!aiPending) {
          aiPending = true;
          ai.step(game).then(() => {
            game.tick();
            aiPending = false;
          });
          lastTick = timestamp;
        }
      } else if (mode === 'human') {
        game.tick();
        lastTick = timestamp;
      }
    }
  }

  particles.update(dt);
  render(timestamp);
}

function render(timestamp) {
  renderer.clear();
  renderer.drawSnake(game.snake);
  renderer.drawFood(game.food, timestamp);
  particles.draw(renderer.ctx);

  switch (game.state) {
    case STATE.IDLE:
      renderer.drawOverlay('Snake', overlayHint());
      break;
    case STATE.PAUSED:
      renderer.drawOverlay('Paused', 'Press Space to Resume');
      break;
    case STATE.GAME_OVER:
      renderer.drawOverlay(
        'Game Over',
        `Score: ${game.score}  \u2014  ${overlayHint()}`,
      );
      break;
  }
}

function overlayHint() {
  if (mode === 'ai' && !ai.ready) return 'Load an ONNX model first';
  return 'Press Space to Start';
}

requestAnimationFrame(loop);
