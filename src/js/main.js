import { Game, STATE } from './game.js';
import { Renderer } from './renderer.js';
import { InputHandler } from './input.js';
import { ParticleSystem } from './particles.js';
import { COLORS } from './constants.js';

/**
 * Bootstrap — wires Game, Renderer, InputHandler, and ParticleSystem
 * together and runs a variable-timestep game loop.
 */
const canvas = document.getElementById('game-canvas');
const scoreEl = document.getElementById('score');
const highScoreEl = document.getElementById('high-score');

const game = new Game();
const renderer = new Renderer(canvas);
const input = new InputHandler();
const particles = new ParticleSystem();

game.on('score', (score, highScore) => {
  scoreEl.textContent = score;
  highScoreEl.textContent = highScore;
});

game.on('eat', (pos) => {
  particles.burst(pos.x, pos.y, COLORS.FOOD, 10);
});

scoreEl.textContent = game.score;
highScoreEl.textContent = game.highScore;

input.bind({
  onDirection(dir) {
    if (game.state === STATE.RUNNING) {
      game.snake.setDirection(dir);
    }
  },
  onAction() {
    if (game.state === STATE.IDLE || game.state === STATE.GAME_OVER) {
      game.start();
    } else if (game.state === STATE.RUNNING || game.state === STATE.PAUSED) {
      game.togglePause();
    }
  },
  onPause() {
    if (game.state === STATE.RUNNING || game.state === STATE.PAUSED) {
      game.togglePause();
    }
  },
});

input.bindTouch(canvas);

let lastTick = 0;
let lastFrame = 0;

function loop(timestamp) {
  requestAnimationFrame(loop);

  const dt = Math.min((timestamp - lastFrame) / 1000, 0.1);
  lastFrame = timestamp;

  if (game.state === STATE.RUNNING) {
    if (timestamp - lastTick >= game.tickInterval) {
      game.tick();
      lastTick = timestamp;
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
      renderer.drawOverlay('Snake', 'Press Space to Start');
      break;
    case STATE.PAUSED:
      renderer.drawOverlay('Paused', 'Press Space to Resume');
      break;
    case STATE.GAME_OVER:
      renderer.drawOverlay(
        'Game Over',
        `Score: ${game.score}  \u2014  Press Space to Retry`,
      );
      break;
  }
}

requestAnimationFrame(loop);
