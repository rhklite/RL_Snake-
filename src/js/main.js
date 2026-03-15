import { Game, STATE } from './game.js';
import { Renderer } from './renderer.js';
import { InputHandler } from './input.js';

/**
 * Bootstrap — wires Game, Renderer, and InputHandler together
 * and runs the variable-timestep game loop.
 */
const canvas = document.getElementById('game-canvas');
const scoreEl = document.getElementById('score');
const highScoreEl = document.getElementById('high-score');

const game = new Game();
const renderer = new Renderer(canvas);
const input = new InputHandler();

function updateScoreboard(score, highScore) {
  scoreEl.textContent = score;
  highScoreEl.textContent = highScore;
}

game.onScoreChange = updateScoreboard;
updateScoreboard(game.score, game.highScore);

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
});

input.bindTouch(canvas);

let lastTick = 0;

function loop(timestamp) {
  requestAnimationFrame(loop);

  if (game.state === STATE.RUNNING) {
    if (timestamp - lastTick >= game.tickInterval) {
      game.tick();
      lastTick = timestamp;
    }
  }

  render();
}

function render() {
  renderer.clear();
  renderer.drawSnake(game.snake);
  renderer.drawFood(game.food);
  renderer.drawScore(game.score, game.highScore);

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
        `Score: ${game.score}  —  Press Space to Retry`,
      );
      break;
  }
}

requestAnimationFrame(loop);
