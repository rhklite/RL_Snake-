import {
  INITIAL_SPEED,
  SPEED_INCREMENT,
  MAX_SPEED,
  POINTS_PER_FOOD,
} from './constants.js';
import { Snake } from './snake.js';
import { Food } from './food.js';
import { EventEmitter } from './events.js';

/**
 * Core game state machine.
 *
 *   idle ──▶ running ──▶ gameover
 *               │  ▲         │
 *               ▼  │         │
 *            paused          │
 *               ◄────────────┘ (restart)
 *
 * Emits:
 *   "score"     (score, highScore)
 *   "state"     (newState, prevState)
 *   "eat"       ()
 */
export const STATE = Object.freeze({
  IDLE: 'idle',
  RUNNING: 'running',
  PAUSED: 'paused',
  GAME_OVER: 'gameover',
});

export class Game extends EventEmitter {
  constructor() {
    super();
    this.snake = new Snake();
    this.food = new Food();
    this._state = STATE.IDLE;
    this.score = 0;
    this.highScore = this._loadHighScore();
    this.speed = INITIAL_SPEED;
  }

  get state() {
    return this._state;
  }

  set state(next) {
    const prev = this._state;
    if (prev === next) return;
    this._state = next;
    this.emit('state', next, prev);
  }

  start() {
    this.snake.reset();
    this.food.spawn(this.snake.occupiedSet);
    this.score = 0;
    this.speed = INITIAL_SPEED;
    this.state = STATE.RUNNING;
    this.emit('score', this.score, this.highScore);
  }

  togglePause() {
    if (this.state === STATE.RUNNING) {
      this.state = STATE.PAUSED;
    } else if (this.state === STATE.PAUSED) {
      this.state = STATE.RUNNING;
    }
  }

  /**
   * Advance the simulation by one tick.
   * @returns {boolean} true when something happened.
   */
  tick() {
    if (this.state !== STATE.RUNNING) return false;

    this.snake.step();

    if (this.snake.hasSelfCollision) {
      this._endGame();
      return true;
    }

    if (this.snake.head.equals(this.food.position)) {
      this.snake.grow();
      this.score += POINTS_PER_FOOD;
      this.speed = Math.min(MAX_SPEED, this.speed + SPEED_INCREMENT);
      this.emit('eat', this.food.position);

      if (!this.food.spawn(this.snake.occupiedSet)) {
        this._endGame();
        return true;
      }

      this.emit('score', this.score, this.highScore);
    }

    return true;
  }

  /** Milliseconds between ticks derived from current speed. */
  get tickInterval() {
    return 1000 / this.speed;
  }

  _endGame() {
    this.state = STATE.GAME_OVER;
    if (this.score > this.highScore) {
      this.highScore = this.score;
      this._saveHighScore();
    }
    this.emit('score', this.score, this.highScore);
  }

  _loadHighScore() {
    try {
      return parseInt(localStorage.getItem('snake_high_score') ?? '0', 10);
    } catch {
      return 0;
    }
  }

  _saveHighScore() {
    try {
      localStorage.setItem('snake_high_score', String(this.highScore));
    } catch { /* storage unavailable */ }
  }
}
