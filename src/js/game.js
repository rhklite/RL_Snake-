import {
  INITIAL_SPEED,
  SPEED_INCREMENT,
  MAX_SPEED,
  POINTS_PER_FOOD,
} from './constants.js';
import { Snake } from './snake.js';
import { Food } from './food.js';

/**
 * Core game state machine.
 * States: idle → running → paused / gameover → idle
 *
 * The Game class owns the Snake and Food but knows nothing about
 * rendering or input — those are plugged in externally.
 */
export const STATE = Object.freeze({
  IDLE: 'idle',
  RUNNING: 'running',
  PAUSED: 'paused',
  GAME_OVER: 'gameover',
});

export class Game {
  constructor() {
    this.snake = new Snake();
    this.food = new Food();
    this.state = STATE.IDLE;
    this.score = 0;
    this.highScore = this._loadHighScore();
    this.speed = INITIAL_SPEED;
    this._onScoreChange = null;
  }

  /** Optional callback when score/highScore change (for UI). */
  set onScoreChange(fn) {
    this._onScoreChange = fn;
  }

  start() {
    this.snake.reset();
    this.food.spawn(this.snake.occupiedSet);
    this.score = 0;
    this.speed = INITIAL_SPEED;
    this.state = STATE.RUNNING;
    this._emitScore();
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
   * Returns true if the frame resulted in a meaningful state change.
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

      if (!this.food.spawn(this.snake.occupiedSet)) {
        this._endGame();
        return true;
      }

      this._emitScore();
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
    this._emitScore();
  }

  _emitScore() {
    this._onScoreChange?.(this.score, this.highScore);
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
