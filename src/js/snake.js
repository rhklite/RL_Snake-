import { GRID, DIRECTION } from './constants.js';
import { Vec2 } from './vector.js';

const MAX_QUEUED_TURNS = 3;

/**
 * Manages the snake's body segments, direction, and movement rules.
 * Supports a turn queue so rapid key presses between ticks are not lost.
 */
export class Snake {
  constructor() {
    this.reset();
  }

  reset() {
    const midX = Math.floor(GRID.COLS / 2);
    const midY = Math.floor(GRID.ROWS / 2);
    this.segments = [
      new Vec2(midX, midY),
      new Vec2(midX - 1, midY),
      new Vec2(midX - 2, midY),
    ];
    this.direction = Vec2.from(DIRECTION.RIGHT);
    this._turnQueue = [];
    this._growPending = 0;
  }

  /**
   * Queue a direction change.
   * Prevents 180-degree reversals relative to the last queued (or current)
   * direction, so fast "up-left" sequences between a single tick both register.
   */
  setDirection(dir) {
    if (this._turnQueue.length >= MAX_QUEUED_TURNS) return;

    const next = Vec2.from(dir);
    const ref =
      this._turnQueue.length > 0
        ? this._turnQueue[this._turnQueue.length - 1]
        : this.direction;

    if (next.add(ref).equals(new Vec2(0, 0))) return;
    if (next.equals(ref)) return;

    this._turnQueue.push(next);
  }

  /** Advance one step. Returns the new head position. */
  step() {
    if (this._turnQueue.length > 0) {
      this.direction = this._turnQueue.shift();
    }

    const head = this.head.add(this.direction).wrap(GRID.COLS, GRID.ROWS);
    this.segments.unshift(head);

    if (this._growPending > 0) {
      this._growPending--;
    } else {
      this.segments.pop();
    }

    return head;
  }

  grow(amount = 1) {
    this._growPending += amount;
  }

  get head() {
    return this.segments[0];
  }

  get length() {
    return this.segments.length;
  }

  /** True when the head overlaps any body segment. */
  get hasSelfCollision() {
    const h = this.head;
    for (let i = 1; i < this.segments.length; i++) {
      if (this.segments[i].equals(h)) return true;
    }
    return false;
  }

  /** Set of "x,y" strings for O(1) occupancy checks. */
  get occupiedSet() {
    return new Set(this.segments.map((s) => s.toString()));
  }
}
