import { GRID, DIRECTION } from './constants.js';
import { Vec2 } from './vector.js';

/**
 * Manages the snake's body segments, direction, and movement rules.
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
    this._nextDirection = Vec2.from(DIRECTION.RIGHT);
    this._growPending = 0;
  }

  /** Queue a direction change (prevents 180-degree reversal). */
  setDirection(dir) {
    const next = Vec2.from(dir);
    if (next.add(this.direction).equals(new Vec2(0, 0))) return;
    this._nextDirection = next;
  }

  /** Advance one step. Returns the new head position. */
  step() {
    this.direction = this._nextDirection;
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

  /** True when the head overlaps any body segment. */
  get hasSelfCollision() {
    return this.segments
      .slice(1)
      .some((seg) => seg.equals(this.head));
  }

  /** Set of "x,y" strings for O(1) occupancy checks. */
  get occupiedSet() {
    return new Set(this.segments.map((s) => s.toString()));
  }
}
