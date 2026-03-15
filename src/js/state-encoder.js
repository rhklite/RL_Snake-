import { GRID, DIRECTION } from './constants.js';
import { Vec2 } from './vector.js';

/**
 * Encodes game state into a flat Float32Array suitable for model inference.
 *
 * Default encoding (11 features, all binary 0/1):
 *   [0] danger straight   — collision one step ahead in current direction
 *   [1] danger right       — collision one step to the right of current dir
 *   [2] danger left        — collision one step to the left of current dir
 *   [3] direction left     — snake is currently heading left
 *   [4] direction right
 *   [5] direction up
 *   [6] direction down
 *   [7] food left          — food is to the left of the head
 *   [8] food right
 *   [9] food up
 *  [10] food down
 *
 * The Python training environment must produce the same encoding.
 * Swap this class out (or subclass it) for grid-based / CNN encodings.
 */

const DIR_UP    = Vec2.from(DIRECTION.UP);
const DIR_DOWN  = Vec2.from(DIRECTION.DOWN);
const DIR_LEFT  = Vec2.from(DIRECTION.LEFT);
const DIR_RIGHT = Vec2.from(DIRECTION.RIGHT);

const CLOCKWISE = [DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT];

export class StateEncoder {
  constructor() {
    this.featureSize = 11;
  }

  /**
   * @param {import('./snake.js').Snake} snake
   * @param {import('./food.js').Food}   food
   * @returns {Float32Array} feature vector of length `this.featureSize`
   */
  encode(snake, food) {
    const head = snake.head;
    const dir  = snake.direction;
    const occupied = snake.occupiedSet;

    const cwIdx   = CLOCKWISE.findIndex((d) => d.equals(dir));
    const dirR    = CLOCKWISE[(cwIdx + 1) % 4];
    const dirL    = CLOCKWISE[(cwIdx + 3) % 4];

    const out = new Float32Array(this.featureSize);

    out[0] = this._isDanger(head, dir, occupied)  ? 1 : 0;
    out[1] = this._isDanger(head, dirR, occupied) ? 1 : 0;
    out[2] = this._isDanger(head, dirL, occupied) ? 1 : 0;

    out[3] = dir.equals(DIR_LEFT)  ? 1 : 0;
    out[4] = dir.equals(DIR_RIGHT) ? 1 : 0;
    out[5] = dir.equals(DIR_UP)    ? 1 : 0;
    out[6] = dir.equals(DIR_DOWN)  ? 1 : 0;

    out[7]  = food.position.x < head.x ? 1 : 0;
    out[8]  = food.position.x > head.x ? 1 : 0;
    out[9]  = food.position.y < head.y ? 1 : 0;
    out[10] = food.position.y > head.y ? 1 : 0;

    return out;
  }

  /**
   * Map a model action index back to an absolute direction vector.
   *
   * Actions (relative to current heading):
   *   0 = go straight
   *   1 = turn right
   *   2 = turn left
   *
   * @param {number} actionIdx
   * @param {Vec2}   currentDir - snake's current direction
   * @returns {{ x: number, y: number }}
   */
  actionToDirection(actionIdx, currentDir) {
    const cwIdx = CLOCKWISE.findIndex((d) => d.equals(currentDir));
    switch (actionIdx) {
      case 0: return CLOCKWISE[cwIdx];
      case 1: return CLOCKWISE[(cwIdx + 1) % 4];
      case 2: return CLOCKWISE[(cwIdx + 3) % 4];
      default: return CLOCKWISE[cwIdx];
    }
  }

  /** Check whether moving one step in `dir` from `pos` hits the snake body. */
  _isDanger(pos, dir, occupied) {
    const next = pos.add(dir).wrap(GRID.COLS, GRID.ROWS);
    return occupied.has(next.toString());
  }
}
