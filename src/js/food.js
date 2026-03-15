import { GRID } from './constants.js';
import { Vec2 } from './vector.js';

/**
 * Handles food placement, ensuring it never spawns on the snake.
 */
export class Food {
  constructor() {
    this.position = new Vec2(0, 0);
  }

  /**
   * Place food on a random unoccupied cell.
   * @param {Set<string>} occupiedSet - set of "x,y" strings the snake covers.
   */
  spawn(occupiedSet) {
    const free = [];
    for (let x = 0; x < GRID.COLS; x++) {
      for (let y = 0; y < GRID.ROWS; y++) {
        if (!occupiedSet.has(`${x},${y}`)) {
          free.push(new Vec2(x, y));
        }
      }
    }
    if (free.length === 0) return false;
    this.position = free[Math.floor(Math.random() * free.length)];
    return true;
  }
}
