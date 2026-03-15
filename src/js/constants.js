/**
 * Centralised configuration for the Snake game.
 * All magic numbers live here so every module can import what it needs.
 */

export const GRID = Object.freeze({
  COLS: 20,
  ROWS: 20,
  CELL_SIZE: 24,
});

export const CANVAS = Object.freeze({
  WIDTH: GRID.COLS * GRID.CELL_SIZE,
  HEIGHT: GRID.ROWS * GRID.CELL_SIZE,
});

export const DIRECTION = Object.freeze({
  UP: { x: 0, y: -1 },
  DOWN: { x: 0, y: 1 },
  LEFT: { x: -1, y: 0 },
  RIGHT: { x: 1, y: 0 },
});

export const INITIAL_SPEED = 8;
export const SPEED_INCREMENT = 0.4;
export const MAX_SPEED = 20;
export const POINTS_PER_FOOD = 10;

export const COLORS = Object.freeze({
  BG: '#1a1a2e',
  GRID_LINE: 'rgba(255, 255, 255, 0.03)',
  SNAKE_HEAD: '#00d2ff',
  SNAKE_BODY: '#0f9b8e',
  SNAKE_STROKE: '#0a0a1a',
  FOOD: '#ff6b6b',
  FOOD_GLOW: 'rgba(255, 107, 107, 0.35)',
  TEXT: '#e0e0e0',
  OVERLAY: 'rgba(10, 10, 26, 0.75)',
});

export const KEY_MAP = Object.freeze({
  ArrowUp: DIRECTION.UP,
  ArrowDown: DIRECTION.DOWN,
  ArrowLeft: DIRECTION.LEFT,
  ArrowRight: DIRECTION.RIGHT,
  w: DIRECTION.UP,
  s: DIRECTION.DOWN,
  a: DIRECTION.LEFT,
  d: DIRECTION.RIGHT,
  W: DIRECTION.UP,
  S: DIRECTION.DOWN,
  A: DIRECTION.LEFT,
  D: DIRECTION.RIGHT,
});
